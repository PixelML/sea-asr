# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Optional KenLM n-gram beam search over the CTC log-probs (pyctcdecode). Install with `pip install sea_asr[lm]`.

LM directory layout:
    lm_config.json   {"alpha": 0.4, "beta": 1.5, "beam": 128, "rescore_beam": 128}   optional: "model" (file name)
                     beam = plain LM decode; rescore_beam = beam that builds the N-best list for sea_asr.rescore;
                     optional "compress_blanks": false (default true, see compress_blanks())
    <name>.bin       KenLM binary (or .arpa); picked by "model", else the only .bin / .arpa in the directory
    unigrams.txt     one word per line (optional; without it pyctcdecode cannot score partial words)

The unit labels are the sentencepiece pieces of the Gemma ids ("▁" = word start), so pyctcdecode runs in BPE mode.
Unit 0 (blank) is "" and unit 1 (unk) is "⁇", a plain word-internal piece: a word that takes it becomes an
out-of-vocabulary word for the LM, so the beam avoids unk (0 of 60 FLEURS-id hypotheses kept one). Any "⁇" left is
removed from the output, so unk carries no text, as in greedy decoding. Do not label unk "<unk>": pyctcdecode 0.5.0
then uses its "▁⁇▁" token, whose trailing "▁" sets one force_next_break flag shared by every beam in the frame loop
and splits words in unrelated beams (seen on FLEURS-id: "prefektur" -> "prefekt").
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .decode import FRAME_S

DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_BEAM = 0.5, 1.0, 32
UNK_WORD = "⁇"
log = logging.getLogger("sea_asr")


def usable_cpus(cgroup_dir: str | Path = "/sys/fs/cgroup") -> int:
    """CPUs this process may use: the affinity mask (else os.cpu_count()), capped by a cgroup CPU quota when one is
    readable (v2 cpu.max, else v1 cpu.cfs_quota_us / cpu.cfs_period_us), so a container is not oversubscribed."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # macOS has no sched_getaffinity
        n = os.cpu_count() or 1
    d = Path(cgroup_dir)
    quota = period = None
    try:
        q, p = (d / "cpu.max").read_text().split()[:2]
        if q != "max":
            quota, period = int(q), int(p)
    except (OSError, ValueError):
        try:
            q = int((d / "cpu" / "cpu.cfs_quota_us").read_text())
            if q > 0:
                quota, period = q, int((d / "cpu" / "cpu.cfs_period_us").read_text())
        except (OSError, ValueError):
            pass
    if quota and period:
        n = min(n, max(1, math.ceil(quota / period)))
    return max(1, n)


_logged_workers = False

# Pooled decoding: the parent copies a batch's dense log-probs once into one shared-memory block; each worker maps
# its utterance as a read-only view and runs pyctcdecode's public decode() / decode_beams() on it. Nothing large is
# pickled, the per-utterance work (pyctcdecode's own re-softmax included) stays in the workers, and the result is
# pyctcdecode's own result by construction.
_POOL_DECODERS: dict[int, object] = {}  # filled before the fork; the children find their decoder here


def _shm_work(job):
    from multiprocessing import shared_memory

    key, top1, beam, name, offset, shape, dtype = job
    # fork children share the parent's resource tracker: attaching re-registers the same name (a no-op) and the
    # parent's unlink() unregisters it once, so the child must not unregister it
    shm = shared_memory.SharedMemory(name=name)
    try:
        x = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf, offset=offset)
        x.flags.writeable = False
        dec = _POOL_DECODERS[key]
        if top1:
            out = dec.decode(x, beam_width=beam)
        else:  # the same tuples as decode_beams_batch(): kenlm state dropped
            out = [(t, f, am, sc) for t, _, f, am, sc in dec.decode_beams(x, beam_width=beam, prune_history=False)]
        del x
        return out
    finally:
        shm.close()


DEFAULT_WORKERS = 1  # in-process: fastest on an H100 host (32 vCPU) on FLEURS-id; a pool won on a 12-core Mac


def default_workers() -> int:
    """lm_workers when the caller gives none: 1, in-process (logged once per process with a hint)."""
    global _logged_workers
    if not _logged_workers:
        log.info("sea_asr: LM beam search in-process (lm_workers=1). On some hosts lm_workers=4 is about 2x faster "
                 "(this host: %d usable CPUs); measure before changing.", usable_cpus())
        _logged_workers = True
    return DEFAULT_WORKERS


class _KnownWarningsFilter(logging.Filter):
    """Two pyctcdecode warnings that do not apply here: the coverage check compares unigram characters with whole BPE
    pieces (always fails in BPE mode), and the missing "▁⁇▁" is deliberate (see the module docstring)."""

    def filter(self, record):
        m = record.getMessage()
        return "Unigrams and labels don't seem to agree" not in m and "UNK token ▁⁇▁ not found" not in m


def unit_labels(decoder, unk_label: str = UNK_WORD) -> list[str]:
    """pyctcdecode labels for units 0..N+1: blank, unk, then the Gemma pieces."""
    return ["", unk_label] + [decoder.piece(u) for u in range(2, len(decoder.gemma_ids) + 2)]


def read_lm_dir(lm_dir: str | Path) -> tuple[Path, list[str] | None, dict]:
    d = Path(lm_dir)
    cfg = json.loads((d / "lm_config.json").read_text()) if (d / "lm_config.json").exists() else {}
    if "model" in cfg:
        model = d / cfg["model"]
    else:
        found = sorted(d.glob("*.bin")) or sorted(d.glob("*.arpa"))
        if len(found) != 1:
            raise FileNotFoundError(f"{d}: expected one .bin or .arpa file (or 'model' in lm_config.json), "
                                    f"found {[f.name for f in found]}")
        model = found[0]
    uni = d / "unigrams.txt"
    unigrams = [l.strip() for l in uni.read_text(encoding="utf-8").splitlines() if l.strip()] if uni.exists() else None
    return model, unigrams, cfg


TOKEN_MIN_LOGP = -5.0  # pyctcdecode 0.5.0 DEFAULT_MIN_TOKEN_LOGP: tokens below it are not expanded
BLANK_ONLY_MARGIN = 0.01  # our log-probs vs pyctcdecode's re-softmax of them differ by ~1e-6


def compress_blanks(log_probs: np.ndarray, token_min_logp: float = TOKEN_MIN_LOGP,
                    margin: float = BLANK_ONLY_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each run of blank-only frames into the run's first frame, before the beam search. Blank-only = blank is
    the frame's argmax and every other unit is below token_min_logp - margin, so pyctcdecode expands nothing but blank
    there: every beam takes blank, texts, word frames and the N-best order are unchanged, and every beam's acoustic
    score moves by the same constant. One frame stays per run, so a unit repeated across a blank ("a _ a") is not
    merged. Run collapsing is from sea_asr/lm_fast.py (same authors), whose p(blank) > 0.99 rule could drop a frame
    where a token still had log p >= -5 and shift word starts by a frame. Returns (kept frames, original index of each
    kept frame) so word times map back."""
    if len(log_probs) == 0:
        return log_probs, np.zeros(0, dtype=np.int64)
    other = log_probs[:, 1:].max(axis=1)
    sure = (log_probs[:, 0] >= other) & (other < token_min_logp - margin)
    keep = ~sure
    keep[1:] |= sure[1:] & ~sure[:-1]
    keep[0] |= sure[0]
    idx = np.flatnonzero(keep)
    return log_probs[idx], idx


def _remap_frames(word_frames, idx):
    """pyctcdecode word frames [start, end) on kept frames -> original frames (end = last kept frame + 1)."""
    if idx is None:
        return word_frames
    return [(w, (int(idx[s]), int(idx[e - 1]) + 1)) for w, (s, e) in word_frames]


def clean_text(text: str) -> str:
    """Drop unk like greedy decoding does: its neighbours join ("ma⁇kan" -> "makan")."""
    return " ".join(text.replace(UNK_WORD, "").split())


class LMDecoder:
    """Beam search with a KenLM n-gram. decode() takes per-utterance log-probs [T, units] (float32 numpy)."""

    def __init__(self, labels: Sequence[str], model_path: str | Path, unigrams: Sequence[str] | None = None,
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA, beam: int = DEFAULT_BEAM,
                 rescore_beam: int | None = None, compress_blanks: bool = True):
        """rescore_beam: beam for nbest_batch (the N-best list that sea_asr.rescore reranks); None = beam.
        compress_blanks: collapse blank-only frame runs before every beam search (see compress_blanks())."""
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as e:
            raise ImportError("LM decoding needs the extra: pip install 'sea_asr[lm]'") from e
        logging.getLogger("pyctcdecode.alphabet").addFilter(_KnownWarningsFilter())
        self.alpha, self.beta, self.beam = alpha, beta, beam
        self.rescore_beam = beam if rescore_beam is None else rescore_beam
        self._rescore_beam_fallback = rescore_beam is None
        self._pool, self._pool_size = None, 0
        self.compress_blanks = compress_blanks
        self.model_path = str(model_path)
        self._dec = build_ctcdecoder(list(labels), kenlm_model_path=self.model_path,
                                     unigrams=list(unigrams) if unigrams else None, alpha=alpha, beta=beta)

    @classmethod
    def from_dir(cls, lm_dir: str | Path, decoder, alpha: float | None = None, beta: float | None = None,
                 beam: int | None = None, rescore_beam: int | None = None,
                 compress_blanks: bool | None = None) -> "LMDecoder":
        """Arguments override lm_config.json, which overrides the defaults. beam drives the plain LM decode only;
        rescore_beam (argument, else "rescore_beam" in lm_config.json) drives the N-best list for rescoring. A config
        without "rescore_beam" (v0.2.0) falls back to beam, the old single-beam behaviour, with a warning on use."""
        model, unigrams, cfg = read_lm_dir(lm_dir)
        return cls(unit_labels(decoder), model, unigrams,
                   alpha=alpha if alpha is not None else cfg.get("alpha", DEFAULT_ALPHA),
                   beta=beta if beta is not None else cfg.get("beta", DEFAULT_BETA),
                   beam=beam if beam is not None else cfg.get("beam", DEFAULT_BEAM),
                   rescore_beam=rescore_beam if rescore_beam is not None else cfg.get("rescore_beam"),
                   compress_blanks=compress_blanks if compress_blanks is not None
                   else cfg.get("compress_blanks", True))

    def _prep(self, log_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.compress_blanks:
            return log_probs, None
        return compress_blanks(log_probs)

    def _get_pool(self, workers: int):
        """One fork pool per decoder, made on first use and reused across batches. Forked after build_ctcdecoder, so
        the children inherit the loaded KenLM (pyctcdecode keeps it in a class-level dict)."""
        if self._pool is None or self._pool_size != workers:
            self.close()
            from multiprocessing import get_context, resource_tracker

            resource_tracker.ensure_running()  # before the fork, so the children share it (see _shm_work)
            _POOL_DECODERS[id(self)] = self._dec
            self._pool, self._pool_size = get_context("fork").Pool(workers), workers
        return self._pool

    def _pooled(self, log_probs, workers: int, top1: bool, beam: int):
        """Decode a batch on the pool through one shared-memory block (see _shm_work); results in input order."""
        from multiprocessing import shared_memory

        pool = self._get_pool(workers)
        arrs = [np.ascontiguousarray(lp) for lp in log_probs]
        shm = shared_memory.SharedMemory(create=True, size=max(1, sum(a.nbytes for a in arrs)))
        try:
            jobs, off = [], 0
            for a in arrs:
                np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf, offset=off)[...] = a
                jobs.append((id(self), top1, beam, shm.name, off, a.shape, a.dtype.str))
                off += a.nbytes
            return pool.map(_shm_work, jobs, chunksize=1)
        finally:
            shm.close()
            shm.unlink()

    def close(self):
        """Stop the worker pool (also done on garbage collection and at interpreter exit)."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool, self._pool_size = None, 0
            _POOL_DECODERS.pop(id(self), None)

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  (interpreter shutdown)
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def decode(self, log_probs: np.ndarray) -> str:
        return clean_text(self._dec.decode(self._prep(log_probs)[0], beam_width=self.beam))

    def decode_batch(self, log_probs: list[np.ndarray], workers: int = 1) -> list[str]:
        """workers > 1: the decoder's persistent fork pool (see _get_pool)."""
        if workers <= 1 or len(log_probs) < 2:
            return [self.decode(lp) for lp in log_probs]
        xs = [self._prep(lp)[0] for lp in log_probs]
        return [clean_text(t) for t in self._pooled(xs, workers, True, self.beam)]

    def nbest_batch(self, log_probs: list[np.ndarray], topk: int, workers: int = 1) -> list[list[tuple]]:
        """Per utterance up to topk distinct cleaned hypotheses [(text, am, beam score, word_frames)], best first.
        prune_history=False keeps the list diverse (True collapses it to a few texts); used by sea_asr.rescore."""
        from .rescore import distinct_nbest

        if self._rescore_beam_fallback:
            log.warning("sea_asr: lm_config.json has no \"rescore_beam\"; the rescoring N-best list uses beam %d "
                        "(v0.2.0 behaviour). The measured setting is rescore_beam 128.", self.beam)
            self._rescore_beam_fallback = False
        beam = self.rescore_beam
        preps = [self._prep(lp) for lp in log_probs]
        if workers > 1 and len(log_probs) > 1:
            beams = self._pooled([x for x, _ in preps], workers, False, beam)
        else:
            beams = [[(t, f, am, sc) for t, _, f, am, sc in
                      self._dec.decode_beams(x, beam_width=beam, prune_history=False)] for x, _ in preps]
        return [distinct_nbest([(t, _remap_frames(f, idx), am, sc) for t, f, am, sc in b], clean_text, topk)
                for b, (_, idx) in zip(beams, preps)]

    def words_from_frames(self, word_frames, offset_s: float = 0.0) -> list[dict]:
        return [{"word": c, "start": round(offset_s + s * FRAME_S, 3), "end": round(offset_s + e * FRAME_S, 3)}
                for w, (s, e) in word_frames if (c := w.replace(UNK_WORD, ""))]

    def decode_words(self, log_probs: np.ndarray, offset_s: float = 0.0) -> tuple[str, list[dict]]:
        """Best beam text plus word timestamps from the beam's word frames ([start, end) in 40 ms frames)."""
        # prune_history=True as in pyctcdecode's decode(), so the text equals decode() (its default here is False)
        x, idx = self._prep(log_probs)
        text, _, word_frames, _, _ = self._dec.decode_beams(x, beam_width=self.beam, prune_history=True)[0]
        return clean_text(text), self.words_from_frames(_remap_frames(word_frames, idx), offset_s)
