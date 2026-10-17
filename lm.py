# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Optional KenLM n-gram beam search over the CTC log-probs (pyctcdecode). Install with `pip install sea_asr[lm]`.

LM directory layout:
    lm_config.json   {"alpha": 0.4, "beta": 1.0}   optional keys: "model" (file name), "beam"
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
from pathlib import Path
from typing import Sequence

import numpy as np

from .decode import FRAME_S

DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_BEAM = 0.5, 1.0, 32
UNK_WORD = "⁇"


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


def clean_text(text: str) -> str:
    """Drop unk like greedy decoding does: its neighbours join ("ma⁇kan" -> "makan")."""
    return " ".join(text.replace(UNK_WORD, "").split())


class LMDecoder:
    """Beam search with a KenLM n-gram. decode() takes per-utterance log-probs [T, units] (float32 numpy)."""

    def __init__(self, labels: Sequence[str], model_path: str | Path, unigrams: Sequence[str] | None = None,
                 alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA, beam: int = DEFAULT_BEAM):
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError as e:
            raise ImportError("LM decoding needs the extra: pip install 'sea_asr[lm]'") from e
        logging.getLogger("pyctcdecode.alphabet").addFilter(_KnownWarningsFilter())
        self.alpha, self.beta, self.beam = alpha, beta, beam
        self.model_path = str(model_path)
        self._dec = build_ctcdecoder(list(labels), kenlm_model_path=self.model_path,
                                     unigrams=list(unigrams) if unigrams else None, alpha=alpha, beta=beta)

    @classmethod
    def from_dir(cls, lm_dir: str | Path, decoder, alpha: float | None = None, beta: float | None = None,
                 beam: int | None = None) -> "LMDecoder":
        """Arguments override lm_config.json, which overrides the defaults."""
        model, unigrams, cfg = read_lm_dir(lm_dir)
        return cls(unit_labels(decoder), model, unigrams,
                   alpha=alpha if alpha is not None else cfg.get("alpha", DEFAULT_ALPHA),
                   beta=beta if beta is not None else cfg.get("beta", DEFAULT_BETA),
                   beam=beam if beam is not None else cfg.get("beam", DEFAULT_BEAM))

    def decode(self, log_probs: np.ndarray) -> str:
        return clean_text(self._dec.decode(log_probs, beam_width=self.beam))

    def decode_batch(self, log_probs: list[np.ndarray], workers: int = 1) -> list[str]:
        """workers > 1: a fork pool, created after the LM is loaded (pyctcdecode shares the model with the children)."""
        if workers <= 1 or len(log_probs) < 2:
            return [self.decode(lp) for lp in log_probs]
        from multiprocessing import get_context

        with get_context("fork").Pool(workers) as pool:
            out = self._dec.decode_batch(pool, log_probs, beam_width=self.beam)
        return [clean_text(t) for t in out]

    def nbest_batch(self, log_probs: list[np.ndarray], topk: int, workers: int = 1) -> list[list[tuple]]:
        """Per utterance up to topk distinct cleaned hypotheses [(text, am, beam score, word_frames)], best first.
        prune_history=False keeps the list diverse (True collapses it to a few texts); used by sea_asr.rescore."""
        from .rescore import distinct_nbest

        if workers > 1 and len(log_probs) > 1:
            from multiprocessing import get_context

            with get_context("fork").Pool(workers) as pool:
                beams = self._dec.decode_beams_batch(pool, log_probs, beam_width=self.beam, prune_history=False)
        else:
            beams = [[(t, f, am, sc) for t, _, f, am, sc in
                      self._dec.decode_beams(lp, beam_width=self.beam, prune_history=False)] for lp in log_probs]
        return [distinct_nbest(b, clean_text, topk) for b in beams]

    def words_from_frames(self, word_frames, offset_s: float = 0.0) -> list[dict]:
        return [{"word": c, "start": round(offset_s + s * FRAME_S, 3), "end": round(offset_s + e * FRAME_S, 3)}
                for w, (s, e) in word_frames if (c := w.replace(UNK_WORD, ""))]

    def decode_words(self, log_probs: np.ndarray, offset_s: float = 0.0) -> tuple[str, list[dict]]:
        """Best beam text plus word timestamps from the beam's word frames ([start, end) in 40 ms frames)."""
        # prune_history=True as in pyctcdecode's decode(), so the text equals decode() (its default here is False)
        text, _, word_frames, _, _ = self._dec.decode_beams(log_probs, beam_width=self.beam, prune_history=True)[0]
        words = [{"word": c, "start": round(offset_s + s * FRAME_S, 3), "end": round(offset_s + e * FRAME_S, 3)}
                 for w, (s, e) in word_frames if (c := w.replace(UNK_WORD, ""))]
        return clean_text(text), words
