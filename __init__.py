# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""sea_asr: CTC speech recognition on the EmbeddingGemma 2 audio tower.

    from sea_asr import SeaASR, to_srt
    asr = SeaASR.from_pretrained("path/or/hf-repo")
    text = asr.transcribe("clip.flac")
    out = asr.transcribe("talk.flac", return_words=True)
    print(to_srt(out["words"]))

    asr = SeaASR.from_pretrained("path/or/hf-repo", lm="path/to/lm-dir")   # KenLM beam search (sea_asr[lm])
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from .decode import to_srt, to_vtt

__all__ = ["SeaASR", "to_srt", "to_vtt"]
__version__ = "0.2.1"


def _resolve_dir(path_or_repo: str, revision: str | None) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, revision=revision,
                                  allow_patterns=["config.json", "units.json", "model.safetensors"]))


def _resolve_lm_dir(lm: str, revision: str | None) -> Path:
    """A local LM directory, or a Hugging Face repo id whose LM files sit in its lm/ folder."""
    p = Path(lm)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(lm, revision=revision, allow_patterns=["lm/*"])) / "lm"


class SeaASR:
    def __init__(self, model, frontend, decoder, device, lm=None, lm_workers: int | None = None, rescorer=None):
        self.model = model
        self.frontend = frontend
        self.decoder = decoder
        self.device = device
        self.lm = lm  # sea_asr.lm.LMDecoder or None (greedy)
        if lm is not None and lm_workers is None:
            from .lm import default_workers

            lm_workers = default_workers()
        self.lm_workers = lm_workers or 1  # beam-search processes (plain LM decode and the rescoring N-best list)
        self.rescorer = rescorer  # sea_asr.rescore.NBestRescorer or None (KenLM 1-best)

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str, revision: str | None = None, device: str | None = None,
                        lm: str | None = None, lm_revision: str | None = None, alpha: float | None = None,
                        beta: float | None = None, beam: int | None = None, lm_workers: int | None = None,
                        rescore: bool | str = False, rescore_revision: str | None = None,
                        rescore_beam: int | None = None, compress_blanks: bool | None = None) -> "SeaASR":
        """lm: an LM directory or HF repo id (see sea_asr.lm); alpha / beta / beam / rescore_beam override its
        lm_config.json. beam is the plain LM decode; rescore_beam builds the N-best list when rescore is on.
        lm_workers: beam-search processes; None = 1 (in-process, no pool); N > 1 = a fork pool of N.
        compress_blanks: collapse blank-only frame runs before the beam search (default on; False = v0.2.0 frames).
        rescore: off by default. True = second pass with the pinned default causal LM (sea_asr.rescore), or a model
        id / directory (then give rescore_revision). Needs lm; the measured setting is alpha 0.4, beta 1.5,
        rescore_beam 128."""
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from .decode import UnitDecoder
        from .features import LogMelFrontEnd
        from .model import AUDIO_TOP_PREFIX, SeaASRModel, load_audio_tower

        d = _resolve_dir(path_or_hf_repo, revision)
        cfg = json.loads((d / "config.json").read_text())
        units = json.loads((d / "units.json").read_text())
        enc_id, enc_rev = cfg["encoder_id"], cfg["encoder_revision"]
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = SeaASRModel(load_audio_tower(enc_id, enc_rev), cfg)
        sd = load_file(str(d / "model.safetensors"))
        top = {k[len(AUDIO_TOP_PREFIX):]: v for k, v in sd.items() if k.startswith(AUDIO_TOP_PREFIX)}
        rest = {k: v for k, v in sd.items() if not k.startswith(AUDIO_TOP_PREFIX)}
        for i in cfg["audio_top_layers"]:
            pre = f"{i}."
            model.audio.layers[i].load_state_dict({k[len(pre):]: v for k, v in top.items() if k.startswith(pre)},
                                                  strict=True)
        missing, unexpected = model.load_state_dict(rest, strict=False)
        missing = [k for k in missing if not k.startswith("audio.")]
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
        model.set_grad_flags()
        model.eval().to(device)

        tok = AutoTokenizer.from_pretrained(enc_id, revision=enc_rev)
        decoder = UnitDecoder(tok, units["gemma_ids"])
        lm_dec = None
        if lm is not None:
            from .lm import LMDecoder

            lm_dec = LMDecoder.from_dir(_resolve_lm_dir(lm, lm_revision), decoder, alpha=alpha, beta=beta, beam=beam,
                                        rescore_beam=rescore_beam, compress_blanks=compress_blanks)
        rescorer = None
        if rescore:
            if lm_dec is None:
                raise ValueError("rescore needs lm= (it reranks the KenLM beam search's N-best list)")
            from .rescore import NBestRescorer

            if rescore is True:
                rescorer = NBestRescorer(device=device)
            elif rescore_revision is None:
                raise ValueError("pin the rescoring LM: give rescore_revision (a commit)")
            else:
                rescorer = NBestRescorer(rescore, rescore_revision, device=device)
        return cls(model, LogMelFrontEnd.from_pretrained(enc_id, enc_rev), decoder, device, lm_dec, lm_workers,
                   rescorer)

    def _autocast(self):
        import torch

        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _run_batch(self, batch, log_probs: bool = False):
        """Greedy frame ids per piece, or float32 log-probs [T, units] per piece when log_probs=True."""
        import numpy as np
        import torch

        lengths = [len(p.wav) for p in batch]
        waves = np.zeros((len(batch), max(lengths)), dtype=np.float32)
        for j, p in enumerate(batch):
            waves[j, : lengths[j]] = p.wav
        waves = torch.from_numpy(waves).to(self.device)
        lens = torch.tensor(lengths, dtype=torch.long, device=self.device)
        with torch.no_grad(), self._autocast():
            feats, fmask = self.frontend(waves, lens)
            logits, out_len = self.model(feats, fmask)
        out_len = out_len.cpu().tolist()
        if log_probs:  # row by row: a 30 s piece is ~750 x 34k floats, a full batch would be gigabytes on the host
            return [torch.log_softmax(logits[j, :n].float(), -1).cpu().numpy() for j, n in enumerate(out_len)]
        ids = logits.argmax(-1).cpu().tolist()
        return [row[:n] for row, n in zip(ids, out_len)]

    def transcribe(self, audio, return_words: bool = False):
        """audio: a path, a float32 16 kHz array, an (array, sample_rate) tuple, or a list of these.

        Returns a string per input, or {"text", "words": [{"word", "start", "end"}]} when return_words=True.
        """
        from .audio import Piece, load_audio, make_batches, split_long
        from .decode import greedy_runs, words_from_runs

        single = not isinstance(audio, list)
        items = [audio] if single else audio
        pieces = []
        for i, a in enumerate(items):
            wav = load_audio(a)
            for start, seg in split_long(wav):
                pieces.append(Piece(i, start / 16_000, seg))

        results = {}
        for batch in make_batches(pieces):
            if self.lm is None:
                for p, frames in zip(batch, self._run_batch(batch)):
                    results[id(p)] = greedy_runs(frames)
            elif self.rescorer is not None:
                nb = self.lm.nbest_batch(self._run_batch(batch, log_probs=True), self.rescorer.topk, self.lm_workers)
                for p, (t, _, _, *frames) in zip(batch, self.rescorer.pick(nb)):
                    results[id(p)] = (t, self.lm.words_from_frames(frames[0], p.offset_s)
                                      if return_words and frames else [])
            elif return_words:
                for p, lp in zip(batch, self._run_batch(batch, log_probs=True)):
                    results[id(p)] = self.lm.decode_words(lp, p.offset_s)
            else:
                texts = self.lm.decode_batch(self._run_batch(batch, log_probs=True), self.lm_workers)
                for p, t in zip(batch, texts):
                    results[id(p)] = (t, [])

        out = []
        for i in range(len(items)):
            mine = sorted((p for p in pieces if p.item == i), key=lambda p: p.offset_s)
            texts, words = [], []
            for p in mine:
                if self.lm is not None:
                    t, w = results[id(p)]
                    texts.append(t)
                    words += w
                    continue
                runs = results[id(p)]
                texts.append(self.decoder.text([u for u, _, _ in runs]))
                if return_words:
                    words += words_from_runs(runs, self.decoder.piece, self.decoder.text, p.offset_s)
            text = " ".join(t for t in texts if t)
            out.append({"text": text, "words": words} if return_words else text)
        return out[0] if single else out
