# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""sea_asr: CTC speech recognition on the EmbeddingGemma 2 audio tower.

    from sea_asr import SeaASR, to_srt
    asr = SeaASR.from_pretrained("path/or/hf-repo")
    text = asr.transcribe("clip.flac")
    out = asr.transcribe("talk.flac", return_words=True)
    print(to_srt(out["words"]))
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from .decode import to_srt, to_vtt

__all__ = ["SeaASR", "to_srt", "to_vtt"]
__version__ = "0.1.0"


def _resolve_dir(path_or_repo: str, revision: str | None) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, revision=revision,
                                  allow_patterns=["config.json", "units.json", "model.safetensors"]))


class SeaASR:
    def __init__(self, model, frontend, decoder, device):
        self.model = model
        self.frontend = frontend
        self.decoder = decoder
        self.device = device

    @classmethod
    def from_pretrained(cls, path_or_hf_repo: str, revision: str | None = None, device: str | None = None) -> "SeaASR":
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
        return cls(model, LogMelFrontEnd.from_pretrained(enc_id, enc_rev), UnitDecoder(tok, units["gemma_ids"]),
                   device)

    def _autocast(self):
        import torch

        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _run_batch(self, batch):
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
        ids = logits.argmax(-1).cpu().tolist()
        return [row[:n] for row, n in zip(ids, out_len.cpu().tolist())]

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
            for p, frames in zip(batch, self._run_batch(batch)):
                results[id(p)] = greedy_runs(frames)

        out = []
        for i in range(len(items)):
            mine = sorted((p for p in pieces if p.item == i), key=lambda p: p.offset_s)
            texts, words = [], []
            for p in mine:
                runs = results[id(p)]
                texts.append(self.decoder.text([u for u, _, _ in runs]))
                if return_words:
                    words += words_from_runs(runs, self.decoder.piece, self.decoder.text, p.offset_s)
            text = " ".join(t for t in texts if t)
            out.append({"text": text, "words": words} if return_words else text)
        return out[0] if single else out
