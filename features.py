# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Log-mel front end (spec section 2). Parameters come from the encoder's HF feature extractor."""

from __future__ import annotations

import numpy as np
import torch


class LogMelFrontEnd:
    """Batched torch log-mel features with the frame-validity rule used in training."""

    def __init__(self, frame_length: int, hop_length: int, fft_length: int, window: np.ndarray,
                 mel_filters: np.ndarray, mel_floor: float):
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.fft_length = int(fft_length)
        self.window = torch.as_tensor(np.asarray(window), dtype=torch.float32)
        self.mel_filters = torch.as_tensor(np.asarray(mel_filters), dtype=torch.float32)
        self.mel_floor = float(mel_floor)
        assert self.window.shape == (self.frame_length,)
        assert self.mel_filters.shape[0] == self.fft_length // 2 + 1

    @classmethod
    def from_pretrained(cls, encoder_id: str, revision: str | None = None) -> "LogMelFrontEnd":
        from transformers import AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(encoder_id, revision=revision)
        assert float(fe.preemphasis) == 0.0, "preemphasis must be 0"
        assert float(fe.dither) == 0.0, "dither must be 0"
        assert float(fe.input_scale_factor) == 1.0, "input scale must be 1"
        assert getattr(fe, "per_bin_mean", None) is None, "per-bin mean must be unset"
        assert getattr(fe, "per_bin_stddev", None) is None, "per-bin stddev must be unset"
        return cls(fe.frame_length, fe.hop_length, fe.fft_length, fe.window, fe.mel_filters, fe.mel_floor)

    def num_frames(self, num_samples: int) -> int:
        padded = num_samples + self.frame_length // 2
        return max(0, (padded - (self.frame_length + 1)) // self.hop_length + 1)

    def __call__(self, waves: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """waves: (B, L) float32, zero-padded on the right. lengths: (B,) true sample counts.

        Returns features (B, T, n_mels) and a bool frame mask (B, T).
        """
        half = self.frame_length // 2
        x = torch.nn.functional.pad(waves, (half, 0))
        frames = x.unfold(-1, self.frame_length + 1, self.hop_length)[..., :-1]
        window = self.window.to(frames.device)
        mel_filters = self.mel_filters.to(frames.device)
        mag = torch.fft.rfft(frames * window, n=self.fft_length).abs()
        feats = torch.log(mag @ mel_filters + self.mel_floor)
        t = torch.arange(feats.shape[1], device=feats.device)
        mask = t[None, :] * self.hop_length + self.frame_length < lengths.to(feats.device)[:, None] + half
        feats = feats.masked_fill(~mask[..., None], 0.0)
        return feats, mask
