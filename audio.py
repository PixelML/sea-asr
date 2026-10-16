# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Audio loading, long-input splitting and length batching (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
MAX_PIECE_S = 30.0
SEARCH_S = 5.0
ENERGY_WIN_S = 0.05
MIN_PIECE_S = 0.1
BATCH_BUDGET_S = 600.0


def load_audio(src, sr: int | None = None) -> np.ndarray:
    """Return mono float32 16 kHz audio from a path, an array (assumed 16 kHz unless sr given) or (array, sr)."""
    if isinstance(src, tuple):
        src, sr = src
    if isinstance(src, (str, bytes)) or hasattr(src, "__fspath__"):
        import soundfile as sf

        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
    else:
        wav = np.asarray(src, dtype=np.float32)
        if wav.ndim == 2:  # (channels, samples) or (samples, channels): average the shorter axis
            wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    sr = SAMPLE_RATE if sr is None else int(sr)
    if sr != SAMPLE_RATE:
        import soxr

        wav = soxr.resample(wav, sr, SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype=np.float32)


def split_long(wav: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[int, np.ndarray]]:
    """Cut inputs longer than 30 s at the quietest 50 ms window in the last 5 s before each 30 s mark.

    Returns (start_sample, piece) pairs; pieces shorter than 0.1 s are dropped.
    """
    max_n, search_n, win = int(MAX_PIECE_S * sr), int(SEARCH_S * sr), int(ENERGY_WIN_S * sr)
    pieces, start = [], 0
    while len(wav) - start > max_n:
        lo = start + max_n - search_n
        seg = wav[lo : start + max_n].astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(seg * seg)])
        energy = csum[win:] - csum[:-win]  # energy[i] = sum of squares of seg[i : i + win]
        cut = lo + int(np.argmin(energy)) + win // 2
        pieces.append((start, wav[start:cut]))
        start = cut
    pieces.append((start, wav[start:]))
    return [(s, p) for s, p in pieces if len(p) >= MIN_PIECE_S * sr]


@dataclass
class Piece:
    item: int  # index of the input this piece belongs to
    offset_s: float  # start time of the piece inside its input
    wav: np.ndarray


def make_batches(pieces: list[Piece], sr: int = SAMPLE_RATE) -> list[list[Piece]]:
    """Sort by length; grow a batch while longest_seconds * (n + 1) <= 600."""
    order = sorted(pieces, key=lambda p: len(p.wav))
    batches, cur = [], []
    for p in order:
        longest = max([len(q.wav) for q in cur] + [len(p.wav)]) / sr
        if cur and longest * (len(cur) + 1) > BATCH_BUDGET_S:
            batches.append(cur)
            cur = []
        cur.append(p)
    if cur:
        batches.append(cur)
    return batches
