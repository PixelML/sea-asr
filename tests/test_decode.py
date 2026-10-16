# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""CPU unit tests: greedy collapse, word timestamps, subtitle cue limits, long-input splitting, batching."""

import random

import numpy as np

from sea_asr.audio import Piece, make_batches, split_long
from sea_asr.decode import greedy_runs, make_cues, to_srt, to_vtt, words_from_runs

# toy vocabulary: unit -> sentencepiece piece
PIECES = {2: "▁hello", 3: "▁wor", 4: "ld", 5: "▁a", 6: "bc", 7: "▁x"}


def piece_of(u):
    return PIECES[u]


def text_of(units):
    return "".join(PIECES[u] for u in units if u >= 2).replace("▁", " ").strip()


def test_greedy_runs_collapse_and_blank():
    frames = [0, 2, 2, 0, 2, 3, 3, 4, 0, 0, 1, 5]
    assert greedy_runs(frames) == [(2, 1, 2), (2, 4, 4), (3, 5, 6), (4, 7, 7), (1, 10, 10), (5, 11, 11)]


def test_word_timestamps():
    frames = [0, 2, 2, 0, 0, 3, 3, 4, 0, 1, 5, 6, 6]
    words = words_from_runs(greedy_runs(frames), piece_of, text_of, offset_s=10.0)
    assert [w["word"] for w in words] == ["hello", "world", "abc"]
    assert words[0] == {"word": "hello", "start": 10.04, "end": 10.12}
    assert words[1] == {"word": "world", "start": 10.2, "end": 10.32}


def _random_words(seed, n=400):
    rng = random.Random(seed)
    frames = []
    for _ in range(n):
        frames += [0] * rng.randint(0, 30)
        frames += [rng.choice(list(PIECES))] * rng.randint(1, 4)
    return words_from_runs(greedy_runs(frames), piece_of, text_of)


def test_timestamps_monotonic():
    for seed in range(5):
        words = _random_words(seed)
        assert words
        for w in words:
            assert w["end"] > w["start"]
        for a, b in zip(words, words[1:]):
            assert a["start"] <= b["start"] and a["end"] <= b["start"]


def _parse_srt(srt):
    cues = []
    for block in srt.strip().split("\n\n"):
        lines = block.split("\n")
        t0, t1 = lines[1].split(" --> ")
        sec = lambda s: int(s[:2]) * 3600 + int(s[3:5]) * 60 + int(s[6:8]) + int(s[9:]) / 1000
        cues.append((sec(t0), sec(t1), lines[2:]))
    return cues


def test_srt_limits():
    rng = random.Random(0)
    vocab = ["kamusta", "ka", "na", "salamat", "po", "pagkakaisa", "anak", "selamat", "pagi", "berkelanjutan"]
    words, t = [], 0.0
    for _ in range(500):
        d = rng.uniform(0.1, 1.2)
        words.append({"word": rng.choice(vocab), "start": round(t, 3), "end": round(t + d, 3)})
        t += d + rng.uniform(0, 0.8)
    words.append({"word": "x" * 70, "start": t, "end": t + 2})  # overlong word gets hard-split
    cues = _parse_srt(to_srt(words))
    assert len(cues) > 10
    for start, end, lines in cues:
        assert 1 <= len(lines) <= 2
        assert all(len(l) <= 32 for l in lines)
        assert end - start <= 6.0 + 1e-9
    # every word kept, in order
    joined = " ".join(" ".join(l) for _, _, l in cues)
    assert joined.replace(" ", "") == "".join(w["word"] for w in words)
    starts = [c[0] for c in cues]
    assert starts == sorted(starts)


def test_vtt_header_and_cues():
    words = [{"word": "halo", "start": 0.0, "end": 0.5}, {"word": "dunia", "start": 0.6, "end": 1.0}]
    vtt = to_vtt(words)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.000\nhalo dunia" in vtt
    assert len(make_cues(words)) == 1


def test_split_long_cuts_at_quiet_window():
    sr = 16000
    rng = np.random.default_rng(0)
    wav = rng.uniform(-0.5, 0.5, 75 * sr).astype(np.float32)
    wav[int(27.3 * sr) : int(27.4 * sr)] = 0.0  # silence inside the first search region (25-30 s)
    pieces = split_long(wav)
    assert len(pieces) == 3
    first_cut = pieces[1][0]
    assert int(27.3 * sr) + 400 <= first_cut <= int(27.4 * sr) - 400
    assert all(len(p) <= 30 * sr for _, p in pieces)
    assert sum(len(p) for _, p in pieces) == len(wav)


def test_split_drops_tiny_tail():
    assert split_long(np.zeros(1000, np.float32)) == []
    assert len(split_long(np.zeros(2000, np.float32))) == 1


def test_batching_budget():
    sr = 16000
    rng = np.random.default_rng(1)
    pieces = [Piece(i, 0.0, np.zeros(int(rng.uniform(1, 30) * sr), np.float32)) for i in range(200)]
    batches = make_batches(pieces)
    assert sum(len(b) for b in batches) == 200
    flat = [len(p.wav) for b in batches for p in b]
    assert flat == sorted(flat)
    for b in batches:
        assert max(len(p.wav) for p in b) / sr * len(b) <= 600
