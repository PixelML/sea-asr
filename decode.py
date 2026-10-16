# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Greedy CTC decoding, word timestamps and subtitle cues (spec section 6)."""

from __future__ import annotations

from typing import Callable, Sequence

FRAME_S = 0.04
BLANK, UNK = 0, 1
WORD_START = "▁"  # sentencepiece word-boundary marker


def greedy_runs(frame_ids: Sequence[int]) -> list[tuple[int, int, int]]:
    """Collapse repeats and drop blanks. Returns (unit, first_frame, last_frame) per emitted unit."""
    runs: list[tuple[int, int, int]] = []
    prev, start = None, 0
    for t, u in enumerate(list(frame_ids) + [None]):
        if u != prev:
            if prev is not None and prev != BLANK:
                runs.append((prev, start, t - 1))
            prev, start = u, t
    return runs


class UnitDecoder:
    """Maps CTC units to Gemma token ids and text."""

    def __init__(self, tokenizer, gemma_ids: Sequence[int]):
        self.tok = tokenizer
        self.gemma_ids = list(gemma_ids)
        self._pieces: dict[int, str] = {}

    def gemma_id(self, unit: int) -> int:
        return self.gemma_ids[unit - 2]

    def piece(self, unit: int) -> str:
        if unit not in self._pieces:
            self._pieces[unit] = self.tok.convert_ids_to_tokens(self.gemma_id(unit))
        return self._pieces[unit]

    def text(self, units: Sequence[int]) -> str:
        return self.tok.decode([self.gemma_id(u) for u in units if u >= 2]).strip()


def words_from_runs(runs, piece_of: Callable[[int], str], text_of: Callable[[list[int]], str],
                    offset_s: float = 0.0) -> list[dict]:
    """Group emitted units into words. A word starts at a unit whose piece starts with '▁' (or the first unit)."""
    groups: list[list[tuple[int, int, int]]] = []
    for r in runs:
        if r[0] < 2:  # blank never appears here; unk carries no text
            continue
        if not groups or piece_of(r[0]).startswith(WORD_START):
            groups.append([r])
        else:
            groups[-1].append(r)
    words = []
    for g in groups:
        w = text_of([u for u, _, _ in g])
        if not w:
            continue
        words.append({"word": w,
                      "start": round(offset_s + g[0][1] * FRAME_S, 3),
                      "end": round(offset_s + (g[-1][2] + 1) * FRAME_S, 3)})
    return words


# ---------------------------------------------------------------- subtitles

MAX_LINES, MAX_CHARS, MAX_CUE_S = 2, 32, 6.0


def _split_long_words(words: list[dict]) -> list[dict]:
    out = []
    for w in words:
        text = w["word"]
        if len(text) <= MAX_CHARS:
            out.append(w)
            continue
        chunks = [text[i : i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
        dur = (w["end"] - w["start"]) / len(chunks)
        for i, c in enumerate(chunks):
            out.append({"word": c, "start": w["start"] + i * dur, "end": w["start"] + (i + 1) * dur})
    return out


def make_cues(words: list[dict]) -> list[dict]:
    """Greedy cues: at most 2 lines x 32 characters and at most 6 s; breaks only at word boundaries."""
    cues, lines, start, end = [], [], None, None

    def flush():
        nonlocal lines, start, end
        if lines:
            cues.append({"start": start, "end": end, "lines": [" ".join(l) for l in lines]})
        lines, start, end = [], None, None

    for w in _split_long_words(words):
        if lines and w["end"] - start > MAX_CUE_S:
            flush()
        if not lines:
            lines, start = [[w["word"]]], w["start"]
        elif len(" ".join(lines[-1] + [w["word"]])) <= MAX_CHARS:
            lines[-1].append(w["word"])
        elif len(lines) < MAX_LINES:
            lines.append([w["word"]])
        else:
            flush()
            lines, start = [[w["word"]]], w["start"]
        end = w["end"]
    flush()
    return cues


def _ts(sec: float, sep: str) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(words: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(make_cues(words), 1):
        blocks.append(f"{i}\n{_ts(c['start'], ',')} --> {_ts(c['end'], ',')}\n" + "\n".join(c["lines"]) + "\n")
    return "\n".join(blocks)


def to_vtt(words: list[dict]) -> str:
    blocks = ["WEBVTT\n"]
    for c in make_cues(words):
        blocks.append(f"{_ts(c['start'], '.')} --> {_ts(c['end'], '.')}\n" + "\n".join(c["lines"]) + "\n")
    return "\n".join(blocks)
