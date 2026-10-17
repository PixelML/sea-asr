# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""CPU tests for the optional KenLM beam search (skipped without sea_asr[lm]). Toy units + a hand-written ARPA."""

import json
import math

import numpy as np
import pytest

pytest.importorskip("pyctcdecode")
pytest.importorskip("kenlm")

from sea_asr.lm import LMDecoder, clean_text, read_lm_dir, unit_labels  # noqa: E402

# unit -> piece; unit 0 blank, 1 unk
PIECES = {2: "▁saya", 3: "▁makan", 4: "▁nasi", 5: "▁nas", 6: "i", 7: "▁masi"}

ARPA = """\\data\\
ngram 1=7
ngram 2=4

\\1-grams:
-1.0\t<s>\t-0.3
-1.0\t</s>
-2.0\t<unk>
-0.7\tsaya\t-0.3
-0.7\tmakan\t-0.3
-0.7\tnasi\t-0.3
-3.0\tmasi\t-0.3

\\2-grams:
-0.1\t<s> saya
-0.1\tsaya makan
-0.1\tmakan nasi
-0.1\tnasi </s>

\\end\\
"""


class ToyDecoder:
    gemma_ids = list(range(len(PIECES)))

    def piece(self, u):
        return PIECES[u]


def frames(*cols, n_units=8):
    """cols: dict unit -> prob per frame; the rest of the mass goes to blank. Returns log-probs [T, units]."""
    out = np.full((len(cols), n_units), 1e-6, np.float32)
    for t, c in enumerate(cols):
        for u, p in c.items():
            out[t, u] = p
        out[t, 0] = max(1e-6, 1 - sum(c.values()))
    return np.log(out / out.sum(1, keepdims=True))


@pytest.fixture
def lm_dir(tmp_path):
    (tmp_path / "toy.arpa").write_text(ARPA, encoding="utf-8")
    (tmp_path / "unigrams.txt").write_text("saya\nmakan\nnasi\nmasi\n", encoding="utf-8")
    (tmp_path / "lm_config.json").write_text(json.dumps({"alpha": 1.0, "beta": 0.5, "beam": 16}))
    return tmp_path


def test_labels_and_lm_dir(lm_dir):
    assert unit_labels(ToyDecoder())[:3] == ["", "⁇", "▁saya"]
    model, unigrams, cfg = read_lm_dir(lm_dir)
    assert model.name == "toy.arpa" and unigrams == ["saya", "makan", "nasi", "masi"] and cfg["beam"] == 16


def test_lm_dir_needs_one_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_lm_dir(tmp_path)


def test_lm_fixes_acoustic_confusion(lm_dir):
    # acoustics slightly prefer "masi" over "nasi" in the last word; the LM flips it
    lp = frames({2: 0.9}, {}, {3: 0.9}, {}, {7: 0.5, 4: 0.45}, {})
    greedy = [PIECES[int(u)] for u in lp.argmax(-1) if u >= 2]
    assert greedy[-1] == "▁masi"
    dec = LMDecoder.from_dir(lm_dir, ToyDecoder())
    assert (dec.alpha, dec.beta, dec.beam) == (1.0, 0.5, 16)
    assert dec.decode(lp) == "saya makan nasi"
    assert dec.decode_batch([lp, lp]) == ["saya makan nasi"] * 2


def test_arguments_override_config(lm_dir):
    dec = LMDecoder.from_dir(lm_dir, ToyDecoder(), alpha=0.0, beta=0.0, beam=4)
    assert (dec.alpha, dec.beta, dec.beam) == (0.0, 0.0, 4)
    lp = frames({2: 0.9}, {}, {3: 0.9}, {}, {7: 0.5, 4: 0.45}, {})
    assert dec.decode(lp) == "saya makan masi"  # alpha 0: acoustics only


def test_word_pieces_join_and_word_times(lm_dir):
    # "▁nas" + "i" form one word; frames 0..5 -> 40 ms each
    lp = frames({2: 0.9}, {}, {3: 0.9}, {5: 0.9}, {6: 0.9}, {})
    text, words = LMDecoder.from_dir(lm_dir, ToyDecoder()).decode_words(lp, offset_s=1.0)
    assert text == "saya makan nasi"
    assert [w["word"] for w in words] == ["saya", "makan", "nasi"]
    assert words[2]["start"] == pytest.approx(1.0 + 3 * 0.04) and words[2]["end"] == pytest.approx(1.0 + 5 * 0.04)
    assert all(math.isfinite(w["start"]) and w["end"] > w["start"] for w in words)


def test_unk_carries_no_text():
    assert clean_text("saya ma⁇kan  ⁇ nasi") == "saya makan nasi"
    assert clean_text("⁇") == ""
