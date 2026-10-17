# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""v0.2.1: separate beams for plain LM decode and the rescoring N-best list; default lm_workers = usable CPUs."""

import json
import logging
import os

import pytest

import sea_asr.lm as lm_mod
from sea_asr import SeaASR
from sea_asr.lm import LMDecoder, usable_cpus

pytest.importorskip("pyctcdecode")
pytest.importorskip("kenlm")

from test_lm import ARPA, ToyDecoder, frames  # noqa: E402


def lm_dir(tmp_path, **cfg):
    (tmp_path / "toy.arpa").write_text(ARPA, encoding="utf-8")
    (tmp_path / "unigrams.txt").write_text("saya\nmakan\nnasi\nmasi\n", encoding="utf-8")
    (tmp_path / "lm_config.json").write_text(json.dumps({"alpha": 1.0, "beta": 0.5, **cfg}))
    return tmp_path


class SpyDecoder:
    """Wraps the pyctcdecode decoder and records the beam_width of every call."""

    def __init__(self, inner):
        self.inner, self.widths = inner, []

    def decode(self, lp, beam_width):
        self.widths.append(("decode", beam_width))
        return self.inner.decode(lp, beam_width=beam_width)

    def decode_beams(self, lp, beam_width, prune_history):
        self.widths.append(("beams", beam_width))
        return self.inner.decode_beams(lp, beam_width=beam_width, prune_history=prune_history)


LP = frames({2: 0.9}, {}, {3: 0.9}, {}, {7: 0.5, 4: 0.45}, {})


def spy(dec):
    dec._dec = SpyDecoder(dec._dec)
    return dec._dec


def test_plain_and_rescore_beams_from_config(tmp_path):
    dec = LMDecoder.from_dir(lm_dir(tmp_path, beam=32, rescore_beam=128), ToyDecoder())
    assert (dec.beam, dec.rescore_beam) == (32, 128)
    s = spy(dec)
    assert dec.decode(LP) == "saya makan nasi"
    nb = dec.nbest_batch([LP], topk=4, workers=1)
    assert nb[0][0][0] == "saya makan nasi"
    assert s.widths == [("decode", 32), ("beams", 128)]


def test_arguments_override_both_beams(tmp_path):
    dec = LMDecoder.from_dir(lm_dir(tmp_path, beam=32, rescore_beam=128), ToyDecoder(), beam=8, rescore_beam=16)
    s = spy(dec)
    dec.decode(LP)
    dec.nbest_batch([LP], topk=2, workers=1)
    assert s.widths == [("decode", 8), ("beams", 16)]


def test_old_config_falls_back_to_beam_with_one_warning(tmp_path, caplog):
    dec = LMDecoder.from_dir(lm_dir(tmp_path, beam=64), ToyDecoder())  # v0.2.0 config: no rescore_beam
    assert dec.rescore_beam == 64
    s = spy(dec)
    with caplog.at_level(logging.WARNING, logger="sea_asr"):
        dec.nbest_batch([LP], topk=2, workers=1)
        dec.nbest_batch([LP], topk=2, workers=1)
    assert s.widths == [("beams", 64), ("beams", 64)]
    assert sum("rescore_beam" in r.getMessage() for r in caplog.records) == 1


def test_plain_decode_never_warns(tmp_path, caplog):
    dec = LMDecoder.from_dir(lm_dir(tmp_path, beam=32), ToyDecoder())
    with caplog.at_level(logging.WARNING, logger="sea_asr"):
        dec.decode(LP)
        dec.decode_batch([LP, LP], workers=1)
    assert not [r for r in caplog.records if "rescore_beam" in r.getMessage()]


# ------------------------------------------------------------------ workers


def no_cgroup(tmp_path):
    return tmp_path / "none"


def test_usable_cpus_this_machine(tmp_path):
    n = usable_cpus(no_cgroup(tmp_path))
    expect = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert n == expect >= 1


def test_forced_single_core(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert usable_cpus(no_cgroup(tmp_path)) == 1


@pytest.mark.parametrize("cpu_max, cores, expect", [
    ("max 100000\n", 8, 8),          # no quota
    ("200000 100000\n", 8, 2),       # 2 CPUs
    ("150000 100000\n", 8, 2),       # 1.5 CPUs -> 2 processes
    ("50000 100000\n", 8, 1),        # half a CPU -> 1
    ("1600000 100000\n", 8, 8),      # quota above the affinity mask: the mask wins
])
def test_cgroup_v2_quota(tmp_path, monkeypatch, cpu_max, cores, expect):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cores)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    (tmp_path / "cpu.max").write_text(cpu_max)
    assert usable_cpus(tmp_path) == expect


def test_cgroup_v1_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert usable_cpus(tmp_path) == 4
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")  # v1 "no limit"
    assert usable_cpus(tmp_path) == 16


def test_default_workers_logged_once_and_used_by_seaasr(monkeypatch, caplog):
    monkeypatch.setattr(lm_mod, "usable_cpus", lambda: 6)
    monkeypatch.setattr(lm_mod, "_logged_workers", False)
    with caplog.at_level(logging.INFO, logger="sea_asr"):
        a = SeaASR(None, None, None, "cpu", lm=object())
        b = SeaASR(None, None, None, "cpu", lm=object())
    assert a.lm_workers == b.lm_workers == 6
    assert sum("worker process" in r.getMessage() for r in caplog.records) == 1
    assert SeaASR(None, None, None, "cpu", lm=object(), lm_workers=3).lm_workers == 3  # explicit wins
    assert SeaASR(None, None, None, "cpu").lm_workers == 1  # greedy: no pool, nothing resolved


def test_pool_runs_with_all_cores_default(tmp_path):
    """Both pooled paths run with the default worker count on this machine and give the sequential result."""
    dec = LMDecoder.from_dir(lm_dir(tmp_path, beam=16, rescore_beam=32), ToyDecoder())
    w = lm_mod.default_workers()
    assert dec.decode_batch([LP] * 3, workers=w) == ["saya makan nasi"] * 3
    seq = dec.nbest_batch([LP] * 3, topk=3, workers=1)
    par = dec.nbest_batch([LP] * 3, topk=3, workers=w)
    assert [[h[:3] for h in u] for u in par] == [[h[:3] for h in u] for u in seq]


def test_pool_made_once_and_reused(tmp_path):
    with LMDecoder.from_dir(lm_dir(tmp_path, beam=16, rescore_beam=32), ToyDecoder()) as dec:
        dec.decode_batch([LP, LP], workers=2)
        pool = dec._pool
        dec.decode_batch([LP, LP], workers=2)
        dec.nbest_batch([LP, LP], topk=2, workers=2)
        assert pool is not None and dec._pool is pool  # one pool for both paths and every batch
        dec.decode_batch([LP, LP], workers=3)
        assert dec._pool is not pool and dec._pool_size == 3  # new size -> new pool
    assert dec._pool is None  # closed by the context manager


def test_shared_memory_pool_is_exact(tmp_path):
    """The pooled path (shared-memory views in the workers) must equal pyctcdecode's own in-process decode() /
    decode_beams() bit for bit: text, word frames and scores; also a 0-frame utterance."""
    import numpy as np

    rng = np.random.default_rng(0)
    lps = []
    for t in (3, 9, 17, 1, 0):
        z = rng.normal(0, 1, (t, 8)).astype(np.float32)
        z[np.arange(t), rng.integers(0, 8, t)] += rng.uniform(4, 12, t).astype(np.float32)  # peaky frames
        lps.append(z - np.log(np.exp(z).sum(1, keepdims=True)))
    lps.append(LP)
    with LMDecoder.from_dir(lm_dir(tmp_path, beam=8, rescore_beam=16), ToyDecoder()) as dec:
        assert dec._pooled(lps, 2, True, 8) == [dec._dec.decode(lp, beam_width=8) for lp in lps]
        ref = [[(t, f, am, sc) for t, _, f, am, sc in dec._dec.decode_beams(lp, beam_width=16, prune_history=False)]
               for lp in lps]
        assert dec._pooled(lps, 2, False, 16) == ref


def test_compress_blanks_map_and_repeats():
    import numpy as np

    from sea_asr.lm import compress_blanks

    # frames: sure-blank run (0-2), "a" (3), sure-blank run (4-6), "a" again (7), sure blank (8)
    lp = frames({}, {}, {}, {2: 0.9}, {}, {}, {}, {2: 0.9}, {})
    x, idx = compress_blanks(lp)
    assert idx.tolist() == [0, 3, 4, 7, 8]
    assert np.array_equal(x, lp[idx])
    greedy = [int(u) for u in x.argmax(-1)]
    assert greedy == [0, 2, 0, 2, 0]  # the blank between the two "a" frames survives: "a a", not "a"
    e, ei = compress_blanks(lp[:0])
    assert e.shape == (0, lp.shape[1]) and ei.size == 0


def test_compressed_words_keep_original_times(tmp_path):
    # "saya" at frames 4-5, "makan" at 10, "nasi" at 15-16, long sure-blank runs between
    cols = [{}] * 4 + [{2: 0.9}, {2: 0.9}] + [{}] * 4 + [{3: 0.9}] + [{}] * 4 + [{5: 0.9}, {6: 0.9}] + [{}] * 3
    lp = frames(*cols)
    on = LMDecoder.from_dir(lm_dir(tmp_path, beam=16), ToyDecoder())
    off = LMDecoder.from_dir(lm_dir(tmp_path, beam=16), ToyDecoder(), compress_blanks=False)
    assert on.compress_blanks and not off.compress_blanks
    t_on, w_on = on.decode_words(lp, offset_s=2.0)
    t_off, w_off = off.decode_words(lp, offset_s=2.0)
    assert t_on == t_off == "saya makan nasi"
    assert w_on == w_off
    assert w_on[0]["start"] == pytest.approx(2.0 + 4 * 0.04) and w_on[2]["end"] == pytest.approx(2.0 + 17 * 0.04)
    nb_on = on.nbest_batch([lp], topk=3, workers=1)[0]
    nb_off = off.nbest_batch([lp], topk=3, workers=1)[0]
    assert [(h[0], h[3]) for h in nb_on] == [(h[0], h[3]) for h in nb_off]  # texts, order, word frames
    shift = nb_on[0][1] - nb_off[0][1]  # dropped blank-only frames: one constant for every am and combined score
    for a, b in zip(nb_on, nb_off):
        assert a[1] - b[1] == pytest.approx(shift, abs=1e-6) and a[2] - b[2] == pytest.approx(shift, abs=1e-6)
    assert on.decode_batch([lp, lp], workers=2) == off.decode_batch([lp, lp], workers=2)


def test_compress_blanks_config_and_flag(tmp_path):
    assert LMDecoder.from_dir(lm_dir(tmp_path, compress_blanks=False), ToyDecoder()).compress_blanks is False
    assert LMDecoder.from_dir(lm_dir(tmp_path), ToyDecoder()).compress_blanks is True  # default on
    assert LMDecoder.from_dir(lm_dir(tmp_path, compress_blanks=False), ToyDecoder(),
                              compress_blanks=True).compress_blanks is True  # argument wins
