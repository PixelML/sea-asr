"""sea_asr.rescore without a model: N-best dedup and the score formula."""

from sea_asr.lm import clean_text
from sea_asr.rescore import NBestRescorer, distinct_nbest


def test_distinct_nbest_keeps_first_and_cleans():
    beams = [("ma⁇kan nasi", [], -1.0, -5.0), ("makan nasi", [], -2.0, -6.0), ("makan nasi goreng", [], -3.0, -7.0)]
    out = distinct_nbest(beams, clean_text, topk=5)
    assert [h[0] for h in out] == ["makan nasi", "makan nasi goreng"]
    assert out[0][1:3] == (-1.0, -5.0)
    assert len(distinct_nbest(beams, clean_text, topk=1)) == 1


def test_pick_uses_weights():
    r = NBestRescorer.__new__(NBestRescorer)  # no model load
    r.weights, r.topk = (0.4, 0.4, 1.0), 32
    nlm = {"saya makan": -10.0, "saya makan nasi": -12.0}
    r.nlm_scores = lambda texts: {t: nlm[t] for t in texts}
    nb = [[("saya makan", -1.0, -6.0, []), ("saya makan nasi", -1.5, -7.0, [])]]
    # a: -1 + 0.4*-5 + 0.4*-10 + 2 = -5.0 ; b: -1.5 + 0.4*-5.5 + 0.4*-12 + 3 = -5.5
    assert r.pick(nb)[0][0] == "saya makan"
    r.weights = (0.4, 0.4, 2.0)  # longer hypothesis wins with a larger word bonus
    assert r.pick(nb)[0][0] == "saya makan nasi"
    assert r.pick([[]])[0][0] == ""
