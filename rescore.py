# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Optional second pass: rescore the KenLM beam search's N-best list with a small causal LM. Off by default.

    asr = SeaASR.from_pretrained("PixelML/sea-asr-id", lm="PixelML/sea-asr-id", alpha=0.4, beta=1.5, beam=128,
                                 rescore=True)          # pip install 'sea_asr[lm]'

score(h) = am + a * (kenlm_beam - am) + b * nlm + c * words
    am          acoustic log-prob of the beam (pyctcdecode "logit score")
    kenlm_beam  pyctcdecode's beam score (am + alpha * KenLM + beta * words)
    nlm         sum of the causal LM's log-probs of the hypothesis tokens + EOS after a fixed Indonesian prompt
Defaults (a 0.4, b 0.4, c 1.0, top 32 distinct hypotheses, beam built with prune_history=False) were chosen on FLEURS
id validation only. Measured for id-v4 + id-w4 KenLM: FLEURS test 6.11 -> 5.48, CV27 7.00 -> 6.28, GigaSpeech 2
17.25 -> 16.54 WER. The default LM is Sea AI Lab's Sailor2-1B (Apache-2.0), pinned by commit.
"""

from __future__ import annotations

DEFAULT_MODEL = "sail/Sailor2-1B"
DEFAULT_REVISION = "75c2856ffc7e6a6e64127f40c5f54a953deb682e"
DEFAULT_WEIGHTS = (0.4, 0.4, 1.0)
DEFAULT_TOPK = 32
PROMPT = "Berikut transkripsi ucapan dalam bahasa Indonesia:\n"


class NBestRescorer:
    def __init__(self, model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION, device=None,
                 weights: tuple[float, float, float] = DEFAULT_WEIGHTS, topk: int = DEFAULT_TOPK, batch: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype).to(self.device).eval()
        self.prefix = self.tok(PROMPT, add_special_tokens=False)["input_ids"]
        self.weights, self.topk, self.batch = weights, topk, batch

    def nlm_scores(self, texts: list[str]) -> dict[str, float]:
        """Log-prob of each distinct text (hypothesis tokens + EOS), batched shortest first."""
        import torch

        out, uniq = {}, sorted(set(texts), key=len)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        for s in range(0, len(uniq), self.batch):
            bt = uniq[s:s + self.batch]
            ids = [self.prefix + self.tok(t, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id] for t in bt]
            width = max(map(len, ids))
            x = torch.full((len(ids), width), pad, dtype=torch.long)
            mask = torch.zeros((len(ids), width), dtype=torch.long)
            for j, row in enumerate(ids):
                x[j, : len(row)] = torch.tensor(row)
                mask[j, : len(row)] = 1
            x, mask = x.to(self.device), mask.to(self.device)
            with torch.no_grad():
                lp = torch.log_softmax(self.lm(input_ids=x, attention_mask=mask).logits.float(), -1)
            tl = lp[:, :-1].gather(-1, x[:, 1:, None])[..., 0]
            for j, row in enumerate(ids):
                out[bt[j]] = float(tl[j, len(self.prefix) - 1: len(row) - 1].sum())
        return out

    def pick(self, nbests: list[list[tuple]]) -> list[tuple]:
        """nbests: per utterance [(text, am, kenlm_beam, ...), ...] best first. Returns the chosen tuple per utterance
        (an empty list gives ("", 0, 0))."""
        a, b, c = self.weights
        nbests = [nb[: self.topk] for nb in nbests]
        scores = self.nlm_scores([h[0] for nb in nbests for h in nb])
        return [max(nb, key=lambda h: h[1] + a * (h[2] - h[1]) + b * scores[h[0]] + c * len(h[0].split()))
                if nb else ("", 0.0, 0.0) for nb in nbests]


def distinct_nbest(beams, clean, topk: int) -> list[tuple]:
    """pyctcdecode beams (text, word_frames, logit_score, lm_score) -> [(clean text, am, beam score, word_frames)],
    first occurrence of each cleaned text only (the beams come best first)."""
    seen, out = set(), []
    for text, frames, am, comb in beams:
        t = clean(text)
        if t in seen:
            continue
        seen.add(t)
        out.append((t, float(am), float(comb), frames))
        if len(out) == topk:
            break
    return out


__all__ = ["NBestRescorer", "distinct_nbest", "DEFAULT_MODEL", "DEFAULT_REVISION"]
