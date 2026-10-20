# Implementation notes (clean-room)

Built from a written spec of the checkpoint format and numerics (not published), the transformers / torch / huggingface_hub source, and the checkpoint
files only. Where the spec was silent or ambiguous, the most literal reading was taken; each choice is below.

## Dependency pins

1. The spec pins no library versions. The encoder checkpoint says `transformers_version: 5.18.0.dev0`, but the
   first PyPI release that has `embedding_gemma2` is **5.19.0** (5.18.0 does not). Pinned `transformers==5.19.0`.
2. `torch==2.14.1` (latest stable at build time); the training torch version is not in the spec. Exact
   reproduction was reached with it, so no search was needed.

## Numerics

3. Section 5 "run steps 2-4 above inside one autocast region, including the mel matmul in step 2.4": read as
   sections 2, 3 and 4 (features, encoder, reader and head). So the head matmul `t.float() @ rows.float().T` also
   runs under bf16 autocast (the `.float()` calls do not stop autocast). Consequence: under autocast the
   `+ mel_floor` add happens on the bf16 matmul output.
4. Inference runs under `torch.no_grad()` (spec is silent). The requires_grad layout of spec 3.3 is still applied.
5. Layer mix `Σ w[i] * state[i]`: plain sequential sum over i = 0..12 (Python `sum`), not stack+reduce.
6. `c = x * mask` in the causal conv uses the boolean mask (`x * mask[..., None]`), so no dtype promotion.
7. q/k/v: `Linear(d → 3d)` output reshaped to (B, T, 3, H, d/H), `unbind(2)`, then each transposed to (B, H, T, Dh)
   for SDPA. Default SDPA scale, no dropout.
8. LayerNorm eps is not in the spec: PyTorch default 1e-5 (all reader norms have weight and bias).
9. Invalid feature frames are zeroed with `masked_fill(~mask, 0)`. `window` and `mel_filters` are cast to float32
   (the extractor stores float32 / float64 numpy arrays); `mel_floor` is used as a Python float.

## Splitting, batching, decoding

10. Energy search: sums of squares in float64 via cumulative sum over all 800-sample windows that fit in
    `[start + 25 s, start + 30 s)`; the first minimum wins on ties; cut = window start + 400 samples.
11. Batching rule `longest_piece_seconds × (pieces_in_batch + 1) ≤ 600`: "longest" = longest of the current batch
    plus the candidate piece (with ascending sort this is the candidate). A batch always takes at least one piece.
12. Piece texts are joined with one space; pieces with empty text are skipped so no double spaces appear.
13. Text = `tokenizer.decode(ids)` with default arguments, then `strip()`.
14. Word timestamps: unk units (1) carry no text; they neither start nor join a word. A word whose decoded text is
    empty is dropped. Word text = decode of the word's own Gemma ids, stripped. Times are rounded to 1 ms.
15. Acceptance harness: a hypothesis that normalises to `None` is compared as the empty string.

## Subtitles

16. Cues are filled greedily: a word goes on the current line if the line stays ≤ 32 characters, else on a second
    line; a third line or a cue span > 6 s starts a new cue. A word longer than 32 characters is hard-split into
    32-character chunks with linearly shared timing. A single word that itself lasts > 6 s cannot be split in time,
    so its cue exceeds 6 s (never seen in tests).

## Checkpoint format (new)

17. Tensor names: `mix_logits`; `reader.{norm_in, proj_in, causal_conv, blocks.{i}.{attn_norm, attn_in, attn_out,
    mlp_norm, mlp_up, mlp_down}, norm_out, proj_text}`; `head.{special, units, bias}`; `audio_top.{layer}.*`
    (the fine-tuned top audio-tower layers keep the HF sub-module names below the layer index, because they load
    into the HF module). `config.json` adds `format`, `encoder_revision`, `enc_dim`, `text_dim` and
    `audio_top_layers`, and drops `dropout`, `chunk_frames`, `layer_mix` (must be true) and `unfreeze_top_k`
    (replaced by the explicit layer list). `units.json` is copied unchanged.

## Audio input

18. Multichannel files are averaged to mono. A bare array is assumed to be 16 kHz; pass `(array, sr)` otherwise.

## LM decoding (v0.2, not in the spec)

19. `sea_asr/lm.py` is written from the public pyctcdecode 0.5.0 and kenlm APIs only. Labels: blank `""`, unk `"⁇"`,
    then the sentencepiece piece of each Gemma id, so pyctcdecode runs in BPE mode. unk stays a word-internal `⁇`
    (as in `scripts/eval_lm.py`); `<unk>` would map to pyctcdecode's `▁⁇▁`, whose trailing `▁` sets a
    `force_next_break` flag shared by all beams in a frame and splits words in other beams. `⁇` is removed from the
    output and its neighbours join, as in greedy decoding.
20. Log-probs are `log_softmax` in float32 of the (autocast) logits, computed per piece row so a 600 s batch never
    holds the full [B, T, 34k] float32 tensor on the host.
21. Word timestamps with the LM come from pyctcdecode's word frames `[start, end)` × 40 ms (+ piece offset).

## v0.2.1: beams, workers, pooled decoding

22. Two beams: `beam` (plain LM decode, `decode` / `transcribe`) and `rescore_beam` (the N-best list that
    `sea_asr.rescore` reranks). Arguments override `lm_config.json`. A v0.2.0 config without `"rescore_beam"` falls
    back to `beam` (the old single-beam behaviour) and logs one warning the first time the N-best path runs.
23. `lm_workers=None` (the default) = 1: decode in-process, no pool. Measured with compress_blanks on, beam 128,
    s of beam search per audio hour. H100 host, 32 vCPU, FLEURS-id dev: 1/4/8/16/32 workers = 4.9/6.0/6.1/6.7/8.4
    (plain), 6.0/6.7/6.4/7.4/9.9 (N-best). Mac, 12 cores: 4 workers about 2x faster than 1, both on FLEURS-id dev
    (1.6 vs 0.8 s for 0.4 h) and on one 11-min file (0.8 vs 0.4 s). So the pool helps or hurts by host, not by input
    size; 1 worker is the default and a one-line hint is logged once. `usable_cpus()` (affinity mask, else
    `os.cpu_count()`, capped by a cgroup v2 `cpu.max` or v1 `cpu.cfs_quota_us / cpu.cfs_period_us` quota) is used in
    the hint and is there for callers that size a pool.
24. One fork pool per `LMDecoder`, made on the first pooled call and reused for every batch and both paths (v0.2.0
    made a new pool per call); `close()`, the context manager or garbage collection stops it.
25. Pooled calls do not pickle the dense [T × 34,610] log-probs. The parent copies a batch once into one
    `multiprocessing.shared_memory` block; each worker maps its utterance as a read-only view and runs pyctcdecode's
    public `decode()` / `decode_beams()`, so results equal pyctcdecode's by construction (tests/test_beams_workers.py)
    and pyctcdecode's per-row re-softmax stays parallel. The parent unlinks the block after the batch. The parent starts
    its resource tracker before forking the pool, so the children share it: their attach re-registers the same name
    (a no-op) and the parent's unlink unregisters it once. (A tracker started later gives each child its own tracker,
    which reports the block as leaked; a child-side unregister makes the parent's unlink raise KeyError.)
    A CSR transfer of only pyctcdecode's kept entries (>= token_min_logp plus the frame argmax) was also exact, but it
    moves that re-softmax into the serial parent and was slower than one process on FLEURS-id (Mac, 12 cores); a
    GPU-side kept set cannot be bit-exact, because the re-softmax depends on every entry of the row and on numpy's
    float32 summation order.
26. `compress_blanks` (default on; `compress_blanks=False`, `"compress_blanks": false` in lm_config.json or CLI
    `--no-compress-blanks` turn it off): before every beam search, each run of blank-only frames is collapsed into its
    first frame (run collapsing from `lm_fast.compress_blanks`). Blank-only = blank is the argmax and every other unit
    is below pyctcdecode's `token_min_logp` (−5) minus a 0.01 margin, so pyctcdecode expands only blank there and
    every beam takes it: texts, word frames and the N-best order cannot change, and every hypothesis's acoustic and
    combined scores move by the same constant (so rescoring picks the same text). `lm_fast`'s p(blank) > 0.99 rule
    was not safe: a token can keep log p ≥ −5 at such a frame, and the beam used it one frame early; dropping that
    frame moved 16 of 276 word starts by 40 ms on FLEURS-id. One frame stays per run, so a unit repeated across a
    blank is not merged. Word times use the kept → original frame map: start = original index of the first kept
    frame, end = original index of the last kept frame + 1.

## v0.2.2: rescoring settings in lm_config.json

27. `lm_config.json` may hold a `"rescore"` object with the keys `prompt`, `a`, `b`, `c` and `top_k` (any subset).
    Resolution per key: `SeaASR.from_pretrained` argument (`rescore_prompt`, `rescore_weights=(a, b, c)`,
    `rescore_topk`), then the block, then the v0.2.1 default (Indonesian prompt, a 0.4, b 0.4, c 1.0, top 32). Unknown
    keys raise `ValueError`, so a typo cannot silently fall back to the defaults. `LMDecoder.from_dir` keeps the whole
    file as `LMDecoder.config`; `sea_asr.rescore.rescore_settings()` resolves it.
28. Nothing else changed: with no block and no arguments the rescorer gets the same prompt ids, weights and top-k as
    v0.2.1. Checked on 120 FLEURS-id validation utterances (PixelML/sea-asr-id, Mac CPU fp32): texts, N-best lists
    and picks identical to v0.2.1.
29. Measured Malay setting (PixelML/sea-asr-ms `lm/lm_config.json`): prompt "Berikut ialah transkripsi pertuturan
    dalam bahasa Melayu:\n", a 0.4, b 0.4, c 2.0, top 32, chosen on FLEURS ms_my validation only.
