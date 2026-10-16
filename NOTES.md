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
