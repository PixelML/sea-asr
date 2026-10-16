# sea_asr

CTC speech recognition for Indonesian and Tagalog/Taglish. Inference only (the training recipe follows in a later
release). Apache-2.0. Models: [PixelML/sea-asr-ph](https://huggingface.co/PixelML/sea-asr-ph).

## Install

```bash
pip install "git+https://github.com/PixelML/sea-asr@v0.1.0"   # Python >= 3.12
pip install -e .                                            # from a clone
```

Pinned: torch 2.14.1, transformers 5.19.0, huggingface_hub 2.1.1, safetensors 0.8.0, numpy 2.5.3.

The first load downloads the encoder from the Hugging Face Hub (accept the Gemma terms and log in if needed).

## Use

```python
from sea_asr import SeaASR, to_srt, to_vtt

asr = SeaASR.from_pretrained("PixelML/sea-asr-ph")         # HF repo id (+ revision=...) or a local dir
print(asr.transcribe("clip.flac"))                          # one string
texts = asr.transcribe(["a.flac", "b.flac"])                # list in, list out (batched)

out = asr.transcribe("talk.flac", return_words=True)       # {"text", "words": [{"word", "start", "end"}]}
open("talk.srt", "w").write(to_srt(out["words"]))
open("talk.vtt", "w").write(to_vtt(out["words"]))
```

Example SRT output (id-v3 on the first FLEURS-id test clips, joined into one 80 s file):

```
1
00:00:00,880 --> 00:00:04,120
tim tim virtual memiliki standar
keunggulan yang sama dengan tim

2
00:00:04,320 --> 00:00:10,320
konvensional tetapi ada sedikit
perbedaan segera setelah
```

Cues have at most 2 lines of 32 characters and last at most 6 s. Inputs longer than 30 s are cut at quiet points
and the timestamps stay on the original timeline. Input: a file path, a float32 16 kHz mono array, or an
`(array, sample_rate)` tuple (resampled with soxr). On CUDA the model runs under bf16 autocast; on CPU in float32.

## Convert a training checkpoint

```bash
python -m sea_asr.convert runs/my-run/best out/my-run      # training checkpoint -> sea_asr format
```

## Tests

```bash
pytest tests/test_decode.py                                 # CPU unit tests
```

## Credits

- Base encoder: Google DeepMind, EmbeddingGemma 2 ([914f7f89](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b))
- Encoder implementation: Hugging Face, transformers ([v5.19.0](https://github.com/huggingface/transformers/tree/v5.19.0))
- Data: YODAS (ESPnet/CMU, [52c5a1b9](https://huggingface.co/datasets/espnet/yodas/tree/52c5a1b9730a136bfd7c4513d4962c70d4e50530)); FLEURS (Google, [70bb2e84](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)); CC-BY YouTube creators listed in each model's ATTRIBUTION
- Teachers: Qwen3-ASR-1.7B (Alibaba Qwen, [7278e1e7](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)); Whisper large-v3 (OpenAI, [openai/whisper](https://github.com/openai/whisper)) as [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3/tree/edaa852ec7e145841d8ffdb056a99866b5f0a478) (`edaa852e`) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 (SYSTRAN)
- Test sets: FLEURS test (Google, same revision)
- Recipe: Quan Nguyen ([@qnguyen3](https://github.com/qnguyen3)), QUAN-Transcribe — https://github.com/qnguyen3/QUAN-Transcribe ([4f45e02](https://github.com/qnguyen3/QUAN-Transcribe/tree/4f45e02))

## Licence

Apache-2.0. Base encoder EmbeddingGemma 2 is Apache-2.0; deployments must adhere to the Gemma Prohibited Use Policy.
