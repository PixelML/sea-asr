# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Command line: python -m sea_asr --model PixelML/sea-asr-ph a.flac b.flac [--lm DIR] [--srt OUT_DIR]"""

import argparse
from pathlib import Path

from . import SeaASR, to_srt


def main():
    ap = argparse.ArgumentParser(prog="sea_asr")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", required=True, help="local dir or HF repo id")
    ap.add_argument("--revision")
    ap.add_argument("--device")
    ap.add_argument("--lm", help="KenLM directory or HF repo id with an lm/ folder (needs sea_asr[lm])")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--beta", type=float)
    ap.add_argument("--beam", type=int)
    ap.add_argument("--lm-workers", type=int, default=1)
    ap.add_argument("--srt", help="write <name>.srt files into this directory")
    a = ap.parse_args()
    asr = SeaASR.from_pretrained(a.model, revision=a.revision, device=a.device, lm=a.lm, alpha=a.alpha, beta=a.beta,
                                 beam=a.beam, lm_workers=a.lm_workers)
    outs = asr.transcribe(list(a.audio), return_words=bool(a.srt))
    for path, out in zip(a.audio, outs):
        if a.srt:
            Path(a.srt).mkdir(parents=True, exist_ok=True)
            (Path(a.srt) / (Path(path).stem + ".srt")).write_text(to_srt(out["words"]), encoding="utf-8")
            out = out["text"]
        print(f"{path}\t{out}")


if __name__ == "__main__":
    main()
