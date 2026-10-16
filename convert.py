# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""One-time converter: training checkpoint dir -> sea_asr dir (spec section 7).

    python -m sea_asr.convert OLD_DIR NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

ENCODER_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
FORMAT = "sea_asr/1"

_FIXED = {
    "layer_logits": "mix_logits",
    "special_rows": "head.special",
    "unit_rows": "head.units",
    "head_bias": "head.bias",
}
_PREFIX = {
    "in_norm.": "reader.norm_in.",
    "in_proj.": "reader.proj_in.",
    "pos_conv.": "reader.causal_conv.",
    "out_norm.": "reader.norm_out.",
    "to_text.": "reader.proj_text.",
}
_BLOCK = {"norm1": "attn_norm", "qkv": "attn_in", "proj": "attn_out", "norm2": "mlp_norm",
          "ff.0": "mlp_up", "ff.3": "mlp_down"}
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(norm1|qkv|proj|norm2|ff\.0|ff\.3)\.(weight|bias)$")
_ENC_RE = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")


def rename(old: str) -> str:
    if old in _FIXED:
        return _FIXED[old]
    for a, b in _PREFIX.items():
        if old.startswith(a) and old.count(".") == 1:
            return b + old[len(a):]
    m = _BLOCK_RE.match(old)
    if m:
        return f"reader.blocks.{m[1]}.{_BLOCK[m[2]]}.{m[3]}"
    m = _ENC_RE.match(old)
    if m:
        return f"audio_top.{m[1]}.{m[2]}"
    raise KeyError(f"unknown checkpoint tensor: {old}")


def convert(old_dir: str | Path, new_dir: str | Path) -> Path:
    old_dir, new_dir = Path(old_dir), Path(new_dir)
    cfg = json.loads((old_dir / "config.json").read_text())
    units = json.loads((old_dir / "units.json").read_text())
    sd = load_file(str(old_dir / "model.safetensors"))

    if not cfg.get("layer_mix", False):
        raise ValueError("only layer_mix=true checkpoints are supported")
    new_sd = {rename(k): v.contiguous() for k, v in sd.items()}
    top = sorted({int(_ENC_RE.match(k)[1]) for k in sd if _ENC_RE.match(k)})
    if len(top) != int(cfg["unfreeze_top_k"]):
        raise ValueError(f"expected {cfg['unfreeze_top_k']} fine-tuned encoder layers, found {top}")
    vocab = int(cfg["vocab_size"])
    assert len(units["gemma_ids"]) == vocab - 2, "units.json does not match vocab_size"
    assert new_sd["head.bias"].shape == (vocab,)
    n_layers = int(cfg["n_layers"])
    assert {int(k.split(".")[2]) for k in new_sd if k.startswith("reader.blocks.")} == set(range(n_layers))

    new_cfg = {
        "format": FORMAT,
        "encoder_id": cfg["encoder_id"],
        "encoder_revision": ENCODER_REVISION,
        "enc_dim": int(new_sd["reader.norm_in.weight"].shape[0]),
        "d_model": int(cfg["d_model"]),
        "n_layers": n_layers,
        "n_heads": int(cfg["n_heads"]),
        "ff_mult": int(cfg["ff_mult"]),
        "conv_kernel": int(cfg["conv_kernel"]),
        "text_dim": int(new_sd["head.special"].shape[1]),
        "vocab_size": vocab,
        "audio_top_layers": top,
    }
    new_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_sd, str(new_dir / "model.safetensors"), metadata={"format": FORMAT})
    (new_dir / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    shutil.copyfile(old_dir / "units.json", new_dir / "units.json")
    return new_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old_dir")
    ap.add_argument("new_dir")
    a = ap.parse_args()
    print(convert(a.old_dir, a.new_dir))


if __name__ == "__main__":
    main()
