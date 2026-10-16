# Copyright 2026 Pixel ML. Licensed under the Apache License, Version 2.0.
"""Encoder wrapper, reader stack and CTC head (spec sections 3 and 4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

AUDIO_TOP_PREFIX = "audio_top."  # new-format prefix for the fine-tuned top encoder layers


class ReaderBlock(nn.Module):
    """Pre-norm transformer block: masked self-attention, then a GELU MLP."""

    def __init__(self, width: int, heads: int, mlp_mult: int):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(width)
        self.attn_in = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp_up = nn.Linear(width, mlp_mult * width)
        self.mlp_down = nn.Linear(mlp_mult * width, width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        qkv = self.attn_in(self.attn_norm(x)).reshape(b, t, 3, self.heads, w // self.heads)
        q, k, v = (z.transpose(1, 2) for z in qkv.unbind(2))  # each (B, H, T, Dh)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        a = a.transpose(1, 2).reshape(b, t, w)
        x = x + self.attn_out(a)
        x = x + self.mlp_down(F.gelu(self.mlp_up(self.mlp_norm(x))))
        return x


class Reader(nn.Module):
    def __init__(self, enc_dim: int, width: int, depth: int, heads: int, mlp_mult: int, conv_kernel: int,
                 text_dim: int):
        super().__init__()
        self.conv_kernel = conv_kernel
        self.norm_in = nn.LayerNorm(enc_dim)
        self.proj_in = nn.Linear(enc_dim, width)
        self.causal_conv = nn.Conv1d(width, width, conv_kernel, groups=width)
        self.blocks = nn.ModuleList(ReaderBlock(width, heads, mlp_mult) for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.proj_text = nn.Linear(width, text_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.norm_in(h))
        c = (x * mask[..., None]).transpose(1, 2)
        c = F.pad(c, (self.conv_kernel - 1, 0))
        c = self.causal_conv(c).transpose(1, 2)
        x = x + F.gelu(c)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.proj_text(self.norm_out(x))


class UnitHead(nn.Module):
    """Logits = text-space vector . [special rows; unit rows] + bias."""

    def __init__(self, vocab: int, text_dim: int):
        super().__init__()
        self.special = nn.Parameter(torch.zeros(2, text_dim))
        self.units = nn.Parameter(torch.zeros(vocab - 2, text_dim))
        self.bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        rows = torch.cat([self.special, self.units], dim=0)
        return t.float() @ rows.float().T + self.bias.float()


class SeaASRModel(nn.Module):
    """Gemma audio tower (frozen except the top layers) + layer mix + reader + unit head."""

    def __init__(self, audio_tower: nn.Module, cfg: dict):
        super().__init__()
        self.audio = audio_tower
        self.top_layers = list(cfg["audio_top_layers"])
        n_states = 1 + len(audio_tower.layers)
        self.mix_logits = nn.Parameter(torch.zeros(n_states))
        self.reader = Reader(cfg["enc_dim"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["ff_mult"],
                             cfg["conv_kernel"], cfg["text_dim"])
        self.head = UnitHead(cfg["vocab_size"], cfg["text_dim"])
        self._states: list[torch.Tensor] = []
        self.audio.subsample_conv_projection.register_forward_hook(self._grab)
        for layer in self.audio.layers:
            layer.register_forward_hook(self._grab)

    def _grab(self, _module, _inputs, output):
        self._states.append(output[0] if isinstance(output, tuple) else output)

    def set_grad_flags(self) -> None:
        """Training-time requires_grad layout; it affects bf16 autocast numerics (spec 3.3)."""
        for p in self.audio.parameters():
            p.requires_grad_(False)
        for i in self.top_layers:
            for p in self.audio.layers[i].parameters():
                p.requires_grad_(True)

    def forward(self, feats: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._states = []
        out = self.audio(feats, frame_mask, return_dict=True)
        mask = out.attention_mask.bool()
        states, self._states = self._states, []
        assert len(states) == self.mix_logits.numel(), f"captured {len(states)} states"
        w = torch.softmax(self.mix_logits.float(), dim=0)
        h = sum(w[i] * s.float() for i, s in enumerate(states))
        t = self.reader(h, mask)
        logits = self.head(t)
        return logits, mask.sum(1)


def load_audio_tower(encoder_id: str, revision: str | None) -> nn.Module:
    from transformers import AutoModel

    full = AutoModel.from_pretrained(encoder_id, revision=revision, dtype=torch.bfloat16, vision_config=None)
    tower = full.audio_tower.float()
    del full
    return tower
