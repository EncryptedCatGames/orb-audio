"""Reward model: audio-text matching, realness, and sync."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, d, 3, padding=1),
            nn.GroupNorm(8, d),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(mel)).flatten(1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d: int = 256, layers: int = 4, heads: int = 4, ctx: int = 512):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=4 * d,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(t, device=ids.device))
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))


class RewardModel(nn.Module):
    def __init__(self, vocab_size: int, d: int = 256, text_layers: int = 4, text_heads: int = 4, ctx: int = 512):
        super().__init__()
        self.audio = AudioEncoder(d)
        self.text = TextEncoder(vocab_size, d, text_layers, text_heads, ctx)
        self.match_audio = nn.Linear(d, 128)
        self.match_text = nn.Linear(d, 128)
        self.realness = nn.Linear(d, 1)
        self.sync = nn.Linear(d, 1)

    def forward(self, mel: torch.Tensor, text_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.audio(mel)
        t = self.text(text_ids)
        return {
            "audio": a,
            "text": t,
            "match_audio": F.normalize(self.match_audio(a), dim=-1),
            "match_text": F.normalize(self.match_text(t), dim=-1),
            "realness": self.realness(a).squeeze(-1),
            "sync": self.sync(a).squeeze(-1),
        }
