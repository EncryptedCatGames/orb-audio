"""Multi-scale spectrogram discriminator, training only."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 4),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Conv2d(base * 4, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 32, scales: int = 2):
        super().__init__()
        self.discs = nn.ModuleList([Discriminator(in_ch, base) for _ in range(scales)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        h = x
        for disc in self.discs:
            out.append(disc(h))
            h = F.avg_pool2d(h, 2)
        return out

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        h = x
        for disc in self.discs:
            out.append(disc.features(h))
            h = F.avg_pool2d(h, 2)
        return out
