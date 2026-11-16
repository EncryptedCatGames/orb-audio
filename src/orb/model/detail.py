"""Detail model: a complex-STFT U-Net that predicts a residual.

Input:  (B, 2, F, T) real/imag STFT of the DSP render.
Output: (B, 2, F, T) residual, scaled down so it cannot copy the target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond)[:, :, None, None]
        beta = self.to_beta(cond)[:, :, None, None]
        return x * (1.0 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int | None = None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(cond_dim, out_ch) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(self.conv(x))
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return self.act(x)


class DetailUNet(nn.Module):
    def __init__(
        self,
        cond_dim: int = 128,
        base: int = 64,
        levels: int = 4,
        residual_scale: float = 0.5,
        noise_channels: int = 0,
        energy_gate: bool = False,
        residual_min_bin: int = 0,
        noise_floor: float = 0.0,
    ):
        super().__init__()
        chs = [base * (2 ** min(i, 2)) for i in range(levels + 1)]
        self.residual_scale = residual_scale
        self.noise_channels = noise_channels
        self.energy_gate = energy_gate
        self.residual_min_bin = residual_min_bin
        self.cond_dim = cond_dim
        self.noise_floor = nn.Parameter(torch.tensor(float(noise_floor)))
        self.inc = ConvBlock(2 + noise_channels, chs[0], cond_dim)
        self.downs = nn.ModuleList(
            [nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1) for i in range(levels)]
        )
        self.down_blocks = nn.ModuleList(
            [ConvBlock(chs[i + 1], chs[i + 1], cond_dim) for i in range(levels)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1) for i in range(levels - 1, -1, -1)]
        )
        self.up_blocks = nn.ModuleList(
            [ConvBlock(chs[i] * 2, chs[i], cond_dim) for i in range(levels - 1, -1, -1)]
        )
        self.out = nn.Conv2d(chs[0], 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean = x
        noise_in = noise
        if self.noise_channels:
            if noise is None:
                noise = torch.randn(
                    x.shape[0],
                    self.noise_channels,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
            noise_in = noise
            x = torch.cat([x, noise], dim=1)
        h = self.inc(x, cond)
        skips = [h]
        for down, block in zip(self.downs, self.down_blocks):
            h = block(down(h), cond)
            skips.append(h)
        for i, (up, block) in enumerate(zip(self.ups, self.up_blocks)):
            h = up(h)
            skip = skips[-(i + 2)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), cond)
        out = torch.tanh(self.out(h)) * self.residual_scale
        if self.residual_min_bin:
            # The DSP low band is already right; spend the model on the top.
            out = out.clone()
            out[:, :, : self.residual_min_bin] = 0.0
        if self.noise_channels and noise_in is not None and self.residual_min_bin:
            # A learnable noise floor gives the high band a non-zero starting
            # magnitude. The band loss has zero gradient at exactly zero, so
            # without this the model cannot grow the top from nothing.
            floor = torch.clamp(self.noise_floor, min=0.0)
            out = out.clone()
            out[:, 0, self.residual_min_bin :] += floor * noise_in[:, 0, self.residual_min_bin :]
        if self.energy_gate:
            # Detail may only appear where the input has signal. Without this,
            # the residual fills silence and shifts the onset. The sqrt softens
            # the transition near zero but stays ~1 wherever there is signal.
            env = (clean[:, 0] ** 2 + clean[:, 1] ** 2).mean(dim=1, keepdim=True)
            gate = torch.sqrt(env / (env + 1e-3))
            out = out * gate.unsqueeze(2)  # (B, 1, 1, T), broadcast over frequency
        return out
