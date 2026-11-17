"""Detail model: a complex-STFT U-Net that predicts a residual.

Input:  (B, 2, F, T) real/imag STFT of the DSP render.
Output: (B, 2, F, T) residual, scaled down so it cannot copy the target.

Artifact controls (all off by default so older checkpoints load unchanged):

- ``upsample="resize_conv"`` replaces the stride-2 transposed convolutions in
  the decoder with interpolate + 3x3 conv. A k=4, s=2 transposed conv has
  uneven kernel overlap, which prints a periodic (checkerboard) pattern along
  the frequency axis; on audio that is a comb of narrowband stripes.
- ``residual_taper_hz=(lo, hi)`` fades the residual to zero between ``lo`` and
  ``hi`` with a raised cosine and zeroes it above ``hi``. The model never
  learned to synthesise the top octave, so it is not allowed to touch it.
- ``residual_mode="mask"`` predicts a real gain ``g`` per bin (bounded by
  ``residual_scale``, optionally box-smoothed by ``mask_smooth``) and returns
  ``g * X``. A free complex residual built by convolutions that mix
  neighbouring frames is a copy of the input shifted by one hop: a comb filter
  with teeth every ``sr / hop`` Hz. A per-bin gain cannot mix frames, cannot
  create energy where the input has none, and cannot write a tone.
- ``quiet_gate_db=d`` fades the residual to zero in frames more than ``d`` dB
  below the reference frame level (``frame_ref``, or the loudest frame of the
  crop), over ``quiet_ramp_db``. Quiet passages are left untouched, so the
  stage cannot lift a noise floor between events.
- ``freq_bins=n`` runs the U-Net on the lowest ``n`` STFT bins only and
  returns a zero residual above them (use with a taper below that bin).
- ``residual_env_bound=a`` soft-limits the residual magnitude in every bin to
  ``a`` times the local (smoothed) input magnitude. The model may reshape
  what is there but cannot paint energy into silence or into a dark band.
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
        upsample: str = "transpose",
        upsample_mode: str = "bilinear",
        residual_taper_hz: tuple[float, float] | list[float] | None = None,
        residual_env_bound: float = 0.0,
        env_kernel: tuple[int, int] | list[int] = (9, 5),
        sample_rate: int = 48000,
        freq_bins: int = 0,
        residual_mode: str = "additive",
        mask_smooth: tuple[int, int] | list[int] | None = None,
        quiet_gate_db: float = 0.0,
        quiet_ramp_db: float = 10.0,
    ):
        super().__init__()
        if upsample not in ("transpose", "resize_conv"):
            raise ValueError(f"unknown upsample {upsample!r}")
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        self.residual_taper_hz = tuple(residual_taper_hz) if residual_taper_hz else None
        self.residual_env_bound = float(residual_env_bound)
        self.env_kernel = tuple(int(k) for k in env_kernel)
        self.sample_rate = int(sample_rate)
        self.freq_bins = int(freq_bins)
        if residual_mode not in ("additive", "mask"):
            raise ValueError(f"unknown residual_mode {residual_mode!r}")
        self.residual_mode = residual_mode
        self.mask_smooth = tuple(int(k) for k in mask_smooth) if mask_smooth else None
        self.quiet_gate_db = float(quiet_gate_db)
        self.quiet_ramp_db = float(quiet_ramp_db)
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
        if upsample == "transpose":
            self.ups = nn.ModuleList(
                [nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1) for i in range(levels - 1, -1, -1)]
            )
        else:
            # Resize first, then convolve: every output position sees the same
            # number of kernel taps, so there is no stride-periodic pattern.
            self.ups = nn.ModuleList(
                [nn.Conv2d(chs[i + 1], chs[i], 3, padding=1) for i in range(levels - 1, -1, -1)]
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
        frame_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``frame_ref`` (B,) is the reference frame power (mean |X|^2 over
        frequency, in the units of ``x``) for the quiet gate; the renderer
        passes the whole cue's active level so a chunk of pure background is
        still recognised as quiet."""
        clean = x
        n_freq = x.shape[2]
        if self.freq_bins and n_freq > self.freq_bins:
            x = x[:, :, : self.freq_bins]
            if noise is not None:
                noise = noise[:, :, : self.freq_bins]
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
            skip = skips[-(i + 2)]
            if self.upsample == "transpose":
                h = up(h)
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            else:
                kw = {"align_corners": False} if self.upsample_mode in ("bilinear", "bicubic") else {}
                h = up(F.interpolate(h, size=skip.shape[-2:], mode=self.upsample_mode, **kw))
            h = block(torch.cat([h, skip], dim=1), cond)
        out = torch.tanh(self.out(h)) * self.residual_scale
        if out.shape[2] < n_freq:
            out = F.pad(out, (0, 0, 0, n_freq - out.shape[2]))
            if noise_in is not None:
                noise_in = F.pad(noise_in, (0, 0, 0, n_freq - noise_in.shape[2]))
        if self.residual_mode == "mask":
            gain = out[:, :1]
            if self.mask_smooth:
                kf, kt = self.mask_smooth
                gain = F.avg_pool2d(gain, (kf, kt), stride=1, padding=(kf // 2, kt // 2), count_include_pad=False)
            out = clean * gain
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
        if self.residual_env_bound > 0:
            out = self._env_bound(out, clean)
        if self.quiet_gate_db > 0:
            power = (clean[:, 0] ** 2 + clean[:, 1] ** 2).mean(dim=1)  # (B, T)
            ref = frame_ref.reshape(-1, 1) if frame_ref is not None else power.amax(dim=1, keepdim=True)
            level = 10.0 * torch.log10((power + 1e-12) / (ref + 1e-12))
            w = ((level + self.quiet_gate_db) / self.quiet_ramp_db).clamp(0.0, 1.0)
            out = out * w[:, None, None, :]
        if self.residual_taper_hz is not None:
            out = out * self.taper(out.shape[2], out.device, out.dtype)[None, None, :, None]
        return out

    def taper(self, n_bins: int, device=None, dtype=None) -> torch.Tensor:
        """(F,) frequency weights: 1 below lo, raised-cosine fade, 0 above hi."""
        freqs = torch.linspace(0.0, self.sample_rate / 2.0, n_bins, device=device, dtype=dtype)
        if self.residual_taper_hz is None:
            return torch.ones_like(freqs)
        lo, hi = self.residual_taper_hz
        u = ((freqs - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(torch.pi * u))

    def _env_bound(self, out: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Soft-limit |residual| to a * smoothed |input| in every bin.

        The magnitude goes through a * S * tanh(|r| / (a * S)), which is the
        identity for small residuals and saturates at a * S. Phase is kept.
        """
        kf, kt = self.env_kernel
        mag = torch.sqrt(clean[:, 0] ** 2 + clean[:, 1] ** 2 + 1e-12)[:, None]
        env = F.avg_pool2d(mag, (kf, kt), stride=1, padding=(kf // 2, kt // 2), count_include_pad=False)[:, 0]
        limit = self.residual_env_bound * env + 1e-8
        r = torch.sqrt(out[:, 0] ** 2 + out[:, 1] ** 2 + 1e-12)
        gain = limit * torch.tanh(r / limit) / r
        return out * gain[:, None]
