"""Losses for the detail model."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def istft(x: torch.Tensor, n_fft: int = 1024, hop: int = 256, length: int | None = None) -> torch.Tensor:
    """(B, 2, F, T) real/imag → (B, n) waveform."""
    spec = torch.complex(x[:, 0], x[:, 1])
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft, device=x.device),
        length=length,
    )


def _mag(wav: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft, device=wav.device),
        return_complex=True,
    )
    return spec.abs()


def multi_res_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ffts: tuple[int, ...] = (256, 512, 1024),
    hops: tuple[int, ...] = (64, 128, 256),
) -> torch.Tensor:
    loss = pred.new_zeros(())
    for n_fft, hop in zip(ffts, hops):
        p = _mag(pred, n_fft, hop)
        t = _mag(target, n_fft, hop)
        # log1p is stable near zero; log(p + 1e-5) produces 1e5-scale gradients.
        loss = loss + F.l1_loss(p, t) + F.l1_loss(torch.log1p(p), torch.log1p(t))
    return loss / len(ffts)


def relative_mag_loss(pred: torch.Tensor, target: torch.Tensor, eps_scale: float = 0.1) -> torch.Tensor:
    """Per-bin relative magnitude error.

    Absolute L1 lets the model dump energy into quiet high-frequency bins: the
    error is small in absolute terms but enormous relative to the target. This
    weights every bin by its own target magnitude, so silence stays silent.
    """
    p = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-12)
    t = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + 1e-12)
    eps = torch.clamp(eps_scale * t.mean(), min=1e-3)
    return torch.mean(torch.abs(p - t) / (t + eps))


def envelope_loss(pred: torch.Tensor, target: torch.Tensor, hop: int = 256) -> torch.Tensor:
    p = pred.abs().unfold(1, hop, hop).mean(-1)
    t = target.abs().unfold(1, hop, hop).mean(-1)
    n = min(p.shape[-1], t.shape[-1])
    return F.l1_loss(p[:, :n], t[:, :n])


def log_mag_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Symmetric log-magnitude L1.

    Unlike the relative loss, over- and under-shooting a quiet bin cost the
    same, so the model is not rewarded for silence.
    """
    p = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + eps**2)
    t = torch.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2 + eps**2)
    return F.l1_loss(torch.log(p), torch.log(t))


def band_energy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bands: tuple[tuple[float, float], ...] = (
        (0.0, 500.0),
        (500.0, 1000.0),
        (1000.0, 2000.0),
        (2000.0, 4000.0),
        (4000.0, 8000.0),
        (8000.0, 16000.0),
        (16000.0, 24000.0),
    ),
    band_weights: tuple[float, ...] | None = None,
    sr: int = 48000,
    n_fft: int = 1024,
    eps_scale: float = 1e-4,
) -> torch.Tensor:
    """Match per-frame energy in octave bands, in log space.

    The per-bin log-magnitude loss can be satisfied by a smooth low band while
    the high band stays orders of magnitude too quiet. This puts one term on
    each band's total energy, so a missing high band cannot hide. `band_weights`
    lets the high bands dominate: otherwise the model lowers the (too loud) low
    band instead of synthesizing the missing top.
    """
    p = pred[:, 0] ** 2 + pred[:, 1] ** 2
    t = target[:, 0] ** 2 + target[:, 1] ** 2
    freqs = torch.linspace(0.0, sr / 2.0, p.shape[1], device=p.device)
    ep, et = [], []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            ep.append(p[:, m].mean(dim=1))
            et.append(t[:, m].mean(dim=1))
    ep = torch.stack(ep, dim=1)
    et = torch.stack(et, dim=1)
    eps = eps_scale * et.mean() + 1e-12
    diff = torch.abs(torch.log(ep + eps) - torch.log(et + eps))
    if band_weights is None:
        return diff.mean()
    w = torch.tensor(band_weights, device=diff.device, dtype=diff.dtype)[None, :, None]
    return (diff * w).sum() / w.expand_as(diff).sum()


def d_loss_hinge(disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_logits = disc(real)
    fake_logits = disc(fake.detach())
    loss = real.new_zeros(())
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss / len(real_logits)


def g_adv_loss_hinge(disc, fake: torch.Tensor) -> torch.Tensor:
    logits = disc(fake)
    loss = fake.new_zeros(())
    for f in logits:
        loss = loss - f.mean()
    return loss / len(logits)


def feature_matching_loss(disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    fr = disc.features(real)
    ff = disc.features(fake)
    loss = fake.new_zeros(())
    for a, b in zip(fr, ff):
        loss = loss + F.l1_loss(b, a.detach())
    return loss / len(fr)


def d_loss(disc, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_logits = disc(real)
    fake_logits = disc(fake.detach())
    loss = real.new_zeros(())
    for r, f in zip(real_logits, fake_logits):
        loss = loss + F.binary_cross_entropy_with_logits(r, torch.ones_like(r))
        loss = loss + F.binary_cross_entropy_with_logits(f, torch.zeros_like(f))
    return loss / len(real_logits)


def g_adv_loss(disc, fake: torch.Tensor) -> torch.Tensor:
    logits = disc(fake)
    loss = fake.new_zeros(())
    for f in logits:
        loss = loss + F.binary_cross_entropy_with_logits(f, torch.ones_like(f))
    return loss / len(logits)
