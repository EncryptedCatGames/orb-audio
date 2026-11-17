"""Apply the neural detail model to a rendered cue.

The model is trained on 256-frame STFT crops, so inference runs the same
window size with a 50% overlap and a Hann crossfade. Feeding a whole 12 s cue
in one pass would change the GroupNorm statistics and the effective receptive
field, which is a train/test mismatch.
"""

from __future__ import annotations

import numpy as np
import torch

from ..model.detail import DetailUNet

N_FFT = 1024
HOP = 256
CHUNK = 256
STRIDE = 128


def _stft(x: np.ndarray) -> torch.Tensor:
    return torch.stft(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT),
        return_complex=True,
        center=True,
    )


def _istft(spec: torch.Tensor, length: int) -> np.ndarray:
    return torch.istft(
        spec,
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT, device=spec.device),
        length=length,
    ).cpu().numpy()


def _chunk_starts(t: int, chunk: int, stride: int) -> list[int]:
    if t <= chunk:
        return [0]
    starts = list(range(0, t - chunk + 1, stride))
    if starts[-1] != t - chunk:
        starts.append(t - chunk)
    return starts


def _fade(n: int, device: torch.device) -> torch.Tensor:
    """Hann fade with a floor so the edges are not zeroed out entirely."""
    if n == 1:
        return torch.ones(1, device=device)
    w = torch.hann_window(n + 2, periodic=False, device=device)[1:-1]
    return w.clamp_min(1e-3)


def detail_channel(
    x: np.ndarray,
    model: DetailUNet,
    cond: torch.Tensor | None = None,
    seed: int = 0,
    strength: float = 1.0,
) -> np.ndarray:
    device = next(model.parameters()).device
    spec = _stft(x).to(device)
    t = spec.shape[-1]
    if cond is None:
        cond = torch.zeros(1, model.cond_dim, device=device)
    cond = cond.to(device).reshape(1, -1)

    # The quiet gate needs the cue's active level, not the chunk's: a chunk of
    # pure background must still count as quiet.
    frame_power = torch.mean(torch.abs(spec) ** 2, dim=0)
    active_power = torch.quantile(frame_power.float().cpu(), 0.99).to(device)

    acc = torch.zeros_like(spec)
    weight = torch.zeros(t, device=device)
    with torch.no_grad():
        for k, start in enumerate(_chunk_starts(t, CHUNK, STRIDE)):
            end = min(start + CHUNK, t)
            chunk = spec[:, start:end]
            local = chunk
            if local.shape[-1] < CHUNK:
                local = torch.nn.functional.pad(local, (0, CHUNK - local.shape[-1]))
            scale = torch.sqrt(torch.mean(torch.abs(local) ** 2)) + 1e-6
            inp = torch.stack([local.real, local.imag], dim=0)[None] / scale

            noise = None
            if model.noise_channels:
                gen = torch.Generator(device=device)
                gen.manual_seed((int(seed) * 1000003 + k) & 0x7FFFFFFF)
                noise = torch.randn(
                    1,
                    model.noise_channels,
                    inp.shape[2],
                    inp.shape[3],
                    device=device,
                    generator=gen,
                )
            frame_ref = None
            if getattr(model, "quiet_gate_db", 0.0) > 0:
                frame_ref = (active_power / scale**2).reshape(1)
            residual = model(inp, cond, noise, frame_ref=frame_ref)[0]
            # No DC: a constant offset would leak into silent frames.
            residual[0, 0] = 0.0
            residual = (residual[0] + 1j * residual[1]) * scale
            residual = residual[:, : end - start]

            w = _fade(end - start, device)
            acc[:, start:end] += residual * w
            weight[start:end] += w

    out_spec = spec + strength * acc / weight.clamp_min(1e-6)
    return _istft(out_spec, len(x))


def apply_detail(
    audio: np.ndarray,
    model: DetailUNet,
    cond: torch.Tensor | None = None,
    seed: int = 0,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply detail to mono or stereo audio, preserving shape and length."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return detail_channel(audio, model, cond, seed, strength).astype(np.float32)
    out = np.empty_like(audio)
    for ch in range(audio.shape[1]):
        out[:, ch] = detail_channel(audio[:, ch], model, cond, seed + ch, strength)
    return out
