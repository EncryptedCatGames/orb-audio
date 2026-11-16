"""Small mel implementation, no torchaudio dependency."""

from __future__ import annotations

import numpy as np
import torch


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2)
    hz = mel_to_hz(mel)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - lo) / max(center - lo, 1e-9)
        right = (hi - freqs) / max(hi - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
    return fb


def log_mel(
    x: np.ndarray,
    sr: int,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop: int = 256,
) -> torch.Tensor:
    """(n,) waveform → (1, n_mels, T) log-mel."""
    spec = torch.stft(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs()
    fb = torch.from_numpy(mel_filterbank(sr, n_fft, n_mels))
    mel = fb @ spec
    return torch.log(mel + 1e-5).unsqueeze(0)
