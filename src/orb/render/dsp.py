"""Small, dependency-free DSP helpers.

numpy only. Every function is deterministic and CPU-fast enough for realtime
rendering of a 30 s cue.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TAU = 2.0 * np.pi
LN1000 = 6.907755278982137  # ln(1000), for -60 dB decay over t60


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def normalize_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-9 else x * (peak / m)


def soft_limit(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m <= ceiling or m < 1e-9 else x * (ceiling / m)


def fade(x: np.ndarray, sr: int, fade_in: float = 0.003, fade_out: float = 0.01) -> np.ndarray:
    x = x.copy()
    a = min(len(x), max(1, int(fade_in * sr)))
    b = min(len(x), max(1, int(fade_out * sr)))
    x[:a] *= np.linspace(0.0, 1.0, a)
    x[-b:] *= np.linspace(1.0, 0.0, b)
    return x


def pan(x: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-power pan. p in [-1, 1]."""
    p = float(np.clip(p, -1.0, 1.0))
    angle = (p + 1.0) * np.pi / 4.0
    return x * np.cos(angle), x * np.sin(angle)


# -- spectral ---------------------------------------------------------------


def stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float64)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft), mode="reflect")
    frames = sliding_window_view(xp, n_fft)[::hop]
    win = np.hanning(n_fft)
    return np.fft.rfft(frames * win, axis=1), len(x)


def istft(X: np.ndarray, n_fft: int = 1024, hop: int = 256, length: int | None = None) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = np.fft.irfft(X, n=n_fft, axis=1) * win
    out_len = n_fft + hop * (len(frames) - 1)
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    for i in range(len(frames)):
        s = i * hop
        out[s : s + n_fft] += frames[i]
        wsum[s : s + n_fft] += win**2
    out /= np.maximum(wsum, 1e-8)
    pad = n_fft // 2
    out = out[pad : pad + length] if length is not None else out[pad:]
    if length is not None and len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out


def bandpass_timevarying(
    x: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    """Gaussian bandpass whose center and width move over time.

    `centers` and `widths` are control curves in Hz; they are interpolated to
    the STFT frame rate.
    """
    X, n = stft(x, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    nf = X.shape[0]
    grid = np.linspace(0.0, 1.0, nf)
    c = np.interp(grid, np.linspace(0.0, 1.0, len(centers)), centers)
    w = np.interp(grid, np.linspace(0.0, 1.0, len(widths)), widths)
    mask = np.exp(-0.5 * ((freqs[None, :] - c[:, None]) / np.maximum(w[:, None], 1.0)) ** 2)
    return istft(X * mask, n_fft, hop, length=n)


def lowpass(x: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    n = len(x)
    if n == 0:
        return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    X *= 1.0 / (1.0 + (f / max(cutoff, 1.0)) ** order)
    return np.fft.irfft(X, n)


# -- sources ----------------------------------------------------------------


def modal(
    f0: float,
    dur: float,
    sr: int,
    ratios: tuple[float, ...] = (1.0, 2.01, 3.03, 4.07),
    t60: tuple[float, ...] = (0.4, 0.25, 0.15, 0.1),
    gains: tuple[float, ...] = (1.0, 0.6, 0.35, 0.2),
) -> np.ndarray:
    """A bank of exponentially decaying sinusoids."""
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    out = np.zeros(n)
    for r, d, g in zip(ratios, t60, gains):
        f = f0 * r
        if f >= sr * 0.45:
            continue
        out += g * np.exp(-LN1000 * t / max(d, 1e-3)) * np.sin(TAU * f * t)
    return out


def sweep_sine(f0: float, f1: float, dur: float, sr: int, curve: float = 1.0) -> np.ndarray:
    n = max(1, int(dur * sr))
    t = np.arange(n) / sr
    f = f0 + (f1 - f0) * (t / max(dur, 1e-6)) ** curve
    phase = TAU * np.cumsum(f) / sr
    return np.sin(phase)


def env_apex(n: int, sr: int, apex_s: float, rise_power: float = 1.6, fall_power: float = 3.2) -> np.ndarray:
    """Rise to 1.0 at apex_s, then exponential fall. Peaks exactly at apex."""
    t = np.arange(n) / sr
    a = max(apex_s, 1e-3)
    d = max(n / sr - a, 1e-3)
    rise = np.clip(t / a, 0.0, 1.0) ** rise_power
    fall = np.exp(-fall_power * np.maximum(t - a, 0.0) / d)
    return rise * fall


def saturate(x: np.ndarray, drive: float = 1.5) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def align_peak(x: np.ndarray, sr: int, target_s: float, max_shift: float = 0.05) -> np.ndarray:
    """Shift a clip so its smoothed envelope peak lands on target_s.

    Used for continuous roles (whoosh, riser, swell) where the score's apex is
    a promise about the envelope, not just the onset.
    """
    if len(x) == 0:
        return x
    k = max(1, int(0.003 * sr))
    env = np.convolve(np.abs(x), np.ones(k) / k, mode="same")
    peak = int(np.argmax(env))
    target = int(np.clip(target_s, 0.0, len(x) / sr) * sr)
    shift = target - peak
    if abs(shift) > max_shift * sr:
        return x
    if shift > 0:
        return np.pad(x, (shift, 0))[: len(x)]
    return np.pad(x, (0, -shift))[-shift:]


def reverb(x: np.ndarray, sr: int, decay: float = 0.6, wet: float = 0.2, seed: int = 0) -> np.ndarray:
    """Convolution with a decaying-noise impulse response. Cheap and stable."""
    if wet <= 0 or len(x) == 0:
        return x
    rng = np.random.default_rng(seed)
    n_ir = max(8, int(decay * 2.5 * sr))
    t = np.arange(n_ir) / sr
    ir = rng.standard_normal(n_ir) * np.exp(-LN1000 * t / max(decay, 1e-3))
    ir[: max(1, int(0.01 * sr))] *= np.linspace(0.0, 1.0, max(1, int(0.01 * sr)))
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    n = len(x) + n_ir - 1
    nfft = 1 << (n - 1).bit_length()
    wet_sig = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[: len(x)]
    return (1.0 - wet) * x + wet * wet_sig
