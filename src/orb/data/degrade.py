"""Controlled degradation for the detail harness.

Takes a real recording and reduces it to something a DSP renderer could
plausibly produce: a smoothed spectral envelope rendered as shaped noise plus
a handful of tracked sinusoids, lowpassed, with the original RMS restored.

The detail model is trained to invert this. The pairs are perfectly aligned, so
this is an upper bound on what it can do with fitted DSP pairs.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

N_FFT = 1024
HOP = 256
SR = 48000


def _slaney_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3.0)
    logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / logstep, mel)


def _slaney_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = m * (200.0 / 3.0)
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), f)


def mel_filterbank(sr: int, n_fft: int, n_bands: int, fmin: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Slaney-mel triangular filterbank, shape (n_bands, n_fft // 2 + 1)."""
    fmax = sr / 2.0
    m = np.linspace(_slaney_mel(np.array(fmin)), _slaney_mel(np.array(fmax)), n_bands + 2)
    hz = _slaney_hz(m)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_bands, len(freqs)))
    for i in range(n_bands):
        lo, center, hi = hz[i], hz[i + 1], hz[i + 2]
        left = (freqs - lo) / max(center - lo, 1e-9)
        right = (hi - freqs) / max(hi - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(left, right))
    return fb, hz[1:-1]


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
        window=torch.hann_window(N_FFT),
        length=length,
    ).numpy()


def _smooth_time(x: np.ndarray, frames: int) -> np.ndarray:
    w = np.hanning(max(3, frames))
    w = w / w.sum()
    return np.stack([np.convolve(row, w, mode="same") for row in x])


def _peaks(mag: np.ndarray, sr: int, n_peaks: int, band_med_bins: np.ndarray) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for t in range(mag.shape[1]):
        col = mag[:, t]
        idx = np.where((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:]))[0] + 1
        if idx.size == 0:
            out.append([])
            continue
        idx = idx[col[idx] > 3.98 * band_med_bins[idx, t]]  # at least 12 dB above the band median
        if idx.size == 0:
            out.append([])
            continue
        idx = idx[np.argsort(col[idx])[-n_peaks:]]
        peaks: list[tuple[float, float]] = []
        for i in idx:
            a = np.log(col[i - 1] + 1e-12)
            b = np.log(col[i] + 1e-12)
            c = np.log(col[i + 1] + 1e-12)
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            freq = (i + delta) * sr / N_FFT
            amp = float(np.exp(b - 0.25 * (a - c) * delta))
            peaks.append((freq, amp))
        out.append(peaks)
    return out


def _tracks(peaks: list[list[tuple[float, float]]], max_jump: float = 0.03) -> list[dict]:
    tracks: list[dict] = []
    active: list[dict] = []
    for t, frame in enumerate(peaks):
        used: set[int] = set()
        next_active: list[dict] = []
        for tr in active:
            best = None
            best_d = 1e9
            for j, (f, _) in enumerate(frame):
                if j in used:
                    continue
                d = abs(f - tr["f"][-1]) / max(tr["f"][-1], 1e-9)
                if d < max_jump and d < best_d:
                    best, best_d = j, d
            if best is None:
                tracks.append(tr)
            else:
                used.add(best)
                tr["t"].append(t)
                tr["f"].append(frame[best][0])
                tr["a"].append(frame[best][1])
                next_active.append(tr)
        for j, (f, a) in enumerate(frame):
            if j not in used:
                next_active.append({"t": [t], "f": [f], "a": [a]})
        active = next_active
    tracks.extend(active)
    return [tr for tr in tracks if len(tr["t"]) >= 3]


def _synth_tracks(tracks: list[dict], n: int, sr: int) -> np.ndarray:
    out = np.zeros(n)
    scale = 2.0 / np.sum(np.hanning(N_FFT))
    for tr in tracks:
        t0 = tr["t"][0] * HOP
        span = (tr["t"][-1] - tr["t"][0]) * HOP + HOP
        m = min(n - t0, span)
        if m <= 1:
            continue
        tt = np.arange(m)
        ft = (np.asarray(tr["t"]) - tr["t"][0]) * HOP
        freq = np.interp(tt, ft, tr["f"])
        amp = np.interp(tt, ft, tr["a"]) * scale
        phase = 2.0 * np.pi * np.cumsum(freq) / sr
        out[t0 : t0 + m] += amp * np.sin(phase)
    return out


def _lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(8, min(cutoff, sr / 2 - 100.0), btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def degrade(
    x: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    n_bands: int = 40,
    n_peaks: int = 8,
    smooth: int = 5,
    lowpass_hz: float = 4000.0,
) -> tuple[np.ndarray, dict]:
    """Return (degraded, params). Output is 48 kHz mono, same length as input."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
        sr = SR
    n = len(x)
    if n < N_FFT:
        x = np.pad(x, (0, N_FFT - n))
        n = len(x)

    spec = _stft(x)
    mag = np.abs(spec.numpy())
    fb, centers = mel_filterbank(sr, N_FFT, n_bands)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)

    band_energy = fb @ (mag**2)
    band_energy = _smooth_time(band_energy, smooth)

    # Shaped noise: distribute each band's energy over its bins, then random phase.
    bin_energy = np.zeros_like(mag)
    for i in range(n_bands):
        mask = fb[i] > 0.5
        if mask.any():
            bin_energy[mask] += band_energy[i] / mask.sum()
    noise_mag = np.sqrt(np.maximum(bin_energy, 1e-14))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=noise_mag.shape)
    noise = _istft(torch.from_numpy(noise_mag * np.exp(1j * phase)).to(torch.complex64), n)

    # Tonal layer: tracked peaks, at least 12 dB above the local band median.
    band_masks = [fb[i] > 0.5 for i in range(n_bands)]
    band_med = np.stack([np.median(mag[m], axis=0) for m in band_masks])
    band_med_bins = np.stack(
        [np.interp(freqs, centers, band_med[:, t]) for t in range(mag.shape[1])], axis=1
    )
    peaks = _peaks(mag, sr, n_peaks, band_med_bins)
    tonal = _synth_tracks(_tracks(peaks), n, sr)

    y = _lowpass(noise + tonal, sr, lowpass_hz)
    rms_x = float(np.sqrt(np.mean(x**2) + 1e-12))
    rms_y = float(np.sqrt(np.mean(y**2) + 1e-12))
    y = y * (rms_x / max(rms_y, 1e-9))
    return y[: len(np.asarray(x))], {
        "n_bands": n_bands,
        "n_peaks": n_peaks,
        "smooth": smooth,
        "lowpass_hz": lowpass_hz,
    }
