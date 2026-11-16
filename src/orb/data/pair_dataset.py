"""Paired (DSP render, real recording) dataset for the detail model."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..io import read_wav


def complex_stft(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """(n,) waveform → (2, F, T) real/imag tensor."""
    spec = torch.stft(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
        center=True,
    )
    return torch.stack([spec.real, spec.imag], dim=0)


def _rms_normalize(x: np.ndarray, target: float = 0.1) -> np.ndarray:
    r = float(np.sqrt(np.mean(x**2) + 1e-12))
    return x * (target / max(r, 1e-9))


class PairDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        n_frames: int = 256,
        n_fft: int = 1024,
        hop: int = 256,
        train: bool = True,
        crop_mode: str = "onset",
        normalize_stft: bool = False,
    ):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.n_frames = n_frames
        self.n_fft = n_fft
        self.hop = hop
        self.train = train
        self.crop_mode = crop_mode
        self.normalize_stft = normalize_stft

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        dsp, _ = read_wav(row["dsp"])
        real, _ = read_wav(row["real"])
        if dsp.ndim > 1:
            dsp = dsp.mean(axis=1)
        if real.ndim > 1:
            real = real.mean(axis=1)
        dsp = _rms_normalize(dsp)
        real = _rms_normalize(real)

        x = complex_stft(dsp, self.n_fft, self.hop)
        y = complex_stft(real, self.n_fft, self.hop)
        if self.normalize_stft:
            # Shared scale: the residual only makes sense in one domain.
            power = (y[0] ** 2 + y[1] ** 2).mean()
            scale = torch.sqrt(power) + 1e-6
            x = x / scale
            y = y / scale
        t = min(x.shape[-1], y.shape[-1])
        x, y = x[:, :, :t], y[:, :, :t]

        if t > self.n_frames:
            xc = yc = None
            for _ in range(4):
                if self.train and self.crop_mode == "random":
                    start = random.randint(0, t - self.n_frames)
                elif self.train:
                    # Keep the onset (at ~0.05 s) in every crop; jitter by at most 4 frames.
                    start = random.randint(0, min(4, t - self.n_frames))
                else:
                    start = 0
                xc = x[:, :, start : start + self.n_frames]
                yc = y[:, :, start : start + self.n_frames]
                # A silent crop makes the relative loss explode and teaches nothing.
                if float((xc**2).mean() + (yc**2).mean()) > 1e-6:
                    break
            x, y = xc, yc
        elif t < self.n_frames:
            pad = self.n_frames - t
            x = F.pad(x, (0, pad))
            y = F.pad(y, (0, pad))

        cond = torch.tensor(row.get("cond", [0.0] * 32), dtype=torch.float32)
        return {"dsp": x, "real": y, "cond": cond}


def collate_pairs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "dsp": torch.stack([item["dsp"] for item in batch]),
        "real": torch.stack([item["real"] for item in batch]),
        "cond": torch.stack([item["cond"] for item in batch]),
    }
