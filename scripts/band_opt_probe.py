#!/usr/bin/env python3
"""Can the band loss drive high-frequency energy up at all?

Optimizes a free residual directly (no model) on one batch. If this cannot
match the high band, the loss is the problem; if it can, the model is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.pair_dataset import PairDataset, collate_pairs  # noqa: E402
from orb.train.losses import band_energy_loss, log_mag_loss  # noqa: E402

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 24000)]


def band_energy(spec: torch.Tensor) -> torch.Tensor:
    mag = spec[:, 0] ** 2 + spec[:, 1] ** 2
    freqs = torch.linspace(0, 24000, mag.shape[1], device=mag.device)
    cols = [mag[:, (freqs >= lo) & (freqs < hi)].mean(dim=(1, 2)) for lo, hi in BANDS]
    return torch.stack(cols, dim=1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairDataset(
        ROOT / "data/degrade/eval.jsonl",
        n_frames=256,
        n_fft=1024,
        hop=256,
        train=False,
        normalize_stft=True,
    )
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pairs)
    batch = next(iter(dl))
    x = batch["dsp"].to(device)
    y = batch["real"].to(device)

    real = band_energy(y).mean(0)
    dsp = band_energy(x).mean(0)

    z = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.02)
    for i in range(1000):
        opt.zero_grad()
        fake = x + z
        loss = band_energy_loss(fake, y) + 0.1 * log_mag_loss(fake, y)
        loss.backward()
        opt.step()
        if i % 200 == 0:
            print(f"step {i:4d} loss {float(loss):.4f}")

    with torch.no_grad():
        fake = band_energy(x + z).mean(0)
    print("band      ", [f"{a}-{b}" for a, b in BANDS])
    print("real      ", [f"{v:.4f}" for v in real.tolist()])
    print("dsp       ", [f"{v:.4f}" for v in dsp.tolist()])
    print("opt       ", [f"{v:.4f}" for v in fake.tolist()])
    print("opt/real  ", [f"{a / max(b, 1e-9):.2f}" for a, b in zip(fake.tolist(), real.tolist())])
    print("resid mean", float(z.abs().mean()), "max", float(z.abs().max()))


if __name__ == "__main__":
    main()
