#!/usr/bin/env python3
"""Band-energy diagnostic: real vs DSP vs detail output."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.pair_dataset import PairDataset, collate_pairs  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.train.common import get_device  # noqa: E402

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 24000)]


def band_energy(spec: torch.Tensor) -> torch.Tensor:
    mag = spec[:, 0] ** 2 + spec[:, 1] ** 2
    freqs = torch.linspace(0, 24000, mag.shape[1], device=mag.device)
    cols = [mag[:, (freqs >= lo) & (freqs < hi)].mean(dim=(1, 2)) for lo, hi in BANDS]
    return torch.stack(cols, dim=1)


def main() -> None:
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "runs/detail_gan2/step2000.pt"
    device = get_device()
    ds = PairDataset(
        ROOT / "data/degrade/eval.jsonl",
        n_frames=256,
        n_fft=1024,
        hop=256,
        train=False,
        normalize_stft=True,
    )
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_pairs)
    ck = torch.load(ROOT / ckpt_path, map_location="cpu", weights_only=False)
    model = DetailUNet(**ck["config"]["model"]).to(device)
    model.load_state_dict(ck["generator"])
    model.eval()

    sums = {"real": [], "dsp": [], "fake": []}
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["dsp"].to(device)
            y = batch["real"].to(device)
            c = batch["cond"].to(device)
            fake = x + model(x, c)
            for key, z in (("real", y), ("dsp", x), ("fake", fake)):
                sums[key].append(band_energy(z).mean(dim=0).cpu())
            if i >= 9:
                break
    out = {k: torch.stack(v).mean(dim=0).numpy() for k, v in sums.items()}
    print("band      ", [f"{a}-{b}" for a, b in BANDS])
    for k in ("real", "dsp", "fake"):
        print(f"{k:10s}", [f"{v:.4f}" for v in out[k]])
    print("fake/real ", [f"{a / max(b, 1e-9):.2f}" for a, b in zip(out["fake"], out["real"])])
    print("dsp/real  ", [f"{a / max(b, 1e-9):.2f}" for a, b in zip(out["dsp"], out["real"])])


if __name__ == "__main__":
    main()
