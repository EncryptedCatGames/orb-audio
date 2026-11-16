#!/usr/bin/env python3
"""Evaluate the detail model against the identity baseline.

    python scripts/eval_detail.py --data data/degrade/eval.jsonl
    python scripts/eval_detail.py --data data/degrade/eval.jsonl \
      --checkpoint runs/detail_harness/best.pt

Reports the multi-resolution STFT loss, envelope correlation, onset F1 within
20 ms, and the high-band energy ratio. The go/no-go is a >=25% MR-STFT
improvement over identity on held-out clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.pair_dataset import PairDataset, collate_pairs  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.train.common import get_device  # noqa: E402
from orb.train.losses import envelope_loss, istft, multi_res_stft_loss, relative_mag_loss  # noqa: E402


def _envelope(wav: np.ndarray, hop: int = 256) -> np.ndarray:
    n = len(wav) // hop
    return np.sqrt(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)


def _onsets(wav: np.ndarray, sr: int, hop: int = 256) -> list[float]:
    env = _envelope(wav, hop)
    log_env = np.log(env + 1e-8)
    flux = np.maximum(0.0, np.diff(log_env))
    if flux.size == 0 or flux.max() <= 0:
        return []
    thresh = float(flux.mean() + 0.5 * flux.std())
    peaks = np.where(flux > thresh)[0]
    min_dist = max(1, int(0.02 * sr / hop))
    keep: list[int] = []
    for p in peaks:
        if not keep or p - keep[-1] >= min_dist:
            keep.append(int(p))
    return [p * hop / sr for p in keep]


def _onset_f1(pred: np.ndarray, real: np.ndarray, sr: int) -> float:
    p = _onsets(pred, sr)
    r = _onsets(real, sr)
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    used: set[int] = set()
    tp = 0
    for t in p:
        for j, u in enumerate(r):
            if j not in used and abs(t - u) <= 0.02:
                tp += 1
                used.add(j)
                break
    return 2.0 * tp / (len(p) + len(r))


def _high_band(wav: np.ndarray, sr: int, cutoff: float = 4000.0) -> float:
    spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sr)
    return float(np.sum(spec[freqs >= cutoff] ** 2) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/degrade/eval.jsonl")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--noise-draws", type=int, default=4)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = get_device()
    dataset = PairDataset(
        ROOT / args.data,
        n_frames=256,
        n_fft=1024,
        hop=256,
        train=False,
        crop_mode="onset",
        normalize_stft=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pairs,
    )

    model = None
    discriminator = None
    if args.checkpoint:
        ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
        model = DetailUNet(**ckpt["config"]["model"]).to(device)
        model.load_state_dict(ckpt["generator"])
        model.eval()
        if "discriminator" in ckpt and ckpt["discriminator"]:
            from orb.model.discriminator import MultiScaleDiscriminator

            discriminator = MultiScaleDiscriminator(**ckpt["config"].get("disc", {})).to(device)
            discriminator.load_state_dict(ckpt["discriminator"])
            discriminator.eval()

    totals = {
        "identity": 0.0,
        "model": 0.0,
        "rel_identity": 0.0,
        "rel_model": 0.0,
        "env": 0.0,
        "onset": 0.0,
        "n": 0,
        "clips": 0,
        "noise": 0.0,
        "disc_acc": 0.0,
        "disc_n": 0,
    }
    high_ratios: list[float] = []
    identity_high: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["dsp"].to(device)
            y = batch["real"].to(device)
            cond = batch["cond"].to(device)
            fake = x + model(x, cond) if model is not None else x

            if model is not None and model.noise_channels and args.noise_draws > 1:
                draws = []
                for _ in range(args.noise_draws):
                    noise = torch.randn(
                        x.shape[0],
                        model.noise_channels,
                        x.shape[2],
                        x.shape[3],
                        device=device,
                    )
                    draws.append((x + model(x, cond, noise)).cpu())
                stack = torch.stack(draws)
                # Std across draws relative to the mean magnitude: 0 means the
                # noise channel is dead.
                totals["noise"] += float(stack.std(dim=0).mean() / (stack.mean(dim=0).abs().mean() + 1e-6))

            if discriminator is not None:
                min_bin = int(ckpt["config"].get("disc_min_bin", 0))

                def disc_in(z: torch.Tensor) -> torch.Tensor:
                    return z[:, :, min_bin:] if min_bin else z

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    real_logits = discriminator(disc_in(y))
                    fake_logits = discriminator(disc_in(fake))
                for r, f in zip(real_logits, fake_logits):
                    totals["disc_acc"] += float((r > 0).float().mean() + (f <= 0).float().mean()) / 2.0
                totals["disc_n"] += len(real_logits)

            real_wav = istft(y, 1024, 256)
            id_wav = istft(x, 1024, 256)
            fake_wav = istft(fake, 1024, 256)
            totals["identity"] += float(multi_res_stft_loss(id_wav, real_wav))
            totals["model"] += float(multi_res_stft_loss(fake_wav, real_wav))
            totals["rel_identity"] += float(relative_mag_loss(x, y))
            totals["rel_model"] += float(relative_mag_loss(fake, y))
            totals["env"] += float(envelope_loss(fake_wav, real_wav, 256))
            totals["n"] += 1
            totals["clips"] += int(x.shape[0])

            for j in range(x.shape[0]):
                r = real_wav[j].cpu().numpy()
                f = fake_wav[j].cpu().numpy()
                totals["onset"] += _onset_f1(f, r, 48000)
                high_ratios.append(_high_band(f, 48000) / max(_high_band(r, 48000), 1e-9))
                identity_high.append(_high_band(id_wav[j].cpu().numpy(), 48000) / max(_high_band(r, 48000), 1e-9))

            if args.max_batches and i + 1 >= args.max_batches:
                break

    n = max(totals["n"], 1)
    clips = max(totals["clips"], 1)
    identity = totals["identity"] / n
    model_loss = totals["model"] / n
    improvement = 1.0 - model_loss / max(identity, 1e-9)
    summary = {
        "checkpoint": args.checkpoint,
        "clips": clips,
        "identity_mr_stft": identity,
        "model_mr_stft": model_loss,
        "improvement": improvement,
        "identity_relative": totals["rel_identity"] / n,
        "model_relative": totals["rel_model"] / n,
        "envelope_l1": totals["env"] / n,
        "onset_f1": totals["onset"] / clips,
        "high_band_ratio": float(np.median(high_ratios)) if high_ratios else 0.0,
        "identity_high_band_ratio": float(np.median(identity_high)) if identity_high else 0.0,
        "noise_sensitivity": totals["noise"] / n,
        "disc_accuracy": totals["disc_acc"] / max(totals["disc_n"], 1),
    }
    print(json.dumps(summary, indent=2))
    print()
    print(f"MR-STFT improvement over identity: {100.0 * improvement:.1f}%  (guardrail: >= 10%)")
    print(f"onset F1: {summary['onset_f1']:.3f}  (gate: >= 0.80)")
    print(f"high-band ratio: {summary['high_band_ratio']:.3f}  (gate: 0.7 - 1.3; identity {summary['identity_high_band_ratio']:.3f})")
    print(f"noise sensitivity: {summary['noise_sensitivity']:.4f}  (gate: > 0.01)")
    print(f"discriminator accuracy: {summary['disc_accuracy']:.3f}  (sane: 0.55 - 0.95)")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
