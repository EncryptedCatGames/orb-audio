#!/usr/bin/env python3
"""Build degraded-real pairs for the detail harness.

    python scripts/build_degrade_pairs.py --esc50 /workspace/data/ESC-50-master

Writes `data/degrade/audio/*.wav` plus four manifests:
  train.jsonl     folds 1-4, all classes
  eval.jsonl      fold 5, all classes
  train_ho.jsonl  folds 1-4, minus 10 held-out classes
  eval_ho.jsonl   the 10 held-out classes, all folds

The harness is for a go/no-go only. Do not warm-start the CC0 model from these
weights; ESC-50 is CC BY-NC.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.degrade import SR, degrade  # noqa: E402
from orb.io import read_wav, write_wav  # noqa: E402

SILENCE_RMS = 10 ** (-50.0 / 20.0)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g)


def _process(job: tuple[str, str, int]) -> dict | None:
    src, out_dir, seed = job
    src_path = Path(src)
    stem = src_path.stem
    dsp_path = Path(out_dir) / f"{stem}_dsp.wav"
    real_path = Path(out_dir) / f"{stem}_real.wav"
    if dsp_path.exists() and real_path.exists():
        return {"dsp": str(dsp_path), "real": str(real_path), "cond": [0.0] * 32, "id": stem}

    x, sr = read_wav(src_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = _resample(np.asarray(x, dtype=np.float64), sr)
    if float(np.sqrt(np.mean(x**2) + 1e-12)) < SILENCE_RMS:
        return None

    rng = np.random.default_rng((zlib.crc32(stem.encode()) ^ seed) & 0xFFFFFFFF)
    n_bands = int(rng.choice((24, 40, 64)))
    n_peaks = int(rng.choice((0, 4, 8, 16)))
    smooth = int(rng.integers(3, 10))
    lowpass = float(rng.uniform(2000.0, 6000.0))
    y, _ = degrade(x, SR, rng, n_bands=n_bands, n_peaks=n_peaks, smooth=smooth, lowpass_hz=lowpass)

    write_wav(real_path, x, SR)
    write_wav(dsp_path, y, SR)
    return {"dsp": str(dsp_path), "real": str(real_path), "cond": [0.0] * 32, "id": stem}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", default="/workspace/data/ESC-50-master")
    ap.add_argument("--out", default="data/degrade")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    esc = Path(args.esc50)
    audio_dir = esc / "audio"
    rows = list(csv.DictReader((esc / "meta" / "esc50.csv").open()))
    if not rows:
        raise SystemExit(f"no rows in {esc / 'meta' / 'esc50.csv'}")

    out = Path(args.out)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    classes = sorted({r["category"] for r in rows})
    heldout = set(classes[:: max(1, len(classes) // 10)][:10])

    jobs = [(str(audio_dir / r["filename"]), str(audio_out), args.seed) for r in rows]
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(_process, jobs, chunksize=8):
            if res is not None:
                results[res["id"]] = res

    splits = {
        "train": [r for r in rows if int(r["fold"]) <= 4],
        "eval": [r for r in rows if int(r["fold"]) == 5],
        "train_ho": [r for r in rows if int(r["fold"]) <= 4 and r["category"] not in heldout],
        "eval_ho": [r for r in rows if r["category"] in heldout],
    }
    for name, split_rows in splits.items():
        path = out / f"{name}.jsonl"
        kept = 0
        with path.open("w") as f:
            for r in split_rows:
                res = results.get(Path(r["filename"]).stem)
                if res is None:
                    continue
                f.write(json.dumps(res) + "\n")
                kept += 1
        print(f"{name:9s} {kept:5d} pairs  {path}")

    print(f"held-out classes ({len(heldout)}): {sorted(heldout)}")
    print(f"audio dir: {audio_out}")


if __name__ == "__main__":
    main()
