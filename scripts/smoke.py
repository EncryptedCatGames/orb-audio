#!/usr/bin/env python3
"""Render the demo cue and print the numbers that matter.

    python scripts/smoke.py
    python scripts/smoke.py --sr 48000 --out out/demo_48k.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.io import write_wav  # noqa: E402
from orb.render.events import render_event  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.score import Score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/demo.wav")
    ap.add_argument("--sr", type=int, default=0, help="override sample rate")
    args = ap.parse_args()

    score = Score.demo()
    if args.sr:
        score.sample_rate = args.sr
    score.validate()

    t0 = time.time()
    audio = render(score)
    dt = time.time() - t0

    out = ROOT / args.out
    write_wav(out, audio, score.sample_rate)

    peak = float(np.max(np.abs(audio)))
    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(audio**2))) + 1e-12)
    print(f"wrote      {out}")
    print(f"duration   {score.duration:.2f} s @ {score.sample_rate} Hz")
    print(f"render     {dt:.2f} s ({score.duration / max(dt, 1e-9):.1f}x realtime)")
    print(f"peak       {peak:.4f}")
    print(f"rms        {rms_db:.1f} dBFS")

    whoosh = next(e for e in score.events if e.role == "whoosh")
    one = render_event(whoosh, score.sample_rate, seed=score.seed + 200)
    env = np.abs(one.mean(axis=1))
    k = max(1, int(0.005 * score.sample_rate))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    peak_t = float(np.argmax(env) / score.sample_rate) + whoosh.t
    print(f"whoosh apex {whoosh.apex:.3f} s, measured {peak_t:.3f} s, error {abs(peak_t - whoosh.apex) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
