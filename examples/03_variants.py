#!/usr/bin/env python3
"""Render N non-repeating variants of one Spec.

A sample library gives you a fixed number of takes; the seed gives you as many
as you want. Useful for games, where the third reuse of the same file is
audible.

    python examples/03_variants.py --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.api import render_spec, save_wav  # noqa: E402

SPEC = {
    "duration": 4.0,
    "texture": "room_cold",
    "mood": "dark",
    "events": [
        {"role": "impact", "t": 0.8, "dur": 0.9, "gain_db": -4.0, "pan": 0.0, "seed": 31},
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="out/variants")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        audio = render_spec(SPEC, seed=i)
        path = out_dir / f"impact_{i:02d}.wav"
        save_wav(audio, path)
        print(f"{path.relative_to(ROOT)}  {audio.shape[0] / 48000:.2f}s")


if __name__ == "__main__":
    main()
