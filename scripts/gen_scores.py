#!/usr/bin/env python3
"""Generate random (caption, score) pairs for planner pretraining.

    python scripts/gen_scores.py --n 10000 --out data/scores.jsonl

Scores are ~1 KB of JSON each. Audio is rendered in the dataloader, never
stored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import caption, sample_score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/scores.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-music", action="store_true", help="v1 scope: SFX + texture only")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for _ in range(args.n):
            score = sample_score(rng, with_music=not args.no_music)
            score.validate()
            f.write(json.dumps({"caption": caption(score), "score": score.to_dict()}) + "\n")

    print(f"wrote {args.n} pairs to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
