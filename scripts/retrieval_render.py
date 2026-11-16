#!/usr/bin/env python3
"""Retrieval baseline: prompt → nearest score → render.

    python scripts/retrieval_render.py --query "a fast airy whoosh" --out out/retrieved.wav

This is the bar the planner has to beat. It is deliberately simple.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.retrieval import RetrievalIndex  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.score import Score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scores.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--out", default="out/retrieved.wav")
    ap.add_argument("--top", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(ROOT / args.data).read_text().splitlines() if line.strip()]
    index = RetrievalIndex([row["caption"] for row in rows])
    hits = index.query(args.query, args.top)
    for rank, (i, score) in enumerate(hits):
        print(f"{rank + 1}. {score:.3f}  {rows[i]['caption']}")

    best = Score.from_dict(rows[hits[0][0]]["score"])
    audio = render(best)
    write_wav(ROOT / args.out, audio, best.sample_rate)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
