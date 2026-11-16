#!/usr/bin/env python3
"""Retrieval baseline over synthetic scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.retrieval import RetrievalIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scores.jsonl")
    ap.add_argument("--query", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    index = RetrievalIndex([row["caption"] for row in rows])

    queries = [args.query] if args.query else [row["caption"] for row in rows[:5]]
    for query in queries:
        print(f"\nquery: {query}")
        for i, score in index.query(query, args.top):
            print(f"  {score:.3f}  {rows[i]['caption']}")


if __name__ == "__main__":
    main()
