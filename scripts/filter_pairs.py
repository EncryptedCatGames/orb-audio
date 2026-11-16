#!/usr/bin/env python3
"""Keep the cleanest pairs for the detail model.

    python scripts/filter_pairs.py --manifest data/pairs/manifest.jsonl --keep 0.3

The fit is heuristic, so the residual contains fitting error as well as
timbre. A smaller clean set beats a large noisy one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/pairs/manifest.jsonl")
    ap.add_argument("--out", default="data/pairs/manifest_filtered.jsonl")
    ap.add_argument("--keep", type=float, default=0.3)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty manifest")
    rows.sort(key=lambda row: row.get("fit_score", 0.0), reverse=True)
    n = max(1, int(len(rows) * args.keep))
    kept = rows[:n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    scores = [row.get("fit_score", 0.0) for row in rows]
    print(f"kept {n}/{len(rows)} (fit >= {kept[-1].get('fit_score', 0.0):.3f})")
    print(f"fit mean {sum(scores) / len(scores):.3f}, max {max(scores):.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
