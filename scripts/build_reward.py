#!/usr/bin/env python3
"""Build a reward-training manifest from a detail-pair manifest.

    python scripts/build_reward.py --pairs data/pairs/manifest.jsonl --out data/reward/manifest.jsonl

Each pair becomes two rows: the DSP render (real=0) and the recording
(real=1), with a caption derived from the role.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROLE_CAPTION = {
    "whoosh": "a fast airy whoosh",
    "impact": "a short deep impact",
    "ui_pop": "a soft rounded pop",
    "riser": "a dark tense riser",
    "swell": "a wide shimmering swell",
    "click": "a tight clean click",
    "bell": "a bright bell",
    "sub_drop": "a deep sub drop",
    "glitch": "a short digital glitch",
    "noise_sweep": "a wide airy noise sweep",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/manifest.jsonl")
    ap.add_argument("--out", default="data/reward/manifest.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            caption = ROLE_CAPTION.get(row.get("role", ""), "a sound effect")
            f.write(json.dumps({"audio": row["dsp"], "caption": caption, "real": 0.0, "sync": 0.0}) + "\n")
            f.write(json.dumps({"audio": row["real"], "caption": caption, "real": 1.0, "sync": 0.0}) + "\n")
    print(f"wrote {2 * len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
