"""Write free-text training rows: {"caption": <prompt>, "spec": <Spec dict>}.

    python scripts/gen_freetext.py --n 150000 --seed 11 --out data/freetext_150k.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.freetext import sample_pair  # noqa: E402


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def spec_dict(spec) -> dict:
    return {
        "duration": float(spec.duration), "texture": spec.texture, "mood": spec.mood,
        "events": [{"role": e.role, "t": float(e.t), "dur": float(e.dur), "gain_db": float(e.gain_db),
                    "pan": float(e.pan), "seed": int(e.seed)} for e in spec.events],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=str(ROOT / "data" / "freetext_eval.jsonl"),
                    help="never emit a prompt that is in the hand-written eval set")
    args = ap.parse_args()
    banned = set()
    if args.exclude and Path(args.exclude).exists():
        banned = {norm(json.loads(ln)["prompt"]) for ln in open(args.exclude) if ln.strip()}
    rng = np.random.default_rng(args.seed)
    n = skipped = 0
    with open(args.out, "w") as f:
        while n < args.n:
            text, spec = sample_pair(rng)
            if norm(text) in banned:
                skipped += 1
                continue
            f.write(json.dumps({"caption": text, "spec": spec_dict(spec)}) + "\n")
            n += 1
    print(f"wrote {n} rows to {args.out} (skipped {skipped} eval prompts)")


if __name__ == "__main__":
    main()
