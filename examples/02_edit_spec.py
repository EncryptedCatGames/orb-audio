#!/usr/bin/env python3
"""Edit one number and re-render.

The point of a score-first engine: the sound is data. Moving the impact from
5.00 s to 5.40 s is a dict edit, not a re-record, and the result is
deterministic. No model is involved in this example at all.

    python examples/02_edit_spec.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.api import render_spec, save_wav  # noqa: E402

SPEC = {
    "duration": 8.0,
    "texture": "drone",
    "mood": "intense",
    "events": [
        {"role": "riser", "t": 0.5, "dur": 2.0, "gain_db": -18.0, "pan": 0.0, "seed": 85},
        {"role": "impact", "t": 3.2, "dur": 0.9, "gain_db": -4.0, "pan": 0.0, "seed": 102},
        {"role": "whoosh", "t": 5.0, "dur": 1.0, "gain_db": -10.0, "pan": -0.5, "seed": 119},
    ],
}


def main() -> None:
    before = copy.deepcopy(SPEC)
    moved = copy.deepcopy(SPEC)
    moved["events"][1]["t"] = 5.4

    a = render_spec(before)
    b = render_spec(moved)
    save_wav(a, ROOT / "out/example_02_before.wav")
    save_wav(b, ROOT / "out/example_02_after.wav")

    print(f"impact t: {before['events'][1]['t']} -> {moved['events'][1]['t']}")
    print(f"before   {a.shape}  {a.dtype}")
    print(f"after    {b.shape}  {b.dtype}")
    print("wrote    out/example_02_before.wav, out/example_02_after.wav")


if __name__ == "__main__":
    main()
