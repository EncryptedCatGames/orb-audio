#!/usr/bin/env python3
"""Caption -> WAV. The full pipeline: planner -> Spec -> DSP -> WAV.

    python examples/01_generate.py
    python examples/01_generate.py "8.00-second tense cue, drone bed, with riser @1.00 2.40s -16dB C #51"

Add ``--detail weights/detail.pt`` to run the optional neural texture stage
(CC BY-NC-4.0, not included in the open-source release).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.api import generate, save_wav  # noqa: E402

DEFAULT_CAPTION = "6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caption", nargs="?", default=DEFAULT_CAPTION)
    ap.add_argument("--out", default="out/example_01.wav")
    ap.add_argument("--planner", default=str(ROOT / "weights/planner.pt"))
    ap.add_argument("--detail", default=None)
    args = ap.parse_args()

    out = generate(args.caption, planner_path=args.planner, detail_path=args.detail)
    save_wav(out["audio"], ROOT / args.out)

    audit = out["audit"]
    print(f"caption  {args.caption}")
    print(f"spec     {out['spec']}")
    print(
        f"audit    peak={audit['peak']:.3f}  rms={audit['rms_dbfs']:.1f} dBFS  "
        f"pass={audit['passes']}"
    )
    print(f"wrote    {args.out}")


if __name__ == "__main__":
    main()
