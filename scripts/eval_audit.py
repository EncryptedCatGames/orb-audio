#!/usr/bin/env python3
"""Audit a WAV or a score JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.audit import audit_file, audit_waveform, passes  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.score import Score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--score")
    args = ap.parse_args()

    if args.wav:
        result = audit_file(args.wav)
    elif args.score:
        score = Score.load(args.score)
        result = audit_waveform(render(score), score.sample_rate)
    else:
        ap.error("pass --wav or --score")

    result["passes"] = passes(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
