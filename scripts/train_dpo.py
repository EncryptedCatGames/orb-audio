#!/usr/bin/env python3
"""DPO for the planner from a JSON config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.train.dpo import train_dpo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    train_dpo(cfg, args.resume)


if __name__ == "__main__":
    main()
