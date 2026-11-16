#!/usr/bin/env python3
"""Train the compact Spec planner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.spec_dataset import SpecDataset  # noqa: E402
from orb.model.spec_tokenizer import SpecTokenizer  # noqa: E402
from orb.train.planner import train_planner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    train_planner(cfg, args.resume, tokenizer=SpecTokenizer(), dataset_cls=SpecDataset)


if __name__ == "__main__":
    main()
