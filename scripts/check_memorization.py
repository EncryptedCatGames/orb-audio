#!/usr/bin/env python3
"""Nearest-neighbour memorization check.

    python scripts/check_memorization.py --train /data/cc0 --generated out/generated --threshold 0.98

Flags generated clips whose mean log-mel is closer than the threshold to a
training clip. A crude but useful guard before shipping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.mel import log_mel  # noqa: E402
from orb.io import read_wav  # noqa: E402

SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}


def embed(path: Path) -> np.ndarray:
    audio, sr = read_wav(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mel = log_mel(audio, sr).numpy().mean(axis=-1).reshape(-1)
    return mel / (np.linalg.norm(mel) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--threshold", type=float, default=0.98)
    args = ap.parse_args()

    train_files = [p for p in Path(args.train).rglob("*") if p.suffix.lower() in SUFFIXES]
    gen_files = [p for p in Path(args.generated).rglob("*") if p.suffix.lower() in SUFFIXES]
    if not train_files or not gen_files:
        raise SystemExit("need at least one file in each folder")

    train = np.stack([embed(p) for p in train_files])
    flagged = 0
    for path in gen_files:
        vec = embed(path)
        sims = train @ vec
        best = float(sims.max())
        if best >= args.threshold:
            flagged += 1
            print(f"FLAG {path}  max cosine {best:.4f}  ({train_files[int(sims.argmax())].name})")

    print(f"checked {len(gen_files)} generated clips against {len(train_files)} training clips, flagged {flagged}")


if __name__ == "__main__":
    main()
