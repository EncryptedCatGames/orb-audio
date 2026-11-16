#!/usr/bin/env python3
"""Render a Spec with no model and no torch.

The DSP path is the shippable engine: numpy only, deterministic, and fast
enough to render several times faster than real time. This example never
imports torch and never touches a checkpoint.

    python examples/04_no_torch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.api import render_spec, save_wav  # noqa: E402

SPEC = {
    "duration": 12.0,
    "texture": "drone",
    "mood": "intense",
    "events": [
        {"role": "riser", "t": 1.0, "dur": 2.5, "gain_db": -18.0, "pan": 0.0, "seed": 85},
        {"role": "whoosh", "t": 4.2, "dur": 1.2, "gain_db": -10.0, "pan": -0.75, "seed": 119},
        {"role": "impact", "t": 5.0, "dur": 0.9, "gain_db": -4.0, "pan": 0.0, "seed": 102},
        {"role": "sub_drop", "t": 5.0, "dur": 1.6, "gain_db": -8.0, "pan": 0.0, "seed": 136},
        {"role": "noise_sweep", "t": 6.5, "dur": 2.0, "gain_db": -16.0, "pan": 0.5, "seed": 151},
        {"role": "impact", "t": 9.5, "dur": 0.9, "gain_db": -4.0, "pan": 0.0, "seed": 153},
        {"role": "bell", "t": 9.5, "dur": 2.0, "gain_db": -12.0, "pan": 0.25, "seed": 61},
    ],
}


def main() -> None:
    audio = render_spec(SPEC)
    save_wav(audio, ROOT / "out/example_04_trailer.wav")
    assert "torch" not in sys.modules, "the DSP path must not need torch"
    print(f"rendered {audio.shape[0] / 48000:.2f}s stereo at 48 kHz, torch not imported")
    print("wrote    out/example_04_trailer.wav")


if __name__ == "__main__":
    main()
