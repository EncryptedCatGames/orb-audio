"""Render the Destruction Cars UFO set to WAV (fork addition).

    .venv/bin/python scripts/render_ufo.py [out_dir]

Each sound is rendered straight from its role (no planner, no bed), peak-
normalised to -1 dBFS, 48 kHz stereo. The hover hum is a seamless loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orb.api import save_wav
from orb.render.events import render_event
from orb.score import Event

SR = 48000

# name: (role, dur, params, seed)
SET: dict[str, tuple[str, float, dict, int]] = {
    "ufo_hover_loop": ("ufo_hover", 4.0, {}, 1),
    "ufo_hover_loop_angry": ("ufo_hover", 4.0, dict(f0=62.0, theremin=1100.0, warble=1.8), 2),
    "ufo_tractor_beam": ("tractor_beam", 3.0, {}, 3),
    "ufo_laser_zap": ("ufo_laser", 0.45, {}, 4),
    "ufo_laser_burst": ("ufo_laser", 0.9, dict(pulses=3.0, f_start=3800.0, f_end=260.0), 5),
    "ufo_laser_charge": ("ufo_charge", 1.4, {}, 6),
    "ufo_abduction": ("abduction", 1.6, {}, 7),
}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out/ufo")
    out.mkdir(parents=True, exist_ok=True)
    for name, (role, dur, params, seed) in SET.items():
        ev = Event(t=0.0, role=role, dur=dur, params=params, gain_db=0.0, seed=seed)
        looping = float(params.get("loop", 1.0 if role == "ufo_hover" else 0.0)) >= 0.5
        audio = render_event(ev, SR, seed=seed, tail=not looping)
        audio = audio - audio.mean(axis=0, keepdims=True) if not looping else audio
        audio *= 10 ** (-1 / 20) / max(float(np.abs(audio).max()), 1e-9)
        if not looping:
            # Trim the reverb tail once it is 60 dB down, with a 20 ms fade.
            loud = np.flatnonzero(np.abs(audio).max(axis=1) > 1e-3)
            end = min(len(audio), int(loud[-1]) + int(0.02 * SR)) if len(loud) else len(audio)
            audio = audio[:end]
            b = min(end, int(0.02 * SR))
            audio[end - b :] *= np.linspace(1.0, 0.0, b)[:, None]
        save_wav(audio.astype(np.float32), out / f"{name}.wav", sample_rate=SR)
        print(f"{name:24s} {len(audio) / SR:5.2f} s  {role}")


if __name__ == "__main__":
    main()
