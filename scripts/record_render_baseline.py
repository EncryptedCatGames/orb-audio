#!/usr/bin/env python3
"""Freeze what the DSP renderer produces today, so a change cannot drift it.

Writes ``tests/data/render_baseline.json``: a fixed set of Specs (every site
library cue, building block and playground preset, the noise-eval suite, the
README trailer, and every role and texture alone) and single events rendered
straight through ``render_event``, each with

* ``sha256``: SHA-256 of the float32 render (exact, bit-for-bit), and
* ``fp``: a fingerprint (RMS of 50 ms frames, per channel), so platforms whose
  FFT rounds differently can still be held to 1e-6 relative.

The specs are stored in the file itself, so editing the site's cue lists later
does not change what the test checks. ``tests/test_render_baseline.py``
enforces it. Re-run this only for an intentional change to an existing sound,
and say so in the changelog:

    .venv/bin/python scripts/record_render_baseline.py            # write
    .venv/bin/python scripts/record_render_baseline.py --check    # compare only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True

OUT = ROOT / "tests" / "data" / "render_baseline.json"

# The roles and textures that existed when the baseline was frozen. New roles
# are appended to EVENT_ROLES and never enter this list.
FROZEN_ROLES = ("whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop", "glitch", "noise_sweep")
FROZEN_TEXTURES = ("room_cold", "wind", "city_hum", "vinyl", "drone", "rain", "fire", "sci_fi")
ROLE_DUR = {"whoosh": 0.8, "impact": 0.7, "ui_pop": 0.16, "riser": 2.0, "swell": 1.5, "click": 0.1,
            "bell": 1.4, "sub_drop": 1.0, "glitch": 0.2, "noise_sweep": 1.5}


def _ev(role, t, dur, gain, pan, seed):
    return {"role": role, "t": t, "dur": dur, "gain_db": gain, "pan": pan, "seed": seed}


def baseline_specs() -> list[tuple[str, dict]]:
    import build_site
    import eval_noise

    out: list[tuple[str, dict]] = []
    for item in build_site.LIBRARY:
        out.append((f"site/library/{item['name']}", item["spec"]))
    for item in build_site.BLOCKS:
        out.append((f"site/block/{item['name']}", item["spec"]))
    for item in build_site.PLAYGROUND:
        out.append((f"site/playground/{item['name']}", item["spec"]))
    for name, spec in eval_noise.suite():
        out.append((f"noise/{name}", spec))
    for role in FROZEN_ROLES:
        for pan in (-0.75, 0.0, 0.5):
            out.append((f"role/{role}/pan{pan:+.2f}", {
                "duration": 4.0, "texture": "none", "mood": "neutral",
                "events": [_ev(role, 0.4, ROLE_DUR[role], -6.0, pan, 0x5A)]}))
    for tex in FROZEN_TEXTURES:
        out.append((f"texture/{tex}/bed", {"duration": 5.0, "texture": tex, "mood": "dark",
                                            "events": [_ev("impact", 1.0, 0.7, -6.0, 0.0, 3)]}))
    return out


def baseline_events() -> list[tuple[str, dict]]:
    out = []
    for role in FROZEN_ROLES:
        for seed in (1, 77, 200):
            for tail in (False, True):
                out.append((f"event/{role}/s{seed}/{'tail' if tail else 'dry'}",
                            {"role": role, "t": 0.25, "dur": ROLE_DUR[role] * 1.3, "gain_db": -4.0,
                             "pan": 0.25, "seed": seed, "tail": tail}))
    return out


def fingerprint(x: np.ndarray, sr: int = 48000) -> list[list[float]]:
    x = np.asarray(x, dtype=np.float64)
    win = int(0.05 * sr)
    n = len(x) // win * win
    if n == 0:
        return [[float(np.sqrt(np.mean(x[:, c] ** 2))) for c in range(x.shape[1])]]
    fr = x[:n].reshape(-1, win, x.shape[1])
    return np.sqrt(np.mean(fr**2, axis=1)).tolist()


def render_spec_case(spec: dict) -> np.ndarray:
    from orb.api import render_spec

    return np.ascontiguousarray(render_spec(spec), dtype=np.float32)


def render_event_case(e: dict) -> np.ndarray:
    from orb.grammar import EVENT_STYLE
    from orb.render.events import render_event
    from orb.score import Event

    apex = e["t"] + EVENT_STYLE[e["role"]][1] * e["dur"]
    ev = Event(e["t"], e["role"], e["dur"], apex=apex, gain_db=e["gain_db"], pan=e["pan"], seed=e["seed"])
    return np.ascontiguousarray(render_event(ev, 48000, seed=e["seed"], tail=e["tail"]), dtype=np.float32)


def sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against the stored file, write nothing")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cases = []
    for name, spec in baseline_specs():
        x = render_spec_case(spec)
        cases.append({"name": name, "kind": "spec", "spec": spec, "sha256": sha(x), "fp": fingerprint(x)})
    for name, e in baseline_events():
        x = render_event_case(e)
        cases.append({"name": name, "kind": "event", "event": e, "sha256": sha(x), "fp": fingerprint(x)})

    if args.check:
        stored = {c["name"]: c for c in json.loads(Path(args.out).read_text())["cases"]}
        bad = [c["name"] for c in cases if stored.get(c["name"], {}).get("sha256") != c["sha256"]]
        print(f"{len(cases) - len(bad)}/{len(cases)} bit-identical")
        for b in bad:
            print("  DIFF", b)
        return 1 if bad else 0

    payload = {
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "numpy": np.__version__, "python": platform.python_version()},
        "cases": cases,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
