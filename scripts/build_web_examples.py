#!/usr/bin/env python3
"""Pre-generate the web example set.

Each example starts as an intended Spec. The caption is derived from that Spec
with the same grammar the planner was trained on, the planner is run on the
caption, and `exact` records whether the planner reproduced the intended Spec.
The audio itself is rendered from the planned Spec (what the model actually
said), so the site never shows audio that disagrees with its spec panel.

Run:  .venv-torch/bin/python scripts/build_web_examples.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import Spec, SpecEvent, caption, score_from_spec  # noqa: E402
from orb.io import read_wav  # noqa: E402
from web.engine import (  # noqa: E402
    DETAIL_PATH,
    PLANNER_PATH,
    SAMPLE_RATE,
    Engine,
    spec_to_dict,
)

EXAMPLES_PATH = ROOT / "web" / "examples.json"
MANIFEST_PATH = ROOT / "artifacts" / "final" / "manifest.json"

# Pan values are exact `orb.grammar.pan_value` outputs:
# L3 -0.75, L2 -0.5, L1 -0.25, C 0.0, R1 0.25, R2 0.5, R3 0.75.
EXAMPLES: list[tuple[str, Spec, bool]] = [
    (
        "whoosh_air",
        Spec(6.0, "wind", "bright", [SpecEvent("whoosh", 1.00, 0.90, -6.0, -0.5, 0x2A)]),
        True,
    ),
    (
        "impact_deep",
        Spec(6.0, "room_cold", "dark", [SpecEvent("impact", 1.20, 0.80, -4.0, 0.0, 0x1F)]),
        True,
    ),
    (
        "ui_pop_soft",
        Spec(4.0, "vinyl", "playful", [SpecEvent("ui_pop", 0.60, 0.16, -12.0, 0.25, 0x73)]),
        True,
    ),
    (
        "riser_tension",
        Spec(8.0, "drone", "tense", [SpecEvent("riser", 1.00, 2.40, -16.0, 0.0, 0x51)]),
        True,
    ),
    (
        "bell_metal",
        Spec(8.0, "room_cold", "eerie", [SpecEvent("bell", 1.00, 1.60, -12.0, 0.5, 0x3D)]),
        True,
    ),
    (
        "sub_drop_clean",
        Spec(8.0, "sci_fi", "intense", [SpecEvent("sub_drop", 2.00, 1.20, -8.0, 0.0, 0x66)]),
        True,
    ),
    (
        "glitch_digital",
        Spec(4.0, "sci_fi", "dark", [SpecEvent("glitch", 0.80, 0.24, -14.0, -0.75, 0xB4)]),
        True,
    ),
    (
        "noise_sweep_air",
        Spec(8.0, "fire", "warm", [SpecEvent("noise_sweep", 1.00, 1.80, -16.0, 0.0, 0x97)]),
        True,
    ),
    (
        "scene_ui_pack",
        Spec(
            6.0,
            "vinyl",
            "playful",
            [
                SpecEvent("ui_pop", 0.40, 0.16, -12.0, -0.5, 0x11),
                SpecEvent("click", 1.20, 0.10, -16.0, 0.5, 0x22),
                SpecEvent("ui_pop", 2.00, 0.16, -10.0, 0.0, 0x33),
                SpecEvent("bell", 3.20, 1.40, -16.0, 0.25, 0x44),
            ],
        ),
        True,
    ),
    (
        "scene_trailer",
        Spec(
            12.0,
            "drone",
            "intense",
            [
                SpecEvent("riser", 1.00, 2.00, -18.0, 0.0, 0x55),
                SpecEvent("impact", 3.20, 0.80, -4.0, 0.0, 0x66),
                SpecEvent("whoosh", 5.00, 0.90, -8.0, -0.5, 0x77),
                SpecEvent("sub_drop", 7.00, 1.20, -8.0, 0.0, 0x88),
                SpecEvent("impact", 9.50, 0.90, -4.0, 0.0, 0x99),
            ],
        ),
        True,
    ),
    (
        "ambience_only",
        Spec(8.0, "rain", "calm", []),
        False,
    ),
    (
        "scene_trailer_dsp",
        Spec(
            12.0,
            "drone",
            "intense",
            [
                SpecEvent("riser", 1.00, 2.00, -18.0, 0.0, 0x55),
                SpecEvent("impact", 3.20, 0.80, -4.0, 0.0, 0x66),
                SpecEvent("whoosh", 5.00, 0.90, -8.0, -0.5, 0x77),
                SpecEvent("sub_drop", 7.00, 1.20, -8.0, 0.0, 0x88),
                SpecEvent("impact", 9.50, 0.90, -4.0, 0.0, 0x99),
            ],
        ),
        False,
    ),
]


def high_band_ratio(path: Path, cutoff: float = 2000.0) -> float:
    """Fraction of spectral power above `cutoff`. Cheap diagnostic only."""
    audio, sr = read_wav(path)
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    total = float(power.sum()) + 1e-12
    return float(power[freqs >= cutoff].sum() / total)


def model_info() -> dict:
    params = {}
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        params = manifest.get("params", {})
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "planner": PLANNER_PATH,
        "detail": DETAIL_PATH,
        "params": params,
        "sample_rate": SAMPLE_RATE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate every example even if its cached WAV already exists",
    )
    args = parser.parse_args()

    engine = Engine()
    engine.warmup()

    entries: list[dict] = []
    rows: list[tuple[str, bool, bool, float, float, float, int, bool]] = []

    for index, (name, intended, detail) in enumerate(EXAMPLES, start=1):
        print(f"[{index}/{len(EXAMPLES)}] {name}", file=sys.stderr, flush=True)
        cap = caption(score_from_spec(intended, with_music=False))
        result = engine.generate(cap, detail=detail, strength=1.0, force=args.force)
        exact = result["spec"] == spec_to_dict(intended)
        hf = high_band_ratio(engine.out_dir / f"{result['id']}.wav")
        entry = {"name": name, **result, "exact": exact}
        entries.append(entry)
        rows.append(
            (
                name,
                detail,
                exact,
                bool(result["passes"]),
                float(result["audit"]["peak"]),
                hf,
                float(result["audit"]["spectral_centroid"]),
                int(result["ms"]),
            )
        )

    payload = {
        "examples": entries,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_info(),
    }
    EXAMPLES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    header = (
        f"{'name':<20} {'detail':<7} {'exact':<6} {'pass':<5} "
        f"{'peak':>6} {'hf>2k':>7} {'centroid':>9} {'ms':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for name, detail, exact, ok, peak, hf, centroid, ms in rows:
        print(
            f"{name:<20} {'on' if detail else 'off':<7} "
            f"{'yes' if exact else 'NO':<6} {'yes' if ok else 'NO':<5} "
            f"{peak:>6.3f} {hf * 100:>6.1f}% {centroid / 1000:>7.2f}k {ms:>7d}"
        )
    exact_count = sum(1 for row in rows if row[2])
    pass_count = sum(1 for row in rows if row[3])
    print()
    print(
        f"{len(rows)} cues · {exact_count} exact · {pass_count} audit pass · "
        f"wrote {EXAMPLES_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
