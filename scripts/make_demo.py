#!/usr/bin/env python3
"""Generate a curated demo set end to end and audit every cue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.audit import audit_waveform, passes  # noqa: E402
from orb.grammar import Spec, SpecEvent, caption, score_from_spec  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.render.mix import apply_detail_routed, render_stems  # noqa: E402
from orb.train.common import get_device  # noqa: E402

# name, duration, texture, mood, events (role, t, dur, gain_db, pan, seed)
CASES = [
    ("whoosh_air", 6.0, "wind", "bright", [("whoosh", 1.00, 0.90, -6.0, -0.5, 0x2A)]),
    ("impact_deep", 6.0, "room_cold", "dark", [("impact", 1.20, 0.80, -4.0, 0.0, 0x1F)]),
    ("ui_pop_soft", 4.0, "vinyl", "playful", [("ui_pop", 0.60, 0.16, -12.0, 0.25, 0x73)]),
    ("riser_tension", 8.0, "drone", "tense", [("riser", 1.00, 2.40, -16.0, 0.0, 0x51)]),
    ("swell_wide", 8.0, "rain", "hopeful", [("swell", 1.50, 1.80, -14.0, 0.0, 0x0C)]),
    ("click_tight", 4.0, "city_hum", "neutral", [("click", 0.40, 0.10, -14.0, -0.25, 0x88)]),
    ("bell_metal", 8.0, "room_cold", "eerie", [("bell", 1.00, 1.60, -12.0, 0.5, 0x3D)]),
    ("sub_drop_clean", 8.0, "sci_fi", "intense", [("sub_drop", 2.00, 1.20, -8.0, 0.0, 0x66)]),
    ("glitch_digital", 4.0, "sci_fi", "dark", [("glitch", 0.80, 0.24, -14.0, -0.75, 0xB4)]),
    ("noise_sweep_air", 8.0, "fire", "warm", [("noise_sweep", 1.00, 1.80, -16.0, 0.0, 0x97)]),
    (
        "scene_ui_pack",
        6.0,
        "vinyl",
        "playful",
        [
            ("ui_pop", 0.40, 0.16, -12.0, -0.5, 0x11),
            ("click", 1.20, 0.10, -16.0, 0.5, 0x22),
            ("ui_pop", 2.00, 0.16, -10.0, 0.0, 0x33),
            ("bell", 3.20, 1.40, -16.0, 0.25, 0x44),
        ],
    ),
    (
        "scene_trailer",
        12.0,
        "drone",
        "intense",
        [
            ("riser", 1.00, 2.00, -18.0, 0.0, 0x55),
            ("impact", 3.20, 0.80, -4.0, 0.0, 0x66),
            ("whoosh", 5.00, 0.90, -8.0, -0.5, 0x77),
            ("sub_drop", 7.00, 1.20, -6.0, 0.0, 0x88),
            ("impact", 9.50, 0.90, -4.0, 0.0, 0x99),
        ],
    ),
    (
        "scene_ambience",
        16.0,
        "rain",
        "calm",
        [
            ("swell", 2.00, 2.00, -18.0, -0.25, 0xAA),
            ("bell", 6.00, 1.60, -18.0, 0.5, 0xBB),
            ("noise_sweep", 10.00, 2.00, -20.0, 0.0, 0xCC),
        ],
    ),
]


def load_planner(path: str, device: torch.device):
    tok = SpecTokenizer()
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = Planner(tok.vocab_size, **ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return tok, model


def load_detail(path: str, device: torch.device) -> DetailUNet:
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = DetailUNet(**ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model


def plan(tok: SpecTokenizer, planner: Planner, text: str, device: torch.device) -> Spec:
    prompt = tok.prompt(text)
    out = planner.generate(
        torch.tensor([prompt], device=device),
        max_new_tokens=1024,
        temperature=1e-6,
        top_p=1.0,
        eos_id=tok.eos_id,
        grammar=SpecGrammar(),
    )
    return tok.decode_spec(out[0, len(prompt) :].tolist())


def high_band(x: np.ndarray, sr: int, cutoff: float = 4000.0) -> float:
    if x.ndim > 1:
        x = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(spec[freqs >= cutoff] ** 2) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="weights/planner.pt")
    ap.add_argument("--detail", default="weights/detail.pt")
    ap.add_argument("--out", default="out/demo")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--sweep", default="scene_trailer")
    args = ap.parse_args()

    device = get_device()
    tok, planner = load_planner(args.planner, device)
    detail = load_detail(args.detail, device) if args.detail else None
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, duration, texture, mood, events in CASES:
        spec = Spec(
            duration,
            texture,
            mood,
            [SpecEvent(r, t, d, g, p, s) for (r, t, d, g, p, s) in events],
        )
        text = caption(score_from_spec(spec))
        got = plan(tok, planner, text, device)
        score = score_from_spec(got, with_music=False)
        score.seed = zlib_crc(text)
        score.validate()

        stems = render_stems(score)
        dsp = stems["mix"]
        audio = apply_detail_routed(stems, detail, seed=score.seed, strength=args.strength) if detail else dsp
        if not np.isfinite(audio).all():
            raise SystemExit(f"non-finite output for {name}")
        write_wav(out_dir / f"{name}.wav", audio, score.sample_rate)

        a = audit_waveform(audio, score.sample_rate)
        rows.append(
            {
                "name": name,
                "caption": text,
                "exact": got == spec,
                "got": {
                    "duration": got.duration,
                    "texture": got.texture,
                    "mood": got.mood,
                    "events": [e.__dict__ for e in got.events],
                },
                "audit": a,
                "passes": passes(a),
                "high_band_ratio": high_band(audio, score.sample_rate)
                / max(high_band(dsp, score.sample_rate), 1e-12),
                "rms_dbfs_dsp": 20.0 * float(np.log10(np.sqrt(np.mean(dsp**2)) + 1e-12)),
                "rms_dbfs_out": 20.0 * float(np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)),
            }
        )

    if detail is not None and args.sweep:
        case = next((c for c in CASES if c[0] == args.sweep), None)
        if case is not None:
            name, duration, texture, mood, events = case
            spec = Spec(duration, texture, mood, [SpecEvent(*e) for e in events])
            text = caption(score_from_spec(spec))
            score = score_from_spec(plan(tok, planner, text, device), with_music=False)
            score.seed = zlib_crc(text)
            stems = render_stems(score)
            for s in (0.0, 0.35, 0.7, 1.0):
                audio = apply_detail_routed(stems, detail, seed=score.seed, strength=s)
                write_wav(out_dir / f"{name}_strength{s:.2f}.wav", audio, score.sample_rate)

    summary = {
        "n": len(rows),
        "exact": sum(r["exact"] for r in rows) / len(rows),
        "audit_pass": sum(r["passes"] for r in rows) / len(rows),
        "mean_high_band_ratio": float(np.mean([r["high_band_ratio"] for r in rows])),
        "min_high_band_ratio": float(np.min([r["high_band_ratio"] for r in rows])),
        "max_high_band_ratio": float(np.max([r["high_band_ratio"] for r in rows])),
        "max_peak": float(max(r["audit"]["peak"] for r in rows)),
        "max_abs_dc": float(max(abs(r["audit"]["dc"]) for r in rows)),
    }
    print(json.dumps(summary, indent=2))
    for r in rows:
        print(
            f"{r['name']:18s} exact={r['exact']!s:5s} pass={r['passes']!s:5s} "
            f"peak={r['audit']['peak']:.3f} hb={r['high_band_ratio']:.2f} "
            f"rms {r['rms_dbfs_dsp']:.1f} -> {r['rms_dbfs_out']:.1f} dBFS"
        )
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(f"wrote {out_dir}")


def zlib_crc(text: str) -> int:
    import zlib

    return zlib.crc32(text.encode()) & 0xFFFF


if __name__ == "__main__":
    main()
