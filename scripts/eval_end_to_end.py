#!/usr/bin/env python3
"""End-to-end check: held-out captions → Spec → Score → DSP → detail → audit."""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.spec_dataset import SpecDataset  # noqa: E402
from orb.eval.audit import audit_waveform, passes  # noqa: E402
from orb.grammar import score_from_spec, spec_from_score  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.render.detail import apply_detail  # noqa: E402
from orb.render.mix import render_stems  # noqa: E402
from orb.score import Score  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def _high_band(x: np.ndarray, sr: int, cutoff: float = 4000.0) -> float:
    if x.ndim > 1:
        x = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(spec[freqs >= cutoff] ** 2) + 1e-12)


def _onset(x: np.ndarray, sr: int, threshold: float = 0.02) -> float:
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    above = np.where(np.abs(x) > threshold * max(peak, 1e-9))[0]
    return float(above[0] / sr) if len(above) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="weights/planner.pt")
    ap.add_argument("--detail", default="weights/detail.pt")
    ap.add_argument("--data", default="data/heldout.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--detail-strength", type=float, default=1.0)
    ap.add_argument("--out", default="out/e2e")
    args = ap.parse_args()

    device = get_device()
    tok = SpecTokenizer()
    pck = torch.load(ROOT / args.planner, map_location="cpu", weights_only=False)
    planner = Planner(tok.vocab_size, **pck["config"]["model"]).to(device)
    planner.load_state_dict(pck["model"])
    planner.eval()

    detail = None
    if args.detail:
        dck = torch.load(ROOT / args.detail, map_location="cpu", weights_only=False)
        detail = DetailUNet(**dck["config"]["model"]).to(device)
        detail.load_state_dict(dck["generator"])
        detail.eval()

    dataset = SpecDataset(ROOT / args.data, tok, max_len=1024)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(min(args.n, len(dataset))):
        row = dataset.rows[i]
        true = spec_from_score(Score.from_dict(row["score"]))
        prompt = tok.prompt(row["caption"])
        grammar = SpecGrammar()
        gen = planner.generate(
            torch.tensor([prompt], device=device),
            max_new_tokens=1024,
            temperature=1e-6,
            top_p=1.0,
            eos_id=tok.eos_id,
            grammar=grammar,
        )
        spec = tok.decode_spec(gen[0, len(prompt) :].tolist())
        score = score_from_spec(spec, with_music=False)
        score.seed = zlib.crc32(row["caption"].encode()) & 0xFFFF
        score.validate()

        stems = render_stems(score)
        dsp = stems["mix"]
        det = (
            apply_detail(dsp, detail, seed=score.seed, strength=args.detail_strength)
            if detail is not None
            else dsp
        )
        a_dsp = audit_waveform(dsp, score.sample_rate)
        a_det = audit_waveform(det, score.sample_rate)
        first_t = score.events[0].t if score.events else 0.0
        onset_err = abs(_onset(stems["events"], score.sample_rate) - first_t) if score.events else 0.0
        if score.events:
            sr = score.sample_rate
            pre = stems["events"][: int(first_t * sr)]
            pre_rms = float(np.sqrt(np.mean(pre**2) + 1e-12))
            ev_rms = float(np.sqrt(np.mean(stems["events"] ** 2) + 1e-12))
            pre_event_db = 20.0 * float(np.log10(pre_rms + 1e-12)) - 20.0 * float(
                np.log10(ev_rms + 1e-12)
            )
        else:
            pre_event_db = -120.0
        rows.append(
            {
                "i": i,
                "exact": spec == true,
                "dsp_pass": passes(a_dsp),
                "det_pass": passes(a_det),
                "dsp_peak": a_dsp["peak"],
                "det_peak": a_det["peak"],
                "det_dc": a_det["dc"],
                "det_click": a_det["click_score"],
                "onset_err": onset_err,
                "pre_event_db": pre_event_db,
                "high_ratio": _high_band(det, score.sample_rate) / _high_band(dsp, score.sample_rate),
                "centroid_dsp": a_dsp["spectral_centroid"],
                "centroid_det": a_det["spectral_centroid"],
                "finite": bool(np.isfinite(det).all()),
            }
        )
        np.save(out_dir / f"cue_{i:03d}_dsp.npy", dsp)
        np.save(out_dir / f"cue_{i:03d}_det.npy", det)

    n = len(rows)
    summary = {
        "n": n,
        "exact_spec": sum(r["exact"] for r in rows) / n,
        "dsp_audit_pass": sum(r["dsp_pass"] for r in rows) / n,
        "detail_audit_pass": sum(r["det_pass"] for r in rows) / n,
        "finite": all(r["finite"] for r in rows),
        "mean_onset_err_ms": 1000.0 * float(np.mean([r["onset_err"] for r in rows])),
        "max_onset_err_ms": 1000.0 * float(np.max([r["onset_err"] for r in rows])),
        "max_pre_event_db": float(np.max([r["pre_event_db"] for r in rows])),
        "mean_high_ratio": float(np.mean([r["high_ratio"] for r in rows])),
        "mean_centroid_dsp": float(np.mean([r["centroid_dsp"] for r in rows])),
        "mean_centroid_det": float(np.mean([r["centroid_det"] for r in rows])),
        "max_peak": float(max(max(r["dsp_peak"], r["det_peak"]) for r in rows)),
        "max_abs_dc": float(max(abs(r["det_dc"]) for r in rows)),
        "max_click": float(max(r["det_click"] for r in rows)),
    }
    print(json.dumps(summary, indent=2))
    bad = [r for r in rows if not r["det_pass"] or not r["finite"]]
    if bad:
        print("failures:")
        for r in bad[:8]:
            print(" ", r)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
