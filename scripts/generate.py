#!/usr/bin/env python3
"""Generate a cue end to end: caption → Spec → Score → DSP → detail → WAV."""

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
from orb.grammar import score_from_spec  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.render.detail import apply_detail  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.score import Score  # noqa: E402
from orb.train.common import get_device  # noqa: E402


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


def spec_from_caption(tok: SpecTokenizer, planner: Planner, caption: str, device: torch.device):
    prompt = tok.prompt(caption)
    grammar = SpecGrammar()
    out = planner.generate(
        torch.tensor([prompt], device=device),
        max_new_tokens=1024,
        temperature=1e-6,
        top_p=1.0,
        eos_id=tok.eos_id,
        grammar=grammar,
    )
    return tok.decode_spec(out[0, len(prompt) :].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="artifacts/final/planner.pt")
    ap.add_argument("--detail", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--score", default=None)
    ap.add_argument("--out", default="out/generated.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detail-strength", type=float, default=1.0)
    args = ap.parse_args()

    device = get_device()
    if args.score:
        score = Score.load(ROOT / args.score)
        spec = None
    else:
        if not args.prompt:
            ap.error("pass --prompt or --score")
        tok, planner = load_planner(args.planner, device)
        spec = spec_from_caption(tok, planner, args.prompt, device)
        score = score_from_spec(spec, with_music=False)
    score.validate()

    detail = load_detail(args.detail, device) if args.detail else None
    dsp = render(score)
    audio = (
        apply_detail(dsp, detail, seed=score.seed, strength=args.detail_strength)
        if detail is not None
        else dsp
    )
    if not np.isfinite(audio).all():
        raise SystemExit("non-finite samples in output")
    write_wav(ROOT / args.out, audio, score.sample_rate)

    result = audit_waveform(audio, score.sample_rate)
    result["passes"] = passes(result)
    print(json.dumps(result, indent=2))
    if spec is not None:
        print(f"spec  {spec}")
    print(f"events {[(e.role, round(e.t, 2), round(e.dur, 2)) for e in score.events]}")
    print(f"dsp rms {float(np.sqrt(np.mean(dsp**2))):.4f}  out rms {float(np.sqrt(np.mean(audio**2))):.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
