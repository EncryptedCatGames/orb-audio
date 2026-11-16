#!/usr/bin/env python3
"""Generate a Spec with the compact planner, compile it, and render it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import score_from_spec  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="a fast airy whoosh into a deep impact")
    ap.add_argument("--out", default="out/spec_generated.wav")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--tries", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    tok = SpecTokenizer()
    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = Planner(tok.vocab_size, **ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = tok.prompt(args.prompt)
    spec = None
    for attempt in range(args.tries):
        grammar = SpecGrammar()
        out = model.generate(
            torch.tensor([prompt], device=device),
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_p=0.95,
            eos_id=tok.eos_id,
            grammar=grammar,
        )
        generated = out[0, len(prompt) :].tolist()
        try:
            spec = tok.decode_spec(generated)
            break
        except ValueError as exc:
            print(f"attempt {attempt + 1}: decode failed: {exc}")
    if spec is None:
        print("all attempts failed")
        return

    score = score_from_spec(spec, with_music=False)
    score.validate()
    audio = render(score)
    write_wav(ROOT / args.out, audio, score.sample_rate)
    print(f"spec     {spec}")
    print(f"events   {[(e.role, round(e.t, 2), round(e.dur, 2), round(e.gain_db, 1), round(e.pan, 2)) for e in score.events]}")
    print(f"texture  {[(t.character, round(t.dur, 2)) for t in score.textures]}")
    print(f"wrote    {args.out}")


if __name__ == "__main__":
    main()
