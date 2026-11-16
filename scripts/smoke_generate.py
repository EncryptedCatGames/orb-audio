#!/usr/bin/env python3
"""Generate a score with the grammar mask and render it.

    python scripts/smoke_generate.py --checkpoint out/smoke_train/planner/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.io import write_wav  # noqa: E402
from orb.model.grammar import GrammarMask  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.tokenizer import EVT, ScoreTokenizer  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/smoke_train/planner/last.pt")
    ap.add_argument("--prompt", default="a fast airy whoosh into a deep impact")
    ap.add_argument("--out", default="out/generated.wav")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    device = get_device()
    tok = ScoreTokenizer()
    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = Planner(tok.vocab_size, **ckpt["config"].get("model", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = tok.prompt(args.prompt)
    grammar = GrammarMask(start="SCORE", min_events=1)
    out = model.generate(
        torch.tensor([prompt], device=device),
        max_new_tokens=args.max_new,
        grammar=grammar,
        temperature=args.temperature,
        top_p=0.95,
        eos_id=tok.eos_id,
    )
    generated = out[0, len(prompt) :].tolist()
    print(f"tokens   {len(generated)}  (event tokens: {generated.count(EVT)})")
    try:
        score = tok.decode_score(generated)
    except ValueError as exc:
        print(f"decode failed: {exc}")
        print("generation likely hit --max-new or the model context; increase either")
        return
    score.validate()
    audio = render(score)
    write_wav(ROOT / args.out, audio, score.sample_rate)

    print(f"prompt   {args.prompt}")
    print(f"tokens   {len(generated)}")
    print(f"duration {score.duration:.2f} s @ {score.sample_rate} Hz")
    print(f"events   {[e.role for e in score.events]}")
    print(f"music    {[t.instrument for t in score.music]}")
    print(f"textures {[t.character for t in score.textures]}")
    print(f"wrote    {args.out}")


if __name__ == "__main__":
    main()
