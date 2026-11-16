#!/usr/bin/env python3
"""Check planner edge cases: 0 events, 1 event, and MAX_EVENTS."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import Spec, SpecEvent, caption, score_from_spec  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def main() -> None:
    dev = get_device()
    tok = SpecTokenizer()
    ck = torch.load(ROOT / "runs/spec_v4/best.pt", map_location="cpu", weights_only=False)
    model = Planner(tok.vocab_size, **ck["config"]["model"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    def run(spec: Spec):
        text = caption(score_from_spec(spec))
        prompt = tok.prompt(text)
        out = model.generate(
            torch.tensor([prompt], device=dev),
            max_new_tokens=1024,
            temperature=1e-6,
            top_p=1.0,
            eos_id=tok.eos_id,
            grammar=SpecGrammar(),
        )
        got = tok.decode_spec(out[0, len(prompt) :].tolist())
        return text, got, got == spec

    cases = [
        Spec(8.0, "wind", "bright", []),
        Spec(8.0, "wind", "bright", [SpecEvent("whoosh", 1.0, 0.9, -6.0, -0.5, 0x2A)]),
        Spec(12.0, "sci_fi", "tense", [SpecEvent("impact", 2.0, 0.7, -6.0, 0.0, 0x1F)]),
        Spec(
            16.0,
            "rain",
            "calm",
            [
                SpecEvent("bell", 1.0, 1.4, -14.0, 0.5, 0x33),
                SpecEvent("swell", 4.0, 1.5, -18.0, -0.25, 0x44),
            ],
        ),
        Spec(
            24.0,
            "drone",
            "dark",
            [
                SpecEvent(role, 1.0 + i * 2.0, 0.6, -10.0, 0.0, i)
                for i, role in enumerate(
                    ("whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop")
                )
            ],
        ),
    ]
    bad = 0
    for spec in cases:
        text, got, ok = run(spec)
        print(f"n={len(spec.events)} dur={spec.duration} exact={ok} got_n={len(got.events)} got_dur={got.duration}")
        if not ok:
            bad += 1
            print("  want", spec)
            print("  got ", got)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
