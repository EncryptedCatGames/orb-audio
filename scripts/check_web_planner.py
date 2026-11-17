#!/usr/bin/env python3
"""Parity check: the browser's numpy planner vs ``orb.api.plan`` in torch.

Decodes N fresh grammar captions (``orb.grammar.sample_spec`` with a seed the
training data never used) plus the page's example chips and ``--free`` plain-text prompts with both, from the
fp16 export in ``site/assets/planner/``, and reports exact Spec agreement,
agreement with the caption's own Spec, and decode time.

    .venv-torch/bin/python scripts/check_web_planner.py --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "site" / "py"))
sys.dont_write_bytecode = True

from orb.api import load_planner, plan, spec_to_dict  # noqa: E402
from orb.grammar import NO_BED, caption, sample_spec, score_from_spec  # noqa: E402

import orb_planner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--free", type=int, default=20,
                    help="also decode this many plain-text prompts from data/freetext_eval.jsonl (parity only)")
    args = ap.parse_args()

    pdir = ROOT / "site" / "assets" / "planner"
    index = json.loads((pdir / "planner.json").read_text())
    orb_planner.load(json.dumps(index), (pdir / index.get("file", "planner.f16.bin")).read_bytes())
    tok, model = load_planner(ROOT / "weights" / "planner.pt")

    rng = np.random.default_rng(args.seed)
    cases = []
    while len(cases) < args.n:
        spec = sample_spec(rng)
        if spec.texture != NO_BED:
            cases.append(("heldout", caption(score_from_spec(spec, with_music=False)), spec_to_dict(spec)))
    cues = json.loads((ROOT / "site" / "assets" / "cues.json").read_text())
    for ex in cues.get("prompts", []):
        cases.append((f"chip:{ex['name']}", ex["caption"], None))
    free = [json.loads(ln)["prompt"] for ln in open(ROOT / "data" / "freetext_eval.jsonl") if ln.strip()]
    for p in free[:: max(1, len(free) // max(args.free, 1))][: args.free]:
        cases.append(("free", p, None))

    same = truth_np = truth_t = 0
    n_truth = 0
    t_np, t_torch = [], []
    for kind, cap, truth in cases:
        t0 = time.perf_counter()
        ref = spec_to_dict(plan(cap, tok, model))
        t_torch.append(time.perf_counter() - t0)
        try:
            res = orb_planner.generate(cap)
        except Exception as exc:  # noqa: BLE001
            print(f"{kind:16s} NUMPY ERROR {exc}")
            continue
        got = res["spec"]
        if res["planned"]:
            t_np.append(res["ms"] / 1000.0)
        ok = got == ref if res["planned"] else True
        same += ok
        if truth is not None:
            n_truth += 1
            truth_np += got == truth
            truth_t += ref == truth
        print(f"{kind:16s} parity={'OK ' if ok else 'DIFF'} planned={res['planned']} "
              f"exact_np={'-' if truth is None else got == truth} {res['ms']:7.0f} ms  {cap[:70]}")
        if not ok:
            print("   torch:", json.dumps(ref))
            print("   numpy:", json.dumps(got))
    print(f"\nparity {same}/{len(cases)}; caption->Spec exact: numpy {truth_np}/{n_truth}, torch {truth_t}/{n_truth}")
    if t_np:
        print(f"numpy decode: median {np.median(t_np) * 1000:.0f} ms, max {max(t_np) * 1000:.0f} ms (native CPython)")
    print(f"torch decode: median {np.median(t_torch) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
