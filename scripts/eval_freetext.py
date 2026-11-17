"""Score the planner on hand-written free-text prompts (data/freetext_eval.jsonl).

Each row says what a reasonable Spec must contain: required roles with a count
range, optional roles that may appear, accepted textures ("none" = no bed), an
optional mood set, an optional intensity (soft / loud) and an optional stated
duration. A prompt passes when every stated check holds; nothing else is
scored, so any sensible layout passes.

Routes:
  raw     the prompt goes straight to the planner (what `orb.generate` does)
  helper  the site's keyword helper (site/app.js `translate`, run in node)
          writes a grammar caption first, which then goes through the same
          routing as `orb.generate` (no-bed captions read exactly, the rest
          planned). A prompt the helper cannot read fails.

The set is split by a hash of the prompt into "dev" and "test" halves; report
both, tune nothing on "test".

    python scripts/eval_freetext.py --planner weights/planner.pt --route raw helper
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import EVENT_STYLE, NO_BED, parse_caption  # noqa: E402

CHECKS = ("roles", "extras", "texture", "mood", "intensity", "duration")


def split_of(prompt: str, blind: bool = False) -> str:
    if blind:
        return "test"
    return "dev" if int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 2 == 0 else "test"


def score(row: dict, spec) -> dict[str, bool]:
    counts: dict[str, int] = defaultdict(int)
    for e in spec.events:
        counts[e.role] += 1
    req = row["roles"]
    ok = {"roles": all(lo <= counts.get(r, 0) <= hi for r, (lo, hi) in req.items())}
    if row.get("any_of"):  # at least one of these (e.g. "a notification blip": pop, chime or click)
        ok["roles"] = ok["roles"] and any(counts.get(r, 0) > 0 for r in row["any_of"])
    allowed = set(req) | set(row.get("optional") or [])
    ok["extras"] = all(r in allowed for r in counts)
    ok["texture"] = spec.texture in row["texture"]
    if row.get("mood"):
        ok["mood"] = spec.mood in row["mood"]
    if row.get("intensity"):
        evs = [e for e in spec.events if e.role in req] or list(spec.events)
        if not evs:
            ok["intensity"] = False
        else:
            rel = sum(e.gain_db - EVENT_STYLE[e.role][2] for e in evs) / len(evs)
            ok["intensity"] = rel <= -3.0 if row["intensity"] == "soft" else rel >= 2.0
    if row.get("duration"):
        d = float(row["duration"])
        ok["duration"] = abs(spec.duration - d) <= max(0.5, 0.1 * d)
    ok["all"] = all(ok.values())
    return ok


def helper_captions(prompts: list[str]) -> list[str | None]:
    out = subprocess.run(
        ["node", str(ROOT / "scripts" / "helper_translate.mjs")],
        input="\n".join(json.dumps(p) for p in prompts) + "\n",
        capture_output=True, text=True, check=True,
    ).stdout.strip().split("\n")
    return [None if json.loads(ln) is None else json.loads(ln)["caption"] for ln in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "freetext_eval.jsonl"))
    ap.add_argument("--planner", default=str(ROOT / "weights" / "planner.pt"))
    ap.add_argument("--route", nargs="+", default=["raw"], choices=["raw", "helper"])
    ap.add_argument("--out", default=None, help="write per-prompt results (jsonl)")
    ap.add_argument("--show-fails", type=int, default=0)
    args = ap.parse_args()

    from orb.api import load_planner, plan, spec_to_dict

    rows = [json.loads(ln) for ln in open(args.data) if ln.strip()]
    tok, planner = load_planner(args.planner)
    results = []
    for route in args.route:
        caps = helper_captions([r["prompt"] for r in rows]) if route == "helper" else [r["prompt"] for r in rows]
        for row, cap in zip(rows, caps):
            spec = None
            if cap is not None:
                direct = None
                if route == "helper":
                    try:
                        direct = parse_caption(cap)
                    except ValueError:
                        direct = None
                spec = direct if direct is not None and direct.texture == NO_BED else plan(cap, tok, planner)
            ok = score(row, spec) if spec is not None else {"all": False, "roles": False, "extras": False, "texture": False}
            results.append({"route": route, "split": split_of(row["prompt"], "blind" in row["tags"]), "prompt": row["prompt"],
                            "tags": row["tags"], "caption": cap, "ok": ok,
                            "spec": spec_to_dict(spec) if spec is not None else None})

    summary = {}
    for route in args.route:
        for split in ("dev", "test", "all"):
            rs = [r for r in results if r["route"] == route and (split == "all" or r["split"] == split)]
            if not rs:
                continue
            line = {"n": len(rs), "pass": round(sum(r["ok"]["all"] for r in rs) / len(rs), 4)}
            for c in CHECKS:
                xs = [r["ok"][c] for r in rs if c in r["ok"]]
                if xs:
                    line[c] = round(sum(xs) / len(xs), 4)
            summary[f"{route}/{split}"] = line
        tags = defaultdict(list)
        for r in results:
            if r["route"] == route:
                for t in r["tags"]:
                    tags[t].append(r["ok"]["all"])
        summary[f"{route}/by_tag"] = {t: round(sum(v) / len(v), 3) for t, v in sorted(tags.items())}
    print(json.dumps({"planner": args.planner, **summary}, indent=1))
    if args.show_fails:
        for r in [r for r in results if not r["ok"]["all"] and r["split"] == "dev"][: args.show_fails]:
            s = r["spec"]
            brief = None if s is None else (s["texture"], s["mood"], s["duration"], [e["role"] for e in s["events"]])
            print(r["route"], "|", r["prompt"], "|", {k: v for k, v in r["ok"].items() if not v}, "|", brief)
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
