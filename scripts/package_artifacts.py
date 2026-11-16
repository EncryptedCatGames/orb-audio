#!/usr/bin/env python3
"""Package the trained artifacts into a small, self-contained directory.

    python scripts/package_artifacts.py --planner runs/spec_v5/best.pt \
      --detail runs/detail_det/best.pt --out artifacts

Writes slim checkpoints (weights + config only), copies the configs and
manifests, and records parameter counts and SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecTokenizer  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="runs/spec_v5/best.pt")
    ap.add_argument("--detail", default="runs/detail_det/best.pt")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"files": {}, "params": {}}

    # Planner: weights + config, no optimizer state.
    tok = SpecTokenizer()
    pck = torch.load(ROOT / args.planner, map_location="cpu", weights_only=False)
    planner = Planner(tok.vocab_size, **pck["config"]["model"])
    planner.load_state_dict(pck["model"])
    planner_path = out / "planner.pt"
    torch.save(
        {"model": planner.state_dict(), "config": pck["config"], "step": pck.get("step"), "vocab": tok.vocab_size},
        planner_path,
    )
    meta["params"]["planner"] = count(planner)
    meta["files"]["planner.pt"] = {"sha256": sha256(planner_path), "bytes": planner_path.stat().st_size}

    # Detail: generator + config, no discriminator or optimizer state.
    total = 0
    if args.detail and (ROOT / args.detail).exists():
        dck = torch.load(ROOT / args.detail, map_location="cpu", weights_only=False)
        detail = DetailUNet(**dck["config"]["model"])
        detail.load_state_dict(dck["generator"])
        detail_path = out / "detail.pt"
        torch.save(
            {"generator": detail.state_dict(), "config": dck["config"], "step": dck.get("step")},
            detail_path,
        )
        meta["params"]["detail"] = count(detail)
        total += count(detail)
        meta["files"]["detail.pt"] = {"sha256": sha256(detail_path), "bytes": detail_path.stat().st_size}

    meta["params"]["total"] = meta["params"]["planner"] + total
    meta["planner_step"] = pck.get("step")

    for name in ("spec_v5.json", "detail_v5.json"):
        src = ROOT / "configs" / name
        if src.exists():
            shutil.copy2(src, out / name)
    for name in ("heldout.jsonl", "train.jsonl", "eval.jsonl"):
        src = ROOT / "data" / "degrade" / name
        if src.exists():
            shutil.copy2(src, out / f"degrade_{name}")
    src = ROOT / "data" / "heldout.jsonl"
    if src.exists():
        shutil.copy2(src, out / "heldout.jsonl")

    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
