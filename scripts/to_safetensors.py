#!/usr/bin/env python3
"""Convert the shipped checkpoints to safetensors for Hugging Face.

    python scripts/to_safetensors.py

Writes ``hf/model-planner/{planner.safetensors,config.json}`` and
``hf/model-detail/{detail.safetensors,config.json}``. The ``.pt`` files stay the
canonical format for the repo; safetensors is what the Hub prefers because it
cannot execute code on load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecTokenizer  # noqa: E402


def _save(state: dict, path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # The planner ties head.weight to tok_emb.weight. safetensors refuses
    # shared storage, so clone each tensor; load_state_dict re-ties them.
    flat = {k: v.detach().clone().contiguous() for k, v in state.items()}
    save_file(flat, str(path))
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    tok = SpecTokenizer()

    pck = torch.load(ROOT / "weights/planner.pt", map_location="cpu", weights_only=False)
    planner = Planner(tok.vocab_size, **pck["config"]["model"])
    planner.load_state_dict(pck["model"])
    _save(planner.state_dict(), ROOT / "hf/model-planner/planner.safetensors")
    (ROOT / "hf/model-planner/config.json").write_text(
        json.dumps(
            {
                "model_type": "orb-planner",
                "architectures": ["Planner"],
                "vocab_size": tok.vocab_size,
                "model": pck["config"]["model"],
                "tokenizer": "byte-level",
                "sample_rate": 48000,
                "parameters": _count(planner),
                "step": pck.get("step"),
                "format": "safetensors",
            },
            indent=2,
        )
        + "\n"
    )

    dck = torch.load(ROOT / "weights/detail.pt", map_location="cpu", weights_only=False)
    detail = DetailUNet(**dck["config"]["model"])
    detail.load_state_dict(dck["generator"])
    _save(detail.state_dict(), ROOT / "hf/model-detail/detail.safetensors")
    (ROOT / "hf/model-detail/config.json").write_text(
        json.dumps(
            {
                "model_type": "orb-detail",
                "architectures": ["DetailUNet"],
                "model": dck["config"]["model"],
                "sample_rate": 48000,
                "n_fft": dck["config"].get("n_fft", 1024),
                "hop": dck["config"].get("hop", 256),
                "parameters": _count(detail),
                "step": dck.get("step"),
                "format": "safetensors",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
