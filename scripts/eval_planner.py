#!/usr/bin/env python3
"""Evaluate a planner checkpoint on a held-out JSONL.

    python scripts/eval_planner.py --checkpoint runs/planner_v1/step5000.pt \
      --data data/heldout.jsonl --max-batches 100

Reports unweighted token cross-entropy and perplexity, which are comparable
across runs (the training loss is weighted).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.score_dataset import ScoreDataset, collate_scores  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.tokenizer import ScoreTokenizer  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--spec", action="store_true")
    args = ap.parse_args()

    device = get_device()
    if args.spec:
        from orb.data.spec_dataset import SpecDataset
        from orb.model.spec_tokenizer import SpecTokenizer

        tok = SpecTokenizer()
        dataset = SpecDataset(ROOT / args.data, tok, max_len=args.max_len)
    else:
        tok = ScoreTokenizer()
        dataset = ScoreDataset(ROOT / args.data, tok, max_len=args.max_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_scores(batch, tok.pad_id),
    )
    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    model = Planner(tok.vocab_size, **ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_loss = 0.0
    total_tokens = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(ids)
            # Same causal shift as Planner.forward: logits at t predict token t + 1.
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            mask = shift_labels.reshape(-1) != -100
            total_loss += float(token_loss[mask].sum())
            total_tokens += float(mask.sum())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break

    mean = total_loss / max(total_tokens, 1.0)
    print(f"checkpoint {args.checkpoint}")
    print(f"batches    {batches}")
    print(f"tokens     {int(total_tokens)}")
    print(f"token ce   {mean:.4f}")
    print(f"perplexity {math.exp(min(mean, 20.0)):.2f}")


if __name__ == "__main__":
    main()
