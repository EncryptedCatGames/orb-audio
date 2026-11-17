#!/usr/bin/env python3
"""Evaluate a Spec planner: teacher-forced CE plus free-running exact match.

    python scripts/eval_spec.py --checkpoint runs/spec_v2/best.pt \
      --data data/heldout.jsonl --n 500

Teacher-forced CE says the model can copy when every previous token is correct.
Free-running exact match is the number that matters for generation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.data.spec_dataset import SpecDataset  # noqa: E402
from orb.grammar import spec_from_score  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import NOBED, SpecGrammar, SpecTokenizer  # noqa: E402
from orb.score import Score  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def teacher_forced_ce(model: Planner, dataset: SpecDataset, device: torch.device, max_batches: int) -> float:
    total = 0.0
    tokens = 0.0
    for start in range(0, min(len(dataset), max_batches * 32), 32):
        batch = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), dataset.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for j, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[j, :n] = b["input_ids"]
            labels[j, :n] = b["labels"]
        ids, labels = ids.to(device), labels.to(device)
        with torch.no_grad():
            logits, _ = model(ids)
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        mask = shift_labels.reshape(-1) != -100
        total += float(loss[mask].sum())
        tokens += float(mask.sum())
    return total / max(tokens, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/heldout.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=1e-6)
    args = ap.parse_args()

    device = get_device()
    tok = SpecTokenizer()
    dataset = SpecDataset(ROOT / args.data, tok, max_len=args.max_len)
    ckpt = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    vocab = int(ckpt["model"]["tok_emb.weight"].shape[0])
    model = Planner(vocab, **ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ce = teacher_forced_ce(model, dataset, device, args.max_batches)
    print(f"checkpoint   {args.checkpoint}")
    print(f"step         {ckpt.get('step')}")
    print(f"token ce     {ce:.4f}")
    print(f"perplexity   {math.exp(min(ce, 20.0)):.3f}")

    n = min(args.n, len(dataset))
    exact = 0
    decode_fail = 0
    fields = {
        "duration": 0,
        "texture": 0,
        "mood": 0,
        "n_events": 0,
        "roles": 0,
        "times": 0,
        "durs": 0,
        "gains": 0,
        "pans": 0,
        "seeds": 0,
    }
    t_err = 0.0
    dur_err = 0.0
    for i in range(n):
        row = dataset.rows[i]
        true = spec_from_score(Score.from_dict(row["score"]))
        prompt = tok.prompt(row["caption"])
        grammar = SpecGrammar(no_bed=vocab > NOBED)
        out = model.generate(
            torch.tensor([prompt], device=device),
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_p=1.0,
            eos_id=tok.eos_id,
            grammar=grammar,
        )
        generated = out[0, len(prompt) :].tolist()
        try:
            pred = tok.decode_spec(generated)
        except ValueError:
            decode_fail += 1
            continue

        ok = True
        if abs(pred.duration - true.duration) < 0.005:
            fields["duration"] += 1
        else:
            ok = False
        if pred.texture == true.texture:
            fields["texture"] += 1
        else:
            ok = False
        if pred.mood == true.mood:
            fields["mood"] += 1
        else:
            ok = False
        if len(pred.events) == len(true.events):
            fields["n_events"] += 1
        else:
            ok = False

        same_roles = len(pred.events) == len(true.events) and all(
            a.role == b.role for a, b in zip(pred.events, true.events)
        )
        if same_roles:
            fields["roles"] += 1
        else:
            ok = False

        if same_roles:
            for a, b in zip(pred.events, true.events):
                t_err += abs(a.t - b.t)
                dur_err += abs(a.dur - b.dur)
                if abs(a.t - b.t) < 0.005:
                    fields["times"] += 1
                if abs(a.dur - b.dur) < 0.005:
                    fields["durs"] += 1
                if abs(a.gain_db - b.gain_db) < 0.01:
                    fields["gains"] += 1
                if abs(a.pan - b.pan) < 0.01:
                    fields["pans"] += 1
                if a.seed == b.seed:
                    fields["seeds"] += 1
                if not (
                    abs(a.t - b.t) < 0.005
                    and abs(a.dur - b.dur) < 0.005
                    and abs(a.gain_db - b.gain_db) < 0.01
                    and abs(a.pan - b.pan) < 0.01
                    and a.seed == b.seed
                ):
                    ok = False
        if ok:
            exact += 1

    print(f"free-run n   {n}")
    print(f"exact        {exact}/{n}  ({100.0 * exact / max(n, 1):.1f}%)")
    print(f"decode fail  {decode_fail}")
    for k, v in fields.items():
        denom = n if k not in ("times", "durs", "gains", "pans", "seeds") else max(1, sum(len(dataset.rows[j]["score"]["events"]) for j in range(n)))
        print(f"  {k:10s} {v:5d}  ({100.0 * v / max(denom, 1):.1f}%)")
    events = sum(len(dataset.rows[j]["score"]["events"]) for j in range(n))
    print(f"mean |t| err {1000.0 * t_err / max(events, 1):.2f} ms")
    print(f"mean |dur| err {1000.0 * dur_err / max(events, 1):.2f} ms")


if __name__ == "__main__":
    main()
