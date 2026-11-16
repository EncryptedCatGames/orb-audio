#!/usr/bin/env python3
"""Build DPO preference pairs with the planner and reward model.

    python scripts/build_preferences.py \
      --planner runs/planner_s/last.pt \
      --reward runs/reward_s/last.pt \
      --prompts data/scores.jsonl \
      --out data/preferences.jsonl \
      --n 1000 --samples 4

For each caption, sample N scores, render them, rank with the reward model,
and keep the best/worst pair. Slow by design: this is an offline data step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.eval.mel import log_mel  # noqa: E402
from orb.model.grammar import GrammarMask  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.reward import RewardModel  # noqa: E402
from orb.model.tokenizer import ScoreTokenizer  # noqa: E402
from orb.render.mix import render  # noqa: E402
from orb.train.common import get_device  # noqa: E402


def load_model(path: str, cls, tok: ScoreTokenizer, device):
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = cls(tok.vocab_size, **ckpt["config"].get("model", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def reward_score(reward, audio, sr, caption, tok, device) -> float:
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    mel = log_mel(mono, sr).unsqueeze(0).to(device)
    text = torch.tensor([tok.encode_caption(caption)[:128]], device=device)
    out = reward(mel, text)
    match = float((out["match_audio"] * out["match_text"]).sum(dim=-1))
    real = float(out["realness"])
    return match + 0.5 * real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True)
    ap.add_argument("--reward", required=True)
    ap.add_argument("--prompts", default="data/scores.jsonl")
    ap.add_argument("--out", default="data/preferences.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    device = get_device()
    tok = ScoreTokenizer()
    planner = load_model(args.planner, Planner, tok, device)
    reward = load_model(args.reward, RewardModel, tok, device)

    rows = [json.loads(line) for line in Path(ROOT / args.prompts).read_text().splitlines() if line.strip()]
    rows = rows[: args.n]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w") as f:
        for i, row in enumerate(rows):
            caption = row["caption"]
            prompt = tok.prompt(caption)
            candidates: list[tuple[float, dict]] = []
            for _ in range(args.samples):
                grammar = GrammarMask(start="SCORE", min_events=1)
                ids = planner.generate(
                    torch.tensor([prompt], device=device),
                    max_new_tokens=args.max_new,
                    grammar=grammar,
                    temperature=args.temperature,
                    top_p=0.95,
                    eos_id=tok.eos_id,
                )
                generated = ids[0, len(prompt) :].tolist()
                try:
                    score = tok.decode_score(generated)
                    score.validate()
                except ValueError:
                    continue
                audio = render(score)
                value = reward_score(reward, audio, score.sample_rate, caption, tok, device)
                candidates.append((value, score.to_dict()))
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda item: item[0])
            f.write(
                json.dumps(
                    {
                        "caption": caption,
                        "chosen": candidates[-1][1],
                        "rejected": candidates[0][1],
                    }
                )
                + "\n"
            )
            written += 1
            if written % 50 == 0:
                print(f"{written} pairs", flush=True)

    print(f"wrote {written} preference pairs to {out}")


if __name__ == "__main__":
    main()
