#!/usr/bin/env python3
"""End-to-end smoke test of every training loop on tiny data.

    python scripts/smoke_train.py

Runs planner SFT, detail training, reward training, and DPO for a few steps
each. It is slow on CPU but it catches shape, loss, and checkpoint bugs before
a GPU run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.grammar import caption, sample_score  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.render.events import render_event  # noqa: E402
from orb.score import Event, Score  # noqa: E402
from orb.train.detail import train_detail  # noqa: E402
from orb.train.dpo import train_dpo  # noqa: E402
from orb.train.planner import train_planner  # noqa: E402
from orb.train.reward import train_reward  # noqa: E402

OUT = ROOT / "out" / "smoke_train"
PLANNER_MODEL = {"d": 64, "layers": 2, "heads": 4, "ctx": 256, "dropout": 0.0}


def build_scores() -> Path:
    path = OUT / "scores.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with path.open("w") as f:
        for _ in range(32):
            score = sample_score(rng)
            f.write(json.dumps({"caption": caption(score), "score": score.to_dict()}) + "\n")
    return path


def build_pairs() -> Path:
    pair_dir = OUT / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "pairs.jsonl"
    sr = 48000
    rng = np.random.default_rng(1)
    with manifest.open("w") as f:
        for i in range(8):
            ev = Event(0.05, "whoosh", 0.8, "fast, airy", apex=0.45, gain_db=0.0)
            dsp = render_event(ev, sr, seed=i)
            dsp = np.pad(dsp, ((int(0.05 * sr), 0), (0, 0)))
            real = dsp + 0.02 * rng.standard_normal(dsp.shape)
            real = real / max(float(np.max(np.abs(real))), 1e-9) * 0.8
            dsp = dsp / max(float(np.max(np.abs(dsp))), 1e-9) * 0.8
            write_wav(pair_dir / f"{i:03d}_dsp.wav", dsp, sr)
            write_wav(pair_dir / f"{i:03d}_real.wav", real, sr)
            f.write(
                json.dumps(
                    {
                        "dsp": str(pair_dir / f"{i:03d}_dsp.wav"),
                        "real": str(pair_dir / f"{i:03d}_real.wav"),
                        "role": "whoosh",
                        "cond": [0.0] * 32,
                    }
                )
                + "\n"
            )
    return manifest


def build_reward(pair_manifest: Path) -> Path:
    manifest = OUT / "reward.jsonl"
    rows = [json.loads(line) for line in pair_manifest.read_text().splitlines()]
    with manifest.open("w") as f:
        for row in rows:
            f.write(json.dumps({"audio": row["dsp"], "caption": "a fast airy whoosh", "real": 0.0, "sync": 0.0}) + "\n")
            f.write(json.dumps({"audio": row["real"], "caption": "a fast airy whoosh", "real": 1.0, "sync": 0.0}) + "\n")
    return manifest


def build_preferences() -> Path:
    path = OUT / "preferences.jsonl"
    a = Score.demo()
    b = Score.demo()
    b.events = [Event(4.2, "bell", 0.8, "soft", apex=4.5, gain_db=-10)]
    with path.open("w") as f:
        for _ in range(8):
            f.write(json.dumps({"caption": "a soft bell cue", "chosen": a.to_dict(), "rejected": b.to_dict()}) + "\n")
    return path


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    scores = build_scores()
    print("planner...")
    train_planner(
        {
            "data": str(scores),
            "out": str(OUT / "planner"),
            "max_len": 256,
            "batch_size": 2,
            "epochs": 1,
            "max_steps": 5,
            "lr": 3e-4,
            "warmup": 2,
            "workers": 0,
            "log_every": 1,
            "save_every": 100,
            "model": PLANNER_MODEL,
        }
    )

    pairs = build_pairs()
    print("detail...")
    train_detail(
        {
            "data": str(pairs),
            "out": str(OUT / "detail"),
            "n_frames": 64,
            "n_fft": 1024,
            "hop": 256,
            "batch_size": 2,
            "epochs": 1,
            "max_steps": 2,
            "lr_g": 1e-4,
            "lr_d": 1e-4,
            "workers": 0,
            "log_every": 1,
            "save_every": 100,
            "model": {"cond_dim": 32, "base": 8, "levels": 2, "residual_scale": 0.5},
            "disc": {"in_ch": 2, "base": 8, "scales": 2},
        }
    )

    reward = build_reward(pairs)
    print("reward...")
    train_reward(
        {
            "data": str(reward),
            "out": str(OUT / "reward"),
            "n_mels": 64,
            "batch_size": 2,
            "epochs": 1,
            "max_steps": 2,
            "lr": 3e-4,
            "warmup": 1,
            "workers": 0,
            "log_every": 1,
            "save_every": 100,
            "model": {"d": 32, "text_layers": 1, "text_heads": 4, "ctx": 64},
        }
    )

    prefs = build_preferences()
    print("dpo...")
    train_dpo(
        {
            "data": str(prefs),
            "sft": str(OUT / "planner" / "last.pt"),
            "out": str(OUT / "dpo"),
            "max_len": 256,
            "batch_size": 2,
            "epochs": 1,
            "max_steps": 2,
            "lr": 5e-6,
            "beta": 0.1,
            "workers": 0,
            "log_every": 1,
            "save_every": 100,
            "model": PLANNER_MODEL,
        }
    )

    print("smoke train ok")


if __name__ == "__main__":
    main()
