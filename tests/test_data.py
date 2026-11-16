import json

import numpy as np
import torch

from orb.data.pair_dataset import PairDataset, collate_pairs
from orb.data.score_dataset import ScoreDataset, collate_scores
from orb.eval.audit import audit_file, passes
from orb.io import write_wav
from orb.model.tokenizer import ScoreTokenizer
from orb.score import Score


def _write(path, kind="sine", sr=48000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    if kind == "sine":
        x = 0.5 * np.sin(2 * np.pi * 220 * t)
    else:
        rng = np.random.default_rng(0)
        x = 0.2 * rng.standard_normal(len(t))
    write_wav(path, x, sr)


def test_score_dataset(tmp_path):
    tok = ScoreTokenizer()
    path = tmp_path / "scores.jsonl"
    score = Score.demo()
    path.write_text(json.dumps({"caption": "a test cue", "score": score.to_dict()}) + "\n")
    ds = ScoreDataset(path, tok, max_len=2048)
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])
    batch = collate_scores([item, item], tok.pad_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].sum() > 0


def test_pair_dataset(tmp_path):
    real = tmp_path / "real.wav"
    dsp = tmp_path / "dsp.wav"
    _write(real, "noise")
    _write(dsp, "sine")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"dsp": str(dsp), "real": str(real), "role": "whoosh", "cond": [0.0] * 32}) + "\n"
    )
    ds = PairDataset(manifest, n_frames=64, train=False)
    item = ds[0]
    assert item["dsp"].shape == (2, 513, 64)
    assert item["real"].shape == (2, 513, 64)
    assert item["cond"].shape == (32,)
    batch = collate_pairs([item, item])
    assert batch["dsp"].shape[0] == 2


def test_audit(tmp_path):
    path = tmp_path / "a.wav"
    _write(path, "sine")
    result = audit_file(path)
    assert result["peak"] > 0
    assert result["rms_dbfs"] < 0
    assert passes(result)
