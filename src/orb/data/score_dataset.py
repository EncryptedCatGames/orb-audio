"""Caption → score dataset for planner SFT."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..model.tokenizer import ScoreTokenizer
from ..score import Score


class ScoreDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: ScoreTokenizer, max_len: int = 1024):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        score = Score.from_dict(row["score"])
        pair = self.tokenizer.encode_pair(row["caption"], score, max_len=self.max_len)
        return {
            "input_ids": torch.tensor(pair.input_ids, dtype=torch.long),
            "labels": torch.tensor(pair.labels, dtype=torch.long),
            "weights": torch.tensor(pair.weights, dtype=torch.float32),
        }


def collate_scores(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.zeros((len(batch), max_len), dtype=torch.float32)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        weights[i, :n] = item["weights"]
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "weights": weights, "attention_mask": attention}
