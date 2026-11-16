"""Preference pairs for DPO."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..model.tokenizer import ScoreTokenizer
from ..score import Score


class PreferenceDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: ScoreTokenizer, max_len: int = 1024):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        chosen = self.tokenizer.encode_pair(
            row["caption"], Score.from_dict(row["chosen"]), max_len=self.max_len
        )
        rejected = self.tokenizer.encode_pair(
            row["caption"], Score.from_dict(row["rejected"]), max_len=self.max_len
        )
        return {
            "chosen_ids": torch.tensor(chosen.input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen.labels, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected.input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected.labels, dtype=torch.long),
        }


def collate_preferences(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    def pad(key_ids: str, key_labels: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(item[key_ids]) for item in batch)
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item[key_ids])
            ids[i, :n] = item[key_ids]
            labels[i, :n] = item[key_labels]
            attn[i, :n] = 1
        return ids, labels, attn

    c_ids, c_labels, c_attn = pad("chosen_ids", "chosen_labels")
    r_ids, r_labels, r_attn = pad("rejected_ids", "rejected_labels")
    return {
        "chosen_ids": c_ids,
        "chosen_labels": c_labels,
        "chosen_attention": c_attn,
        "rejected_ids": r_ids,
        "rejected_labels": r_labels,
        "rejected_attention": r_attn,
    }
