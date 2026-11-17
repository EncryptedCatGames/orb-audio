"""Caption → compact Spec dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..grammar import Spec, SpecEvent, spec_from_score
from ..model.spec_tokenizer import SpecTokenizer
from ..score import Score


class SpecDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: SpecTokenizer, max_len: int = 1024):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        if "spec" in row:  # free-text rows carry the target Spec itself
            d = row["spec"]
            evs = [SpecEvent(e["role"], e["t"], e["dur"], e["gain_db"], e["pan"], e["seed"]) for e in d["events"]]
            spec = Spec(d["duration"], d["texture"], d["mood"], evs, d.get("bed_db"))
        else:
            spec = spec_from_score(Score.from_dict(row["score"]))
        pair = self.tokenizer.encode_pair(row["caption"], spec, max_len=self.max_len)
        return {
            "input_ids": torch.tensor(pair.input_ids, dtype=torch.long),
            "labels": torch.tensor(pair.labels, dtype=torch.long),
            "weights": torch.tensor(pair.weights, dtype=torch.float32),
        }
