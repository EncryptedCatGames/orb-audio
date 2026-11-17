"""Dataset for reward-model training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..eval.mel import log_mel
from ..io import read_wav
from ..model.tokenizer import ScoreTokenizer


class RewardDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        tokenizer: ScoreTokenizer,
        n_mels: int = 64,
        max_text_len: int = 128,
    ):
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.n_mels = n_mels
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        audio, sr = read_wav(row["audio"])
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        mel = log_mel(audio, sr, self.n_mels)
        text = self.tokenizer.encode_caption(row["caption"])[: self.max_text_len]
        return {
            "mel": mel,
            "text_ids": torch.tensor(text, dtype=torch.long),
            "real": torch.tensor(float(row.get("real", 1.0)), dtype=torch.float32),
            "sync": torch.tensor(float(row.get("sync", 0.0)), dtype=torch.float32),
        }


def collate_reward(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_t = max(item["mel"].shape[-1] for item in batch)
    max_text = max(len(item["text_ids"]) for item in batch)
    mels = torch.stack([F.pad(item["mel"], (0, max_t - item["mel"].shape[-1])) for item in batch])
    text = torch.zeros((len(batch), max_text), dtype=torch.long)
    for i, item in enumerate(batch):
        text[i, : len(item["text_ids"])] = item["text_ids"]
    return {
        "mel": mels,
        "text_ids": text,
        "real": torch.stack([item["real"] for item in batch]),
        "sync": torch.stack([item["sync"] for item in batch]),
    }
