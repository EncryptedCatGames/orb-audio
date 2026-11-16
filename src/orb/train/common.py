"""Shared training utilities."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 100, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, step: int, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0), ckpt.get("config", {})


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.header_written = False

    def log(self, **values) -> None:
        if not self.header_written:
            self.writer.writerow(values.keys())
            self.header_written = True
        self.writer.writerow(values.values())
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
