"""Reward-model training: contrastive match + realness + sync."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.reward_dataset import RewardDataset, collate_reward
from ..model.reward import RewardModel
from ..model.tokenizer import ScoreTokenizer
from .common import CSVLogger, cosine_lr, get_device, load_checkpoint, save_checkpoint, set_seed


def train_reward(cfg: dict, resume: str | None = None) -> RewardModel:
    set_seed(cfg.get("seed", 0))
    device = get_device()
    tok = ScoreTokenizer()
    dataset = RewardDataset(cfg["data"], tok, n_mels=cfg.get("n_mels", 64))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("workers", 0),
        collate_fn=collate_reward,
        drop_last=True,
    )
    model = RewardModel(tok.vocab_size, **cfg.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01)
    total = cfg.get("epochs", 1) * max(len(loader), 1)
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(cfg.get("out", "runs/reward"))
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "log.csv")
    step = 0
    if resume:
        step, _ = load_checkpoint(resume, model, optimizer, map_location=device)

    for epoch in range(cfg.get("epochs", 1)):
        model.train()
        for batch in loader:
            lr = cosine_lr(step, total, cfg.get("lr", 3e-4), cfg.get("warmup", 50))
            for group in optimizer.param_groups:
                group["lr"] = lr
            mel = batch["mel"].to(device)
            text = batch["text_ids"].to(device)
            real = batch["real"].to(device)
            sync = batch["sync"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out_dict = model(mel, text)
                logits = out_dict["match_audio"] @ out_dict["match_text"].T / cfg.get("temperature", 0.07)
                labels = torch.arange(mel.shape[0], device=device)
                loss_match = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                loss_real = F.binary_cross_entropy_with_logits(out_dict["realness"], real)
                loss_sync = F.mse_loss(out_dict["sync"], sync)
                loss = (
                    loss_match
                    + cfg.get("real_weight", 0.5) * loss_real
                    + cfg.get("sync_weight", 0.1) * loss_sync
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % cfg.get("log_every", 10) == 0:
                logger.log(
                    step=step,
                    epoch=epoch,
                    loss=float(loss.detach()),
                    match=float(loss_match.detach()),
                    real=float(loss_real.detach()),
                    sync=float(loss_sync.detach()),
                )
                print(f"step {step:6d}  loss {float(loss.detach()):.4f}  match {float(loss_match.detach()):.4f}")
            step += 1
            if step % cfg.get("save_every", 500) == 0:
                save_checkpoint(out / f"step{step}.pt", model, optimizer, step, cfg)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        save_checkpoint(out / "last.pt", model, optimizer, step, cfg)
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break

    logger.close()
    return model
