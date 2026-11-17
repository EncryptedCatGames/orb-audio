"""Planner SFT."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.score_dataset import ScoreDataset, collate_scores
from ..model.planner import Planner
from ..model.tokenizer import ScoreTokenizer
from .common import CSVLogger, cosine_lr, get_device, load_checkpoint, save_checkpoint, set_seed


@torch.no_grad()
def evaluate(model: Planner, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    batches = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(ids)
        # Same causal shift as Planner.forward: logits at t predict token t + 1.
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        mask = shift_labels.reshape(-1) != -100
        total_loss += float(token_loss[mask].sum())
        total_tokens += float(mask.sum())
        batches += 1
        if max_batches and batches >= max_batches:
            break
    model.train()
    return total_loss / max(total_tokens, 1.0)


def train_planner(
    cfg: dict,
    resume: str | None = None,
    tokenizer=None,
    dataset_cls=ScoreDataset,
) -> Planner:
    set_seed(cfg.get("seed", 0))
    device = get_device()
    tok = tokenizer or ScoreTokenizer()
    dataset = dataset_cls(cfg["data"], tok, cfg.get("max_len", 1024))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("workers", 0),
        collate_fn=lambda batch: collate_scores(batch, tok.pad_id),
        drop_last=True,
    )
    val_loader = None
    if cfg.get("val_data"):
        val_dataset = dataset_cls(cfg["val_data"], tok, cfg.get("max_len", 1024))
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=cfg.get("workers", 0),
            collate_fn=lambda batch: collate_scores(batch, tok.pad_id),
        )

    model = Planner(tok.vocab_size, **cfg.get("model", {})).to(device)
    if cfg.get("init_from") and not resume:
        # Warm start from an earlier planner. Rows for tokens it did not have
        # (a grown vocabulary) keep their fresh init; the head is tied to them.
        src = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)["model"]
        own = model.state_dict()
        for k, v in src.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k in own and k in ("tok_emb.weight", "head.weight") and own[k].shape[1:] == v.shape[1:]:
                own[k][: v.shape[0]] = v
        model.load_state_dict(own)
        print(f"init from {cfg['init_from']}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )
    total = cfg.get("epochs", 1) * max(len(loader), 1)
    if cfg.get("max_steps"):
        total = min(total, cfg["max_steps"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out = Path(cfg.get("out", "runs/planner"))
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "log.csv")

    step = 0
    best_val = float("inf")
    if resume:
        step, _ = load_checkpoint(resume, model, optimizer, map_location=device)

    for epoch in range(cfg.get("epochs", 1)):
        model.train()
        for batch in loader:
            lr = cosine_lr(step, total, cfg.get("lr", 3e-4), cfg.get("warmup", 100))
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(ids, labels, weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            val_loss = ""
            if val_loader is not None and step % cfg.get("eval_every", 500) == 0:
                val_loss = evaluate(model, val_loader, device, cfg.get("eval_batches", 50))
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(out / "best.pt", model, optimizer, step, cfg)

            if step % cfg.get("log_every", 10) == 0:
                value = float(loss.detach())
                logger.log(step=step, epoch=epoch, loss=value, val_loss=val_loss, lr=lr)
                suffix = f"  val {val_loss:.4f}" if val_loss != "" else ""
                print(f"step {step:6d}  loss {value:.4f}{suffix}  lr {lr:.2e}")
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
