"""Direct preference optimization for the planner."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.preference_dataset import PreferenceDataset, collate_preferences
from ..model.planner import Planner
from ..model.tokenizer import ScoreTokenizer
from .common import CSVLogger, get_device, load_checkpoint, save_checkpoint, set_seed


def sequence_logprob(model: Planner, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    target = labels[:, 1:]
    mask = target != -100
    target = target.clamp_min(0)
    token_logp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def train_dpo(cfg: dict, resume: str | None = None) -> Planner:
    set_seed(cfg.get("seed", 0))
    device = get_device()
    tok = ScoreTokenizer()
    dataset = PreferenceDataset(cfg["data"], tok, cfg.get("max_len", 1024))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("workers", 0),
        collate_fn=lambda batch: collate_preferences(batch, tok.pad_id),
        drop_last=True,
    )

    policy = Planner(tok.vocab_size, **cfg.get("model", {})).to(device)
    reference = Planner(tok.vocab_size, **cfg.get("model", {})).to(device)
    load_checkpoint(cfg["sft"], policy, map_location=device)
    load_checkpoint(cfg["sft"], reference, map_location=device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.get("lr", 5e-6), weight_decay=0.0)
    beta = cfg.get("beta", 0.1)
    out = Path(cfg.get("out", "runs/dpo"))
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "log.csv")
    step = 0
    if resume:
        step, _ = load_checkpoint(resume, policy, optimizer, map_location=device)

    for epoch in range(cfg.get("epochs", 1)):
        policy.train()
        for batch in loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            chosen_logp = sequence_logprob(policy, chosen_ids, chosen_labels)
            rejected_logp = sequence_logprob(policy, rejected_ids, rejected_labels)
            with torch.no_grad():
                ref_chosen = sequence_logprob(reference, chosen_ids, chosen_labels)
                ref_rejected = sequence_logprob(reference, rejected_ids, rejected_labels)

            margin = (chosen_logp - ref_chosen) - (rejected_logp - ref_rejected)
            loss = -F.logsigmoid(beta * margin).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            if step % cfg.get("log_every", 10) == 0:
                logger.log(
                    step=step,
                    epoch=epoch,
                    loss=float(loss.detach()),
                    margin=float(margin.mean().detach()),
                )
                print(f"step {step:6d}  loss {float(loss.detach()):.4f}  margin {float(margin.mean().detach()):.3f}")
            step += 1
            if step % cfg.get("save_every", 250) == 0:
                save_checkpoint(out / f"step{step}.pt", policy, optimizer, step, cfg)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        save_checkpoint(out / "last.pt", policy, optimizer, step, cfg)
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break

    logger.close()
    return policy
