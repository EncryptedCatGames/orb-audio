"""Planner: a small decoder-only transformer over the score grammar."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Planner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d: int = 512,
        layers: int = 12,
        heads: int = 8,
        ctx: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        if t > self.ctx:
            raise ValueError(f"sequence length {t} exceeds context {self.ctx}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Causal LM: the logits at position t predict the token at t + 1.
            shift_logits = logits[:, :-1, :]
            shift_targets = targets[:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            if weights is not None:
                w = weights[:, 1:].reshape(-1)
                loss = (token_loss * w).sum() / (w.sum() + 1e-9)
            else:
                loss = token_loss.mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        mask_fn=None,
        grammar=None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy/sampled generation for a single sequence (batch size 1)."""
        if idx.shape[0] != 1:
            raise ValueError("generate() supports batch size 1")
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx :])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if grammar is not None:
                allowed = grammar.allowed()
                mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[torch.tensor(sorted(allowed), device=logits.device)] = True
                logits = logits.masked_fill(~mask, float("-inf"))
            elif mask_fn is not None:
                allowed = mask_fn()
                mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[torch.tensor(sorted(allowed), device=logits.device)] = True
                logits = logits.masked_fill(~mask, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_probs[remove] = 0.0
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            if grammar is not None:
                grammar.advance(int(next_tok.item()))
            if eos_id is not None and int(next_tok.item()) == eos_id:
                break
        return idx
