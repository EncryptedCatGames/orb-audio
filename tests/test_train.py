"""Training-loop tests: the validation metric must match the training objective."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # these tests need the torch install (pip install -e ".[train]")

import torch
import torch.nn.functional as F

from orb.model.planner import Planner
from orb.train.planner import evaluate


class _Loader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


def test_evaluate_uses_next_token_targets():
    torch.manual_seed(0)
    model = Planner(vocab_size=64, d=32, layers=2, heads=4, ctx=32)
    ids = torch.randint(0, 64, (2, 12))
    labels = ids.clone()
    labels[:, :4] = -100
    batch = {"input_ids": ids, "labels": labels}

    got = evaluate(model, _Loader([batch]), torch.device("cpu"), max_batches=1)

    with torch.no_grad():
        logits, _ = model(ids)
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    expected = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    assert abs(got - float(expected)) < 1e-5


def test_overshoot_loss_is_one_sided():
    from orb.train.losses import overshoot_loss

    torch.manual_seed(0)
    y = torch.randn(2, 2, 64, 32)
    assert float(overshoot_loss(y, y)) == 0.0
    assert float(overshoot_loss(0.5 * y, y)) == 0.0  # too quiet is not penalised here
    assert float(overshoot_loss(2.0 * y, y)) > 0.0
    # Energy painted into a silent region costs more than the same excess on signal.
    quiet = y.clone()
    quiet[:, :, 32:] *= 1e-3
    noisy = quiet.clone()
    noisy[:, :, 32:] += 0.1 * torch.randn_like(noisy[:, :, 32:])
    assert float(overshoot_loss(noisy, quiet)) > float(overshoot_loss(2.0 * y, y))


def test_residual_idempotence_loss():
    from orb.train.losses import residual_idempotence_loss

    x = torch.randn(2, 2, 16, 8)
    assert float(residual_idempotence_loss(torch.zeros_like(x), x)) == 0.0
    assert abs(float(residual_idempotence_loss(0.1 * x, x)) - 0.01) < 1e-4
