"""Training-loop tests: the validation metric must match the training objective."""

from __future__ import annotations

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
