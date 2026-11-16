import torch
import torch.nn.functional as F

from orb.model.detail import DetailUNet
from orb.model.discriminator import MultiScaleDiscriminator
from orb.model.planner import Planner
from orb.model.reward import RewardModel


def test_planner_forward_and_loss():
    model = Planner(vocab_size=128, d=64, layers=2, heads=4, ctx=64)
    ids = torch.randint(0, 128, (2, 32))
    targets = ids.clone()
    targets[:, :8] = -100
    logits, loss = model(ids, targets)
    assert logits.shape == (2, 32, 128)
    assert loss is not None and torch.isfinite(loss)


def test_planner_loss_uses_next_token():
    model = Planner(vocab_size=128, d=64, layers=2, heads=4, ctx=64)
    ids = torch.randint(0, 128, (2, 16))
    targets = ids.clone()
    logits, loss = model(ids, targets)
    manual = F.cross_entropy(
        logits[:, :-1].reshape(-1, 128),
        targets[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(loss, manual, atol=1e-6)


def test_planner_generate():
    model = Planner(vocab_size=128, d=64, layers=2, heads=4, ctx=64)
    model.eval()
    out = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4, temperature=1.0, top_p=1.0)
    assert out.shape == (1, 7)


def test_detail_forward_shape():
    model = DetailUNet(cond_dim=8, base=8, levels=2)
    x = torch.randn(2, 2, 64, 32)
    cond = torch.randn(2, 8)
    y = model(x, cond)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_reward_forward_shape():
    model = RewardModel(vocab_size=128, d=32, text_layers=1, text_heads=4, ctx=64)
    mel = torch.randn(2, 1, 64, 32)
    text = torch.randint(0, 128, (2, 16))
    out = model(mel, text)
    assert out["match_audio"].shape == (2, 128)
    assert out["realness"].shape == (2,)
    assert out["sync"].shape == (2,)


def test_discriminator_scales():
    disc = MultiScaleDiscriminator(in_ch=2, base=8, scales=2)
    x = torch.randn(2, 2, 64, 32)
    out = disc(x)
    assert len(out) == 2
    assert all(torch.isfinite(o).all() for o in out)
