import pytest

pytest.importorskip("torch")  # these tests need the torch install (pip install -e ".[train]")

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


def test_detail_resize_conv_shape_and_no_transpose():
    model = DetailUNet(cond_dim=8, base=8, levels=3, upsample="resize_conv")
    assert not any(isinstance(m, torch.nn.ConvTranspose2d) for m in model.modules())
    # Odd frequency size, like the real 513 bins.
    x = torch.randn(2, 2, 65, 32)
    y = model(x, torch.randn(2, 8))
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_detail_default_is_transpose_for_old_checkpoints():
    # The v5 config has no "upsample" key; it must build the same module tree
    # so its state dict loads strictly.
    v5_model_cfg = {
        "cond_dim": 8, "base": 8, "levels": 2, "residual_scale": 8.0, "noise_channels": 1,
        "energy_gate": True, "residual_min_bin": 4, "noise_floor": 0.1,
    }
    old = DetailUNet(**v5_model_cfg)
    assert any(isinstance(m, torch.nn.ConvTranspose2d) for m in old.modules())
    new = DetailUNet(**v5_model_cfg)
    new.load_state_dict(old.state_dict(), strict=True)


def test_detail_checkpoint_loads_if_present():
    import json
    from pathlib import Path

    import pytest

    root = Path(__file__).resolve().parents[1]
    path = root / "weights" / "detail.pt"
    if not path.exists():
        pytest.skip("weights/detail.pt not present")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DetailUNet(**ckpt["config"]["model"])
    model.load_state_dict(ckpt["generator"], strict=True)
    # The release weights must use the stripe-free upsampler (v5's ConvTranspose2d left checkerboard tones).
    assert model.upsample == "resize_conv"
    v6 = json.loads((root / "configs" / "detail_v6.json").read_text())["model"]
    assert DetailUNet(**v6).upsample == "resize_conv"


def test_detail_taper_zeroes_residual_above_cutoff():
    torch.manual_seed(0)
    model = DetailUNet(cond_dim=8, base=8, levels=2, upsample="resize_conv",
                       residual_taper_hz=(8000.0, 9500.0), sample_rate=48000)
    torch.nn.init.normal_(model.out.weight, std=0.1)  # non-zero residual
    x = torch.randn(1, 2, 513, 16)
    r = model(x, torch.zeros(1, 8))
    freqs = torch.linspace(0, 24000, 513)
    assert r[:, :, freqs > 9500.0].abs().max() == 0.0
    assert r[:, :, freqs < 8000.0].abs().max() > 0.0


def test_detail_env_bound_limits_residual():
    torch.manual_seed(0)
    model = DetailUNet(cond_dim=8, base=8, levels=2, upsample="resize_conv",
                       residual_env_bound=0.5, residual_scale=50.0)
    torch.nn.init.normal_(model.out.weight, std=1.0)
    x = torch.randn(1, 2, 64, 16)
    x[:, :, 32:] *= 1e-4  # a near-silent upper half
    r = model(x, torch.zeros(1, 8))
    mag = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)[:, None]
    env = torch.nn.functional.avg_pool2d(mag, (9, 5), stride=1, padding=(4, 2), count_include_pad=False)[:, 0]
    rmag = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
    assert (rmag <= 0.5 * env + 1e-6).all()


def test_detail_freq_bins_crop_pads_zero_residual():
    torch.manual_seed(0)
    model = DetailUNet(cond_dim=8, base=8, levels=2, upsample="resize_conv", freq_bins=33)
    torch.nn.init.normal_(model.out.weight, std=0.1)
    x = torch.randn(2, 2, 65, 16)
    r = model(x, torch.zeros(2, 8))
    assert r.shape == x.shape
    assert r[:, :, 33:].abs().max() == 0.0
    assert r[:, :, :33].abs().max() > 0.0


def test_detail_mask_mode_is_bounded_per_bin_gain():
    torch.manual_seed(0)
    model = DetailUNet(cond_dim=8, base=8, levels=2, upsample="resize_conv", residual_mode="mask",
                       residual_scale=0.5, mask_smooth=(3, 3))
    torch.nn.init.normal_(model.out.weight, std=1.0)
    x = torch.randn(1, 2, 64, 16)
    x[:, :, 40:] = 0.0  # silence stays silent
    r = model(x, torch.zeros(1, 8))
    assert r.shape == x.shape
    assert r[:, :, 40:].abs().max() == 0.0
    # Residual is a real gain times the input: same phase, |g| <= 0.5.
    cross = r[:, 0] * x[:, 1] - r[:, 1] * x[:, 0]
    assert cross.abs().max() < 1e-5
    mag_x = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    mag_r = torch.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
    assert (mag_r <= 0.5 * mag_x + 1e-6).all()


def test_detail_quiet_gate_zeroes_residual_in_quiet_frames():
    import torch
    from orb.model.detail import DetailUNet

    torch.manual_seed(0)
    m = DetailUNet(cond_dim=8, base=8, levels=2, residual_scale=0.5, noise_channels=0,
                   upsample="resize_conv", residual_mode="mask", quiet_gate_db=30.0, quiet_ramp_db=10.0).eval()
    torch.nn.init.normal_(m.out.weight, std=0.1)  # output layer is zero-initialised
    torch.nn.init.normal_(m.out.bias, std=0.1)
    x = torch.randn(1, 2, 513, 32)
    x[..., 16:] *= 1e-3  # second half is 60 dB down
    with torch.no_grad():
        r = m(x, torch.zeros(1, 8))
    assert r[..., 16:].abs().max() == 0
    assert r[..., :16].abs().max() > 0
    # An explicit cue-level reference makes even the loud half count as quiet.
    with torch.no_grad():
        r2 = m(x, torch.zeros(1, 8), frame_ref=torch.tensor([1e6]))
    assert r2.abs().max() == 0
