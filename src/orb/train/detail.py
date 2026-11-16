"""Detail-model training: reconstruction + adversarial."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.pair_dataset import PairDataset, collate_pairs
from ..model.detail import DetailUNet
from ..model.discriminator import MultiScaleDiscriminator
from .common import CSVLogger, get_device, set_seed
from .losses import (
    band_energy_loss,
    d_loss,
    d_loss_hinge,
    envelope_loss,
    feature_matching_loss,
    g_adv_loss,
    g_adv_loss_hinge,
    istft,
    log_mag_loss,
    multi_res_stft_loss,
    relative_mag_loss,
)


def _noise_like(x: torch.Tensor, channels: int) -> torch.Tensor | None:
    if not channels:
        return None
    return torch.randn(x.shape[0], channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)


def _save_detail(path, generator, discriminator, opt_g, opt_d, step, cfg) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "step": step,
            "config": cfg,
        },
        path,
    )


def train_detail(cfg: dict, resume: str | None = None) -> DetailUNet:
    set_seed(cfg.get("seed", 0))
    device = get_device()
    dataset = PairDataset(
        cfg["data"],
        n_frames=cfg.get("n_frames", 256),
        n_fft=cfg.get("n_fft", 1024),
        hop=cfg.get("hop", 256),
        train=True,
        crop_mode=cfg.get("crop_mode", "onset"),
        normalize_stft=cfg.get("normalize_stft", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("workers", 0),
        collate_fn=collate_pairs,
        drop_last=True,
    )
    generator = DetailUNet(**cfg.get("model", {})).to(device)
    discriminator = MultiScaleDiscriminator(**cfg.get("disc", {})).to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.get("lr_g", 1e-4), betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.get("lr_d", 1e-4), betas=(0.5, 0.999))
    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(cfg.get("out", "runs/detail"))
    out.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(out / "log.csv")
    step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt.get("step", 0)
        # A resume may deliberately change the schedule.
        for group in opt_g.param_groups:
            group["lr"] = cfg.get("lr_g", group["lr"])
        for group in opt_d.param_groups:
            group["lr"] = cfg.get("lr_d", group["lr"])

    use_disc = cfg.get("adv_weight", 0.1) > 0.0
    warmup = int(cfg.get("warmup_steps", 0))
    ramp_steps = max(1, int(cfg.get("ramp_steps", 0)))
    gan_loss = cfg.get("gan_loss", "bce")
    min_bin = int(cfg.get("disc_min_bin", 0))

    def disc_in(z: torch.Tensor) -> torch.Tensor:
        """Optional high-pass: the discriminator only judges the band the
        degradation destroyed, so it cannot win on the easy low band."""
        return z[:, :, min_bin:] if min_bin else z

    for epoch in range(cfg.get("epochs", 1)):
        generator.train()
        discriminator.train()
        for batch in loader:
            x = batch["dsp"].to(device)
            y = batch["real"].to(device)
            cond = batch["cond"].to(device)
            disc_on = use_disc and step >= warmup

            loss_d = torch.zeros((), device=device)
            if disc_on:
                # Discriminator step
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    fake = x + generator(x, cond, _noise_like(x, generator.noise_channels))
                    loss_d = (
                        d_loss_hinge(discriminator, disc_in(y), disc_in(fake))
                        if gan_loss == "hinge"
                        else d_loss(discriminator, disc_in(y), disc_in(fake))
                    )
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

            # Generator step
            opt_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                fake = x + generator(x, cond, _noise_like(x, generator.noise_channels))
                fake_wav = istft(fake, cfg.get("n_fft", 1024), cfg.get("hop", 256))
                real_wav = istft(y, cfg.get("n_fft", 1024), cfg.get("hop", 256))
                log_weight = cfg.get("log_weight", 0.0)
                if log_weight > 0:
                    loss_mag = log_mag_loss(fake, y)
                    mag_weight = log_weight
                else:
                    loss_mag = relative_mag_loss(fake, y)
                    mag_weight = cfg.get("rel_weight", 0.0)
                loss_rec = multi_res_stft_loss(fake_wav, real_wav)
                loss_env = envelope_loss(fake_wav, real_wav, cfg.get("hop", 256))
                loss_band = band_energy_loss(
                    fake,
                    y,
                    band_weights=cfg.get("band_weights"),
                    sr=cfg.get("sample_rate", 48000),
                    n_fft=cfg.get("n_fft", 1024),
                )
                if disc_on:
                    loss_adv = (
                        g_adv_loss_hinge(discriminator, disc_in(fake))
                        if gan_loss == "hinge"
                        else g_adv_loss(discriminator, disc_in(fake))
                    )
                    ramp = min(1.0, max(0.0, (step - warmup) / ramp_steps))
                    adv_eff = cfg.get("adv_weight", 0.1) * ramp
                    fm_weight = cfg.get("fm_weight", 0.0)
                    loss_fm = (
                        feature_matching_loss(discriminator, disc_in(y), disc_in(fake))
                        if fm_weight > 0
                        else torch.zeros((), device=device)
                    )
                else:
                    loss_adv = torch.zeros((), device=device)
                    loss_fm = torch.zeros((), device=device)
                    adv_eff = 0.0
                    fm_weight = 0.0
                loss_g = (
                    mag_weight * loss_mag
                    + cfg.get("rec_weight", 1.0) * loss_rec
                    + cfg.get("env_weight", 0.2) * loss_env
                    + cfg.get("band_weight", 0.0) * loss_band
                    + adv_eff * loss_adv
                    + fm_weight * loss_fm
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

            if step % cfg.get("log_every", 10) == 0:
                logger.log(
                    step=step,
                    epoch=epoch,
                    loss_g=float(loss_g.detach()),
                    loss_d=float(loss_d.detach()),
                    rec=float(loss_rec.detach()),
                    mag=float(loss_mag.detach()),
                    env=float(loss_env.detach()),
                    band=float(loss_band.detach()),
                    adv=float(loss_adv.detach()),
                    fm=float(loss_fm.detach()),
                )
                print(
                    f"step {step:6d}  g {float(loss_g.detach()):.4f}  d {float(loss_d.detach()):.4f} "
                    f"rec {float(loss_rec.detach()):.4f}  mag {float(loss_mag.detach()):.4f} "
                    f"band {float(loss_band.detach()):.4f}  fm {float(loss_fm.detach()):.4f}"
                )
            step += 1
            if step % cfg.get("save_every", 500) == 0:
                _save_detail(out / f"step{step}.pt", generator, discriminator, opt_g, opt_d, step, cfg)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        _save_detail(out / "last.pt", generator, discriminator, opt_g, opt_d, step, cfg)
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break

    logger.close()
    return generator
