#!/usr/bin/env python3
"""No-noise gate for the detail model.

The detail stage must never add audible noise or tonal artifacts to a DSP
render. This script measures exactly that, on fixed, deterministic inputs:

    python scripts/eval_detail_artifacts.py --checkpoint ''                 # identity
    python scripts/eval_detail_artifacts.py --checkpoint weights/detail.pt  # released (v6)
    python scripts/eval_detail_artifacts.py --checkpoint runs/detail_v6/last.pt \
        --json-out runs/detail_v6/gate.json --png-dir runs/detail_v6/gate_png

Inputs: 32 fixed Specs rendered with ``orb.api.render_spec`` (all 8 textures,
all 16 roles, quiet sparse cues and dense trailer cues), processed exactly like
production (stereo, ``apply_detail(audio, model, seed=0)``), plus the held-out
degradation pairs in ``data/degrade/eval.jsonl``.

Gates (every DSP cue, both channels, worst case):

G1 tonal artifacts
   a. Prominence: the time-averaged spectrum in dB minus its 1/3-octave
      running median. ``max_f (prom_out - prom_in)`` over 100 Hz - 20 kHz,
      in bins where the output is within 60 dB of its own peak, must be
      < 6 dB: no new narrowband lines.
   b. Periodicity: comb score of the detrended dB spectrum at the stride
      lags (4, 8, 16 bins; 4 bins = 187.5 Hz = sr / hop, the one-hop echo of
      frame-mixing convolutions): ``ac(L) - (ac(L-2) + ac(L+2)) / 2``. The
      output's must not exceed the input's (floored at 0) by more than 0.20,
      and the residual's (in bins where it is within 30 dB of the input) by
      more than 0.30.
   G1a is also measured on the eval.jsonl pairs (first clip of every batch)
   and must stay < 6 dB there too.
G2 high-frequency noise: output/input energy above 10 kHz and above 12 kHz,
   each <= 1.25x (both energies are floored at -60 dB re the cue's total).
G3 quiet regions: in frames >= 30 dB below the input's active (99th
   percentile) frame level, the output's level rises by <= 0.5 dB (floored at
   -70 dB re the active level).
G4 usefulness, on eval.jsonl: MR-STFT improvement over identity >= 0 and
   onset F1 no more than 0.02 below identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orb.api import render_spec  # noqa: E402
from orb.io import write_wav  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.render.detail import apply_detail  # noqa: E402

SR = 48000
N_FFT = 1024
HOP = 256
FREQS = np.arange(N_FFT // 2 + 1) * SR / N_FFT
STRIDE_LAGS = (4, 8, 16)

G1_PROM_DB = 6.0
G1_PERIOD_OUT = 0.20
G1_PERIOD_RES = 0.30
G2_RATIO = 1.25
G3_DB = 0.5
G4_ONSET_DROP = 0.02


def _ev(role, t, dur, gain_db, pan=0.0, seed=7):
    return {"role": role, "t": t, "dur": dur, "gain_db": gain_db, "pan": pan, "seed": seed}


# Fixed cues. Names are stable so reports can be diffed across checkpoints.
CUES: list[tuple[str, dict]] = [
    ("quiet_ui_pop_room_cold", {"duration": 3.0, "texture": "room_cold", "mood": "calm",
                                "events": [_ev("ui_pop", 1.0, 0.16, -14.0)]}),
    ("quiet_click_vinyl", {"duration": 3.0, "texture": "vinyl", "mood": "neutral",
                           "events": [_ev("click", 0.8, 0.10, -16.0, 0.25)]}),
    ("bed_room_cold", {"duration": 4.0, "texture": "room_cold", "mood": "calm", "events": []}),
    ("bed_wind", {"duration": 4.0, "texture": "wind", "mood": "eerie", "events": []}),
    ("bed_city_hum", {"duration": 4.0, "texture": "city_hum", "mood": "neutral", "events": []}),
    ("bed_vinyl", {"duration": 4.0, "texture": "vinyl", "mood": "warm", "events": []}),
    ("bed_drone", {"duration": 4.0, "texture": "drone", "mood": "dark", "events": []}),
    ("bed_rain", {"duration": 4.0, "texture": "rain", "mood": "calm", "events": []}),
    ("bed_fire", {"duration": 4.0, "texture": "fire", "mood": "warm", "events": []}),
    ("bed_sci_fi", {"duration": 4.0, "texture": "sci_fi", "mood": "tense", "events": []}),
    ("whoosh_wind", {"duration": 3.0, "texture": "wind", "mood": "bright",
                     "events": [_ev("whoosh", 0.8, 0.8, -8.0, -0.5)]}),
    ("impact_drone", {"duration": 3.0, "texture": "drone", "mood": "intense",
                      "events": [_ev("impact", 0.5, 0.7, -6.0)]}),
    ("riser_sci_fi", {"duration": 4.0, "texture": "sci_fi", "mood": "tense",
                      "events": [_ev("riser", 0.5, 2.0, -18.0)]}),
    ("swell_rain", {"duration": 4.0, "texture": "rain", "mood": "hopeful",
                    "events": [_ev("swell", 0.5, 1.5, -16.0, 0.5)]}),
    ("bell_room_cold", {"duration": 3.0, "texture": "room_cold", "mood": "bright",
                        "events": [_ev("bell", 0.5, 1.4, -14.0)]}),
    ("sub_drop_city_hum", {"duration": 3.0, "texture": "city_hum", "mood": "dark",
                           "events": [_ev("sub_drop", 0.5, 1.0, -12.0)]}),
    ("glitch_fire", {"duration": 3.0, "texture": "fire", "mood": "playful",
                     "events": [_ev("glitch", 1.0, 0.2, -16.0, -0.25)]}),
    ("noise_sweep_vinyl", {"duration": 4.0, "texture": "vinyl", "mood": "eerie",
                           "events": [_ev("noise_sweep", 0.5, 1.5, -18.0, 0.75)]}),
    ("ui_sequence_room_cold", {"duration": 4.0, "texture": "room_cold", "mood": "playful",
                               "events": [_ev("ui_pop", 0.5, 0.16, -14.0, -0.25, 3),
                                          _ev("click", 1.2, 0.1, -16.0, 0.25, 9),
                                          _ev("ui_pop", 2.0, 0.16, -12.0, 0.0, 41),
                                          _ev("bell", 2.8, 1.0, -16.0, 0.0, 12)]}),
    ("trailer_dense_sci_fi", {"duration": 8.0, "texture": "sci_fi", "mood": "intense",
                              "events": [_ev("riser", 0.0, 2.0, -18.0, 0.0, 1),
                                         _ev("whoosh", 1.8, 0.8, -8.0, -0.75, 2),
                                         _ev("impact", 2.0, 0.7, -6.0, 0.0, 3),
                                         _ev("sub_drop", 2.0, 1.0, -12.0, 0.0, 4),
                                         _ev("glitch", 3.5, 0.2, -16.0, 0.5, 5),
                                         _ev("noise_sweep", 4.0, 1.5, -18.0, -0.5, 6),
                                         _ev("swell", 5.5, 1.5, -18.0, 0.25, 7),
                                         _ev("impact", 7.0, 0.7, -6.0, 0.0, 8)]}),
    ("trailer_dense_drone", {"duration": 10.0, "texture": "drone", "mood": "dark",
                             "events": [_ev("swell", 0.0, 1.5, -18.0, 0.0, 11),
                                        _ev("riser", 1.0, 2.0, -20.0, 0.0, 12),
                                        _ev("impact", 3.0, 0.7, -6.0, 0.0, 13),
                                        _ev("bell", 3.2, 1.4, -14.0, 0.5, 14),
                                        _ev("whoosh", 5.0, 0.8, -8.0, 0.75, 15),
                                        _ev("sub_drop", 6.0, 1.0, -12.0, 0.0, 16),
                                        _ev("click", 7.5, 0.1, -16.0, -0.5, 17),
                                        _ev("impact", 8.5, 0.7, -6.0, 0.0, 18)]}),
    ("trailer_rain_storm", {"duration": 8.0, "texture": "rain", "mood": "intense",
                            "events": [_ev("noise_sweep", 0.0, 1.5, -18.0, -0.5, 21),
                                       _ev("impact", 1.5, 0.7, -6.0, 0.0, 22),
                                       _ev("whoosh", 3.0, 0.8, -8.0, 0.5, 23),
                                       _ev("impact", 4.0, 0.7, -8.0, 0.0, 24),
                                       _ev("sub_drop", 5.0, 1.0, -12.0, 0.0, 25),
                                       _ev("swell", 6.0, 1.5, -18.0, 0.0, 26)]}),
    ("bells_fire", {"duration": 5.0, "texture": "fire", "mood": "hopeful",
                    "events": [_ev("bell", 0.5, 1.4, -14.0, -0.5, 31),
                               _ev("bell", 2.0, 1.4, -14.0, 0.5, 32),
                               _ev("swell", 3.0, 1.5, -18.0, 0.0, 33)]}),
    ("glitch_city_hum", {"duration": 4.0, "texture": "city_hum", "mood": "tense",
                         "events": [_ev("glitch", 0.5, 0.2, -16.0, -0.5, 41),
                                    _ev("glitch", 1.5, 0.2, -16.0, 0.5, 42),
                                    _ev("click", 2.5, 0.1, -16.0, 0.0, 43),
                                    _ev("glitch", 3.0, 0.2, -14.0, 0.0, 44)]}),
    # natural and foley roles (v0.2)
    ("thunder_close_rain", {"duration": 8.0, "texture": "rain", "mood": "dark",
                            "events": [_ev("thunder", 0.5, 6.0, -4.0, 0.25, 0x00)]}),
    ("thunder_far_wind", {"duration": 9.0, "texture": "wind", "mood": "dark",
                          "events": [_ev("thunder", 0.5, 7.0, -8.0, -0.5, 0x03)]}),
    ("door_slam_room_cold", {"duration": 3.0, "texture": "room_cold", "mood": "tense",
                             "events": [_ev("door_slam", 0.5, 0.9, -6.0, 0.0, 0x01)]}),
    ("knock_room_cold", {"duration": 3.0, "texture": "room_cold", "mood": "tense",
                         "events": [_ev("knock", 0.5, 0.7, -10.0, 0.25, 0x03)]}),
    ("footsteps_city_hum", {"duration": 5.0, "texture": "city_hum", "mood": "neutral",
                            "events": [_ev("footsteps", 0.5, 3.0, -12.0, -0.25, 0x04)]}),
    ("footsteps_gravel_wind", {"duration": 5.0, "texture": "wind", "mood": "calm",
                               "events": [_ev("footsteps", 0.5, 3.0, -12.0, 0.25, 0x02)]}),
    ("water_drops_drone", {"duration": 4.0, "texture": "drone", "mood": "eerie",
                           "events": [_ev("water_drop", 0.5, 0.35, -14.0, -0.5, 0x02),
                                      _ev("water_drop", 1.6, 0.35, -16.0, 0.5, 0x05),
                                      _ev("water_drop", 2.7, 0.35, -12.0, 0.0, 0x08)]}),
    ("explosion_fire", {"duration": 5.0, "texture": "fire", "mood": "intense",
                        "events": [_ev("explosion", 0.5, 3.0, -4.0, 0.0, 0x00)]}),
]
PNG_CUES = ("quiet_ui_pop_room_cold", "bed_room_cold", "bed_rain", "trailer_dense_sci_fi")


# -- spectral helpers --------------------------------------------------------


def _stft_power(x: np.ndarray) -> np.ndarray:
    """(n,) -> (F, T) power."""
    spec = torch.stft(
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        n_fft=N_FFT,
        hop_length=HOP,
        window=torch.hann_window(N_FFT),
        return_complex=True,
        center=True,
    )
    return (spec.abs() ** 2).numpy().astype(np.float64)


def _third_octave_median(db: np.ndarray, min_width: int = 5) -> np.ndarray:
    out = np.empty_like(db)
    n = len(db)
    for k in range(n):
        lo = int(np.floor(k * 2 ** (-1 / 6)))
        hi = int(np.ceil(k * 2 ** (1 / 6)))
        if hi - lo + 1 < min_width:
            lo, hi = k - min_width // 2, k + min_width // 2
        lo, hi = max(lo, 0), min(hi, n - 1)
        out[k] = np.median(db[lo : hi + 1])
    return out


def _prominence(avg_power: np.ndarray, floor: float) -> np.ndarray:
    db = 10.0 * np.log10(avg_power + floor)
    return db - _third_octave_median(db)


def _comb_score(db: np.ndarray, lags=STRIDE_LAGS) -> dict[int, float]:
    """Comb score of a dB spectrum at each lag (in bins).

    The spectrum is detrended (minus a 9-bin moving average) and
    autocorrelated; the score at lag L is ``ac(L) - (ac(L-2) + ac(L+2)) / 2``,
    i.e. how much L stands out from its neighbours. A comb with period L
    scores ~1 (period 4: ~2); a smooth or white spectrum scores ~0. Taking
    the plain ``ac(L)`` instead flags any spectrum smoother than the input.
    """
    if len(db) < 4 * (max(lags) + 2):
        return {lag: 0.0 for lag in lags}
    trend = np.convolve(db, np.ones(9) / 9.0, mode="same")
    d = (db - trend)[5:-5]
    d = d - d.mean()
    denom = float(np.dot(d, d))
    if denom <= 1e-12:
        return {lag: 0.0 for lag in lags}

    def ac(lag: int) -> float:
        return float(np.dot(d[:-lag], d[lag:]) / denom)

    return {lag: ac(lag) - 0.5 * (ac(lag - 2) + ac(lag + 2)) for lag in lags}


def channel_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """All G1-G3 metrics for one channel: input x, output y."""
    px = _stft_power(x)
    py = _stft_power(y)
    pr = _stft_power(y - x)
    ax, ay, ar = px.mean(axis=1), py.mean(axis=1), pr.mean(axis=1)
    peak = max(float(ax.max()), 1e-20)
    floor = peak * 1e-10  # -100 dB re the input peak bin

    # G1a: new narrowband lines in the output.
    prom_x = _prominence(ax, floor)
    prom_y = _prominence(ay, floor)
    band = (FREQS >= 100.0) & (FREQS <= 20000.0)
    audible = ay >= float(ay.max()) * 1e-6  # within 60 dB of the output's own peak
    sel = band & audible
    prom_delta = float(np.max((prom_y - prom_x)[sel])) if sel.any() else 0.0
    prom_delta_freq = float(FREQS[sel][np.argmax((prom_y - prom_x)[sel])]) if sel.any() else 0.0

    # G1b: stride periodicity, from 2 kHz up (where the residual lives).
    hb = FREQS >= 2000.0
    c_x = _comb_score(10 * np.log10(ax[hb] + floor))
    c_y = _comb_score(10 * np.log10(ay[hb] + floor))
    period_out = max(c_y[lag] - max(c_x[lag], 0.0) for lag in STRIDE_LAGS)
    relevant = hb & (ar >= ax * 1e-3)  # residual within 30 dB of the input
    if relevant.sum() >= 64:
        idx = np.where(relevant)[0]
        lo, hi = idx.min(), idx.max() + 1
        c_r = _comb_score(10 * np.log10(ar[lo:hi] + floor))
        c_xr = _comb_score(10 * np.log10(ax[lo:hi] + floor))
        # A residual g * X inherits a comb the input already has; only a
        # comb the input does not have counts.
        period_res = max(c_r[lag] - max(c_xr[lag], 0.0) for lag in STRIDE_LAGS)
    else:
        period_res = 0.0

    # G2: energy above 10 / 12 kHz, floored at -60 dB re the total.
    total = float(px.sum()) + 1e-20
    e_floor = total * 1e-6

    def hf_ratio(cut: float) -> float:
        m = FREQS >= cut
        return float((py[m].sum() + e_floor) / (px[m].sum() + e_floor))

    # G3: quiet frames.
    fx = px.sum(axis=0)
    fy = py.sum(axis=0)
    fx_db = 10 * np.log10(fx + 1e-20)
    active = float(np.percentile(fx_db, 99))
    quiet = fx_db <= active - 30.0
    q_floor = 10 ** (active / 10.0) * 1e-7
    if quiet.any():
        quiet_rise = float(10 * np.log10((fy[quiet].sum() + q_floor * quiet.sum()) / (fx[quiet].sum() + q_floor * quiet.sum())))
    else:
        quiet_rise = 0.0

    return {
        "g1_prom_delta_db": prom_delta,
        "g1_prom_delta_hz": prom_delta_freq,
        "g1_period_out": float(period_out),
        "g1_period_res": float(period_res),
        "g2_hf10_ratio": hf_ratio(10000.0),
        "g2_hf12_ratio": hf_ratio(12000.0),
        "g3_quiet_rise_db": quiet_rise,
        "g3_quiet_frames": int(quiet.sum()),
        "residual_rel_db": float(10 * np.log10((pr.sum() + 1e-20) / total)),
    }


# -- model -------------------------------------------------------------------


def load_model(path: str, device: torch.device, override: dict | None = None) -> DetailUNet | None:
    """``override`` changes parameter-free inference options (e.g.
    ``quiet_gate_db``) of a checkpoint's model config."""
    if not path:
        return None
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = DetailUNet(**{**ckpt["config"]["model"], **(override or {})})
    model.load_state_dict(ckpt["generator"])
    return model.to(device).eval()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _spectrogram_png(wav: Path, png: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
            "-lavfi", "showspectrumpic=s=1024x512:legend=1:color=intensity:scale=log:fscale=lin:stop=24000",
            str(png),
        ],
        check=True,
    )


# -- G4 ----------------------------------------------------------------------


def restoration_metrics(model: DetailUNet | None, data: str, device: torch.device, max_batches: int) -> dict:
    from torch.utils.data import DataLoader

    from eval_detail import _onset_f1
    from orb.data.pair_dataset import PairDataset, collate_pairs
    from orb.train.losses import istft, multi_res_stft_loss

    ds = PairDataset(ROOT / data, n_frames=256, n_fft=1024, hop=256, train=False, crop_mode="onset", normalize_stft=True)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_pairs)
    tot_id = tot_m = 0.0
    onset_id = onset_m = 0.0
    n = clips = 0
    prom: list[float] = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["dsp"].to(device)
            y = batch["real"].to(device)
            cond = batch["cond"].to(device)
            if model is not None:
                noise = None
                if model.noise_channels:
                    noise = torch.randn(x.shape[0], model.noise_channels, x.shape[2], x.shape[3], generator=gen).to(device)
                fake = x + model(x, cond, noise)
            else:
                fake = x
            real_wav = istft(y, 1024, 256).cpu()
            id_wav = istft(x, 1024, 256).cpu()
            fake_wav = istft(fake, 1024, 256).cpu()
            tot_id += float(multi_res_stft_loss(id_wav, real_wav))
            tot_m += float(multi_res_stft_loss(fake_wav, real_wav))
            n += 1
            for j in range(x.shape[0]):
                r = real_wav[j].numpy()
                onset_id += _onset_f1(id_wav[j].numpy(), r, SR)
                onset_m += _onset_f1(fake_wav[j].numpy(), r, SR)
                clips += 1
                if model is not None and j == 0:
                    prom.append(channel_metrics(id_wav[j].numpy(), fake_wav[j].numpy())["g1_prom_delta_db"])
            if max_batches and i + 1 >= max_batches:
                break
    return {
        "eval_clips": clips,
        "identity_mr_stft": tot_id / max(n, 1),
        "model_mr_stft": tot_m / max(n, 1),
        "mr_stft_improvement": 1.0 - tot_m / max(tot_id, 1e-9),
        "identity_onset_f1": onset_id / max(clips, 1),
        "onset_f1": onset_m / max(clips, 1),
        "eval_g1_prom_delta_db_max": max(prom) if prom else 0.0,
        "eval_g1_prom_delta_db_median": float(np.median(prom)) if prom else 0.0,
    }


# -- main --------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="", help="detail checkpoint; '' = identity")
    ap.add_argument("--data", default="data/degrade/eval.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-batches", type=int, default=0, help="G4 batches (0 = all 400 clips)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--png-dir", default=None, help="write input/output spectrogram PNGs here")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--model-override", default=None,
                    help='JSON of inference-only model options, e.g. \'{"quiet_gate_db": 30}\'')
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    model = load_model(args.checkpoint, device, json.loads(args.model_override) if args.model_override else None)

    png_dir = Path(args.png_dir) if args.png_dir else None
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)

    per_cue: dict[str, dict] = {}
    for name, spec in CUES:
        x = render_spec(spec, seed=0)
        y = apply_detail(x, model, seed=0) if model is not None else x.copy()
        chans = [channel_metrics(x[:, c], y[:, c]) for c in range(x.shape[1])]
        worst = {}
        for key in chans[0]:
            vals = [c[key] for c in chans]
            worst[key] = vals[0] if key == "g1_prom_delta_hz" else max(vals)
        if worst["g1_prom_delta_db"] != chans[0]["g1_prom_delta_db"]:
            worst["g1_prom_delta_hz"] = chans[1]["g1_prom_delta_hz"]
        per_cue[name] = worst
        if png_dir and name in PNG_CUES:
            write_wav(png_dir / f"{name}_in.wav", x, SR)
            write_wav(png_dir / f"{name}_out.wav", y, SR)
            _spectrogram_png(png_dir / f"{name}_in.wav", png_dir / f"{name}_in.png")
            _spectrogram_png(png_dir / f"{name}_out.wav", png_dir / f"{name}_out.png")
        if not args.quiet:
            m = worst
            print(
                f"{name:26s} prom {m['g1_prom_delta_db']:5.2f} dB @{m['g1_prom_delta_hz']:6.0f}  "
                f"per_out {m['g1_period_out']:5.2f} per_res {m['g1_period_res']:5.2f}  "
                f"hf10 {m['g2_hf10_ratio']:6.2f} hf12 {m['g2_hf12_ratio']:6.2f}  "
                f"quiet {m['g3_quiet_rise_db']:+5.2f} dB ({m['g3_quiet_frames']})  res {m['residual_rel_db']:6.1f} dB",
                flush=True,
            )

    def worst(key: str) -> float:
        return max(m[key] for m in per_cue.values())

    g4 = restoration_metrics(model, args.data, device, args.max_batches)
    summary = {
        "checkpoint": args.checkpoint,
        "model_override": args.model_override,
        "g1_prom_delta_db": worst("g1_prom_delta_db"),
        "g1_period_out": worst("g1_period_out"),
        "g1_period_res": worst("g1_period_res"),
        "g2_hf10_ratio": worst("g2_hf10_ratio"),
        "g2_hf12_ratio": worst("g2_hf12_ratio"),
        "g3_quiet_rise_db": worst("g3_quiet_rise_db"),
        **g4,
    }
    checks = {
        "G1_tonal": summary["g1_prom_delta_db"] < G1_PROM_DB
        and summary["g1_period_out"] <= G1_PERIOD_OUT
        and summary["g1_period_res"] <= G1_PERIOD_RES
        and summary["eval_g1_prom_delta_db_max"] < G1_PROM_DB,
        "G2_hf_noise": summary["g2_hf10_ratio"] <= G2_RATIO and summary["g2_hf12_ratio"] <= G2_RATIO,
        "G3_quiet": summary["g3_quiet_rise_db"] <= G3_DB,
        "G4_useful": summary["mr_stft_improvement"] >= 0.0
        and summary["onset_f1"] >= summary["identity_onset_f1"] - G4_ONSET_DROP,
    }
    summary["checks"] = checks
    summary["pass"] = all(checks.values())
    summary["per_cue"] = per_cue

    print()
    print(f"checkpoint: {args.checkpoint or '(identity)'}")
    print(f"G1 prominence  {summary['g1_prom_delta_db']:.2f} dB (< {G1_PROM_DB})   "
          f"periodicity out {summary['g1_period_out']:.3f} (<= {G1_PERIOD_OUT})  res {summary['g1_period_res']:.3f} (<= {G1_PERIOD_RES})"
          f"   eval-pairs prom max {summary['eval_g1_prom_delta_db_max']:.2f} median {summary['eval_g1_prom_delta_db_median']:.2f}")
    print(f"G2 HF >10k {summary['g2_hf10_ratio']:.3f}x  >12k {summary['g2_hf12_ratio']:.3f}x (<= {G2_RATIO})")
    print(f"G3 quiet-frame rise {summary['g3_quiet_rise_db']:+.3f} dB (<= {G3_DB})")
    print(f"G4 MR-STFT improvement {100 * summary['mr_stft_improvement']:+.2f}%  onset F1 {summary['onset_f1']:.3f} "
          f"(identity {summary['identity_onset_f1']:.3f})")
    print("checks:", " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()))
    print("GATE:", "PASS" if summary["pass"] else "FAIL")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
