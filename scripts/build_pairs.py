#!/usr/bin/env python3
"""Build paired (DSP render, real recording) clips for the detail model.

    python scripts/build_pairs.py --input /data/cc0 --out data/pairs --workers 16

The fit here is heuristic: onset, duration, spectral centroid, zero-crossing
rate. It is deliberately simple and replaceable by earshot `fit`. Every pair
is written with a manifest line and a 32-d conditioning vector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.io import read_wav, write_wav  # noqa: E402
from orb.render.events import render_event  # noqa: E402
from orb.score import EVENT_ROLES, Event  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}


def convert(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "48000", "-loglevel", "error", str(dst)],
        check=True,
    )


def find_onset(x: np.ndarray, sr: int) -> int:
    env = np.abs(x)
    k = max(1, int(0.005 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    peak = float(env.max()) if len(env) else 0.0
    if peak <= 0:
        return 0
    idx = np.where(env > 0.05 * peak)[0]
    return int(idx[0]) if len(idx) else 0


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(np.sign(x))) > 0))


def guess_role(duration: float, centroid: float, zcr: float) -> str:
    if duration < 0.12:
        return "click"
    if duration < 0.25 and centroid > 1500:
        return "ui_pop"
    if centroid < 400 and duration < 1.2:
        return "impact"
    if duration > 1.5:
        return "noise_sweep"
    return "whoosh"


def cond_vector(role: str, duration: float, apex_frac: float) -> list[float]:
    v = [0.0] * 32
    v[EVENT_ROLES.index(role)] = 1.0
    v[10] = min(duration / 3.0, 1.0)
    v[11] = apex_frac
    v[12] = 0.9
    v[13] = 0.5
    v[14] = 0.5
    v[15] = 0.75
    return v


def rms_normalize(x: np.ndarray, target: float = 0.1) -> np.ndarray:
    r = float(np.sqrt(np.mean(x**2) + 1e-12))
    return x * (target / max(r, 1e-9))


def spectral_fit(real: np.ndarray, dsp: np.ndarray) -> float:
    """1.0 = identical spectral shape, 0.0 = unrelated. Cheap pair filter."""
    n = min(len(real), len(dsp))
    a = np.abs(np.fft.rfft(real[:n]))
    b = np.abs(np.fft.rfft(dsp[:n]))
    a = a / (np.sum(a) + 1e-9)
    b = b / (np.sum(b) + 1e-9)
    return float(max(0.0, 1.0 - np.sum(np.abs(a - b))))


def process_one(task: tuple[int, str, str, float]) -> dict | None:
    idx, src_str, out_str, max_dur = task
    src = Path(src_str)
    out_dir = Path(out_str)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_wav = Path(tmp) / "in.wav"
            convert(src, tmp_wav)
            audio, sr = read_wav(tmp_wav)
    except Exception as exc:  # noqa: BLE001
        print(f"skip {src}: {exc}", flush=True)
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < int(0.05 * sr):
        return None

    audio = audio / max(float(np.max(np.abs(audio))), 1e-9) * 0.9
    onset = find_onset(audio, sr)
    start = max(0, onset - int(0.05 * sr))
    end = min(len(audio), start + int(max_dur * sr))
    real = audio[start:end]
    if len(real) < int(0.06 * sr):
        return None

    duration = len(real) / sr
    centroid = spectral_centroid(real, sr)
    zcr = zero_crossing_rate(real)
    role = guess_role(duration, centroid, zcr)
    event_dur = max(0.02, duration - 0.05)
    apex = 0.05 + 0.5 * event_dur
    event = Event(0.05, role, event_dur, "fitted", apex=apex, gain_db=0.0)
    dsp = render_event(event, sr, seed=0)
    dsp = np.pad(dsp, ((int(0.05 * sr), 0), (0, 0)))
    if len(dsp) < len(real):
        dsp = np.pad(dsp, ((0, len(real) - len(dsp)), (0, 0)))
    dsp = dsp[: len(real)]

    real = rms_normalize(real)
    dsp = rms_normalize(dsp)

    stem = f"{idx:06d}"
    real_path = out_dir / f"{stem}_real.wav"
    dsp_path = out_dir / f"{stem}_dsp.wav"
    write_wav(real_path, real, sr)
    write_wav(dsp_path, dsp, sr)
    return {
        "dsp": str(dsp_path),
        "real": str(real_path),
        "role": role,
        "duration": duration,
        "fit_score": spectral_fit(real, dsp),
        "cond": cond_vector(role, duration, 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/pairs")
    ap.add_argument("--manifest", default="data/pairs/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-dur", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    src_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if args.limit:
        files = files[: args.limit]
    tasks = [(i, str(p), str(out_dir), args.max_dur) for i, p in enumerate(files)]

    count = 0
    with manifest_path.open("w") as manifest:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for row in pool.map(process_one, tasks, chunksize=4):
                    if row is None:
                        continue
                    manifest.write(json.dumps(row) + "\n")
                    count += 1
                    if count % 100 == 0:
                        print(f"{count} pairs", flush=True)
        else:
            for task in tasks:
                row = process_one(task)
                if row is None:
                    continue
                manifest.write(json.dumps(row) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"{count} pairs", flush=True)

    print(f"wrote {count} pairs to {out_dir} and {manifest_path}")


if __name__ == "__main__":
    main()
