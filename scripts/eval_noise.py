#!/usr/bin/env python3
"""Measure the noise floor of the DSP renderer on a fixed suite of cues.

The suite covers every role alone (no bed), every texture alone, sparse UI
cues, and the README trailer cue. For each cue it reports:

* ``pre``    RMS dBFS of the mix before the first event (silence or bed)
* ``floor``  noise floor: 10th percentile of 20 ms frame RMS, dBFS
* ``>8k``    share of the cue's energy above 8 kHz, %
* ``>12k``   share of the cue's energy above 12 kHz, %
* ``flat``   spectral flatness (0 = tonal/coloured, 1 = white) of the
             quietest 10% of frames; ``-`` when those frames are silent
* ``audit``  `orb.eval.audit.passes`
* ``timing`` each event rendered alone: its start (first sample above
             -100 dBFS) and its onset (the 5 ms envelope first reaches -20 dB
             of its maximum). Transient roles must keep both within 0.5 ms /
             1 ms. The continuous roles (whoosh, riser, swell, noise_sweep)
             fade in over 5 ms, so their -100 dBFS start may move by up to
             5 ms, and the whoosh onset by up to 5 ms (it used to be the
             click at the cut-off start). The onset is not gated for riser,
             swell and noise_sweep: their -20 dB point lies on a ramp of a
             second or more, so a 1 dB level change moves it by tens of ms
             without any change in timing.

Two more targets are gated:

* onset step: for the continuous roles, rendered alone on silence
  with 16 seeds, the loudest sample in the first 0.5 ms after the event starts
  must stay <= -40 dBFS (a clip that begins mid-waveform is a click).
* default bed: a cue with events and a bed at the default level (what every
  caption gives) must sit <= -50 dBFS before its first event.

Run it against the working tree, or compare with a frozen copy of an older
``src`` directory (``--baseline-src``). Cues that use ``texture: "none"`` or
``bed_db`` cannot be expressed by an older renderer; the baseline renders them
with ``room_cold`` at the default level, which is what a user got before the
bed could be switched off.

    .venv/bin/python scripts/eval_noise.py
    .venv/bin/python scripts/eval_noise.py --baseline-src /path/to/old/src --wav-dir out/noise
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 48000
FRAME = 960  # 20 ms
SILENT_DBFS = -200.0

ROLE_DUR = {
    "whoosh": 0.8,
    "impact": 0.7,
    "ui_pop": 0.16,
    "riser": 2.0,
    "swell": 1.5,
    "click": 0.1,
    "bell": 1.4,
    "sub_drop": 1.0,
    "glitch": 0.2,
    "noise_sweep": 1.5,
    # natural and foley roles (v0.2)
    "thunder": 6.0,
    "door_slam": 0.9,
    "knock": 0.7,
    "footsteps": 3.0,
    "water_drop": 0.35,
    "explosion": 3.0,
}
TEXTURE_NAMES = ("room_cold", "wind", "city_hum", "vinyl", "drone", "rain", "fire", "sci_fi")


def _ev(role, t, dur, gain, pan, seed):
    return {"role": role, "t": t, "dur": dur, "gain_db": gain, "pan": pan, "seed": seed}


def suite() -> list[tuple[str, dict]]:
    cues: list[tuple[str, dict]] = []
    for role, dur in ROLE_DUR.items():
        cues.append(
            (
                f"role/{role}",
                {"duration": max(4.0, float(np.ceil(dur + 1.5))), "texture": "none", "mood": "neutral",
                 "events": [_ev(role, 0.5, dur, -8.0, 0.0, 7)]},
            )
        )
    for tex in TEXTURE_NAMES:
        cues.append((f"texture/{tex}", {"duration": 6.0, "texture": tex, "mood": "calm", "events": []}))
    pop = [_ev("ui_pop", 0.6, 0.16, -12.0, 0.25, 0x73)]
    cues.append(("ui/pop_vinyl_bed", {"duration": 4.0, "texture": "vinyl", "mood": "playful", "events": pop}))
    # What a caption gives a UI sound: the planner always adds a bed at the
    # default level.
    cues.append(("ui/pop_room_default", {"duration": 2.0, "texture": "room_cold", "mood": "neutral", "events": pop}))
    cues.append(("ui/pop_none", {"duration": 4.0, "texture": "none", "mood": "playful", "events": pop}))
    pack = [
        _ev("ui_pop", 0.40, 0.16, -12.0, -0.5, 0x11),
        _ev("click", 1.20, 0.10, -16.0, 0.5, 0x22),
        _ev("ui_pop", 2.00, 0.16, -10.0, 0.0, 0x33),
        _ev("bell", 3.20, 1.40, -16.0, 0.25, 0x44),
    ]
    cues.append(("ui/pack_none", {"duration": 6.0, "texture": "none", "mood": "playful", "events": pack}))
    cues.append(("ui/click_glitch_none", {
        "duration": 2.0, "texture": "none", "mood": "neutral",
        "events": [_ev("click", 0.30, 0.10, -10.0, 0.0, 3), _ev("glitch", 1.00, 0.20, -12.0, 0.0, 9)],
    }))
    trailer = [
        _ev("riser", 1.00, 2.50, -18.0, 0.0, 0x55),
        _ev("whoosh", 4.20, 1.20, -10.0, -0.75, 0x77),
        _ev("impact", 5.00, 0.90, -4.0, 0.0, 0x66),
        _ev("sub_drop", 5.00, 1.60, -8.0, 0.0, 0x88),
        _ev("noise_sweep", 6.50, 2.00, -16.0, 0.5, 0x97),
        _ev("impact", 9.50, 0.90, -4.0, 0.0, 0x99),
        _ev("bell", 9.50, 2.00, -12.0, 0.25, 0x3D),
    ]
    cues.append(("readme/trailer_drone", {"duration": 12.0, "texture": "drone", "mood": "intense", "events": trailer}))
    cues.append(("readme/trailer_none", {"duration": 12.0, "texture": "none", "mood": "intense", "events": trailer}))
    # natural and foley scenes (v0.2)
    storm = [
        _ev("thunder", 1.00, 7.00, -10.0, -0.5, 0x03),  # far
        _ev("thunder", 7.50, 6.00, -4.0, 0.25, 0x00),  # close crack
        _ev("water_drop", 4.00, 0.35, -18.0, 0.5, 0x11),
    ]
    cues.append(("foley/storm_rain", {"duration": 14.0, "texture": "rain", "mood": "dark", "events": storm}))
    cues.append(("foley/storm_none", {"duration": 14.0, "texture": "none", "mood": "dark", "events": storm}))
    arrival = [
        _ev("footsteps", 0.40, 3.20, -14.0, -0.5, 0x04),
        _ev("door_slam", 3.90, 0.90, -6.0, 0.0, 0x01),
        _ev("knock", 5.60, 0.70, -10.0, 0.25, 0x03),
    ]
    cues.append(("foley/arrival_none", {"duration": 8.0, "texture": "none", "mood": "tense", "events": arrival}))
    cues.append(("foley/arrival_room", {"duration": 8.0, "texture": "room_cold", "mood": "tense", "events": arrival}))
    cave = [_ev("water_drop", t, 0.35, g, p, sd) for t, g, p, sd in
            ((0.6, -14.0, -0.5, 0x02), (1.9, -18.0, 0.5, 0x05), (3.1, -12.0, 0.0, 0x08), (4.4, -16.0, -0.25, 0x0b))]
    cues.append(("foley/cave_drone", {"duration": 6.0, "texture": "drone", "mood": "eerie", "events": cave}))
    cues.append(("foley/cave_none", {"duration": 6.0, "texture": "none", "mood": "eerie", "events": cave}))
    battle = [
        _ev("explosion", 0.50, 3.00, -2.0, 0.0, 0x00),
        _ev("explosion", 3.80, 3.00, -10.0, 0.75, 0x02),
        _ev("whoosh", 3.00, 0.80, -10.0, -0.5, 0x21),
    ]
    cues.append(("foley/battle_none", {"duration": 8.0, "texture": "none", "mood": "intense", "events": battle}))
    return cues


# -- metrics ---------------------------------------------------------------


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-10 else SILENT_DBFS


def _frames(mono: np.ndarray, size: int) -> np.ndarray:
    n = len(mono) // size * size
    return mono[:n].reshape(-1, size)


def metrics(mix: np.ndarray, events_stem: np.ndarray, spec: dict, audit_pass: bool) -> dict:
    mono = mix.astype(np.float64).mean(axis=1)
    out: dict = {"audit": bool(audit_pass), "peak_db": _db(float(np.abs(mix).max()))}

    starts = [int(round(e["t"] * SR)) for e in spec["events"]]
    if starts and min(starts) > 0:
        pre = mono[: min(starts)]
        out["pre_db"] = _db(float(np.sqrt(np.mean(pre**2))))
    else:
        out["pre_db"] = None

    fr = np.sqrt(np.mean(_frames(mono, FRAME) ** 2, axis=1))
    out["floor_db"] = _db(float(np.percentile(fr, 10)))

    power = np.abs(np.fft.rfft(mono)) ** 2
    f = np.fft.rfftfreq(len(mono), 1.0 / SR)
    tot = float(power.sum()) + 1e-30
    out["hf8"] = 100.0 * float(power[f > 8000].sum()) / tot
    out["hf12"] = 100.0 * float(power[f > 12000].sum()) / tot

    # Spectral flatness of the quietest 10% of 2048-sample frames.
    big = _frames(mono, 2048)
    rms_big = np.sqrt(np.mean(big**2, axis=1))
    k = max(1, int(0.1 * len(big)))
    quiet = big[np.argsort(rms_big)[:k]]
    if float(np.sqrt(np.mean(quiet**2))) < 10 ** (-120 / 20):
        out["flat"] = None
        out["gap_hf8"] = None
    else:
        P = np.mean(np.abs(np.fft.rfft(quiet * np.hanning(2048), axis=1)) ** 2, axis=0)[1:] + 1e-30
        out["flat"] = float(np.exp(np.mean(np.log(P))) / np.mean(P))
        fq = np.fft.rfftfreq(2048, 1.0 / SR)[1:]
        out["gap_hf8"] = 100.0 * float(P[fq > 8000].sum() / P.sum())

    out["event_starts"] = starts
    return out


SLOW_ROLES = {"riser", "swell", "noise_sweep"}
CONTINUOUS = SLOW_ROLES | {"whoosh"}


def event_timing(stem: np.ndarray) -> tuple[int, int]:
    """(start, -20 dB onset) in samples for one event's stem."""
    x = np.abs(stem.astype(np.float64)).max(axis=1)
    start = int(np.argmax(x > 1e-5))
    k = int(0.005 * SR)
    env = np.convolve(x, np.ones(k) / k, mode="same")
    return start, int(np.argmax(env > 0.1 * env.max()))


# Transient roles (impact, click, ui_pop, glitch) reach their peak within
# 0.5 ms by design; the step check is for roles that should swell in.
ONSET_ROLES = ("whoosh", "riser", "swell", "noise_sweep")
ONSET_SEEDS = range(0, 256, 16)
ONSET_STEP_MAX_DB = -40.0  # the first 0.5 ms of any event, dBFS
DEFAULT_BED_PRE_MAX_DB = -50.0  # a default bed before the first event, dBFS


def onset_steps(stems_of, used_texture: str) -> dict:
    """Worst level of the first 0.5 ms of each role over 16 seeds, dBFS.

    A clip that begins mid-waveform (whoosh, before the align/fade order was
    fixed) shows up as a step of -15..-23 dBFS here; a faded onset is far
    below -40.
    """
    out = {}
    for role in ONSET_ROLES:
        worst = SILENT_DBFS
        for seed in ONSET_SEEDS:
            ev = _ev(role, 0.5, ROLE_DUR[role], -6.0, 0.0, seed)
            stem = stems_of({"duration": 3.0, "texture": used_texture, "mood": "neutral", "events": [ev]})["events"]
            x = np.abs(stem.astype(np.float64)).max(axis=1)
            nz = np.flatnonzero(x > 1e-7)
            if nz.size:
                worst = max(worst, _db(float(x[nz[0] : nz[0] + 24].max())))
        out[role] = worst
    return out


# -- rendering -------------------------------------------------------------


def render_suite(src: Path, wav_dir: Path | None) -> dict:
    sys.path.insert(0, str(src))
    import orb.api as api  # noqa: E402
    from orb.eval.audit import audit_waveform, passes  # noqa: E402
    from orb.grammar import score_from_spec  # noqa: E402
    from orb.render.mix import render_stems  # noqa: E402

    try:
        api.dict_to_spec({"duration": 1.0, "texture": "none", "mood": "calm", "events": [_ev("click", 0.1, 0.1, -8, 0, 1)]})
        supports_none = True
    except (ValueError, KeyError):
        supports_none = False

    results = {}
    def stems_of(used: dict) -> dict:
        score = score_from_spec(api.dict_to_spec(used), with_music=False)
        score.seed = 0
        return render_stems(score)

    for name, spec in suite():
        used = dict(spec)
        if not supports_none:
            if used.get("texture") == "none":
                used["texture"] = "room_cold"
            used.pop("bed_db", None)
        stems = stems_of(used)
        mix = stems["mix"]
        # The public entry point must give the same audio as the stems.
        assert np.array_equal(mix, api.render_spec(used)), name
        audit = audit_waveform(mix, SR)
        timing = [
            [e["role"], *event_timing(stems_of({**used, "events": [e]})["events"])] for e in used["events"]
        ]
        results[name] = {
            **metrics(mix, stems["events"], spec, passes(audit)),
            "texture_used": used["texture"],
            "timing": timing,
        }
        if wav_dir is not None:
            wav_dir.mkdir(parents=True, exist_ok=True)
            api.save_wav(mix, wav_dir / (name.replace("/", "__") + ".wav"))
    return {"cues": results, "onsets": onset_steps(stems_of, "none" if supports_none else "room_cold")}


# -- report ----------------------------------------------------------------


def _fmt_db(x):
    if x is None:
        return "    -"
    return " -inf" if x <= SILENT_DBFS else f"{x:5.0f}"


def _fmt(x, spec="5.2f"):
    return "    -" if x is None else format(x, spec)


def table(after: dict, before: dict | None) -> str:
    cols = ["pre", "floor", ">8k%", ">12k%", "flat", "pass"]
    head = f"{'cue':<24}" + "".join(f"{c:>7}" for c in cols)
    if before is not None:
        head = f"{'cue':<24} | BEFORE " + "".join(f"{c:>7}" for c in cols) + " | AFTER  " + "".join(
            f"{c:>7}" for c in cols
        ) + " | timing"
    lines = [head, "-" * len(head)]

    def row(r):
        return (
            f"{_fmt_db(r['pre_db']):>7}{_fmt_db(r['floor_db']):>7}{_fmt(r['hf8'], '6.2f'):>7}"
            f"{_fmt(r['hf12'], '6.3f'):>7}{_fmt(r['flat'], '5.3f'):>7}{('yes' if r['audit'] else 'NO'):>7}"
        )

    for name, r in after.items():
        if before is None:
            lines.append(f"{name:<24}" + row(r))
        else:
            b = before[name]
            lines.append(f"{name:<24} |        " + row(b) + " |        " + row(r) + f" | {_timing(b, r)}")
    return "\n".join(lines)


def _timing_delta(b: dict, r: dict) -> tuple[int, float, list[str]]:
    """(worst start shift in samples, worst onset shift in ms, problems)."""
    start = onset = 0.0
    problems = []
    for (role, s0, o0), (_, s1, o1) in zip(b["timing"], r["timing"]):
        ds, do = abs(s1 - s0), 1000.0 * abs(o1 - o0) / SR
        start = max(start, ds)
        if role not in SLOW_ROLES:
            onset = max(onset, do)
        start_tol = 0.005 if role in CONTINUOUS else 0.0005
        onset_tol = 5.0 if role in CONTINUOUS else 1.0
        if ds > start_tol * SR:
            problems.append(f"{role} start moved {ds} samples")
        if role not in SLOW_ROLES and do > onset_tol:
            problems.append(f"{role} onset moved {do:.2f} ms")
    return int(start), onset, problems


def _timing(b: dict, r: dict) -> str:
    if not r["timing"]:
        return "-"
    start, onset, problems = _timing_delta(b, r)
    return f"start {start:d} smp, onset {onset:.2f} ms" + (" MOVED" if problems else "")


def onset_table(after: dict, before: dict | None) -> str:
    lines = ["onset step (worst first 0.5 ms over 16 seeds, dBFS; target <= %g)" % ONSET_STEP_MAX_DB]
    for role, v in after.items():
        b = f"{before[role]:7.1f} -> " if before is not None else ""
        lines.append(f"  {role:<12} {b}{v:7.1f}")
    return "\n".join(lines)


def check_targets(after: dict, onsets: dict | None = None, specs: dict | None = None) -> list[str]:
    problems = []
    for role, v in (onsets or {}).items():
        if v > ONSET_STEP_MAX_DB:
            problems.append(f"onset/{role}: first 0.5 ms reaches {v:.1f} dBFS (target <= {ONSET_STEP_MAX_DB:g})")
    for name, r in after.items():
        if not r["audit"]:
            problems.append(f"{name}: audit fails")
        spec = (specs or {}).get(name, {})
        if (
            spec.get("texture") != "none" and spec.get("events") and "bed_db" not in spec
            and r["pre_db"] is not None and r["pre_db"] > DEFAULT_BED_PRE_MAX_DB
        ):
            problems.append(
                f"{name}: default bed at {r['pre_db']:.1f} dBFS before the first event "
                f"(target <= {DEFAULT_BED_PRE_MAX_DB:g})"
            )
        if r["texture_used"] == "none":
            if r["pre_db"] is not None and r["pre_db"] > -90:
                problems.append(f"{name}: {r['pre_db']:.1f} dBFS before the first event (target <= -90)")
            if r["floor_db"] > -90:
                problems.append(f"{name}: gap floor {r['floor_db']:.1f} dBFS (target <= -90)")
        elif r["flat"] is not None and r["gap_hf8"] is not None and r["flat"] > 0.5 and r["gap_hf8"] > 5:
            problems.append(f"{name}: gaps look like broadband hiss (flatness {r['flat']:.2f})")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "src"), help="src directory holding the `orb` package to measure")
    ap.add_argument("--baseline-src", default=None, help="an older src directory to compare against")
    ap.add_argument("--wav-dir", default=None, help="write every rendered cue here (AFTER in <dir>/after, BEFORE in <dir>/before)")
    ap.add_argument("--json", default=None, help="write the raw measurements here")
    ap.add_argument("--_dump", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._dump:
        wav = Path(args.wav_dir) if args.wav_dir else None
        Path(args._dump).write_text(json.dumps(render_suite(Path(args.src), wav)))
        return 0

    def run(src: str, tag: str) -> dict:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            dump = Path(td) / "m.json"
            cmd = [sys.executable, __file__, "--src", src, "--_dump", str(dump)]
            if args.wav_dir:
                cmd += ["--wav-dir", str(Path(args.wav_dir) / tag)]
            subprocess.run(cmd, check=True)
            return json.loads(dump.read_text())

    after_all = run(args.src, "after")
    before_all = run(args.baseline_src, "before") if args.baseline_src else None
    after = after_all["cues"]
    before = before_all["cues"] if before_all else None
    print(table(after, before))
    print()
    print(onset_table(after_all["onsets"], before_all["onsets"] if before_all else None))
    problems = check_targets(after, after_all["onsets"], dict(suite()))
    if before is not None:
        for n in after:
            problems += [f"{n}: {p}" for p in _timing_delta(before[n], after[n])[2]]
    print()
    print("targets: OK" if not problems else "targets: FAIL\n  " + "\n  ".join(problems))
    if args.json:
        Path(args.json).write_text(json.dumps({"after": after_all, "before": before_all}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
