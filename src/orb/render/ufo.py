"""UFO roles for Destruction Cars (fork addition).

Cheap-B-movie flying-saucer sounds: the hover hum, the tractor beam, the ray
gun, the laser charging up and the abduction "gulp". Each is a small synth
with a handful of knobs, like the roles in `events.py`.

These roles live in `GAME_ROLES`, not `EVENT_ROLES`, so the planner's token
layout (and every already-rendered cue) is untouched. They are reachable from
a hand-written Score, not from a caption.
"""

from __future__ import annotations

import numpy as np

from ..score import Event
from .dsp import TAU, bandlimit, saturate

ROLES = ("ufo_hover", "tractor_beam", "ufo_laser", "ufo_charge", "abduction")

# Roles that sustain: raised-cosine edges instead of the 2 ms transient fade.
SMOOTH_ROLES = ("ufo_hover", "tractor_beam", "ufo_charge", "abduction")

PARAMS: dict[str, dict[str, float]] = {
    # f0: hull hum (Hz). warble: theremin depth (0-1). loop: 1 = seamless loop.
    # A looping clip gets no edge fades and no reverb (both would break the seam).
    "ufo_hover": dict(f0=55.0, theremin=700.0, warble=1.0, loop=1.0, width=0.6, reverb=0.25),
    # f0: beam drone. shimmer: high-partial level. pulse: Hz of the beam's throb.
    "tractor_beam": dict(f0=110.0, shimmer=0.6, pulse=7.0, width=0.7, reverb=0.35),
    # f_start/f_end: the zap's chirp. pulses: zaps in the clip.
    "ufo_laser": dict(f_start=3200.0, f_end=180.0, pulses=1.0, width=0.3, reverb=0.3),
    # f_start/f_end: the whine rising to the shot.
    "ufo_charge": dict(f_start=180.0, f_end=2600.0, width=0.4, reverb=0.2),
    # f_start/f_end: the upward "whoop"; thunk: the hollow gulp at the top.
    "abduction": dict(f_start=140.0, f_end=1300.0, thunk=1.0, width=0.5, reverb=0.3),
}


def _phase(freq: np.ndarray, sr: int, start: float = 0.0) -> np.ndarray:
    return start + TAU * np.cumsum(freq) / sr


def _hover(ev: Event, sr: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    """A saucer hanging in the air: a beating hull drone and a wandering theremin.

    With loop=1 every LFO completes a whole number of cycles over the clip and
    each oscillator ends on a whole cycle, so the clip repeats without a seam.
    """
    n = max(1, int(ev.dur * sr))
    dur = n / sr
    t = np.arange(n) / sr
    loop = p.get("loop", 1.0) >= 0.5

    def lfo_rate(hz: float) -> float:
        return max(1.0, round(hz * dur)) / dur if loop else hz

    def osc_f(hz: float) -> float:
        return round(hz * dur) / dur if loop else hz

    f0 = osc_f(float(p["f0"]))
    beat = osc_f(f0 + lfo_rate(1.3))  # a slow beating second hull tone
    hull = (
        np.sin(TAU * f0 * t + rng.uniform(0, TAU))
        + 0.8 * np.sin(TAU * beat * t + rng.uniform(0, TAU))
        + 0.35 * np.sin(TAU * 2 * f0 * t)
        + 0.2 * np.sin(TAU * 3 * beat * t)
    )
    hull = saturate(0.6 * hull, 1.8)

    # The theremin: a sine gliding around `theremin` Hz with a 6 Hz vibrato.
    # Its phase has to close on the loop too, so the frequency curve is built
    # from loop-periodic sines and its mean is snapped to a whole cycle.
    w = float(p["warble"])
    th = float(p["theremin"])
    glide = 0.18 * w * np.sin(TAU * lfo_rate(0.11) * t + rng.uniform(0, TAU)) + 0.08 * w * np.sin(
        TAU * lfo_rate(0.27) * t + rng.uniform(0, TAU)
    )
    vib = 0.012 * w * np.sin(TAU * lfo_rate(6.0) * t)
    f_th = th * np.exp(glide + vib)
    if loop:
        f_th *= osc_f(float(f_th.mean())) / float(f_th.mean())
    theremin = np.sin(_phase(f_th, sr))
    theremin *= 0.75 + 0.25 * np.sin(TAU * lfo_rate(0.4) * t)

    # Ring-modulated shimmer: the "electronic" in a 1950s saucer.
    ring = np.sin(_phase(f_th * 1.5, sr)) * np.sin(TAU * osc_f(f0 * 9.0) * t)

    # A little air turbulence under it, throbbing with the hull beat.
    k = min(n // 4, int(0.25 * sr)) if loop else 0
    air = bandlimit(rng.standard_normal(n + k), sr, 200.0, 1800.0, order=2)
    if k > 1:
        # Noise isn't periodic: render k extra samples and crossfade that
        # overhang into the head, so the last sample flows into the first.
        f = np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, k))
        air[:k] = air[:k] * f + air[n:] * np.cos(0.5 * np.pi * np.linspace(0.0, 1.0, k))
    air = air[:n]
    air /= max(float(np.sqrt(np.mean(air**2))), 1e-9)
    air *= 0.5 + 0.5 * np.sin(TAU * lfo_rate(1.3) * t)

    return 0.55 * hull + 0.32 * theremin + 0.12 * ring + 0.06 * air


def _beam(ev: Event, sr: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    """The tractor beam switching on and holding: a rising drone, a throbbing
    shimmer of high partials sweeping like a phaser, and a sparkle on top."""
    n = max(1, int(ev.dur * sr))
    t = np.arange(n) / sr
    dur = n / sr
    f0 = float(p["f0"])

    # Switch-on: the pitch slides up an octave into place over 0.35 s.
    on = np.clip(t / 0.35, 0.0, 1.0)
    f = f0 * (0.5 + 0.5 * on**0.6)
    ph = _phase(f, sr)
    drone = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph + 0.4)

    # Shimmer: partials 8-16 of the drone, each faded in and out by a slow
    # sweeping comb (the phaser sound of a beam).
    sweep = 0.5 + 0.5 * np.sin(TAU * 0.35 * t + rng.uniform(0, TAU))
    shimmer = np.zeros(n)
    for k in range(8, 17):
        centre = (k - 8) / 8.0
        gain = np.exp(-((sweep - centre) ** 2) / 0.03)
        shimmer += gain * np.sin(k * ph + rng.uniform(0, TAU)) / k**0.5
    pulse = 0.6 + 0.4 * np.sin(TAU * float(p["pulse"]) * t) ** 2
    shimmer *= pulse * float(p["shimmer"])

    sparkle = bandlimit(rng.standard_normal(n), sr, 3500.0, 7500.0, order=2)
    sparkle /= max(float(np.sqrt(np.mean(sparkle**2))), 1e-9)
    sparkle *= pulse

    env = np.clip(t / 0.12, 0.0, 1.0) * np.clip((dur - t) / 0.25, 0.0, 1.0)
    # The switch-on has a little "thoom" under it.
    thoom = np.sin(_phase(60.0 * np.exp(-t * 3.0) + 30.0, sr)) * np.exp(-t * 5.0)
    return env * (0.5 * saturate(0.6 * drone, 1.5) + 0.35 * shimmer + 0.05 * sparkle) + 0.6 * thoom


def _zap(t: np.ndarray, sr: int, f0: float, f1: float, length: float, rng: np.random.Generator) -> np.ndarray:
    """One ray-gun 'pew': a falling exponential chirp with a fast FM wobble."""
    n = len(t)
    x = np.clip(t / length, 0.0, 1.0)
    f = f0 * (f1 / f0) ** (x**0.7)
    f *= 1.0 + 0.08 * np.sin(TAU * 38.0 * t)  # the wobble that makes it sci-fi
    s = np.sin(_phase(f, sr)) + 0.3 * np.sign(np.sin(_phase(f * 0.5, sr)))
    env = np.exp(-4.0 * x) * (t < length)
    a = max(1, int(0.002 * sr))
    env[:a] *= np.linspace(0.0, 1.0, a)
    crack = bandlimit(rng.standard_normal(n), sr, 2000.0, 9000.0, order=2) * np.exp(-t / 0.012)
    crack /= max(float(np.abs(crack).max()), 1e-9)
    return env * s + 0.4 * crack


def _laser(ev: Event, sr: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    n = max(1, int(ev.dur * sr))
    out = np.zeros(n)
    pulses = max(1, int(round(float(p["pulses"]))))
    gap = ev.dur / pulses
    for i in range(pulses):
        start = int(i * gap * sr)
        m = n - start
        t = np.arange(m) / sr
        f0 = float(p["f_start"]) * rng.uniform(0.94, 1.06)
        out[start:] += _zap(t, sr, f0, float(p["f_end"]), min(gap, ev.dur) * 0.9, rng)
    return saturate(0.7 * out, 1.4)


def _charge(ev: Event, sr: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    """A whine rising to the shot, with a tremolo that speeds up as it fills."""
    n = max(1, int(ev.dur * sr))
    t = np.arange(n) / sr
    x = t / (n / sr)
    f = float(p["f_start"]) * (float(p["f_end"]) / float(p["f_start"])) ** (x**1.6)
    ph = _phase(f, sr)
    tone = np.sin(ph) + 0.4 * np.sin(2.01 * ph) + 0.2 * np.sin(3.0 * ph)
    trem_rate = 4.0 + 36.0 * x**1.5
    trem = 0.55 + 0.45 * np.sin(_phase(trem_rate, sr))
    env = x**1.2
    hiss = bandlimit(rng.standard_normal(n), sr, 1500.0, 8000.0, order=2)
    hiss /= max(float(np.sqrt(np.mean(hiss**2))), 1e-9)
    return env * (0.7 * tone * trem + 0.04 * hiss * x)


def _abduction(ev: Event, sr: int, rng: np.random.Generator, p: dict) -> np.ndarray:
    """Something sucked up into the saucer: a wobbling upward 'whoop' that ends
    in a hollow, metallic gulp as the hatch closes."""
    n = max(1, int(ev.dur * sr))
    t = np.arange(n) / sr
    dur = n / sr
    rise = min(dur * 0.7, 1.2)
    x = np.clip(t / rise, 0.0, 1.0)
    f = float(p["f_start"]) * (float(p["f_end"]) / float(p["f_start"])) ** (x**1.3)
    f *= 1.0 + 0.06 * np.sin(TAU * 9.0 * t)  # the cartoon wobble
    ph = _phase(f, sr)
    whoop = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(5 * ph)
    whoop *= np.clip(t / 0.08, 0.0, 1.0) * np.where(t < rise, 1.0, np.exp(-(t - rise) / 0.05))

    # The gulp: a pipe-like modal "bonk" at the top of the rise.
    tt = np.maximum(t - rise, 0.0)
    gate = (t >= rise).astype(float)
    gulp = np.zeros(n)
    for fr, a, d in ((95.0, 1.0, 0.18), (190.0 * 1.13, 0.45, 0.12), (420.0, 0.25, 0.07), (870.0, 0.12, 0.04)):
        gulp += a * np.sin(TAU * fr * tt * (1.0 - 0.15 * np.exp(-tt / 0.03))) * np.exp(-tt / d)
    gulp *= gate * float(p["thunk"])

    sizzle = bandlimit(rng.standard_normal(n), sr, 2500.0, 7000.0, order=2)
    sizzle /= max(float(np.sqrt(np.mean(sizzle**2))), 1e-9)
    sizzle *= 0.05 * x * (t < rise)
    return 0.6 * whoop + 0.9 * gulp + sizzle


RENDERERS = {
    "ufo_hover": _hover,
    "tractor_beam": _beam,
    "ufo_laser": _laser,
    "ufo_charge": _charge,
    "abduction": _abduction,
}
