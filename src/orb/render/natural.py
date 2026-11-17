"""Everyday, nature and action sounds: the v0.3 library.

Each role is a small physical or acoustic model in numpy, built the same way
as the foley roles in `foley.py` (and rendered through the same
`render_foley` path: high-pass, smooth low-pass to 12 kHz after every
envelope and attack, peak normalisation, a dark room or outdoor reverb, fades
to exactly zero, constant-power pan). Only sounds that are not voices are
here: a dog, a cat, a laugh or a cough needs recordings or a learned model.

The seed picks a documented variant (`VARIANTS`) and seeds the noise. Every
model is deterministic.
"""

from __future__ import annotations

import numpy as np

from ..score import Event
from .dsp import LN1000, TAU, bandlimit, saturate
from .foley import _attack, _burst, _fade_out, _fftconv, _modes_ir, _norm, _pulse, _ring, _smooth, _bubble

ROLES = (
    "bird_chirp",
    "crickets",
    "wave_crash",
    "wind_gust",
    "splash",
    "frog",
    "owl",
    "glass_break",
    "clock_tick",
    "phone_ring",
    "keyboard_typing",
    "mouse_click",
    "camera_shutter",
    "paper_rustle",
    "cork_pop",
    "coin_drop",
    "metal_clang",
    "wood_creak",
    "heartbeat",
    "applause",
    "pouring_water",
    "can_opening",
    "clock_alarm",
    "punch",
    "sword_clash",
    "gunshot",
    "car_horn",
    "siren",
    "engine",
    "church_bell",
    "fireworks",
)

VARIANTS = {
    "bird_chirp": "seed % 4 = song: 0 sparrow chips, 1 trill, 2 warbled phrase, 3 two-note whistle",
    "crickets": "seed % 3 = 0 one cricket close, 1 two crickets, 2 a field chorus",
    "wave_crash": "seed % 3 = 0 small wave on sand, 1 breaking wave, 2 big wave on rocks",
    "wind_gust": "seed % 3 = 0 low buffeting gust, 1 howling gust, 2 gust through leaves",
    "splash": "seed % 3 = 0 small stone, 1 something heavier, 2 a big jump into water",
    "frog": "seed % 3 = 0 two-part ribbit, 1 deep bullfrog, 2 small tree frog peeps",
    "owl": "seed % 2 = 0 hoo hoo-hoo (tawny/great horned), 1 a single soft hoot",
    "glass_break": "seed % 3 = 0 a glass dropped on a hard floor, 1 a window pane, 2 a bottle smashed",
    "clock_tick": "seed % 3 = 0 wall clock (1 tick/s, tick-tock), 1 pocket watch (fast, small), 2 grandfather clock (slow, deep case)",
    "phone_ring": "seed % 4 = 0 classic bell phone (2 s on, 4 s off), 1 classic UK double ring, 2 modern marimba ringtone, 3 phone vibrating on a table",
    "keyboard_typing": "seed % 3 = 0 laptop keys, 1 mechanical keyboard, 2 old typewriter",
    "mouse_click": "seed % 2 = 0 single click, 1 double click",
    "camera_shutter": "seed % 3 = 0 DSLR single frame, 1 DSLR burst, 2 film camera with wind-on lever",
    "paper_rustle": "seed % 2 = 0 handling a sheet, 1 crumpling a sheet into a ball",
    "cork_pop": "seed % 2 = 0 wine cork, 1 champagne cork with fizz",
    "coin_drop": "seed % 3 = 0 one coin bouncing, 1 a coin spinning down, 2 a handful of coins into a dish",
    "metal_clang": "seed % 3 = 0 metal pipe, 1 bin lid or sheet, 2 anvil hit",
    "wood_creak": "seed % 3 = 0 creaking door, 1 floorboard, 2 old ship's timber",
    "heartbeat": "seed % 3 = tempo: 0 resting (62 bpm), 1 nervous (88 bpm), 2 racing (120 bpm)",
    "applause": "seed % 3 = 0 a few people, 1 a room, 2 a big audience",
    "pouring_water": "seed % 2 = 0 into a glass (pitch rises as it fills), 1 from a jug into a pot",
    "can_opening": "seed % 2 = 0 soda can, 1 beer can (bigger hiss)",
    "clock_alarm": "seed % 2 = 0 twin-bell wind-up alarm clock, 1 digital beeping alarm",
    "punch": "seed % 3 = 0 body punch, 1 face punch, 2 heavy movie punch",
    "sword_clash": "seed % 3 = 0 blade clash, 1 sword drawn from its sheath, 2 two quick clashes",
    "gunshot": "seed % 3 = 0 pistol, 1 rifle, 2 distant shot",
    "car_horn": "seed % 3 = 0 one honk, 1 two short honks, 2 long honk",
    "siren": "seed % 3 = 0 wail, 1 yelp, 2 two-tone hi-lo",
    "engine": "seed % 3 = 0 car idling, 1 car revving, 2 car passing by",
    "church_bell": "seed % 2 = 0 one strike, 1 tolling (a strike every 2.5 s)",
    "fireworks": "seed % 3 = 0 one shell (whistle and burst), 1 crackling shell, 2 distant show",
}

# f0 / width / reverb: what the mixer passes as `p`. A role may override the
# reverb per variant (outdoor vs room) by writing into `p`.
PARAMS: dict[str, dict[str, float]] = {
    "bird_chirp": dict(width=0.3, reverb=0.18, reverb_decay=0.9),
    "crickets": dict(width=0.5, reverb=0.12, reverb_decay=0.8),
    "wave_crash": dict(width=0.8, reverb=0.0),
    "wind_gust": dict(width=0.8, reverb=0.0),
    "splash": dict(width=0.4, reverb=0.14, reverb_decay=0.7),
    "frog": dict(width=0.2, reverb=0.16, reverb_decay=0.8),
    "owl": dict(width=0.2, reverb=0.3, reverb_decay=1.4),
    "glass_break": dict(width=0.6, reverb=0.2, reverb_decay=0.6),
    "clock_tick": dict(width=0.0, reverb=0.12, reverb_decay=0.5),
    "phone_ring": dict(width=0.2, reverb=0.16, reverb_decay=0.5),
    "keyboard_typing": dict(width=0.3, reverb=0.08, reverb_decay=0.4),
    "mouse_click": dict(width=0.0, reverb=0.05, reverb_decay=0.3),
    "camera_shutter": dict(width=0.1, reverb=0.08, reverb_decay=0.4),
    "paper_rustle": dict(width=0.4, reverb=0.08, reverb_decay=0.4),
    "cork_pop": dict(width=0.2, reverb=0.14, reverb_decay=0.5),
    "coin_drop": dict(width=0.2, reverb=0.16, reverb_decay=0.6),
    "metal_clang": dict(width=0.4, reverb=0.2, reverb_decay=1.0),
    "wood_creak": dict(width=0.3, reverb=0.16, reverb_decay=0.6),
    "heartbeat": dict(width=0.0, reverb=0.0),
    "applause": dict(width=0.9, reverb=0.2, reverb_decay=0.9),
    "pouring_water": dict(width=0.2, reverb=0.1, reverb_decay=0.4),
    "can_opening": dict(width=0.1, reverb=0.08, reverb_decay=0.4),
    "clock_alarm": dict(width=0.1, reverb=0.12, reverb_decay=0.5),
    "punch": dict(width=0.2, reverb=0.1, reverb_decay=0.5),
    "sword_clash": dict(width=0.4, reverb=0.18, reverb_decay=0.9),
    "gunshot": dict(width=0.5, reverb=0.3, reverb_decay=1.2),
    "car_horn": dict(width=0.2, reverb=0.18, reverb_decay=0.9),
    "siren": dict(width=0.3, reverb=0.22, reverb_decay=1.2),
    "engine": dict(width=0.3, reverb=0.08, reverb_decay=0.6),
    "church_bell": dict(width=0.4, reverb=0.3, reverb_decay=2.0),
    "fireworks": dict(width=0.7, reverb=0.25, reverb_decay=1.6),
}

# Roles that are a stable pitched tone by design (a whistle, a horn, a bell,
# a cricket's song). Their high band carries real harmonics, so the
# "no whistle line" and "no buzzing envelope" checks that guard the noisy
# roles do not apply; the Nyquist check still does.
TONAL = ("bird_chirp", "crickets", "owl", "phone_ring", "clock_alarm", "car_horn", "siren",
         "church_bell", "metal_clang", "sword_clash", "glass_break", "coin_drop", "frog")


# -- helpers -----------------------------------------------------------------


def _n(ev: Event, sr: int) -> int:
    return max(1, int(ev.dur * sr))


def _tone(freq: np.ndarray, sr: int, phase: float = 0.0) -> np.ndarray:
    """A sine following the instantaneous frequency curve `freq` (Hz)."""
    return np.sin(phase + TAU * np.cumsum(freq) / sr)


def _hann_env(n: int, at: int, length: int, rise: float = 0.3) -> np.ndarray:
    """0 outside [at, at+length); a raised-cosine rise over `rise` of it, then
    a raised-cosine fall over the rest."""
    out = np.zeros(n)
    k = max(2, min(length, n - at))
    if at >= n or k < 2:
        return out
    a = max(1, int(rise * k))
    b = max(1, k - a)
    e = np.ones(k)
    e[:a] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(a) + 0.5) / a)
    e[a:] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(b) + 0.5) / b)
    out[at : at + k] = e
    return out


def _strike(n: int, sr: int, at: int, freqs, t60s, gains, rng: np.random.Generator,
            contact: float = 0.0008, noise: float = 0.0) -> np.ndarray:
    """A struck object: a short force pulse (and optional noise) into a bank of
    modes. Unlike `_ring`, the force is not differentiated, so a contact this
    short already carries the highs."""
    exc = _pulse(n, at, max(2, int(contact * sr)))
    if noise > 0:
        exc = exc + noise * _burst(rng, n, at, int(4 * contact * sr) + 2, contact, sr, rise=0.0005)
    return _fftconv(exc, _modes_ir(freqs, t60s, gains, sr, n), n)


def _grains(rng: np.random.Generator, n: int, times: np.ndarray, amps: np.ndarray, sr: int) -> np.ndarray:
    """Impulses at `times` (s) with `amps`, random sign."""
    out = np.zeros(n)
    idx = (np.asarray(times) * sr).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(out, idx[ok], np.asarray(amps)[ok] * rng.choice((-1.0, 1.0), int(ok.sum())))
    return out


def _decorrelate(x: np.ndarray, sr: int, crng: np.random.Generator, taps: int = 6, span: float = 0.012) -> np.ndarray:
    """A second ear: a few short, weak, darkened echoes (a little space), so a
    mono source gets some width without a comb."""
    n = len(x)
    out = x.copy()
    for _ in range(taps):
        d = int(crng.uniform(0.002, span) * sr)
        if d < n:
            out[d:] += crng.uniform(0.05, 0.15) * x[: n - d]
    return out


def _bounces(first_gap: float, ratio: float, count: int, crng: np.random.Generator) -> np.ndarray:
    """Times of a bouncing object: each gap shrinks by `ratio`."""
    t = [0.0]
    g = first_gap
    for _ in range(count - 1):
        t.append(t[-1] + g * crng.uniform(0.92, 1.08))
        g *= ratio
        if g < 0.004:
            break
    return np.array(t)


def _bubble_into(out: np.ndarray, sr: int, at: int, f0: float, damp_scale: float, rise: float, amp: float) -> None:
    """`amp * foley._bubble(...)` added into `out` in place, computing only the
    samples the bubble occupies (a stream has thousands of them)."""
    n = len(out)
    if at >= n or at < 0:
        return
    d = (0.043 * f0 + 0.0014 * f0**1.5) * damp_scale
    k = min(n - at, int(LN1000 / d * 1.2 * sr))
    t = np.arange(k) / sr
    ph = TAU * np.cumsum(f0 * (1.0 + rise * d * t)) / sr
    a = min(k, max(2, int(0.002 * sr)))
    env = np.exp(-d * t)
    env[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, a))
    out[at : at + k] += amp * np.sin(ph) * env


def _harmonic(f: np.ndarray, sr: int, amps, phase: float = 0.0) -> np.ndarray:
    """A sum of harmonics k*f with amplitudes `amps` (k = 1..), each muted
    above 0.45 sr."""
    ph = phase + TAU * np.cumsum(f) / sr
    out = np.zeros(len(f))
    for k, a in enumerate(amps, start=1):
        if a == 0:
            continue
        mask = (k * f) < 0.45 * sr
        out += a * np.sin(k * ph) * mask
    return out


# =============================================================================
# Nature
# =============================================================================


def _syllable(n: int, sr: int, at: int, f_a: float, f_b: float, length: float, shape: str,
              crng: np.random.Generator) -> np.ndarray:
    """One bird syllable: a pure whistle gliding from f_a to f_b, a soft
    second harmonic, a smooth amplitude bump."""
    k = int(length * sr)
    if at >= n or k < 8:
        return np.zeros(n)
    k = min(k, n - at)
    u = np.arange(k) / max(1, k - 1)
    if shape == "down":
        f = f_b + (f_a - f_b) * (1.0 - u) ** 2
    elif shape == "up":
        f = f_a + (f_b - f_a) * u**1.5
    else:  # arch
        f = f_a + (f_b - f_a) * np.sin(np.pi * u)
    f = f * (1.0 + 0.012 * np.sin(TAU * crng.uniform(40, 80) * np.arange(k) / sr))
    ph = TAU * np.cumsum(f) / sr
    env = np.sin(np.pi * u) ** 1.5
    y = (np.sin(ph) + 0.08 * np.sin(2 * ph)) * env
    out = np.zeros(n)
    out[at : at + k] = y
    return out


def _bird_chirp(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 4
    mono = np.zeros(n)
    t = 0.01
    base = crng.uniform(0.85, 1.15)
    while t < dur - 0.08:
        at = int(t * sr)
        if kind == 0:  # sparrow "chip": short down-sweeps in a little group
            for _ in range(int(crng.integers(2, 5))):
                L = crng.uniform(0.035, 0.06)
                f_hi = crng.uniform(3900, 5200) * base
                mono += crng.uniform(0.7, 1.0) * _syllable(n, sr, at, f_hi, f_hi * 0.55, L, "down", crng)
                at += int((L + crng.uniform(0.05, 0.11)) * sr)
            t = at / sr + crng.uniform(0.35, 0.9)
        elif kind == 1:  # trill: a fast run of identical short notes
            reps = int(crng.integers(8, 16))
            L = crng.uniform(0.022, 0.032)
            gap = crng.uniform(0.012, 0.022)
            f_hi = crng.uniform(3500, 4800) * base
            for r in range(reps):
                mono += (0.6 + 0.4 * np.sin(np.pi * (r + 0.5) / reps)) * _syllable(n, sr, at, f_hi, f_hi * 0.7, L, "down", crng)
                at += int((L + gap) * sr)
            t = at / sr + crng.uniform(0.5, 1.2)
        elif kind == 2:  # warbled phrase: varied syllables, rises and falls
            for _ in range(int(crng.integers(4, 8))):
                L = crng.uniform(0.04, 0.11)
                fa = crng.uniform(2400, 3800) * base
                fb = min(5200.0, fa * crng.uniform(0.7, 1.35))
                shape = str(crng.choice(("down", "up", "arch")))
                mono += crng.uniform(0.6, 1.0) * _syllable(n, sr, at, fa, fb, L, shape, crng)
                at += int((L + crng.uniform(0.02, 0.06)) * sr)
            t = at / sr + crng.uniform(0.6, 1.4)
        else:  # two-note whistle "fee-bee": a high note then a lower one
            f1 = crng.uniform(3600, 4100) * base
            mono += _syllable(n, sr, at, f1 * 1.01, f1 * 0.99, 0.28, "down", crng)
            at += int(0.36 * sr)
            mono += 0.9 * _syllable(n, sr, at, f1 * 0.84, f1 * 0.8, 0.3, "down", crng)
            t = at / sr + 0.3 + crng.uniform(0.8, 1.6)
    p["reverb"], p["reverb_decay"] = 0.2, 1.0
    return mono, _decorrelate(mono, sr, crng)


def _crickets(ev, sr, rng, crng, p, seed):
    """Field crickets: a pure carrier near 4.5 kHz (the wing's harp), gated
    into pulses (one per wing closure, ~30 per second), 3-5 pulses per chirp,
    a couple of chirps per second."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    count = (1, 2, 7)[seed % 3]
    left = np.zeros(n)
    right = np.zeros(n)
    for c in range(count):
        fc = crng.uniform(4200, 4900)
        pulse_rate = crng.uniform(26, 34)
        per = int(crng.integers(3, 6))
        chirp_rate = crng.uniform(1.6, 2.6)
        env = np.zeros(n)
        tm = crng.uniform(0.0, 1.0 / chirp_rate)
        while tm < dur:
            for k in range(per):
                pt = tm + k / pulse_rate
                at = int(pt * sr)
                env += _hann_env(n, at, int(0.6 / pulse_rate * sr), 0.25)
            tm += (1.0 / chirp_rate) * crng.uniform(0.9, 1.1)
        f = fc * (1.0 + 0.004 * np.sin(TAU * crng.uniform(0.1, 0.4) * t))
        y = _harmonic(f, sr, (1.0, 0.06, 0.02)) * env
        level = 1.0 if c == 0 else crng.uniform(0.2, 0.55)
        pos = 0.5 if count == 1 else crng.uniform(0.15, 0.85)
        left += level * y * (1.0 - pos) * 2
        right += level * y * pos * 2
    p["reverb"], p["reverb_decay"] = (0.1, 0.15, 0.25)[seed % 3], 0.9
    fo = _fade_out(n, sr, min(0.2, 0.3 * dur))
    return left * fo, right * fo


def _wash(rng, n, sr, lo, hi, order=2):
    return _norm(bandlimit(rng.standard_normal(n), sr, lo, hi, order=order))


def _wave_crash(ev, sr, rng, crng, p, seed):
    """One wave: the swell rises (low rush), breaks (a broadband crash with a
    low thump), then the foam hisses and fizzes as it runs up and drains
    back, slowly darkening."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    size = (0.6, 1.0, 1.3)[seed % 3]
    t_break = min(0.25 * dur, crng.uniform(0.6, 1.1) * size)
    rise = (0.5 - 0.5 * np.cos(np.pi * np.clip(t / max(t_break, 1e-3), 0, 1))) ** 2
    after = np.maximum(t - t_break, 0.0)
    decay_t = max(0.3 * dur, 0.6)
    body_env = np.where(t < t_break, rise, np.exp(-after / (0.45 * decay_t)))
    foam_env = _attack(n, sr, t_break, 0.08) * np.exp(-after / decay_t) * (0.75 + 0.25 * _smooth(crng, n, sr, 3.0))

    def chan():
        rush = _wash(rng, n, sr, 60.0, 900.0) * body_env
        crash = _wash(rng, n, sr, 200.0, 5000.0) * _attack(n, sr, t_break - 0.05, 0.08) * np.exp(-after / (0.12 * size + 0.1))
        foam = _wash(rng, n, sr, 700.0, 7000.0, order=4) * foam_env
        # bubbling in the foam: many tiny bubbles
        cnt = int(400 * dur)
        tt = t_break + crng.exponential(decay_t * 0.8, cnt)
        g = _grains(rng, n, tt, np.exp(-(tt - t_break) / decay_t), sr)
        fizz = bandlimit(g, sr, 1500.0, 6500.0, order=4)
        return 0.8 * rush + 0.9 * size * crash + 0.55 * foam + 0.3 * _norm(fizz)

    thump = bandlimit(_burst(rng, n, int(t_break * sr), int(0.4 * sr), 0.09, sr, rise=0.01), sr, 35.0, 180.0, order=2)
    th = 0.5 * size * _norm(thump)
    fo = _fade_out(n, sr, min(0.4 * dur, 1.5))
    return (chan() + th) * fo, (chan() + th) * fo


def _wind_gust(ev, sr, rng, crng, p, seed):
    """A gust: low turbulent buffeting that swells and dies away, with a
    moving resonance (howl) or a leafy rustle on top."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    peak = crng.uniform(0.3, 0.55) * dur
    env = np.where(t < peak, np.sin(0.5 * np.pi * t / peak) ** 2,
                   np.cos(0.5 * np.pi * np.clip((t - peak) / (dur - peak), 0, 1)) ** 1.5)
    flutter = 0.6 + 0.4 * _smooth(crng, n, sr, 5.0)
    env = env * flutter

    def chan():
        low = _wash(rng, n, sr, 25.0, 220.0) * (0.7 + 0.3 * _smooth(crng, n, sr, 9.0))
        mid = _wash(rng, n, sr, 150.0, 1200.0)
        out = low + (0.15 if kind == 0 else 0.35) * mid
        if kind == 1:
            # howl: noise through a narrow band whose centre follows the gust
            fc = 380.0 + 420.0 * env / max(env.max(), 1e-9)
            noise = rng.standard_normal(n)
            howl = np.zeros(n)
            # time-varying resonance by short-block band-passing
            blk = int(0.05 * sr)
            for s in range(0, n, blk):
                e = min(n, s + blk + blk)
                c = float(np.mean(fc[s : min(n, s + blk)]))
                seg = bandlimit(noise[s:e], sr, c * 0.93, c * 1.07, order=2)
                w = np.hanning(e - s)
                howl[s:e] += seg * w
            out = out + 0.9 * _norm(howl)
        elif kind == 2:
            leaves = _wash(rng, n, sr, 1500.0, 7000.0, order=4) * (0.4 + 0.6 * _smooth(crng, n, sr, 14.0))
            out = 0.7 * out + 0.5 * leaves
        return out * env

    fo = _fade_out(n, sr, 0.1)
    return chan() * fo, chan() * fo


def _splash(ev, sr, rng, crng, p, seed):
    """Something enters water: the hit (a short, mid-bright slap), the crown
    and droplets falling back (many small bubbles), and the cavity closing
    (a low bloop)."""
    n = _n(ev, sr)
    dur = n / sr
    size = (0.4, 0.8, 1.4)[seed % 3]
    at = int(0.004 * sr)
    slap = bandlimit(_burst(rng, n, at, int(0.12 * sr), 0.018 * size + 0.01, sr, rise=0.002), sr, 250.0, 4500.0, order=4)
    body = bandlimit(_burst(rng, n, at, int(0.6 * sr), 0.08 * size + 0.04, sr, rise=0.006), sr, 120.0, 2500.0, order=2)
    bloop_f = 380.0 / size**0.6 * crng.uniform(0.9, 1.1)
    bloop = _bubble(n, sr, at + int((0.03 + 0.05 * size) * sr), bloop_f, 0.55, 0.25)
    # droplets falling back over 0.2-1 s: small bubbles at random pitches
    drops = np.zeros(n)
    k = int(crng.integers(10, 22) * size)
    span = min(dur - 0.05, 0.25 + 0.5 * size)
    for _ in range(k):
        dt = at / sr + 0.05 + crng.exponential(span * 0.35)
        if dt < dur - 0.05:
            a_ = crng.uniform(0.2, 1.0) * np.exp(-(dt / span))
            _bubble_into(drops, sr, int(dt * sr), crng.uniform(900, 3200), 0.7, 0.12, a_)
    mono = _norm(slap) + 0.8 * _norm(body) + 0.5 * size**0.5 * _norm(bloop) + 0.4 * _norm(drops)
    return mono, _decorrelate(mono, sr, crng)


def _frog(ev, sr, rng, crng, p, seed):
    """A frog call: a vocal sac buzz, i.e. a low carrier pulsed fast, in
    one or two bursts (rib-bit), repeated."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    mono = np.zeros(n)
    t = 0.02
    while t < dur - 0.1:
        if kind == 0:
            f0, rate, parts = crng.uniform(480, 620), crng.uniform(80, 110), ((0.0, 0.07), (0.12, 0.09))
        elif kind == 1:
            f0, rate, parts = crng.uniform(110, 150), crng.uniform(14, 20), ((0.0, 0.55),)
        else:
            f0, rate, parts = crng.uniform(2600, 3000), 0.0, ((0.0, 0.06),)
        for off, L in parts:
            at = int((t + off) * sr)
            k = int(L * sr)
            if at + k >= n:
                break
            tt = np.arange(k) / sr
            f = f0 * (1.0 + 0.06 * (tt / L))
            if kind == 1:
                y = _harmonic(f, sr, (1.0, 0.7, 0.5, 0.25, 0.15))
            elif kind == 0:
                y = _harmonic(f, sr, (1.0, 0.6, 0.45, 0.3, 0.15, 0.08))
            else:
                y = _harmonic(f, sr, (1.0, 0.04))
            if rate > 0:  # pulsed: the sac buzzes
                y *= np.clip(np.sin(np.pi * rate * tt), 0, 1) ** 2
            y *= np.sin(np.pi * np.arange(k) / k) ** 0.6
            mono[at : at + k] += y
        t += (0.35 if kind == 0 else 1.0 if kind == 1 else 0.3) + crng.uniform(0.25, 0.9)
    if kind == 1:
        mono = bandlimit(mono, sr, 60.0, 1800.0, order=2)
    elif kind == 0:
        mono = bandlimit(mono, sr, 250.0, 3500.0, order=2)
    return mono, _decorrelate(mono, sr, crng)


def _owl(ev, sr, rng, crng, p, seed):
    """Owl hoots: soft, breathy low whistles (~350-420 Hz) that swell in and
    glide down a little at the end."""
    n = _n(ev, sr)
    dur = n / sr
    f0 = crng.uniform(330, 420)
    pattern = ((0.0, 0.45, 1.0), (0.75, 0.22, 0.75), (1.05, 0.55, 0.95)) if seed % 2 == 0 else ((0.0, 0.7, 1.0),)
    mono = np.zeros(n)
    start = 0.05
    while start < dur - 0.4:
        for off, L, a in pattern:
            at = int((start + off) * sr)
            k = int(L * sr)
            if at + k >= n:
                break
            u = np.arange(k) / k
            f = f0 * (1.0 + 0.03 * np.sin(np.pi * u) - 0.06 * u**3)
            y = _harmonic(f, sr, (1.0, 0.18, 0.05))
            breath = bandlimit(rng.standard_normal(k), sr, 250.0, 1400.0, order=2)
            y = y + 0.25 * _norm(breath)
            env = np.sin(np.pi * np.clip(u / 0.3, 0, 1) / 2) ** 2 * np.cos(np.pi * np.clip((u - 0.6) / 0.4, 0, 1) / 2) ** 2
            mono[at : at + k] += a * y * env
        start += (1.8 if seed % 2 == 0 else 1.2) + crng.uniform(1.0, 2.5)
    if not mono.any() and n > int(0.15 * sr):
        # Slot too short for the full pattern: one hoot squeezed to fit.
        at, k = int(0.02 * sr), n - int(0.05 * sr)
        u = np.arange(k) / k
        f = f0 * (1.0 + 0.03 * np.sin(np.pi * u) - 0.06 * u**3)
        y = _harmonic(f, sr, (1.0, 0.18, 0.05)) + 0.25 * _norm(bandlimit(rng.standard_normal(k), sr, 250.0, 1400.0, order=2))
        env = np.sin(np.pi * np.clip(u / 0.3, 0, 1) / 2) ** 2 * np.cos(np.pi * np.clip((u - 0.6) / 0.4, 0, 1) / 2) ** 2
        mono[at : at + k] += y * env
    return mono, _decorrelate(mono, sr, crng, span=0.03)


# =============================================================================
# Everyday
# =============================================================================


def _shard(n, sr, at, rng, crng, bright=1.0):
    """A small glass fragment ringing: 3-4 high inharmonic modes, short."""
    f = crng.uniform(3500, 8000) * bright
    freqs = f * np.array((1.0, crng.uniform(1.4, 1.7), crng.uniform(2.1, 2.6), crng.uniform(2.9, 3.4)))
    t60 = crng.uniform(0.04, 0.25) * np.array((1.0, 0.7, 0.5, 0.35))
    return _strike(n, sr, at, freqs, t60, (1.0, 0.6, 0.4, 0.25), rng, contact=0.0002, noise=0.5)


def _glass_break(ev, sr, rng, crng, p, seed):
    """Glass shattering: a sharp crack, the body's ring as it fails, then
    dozens of shards hitting the floor and each other, thinning out."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    at = int(0.003 * sr)
    crack = bandlimit(_burst(rng, n, at, int(0.05 * sr), 0.006, sr, rise=0.0005), sr, 800.0, 9000.0, order=4)
    ring_f = crng.uniform(1100, 2200) * (0.6 if kind == 1 else 1.0)
    ring = _strike(n, sr, at, ring_f * np.array((1.0, 2.3, 3.9, 5.6)), (0.12, 0.09, 0.06, 0.04), (1, 0.7, 0.5, 0.3), rng, 0.0003)
    thud = bandlimit(_burst(rng, n, at, int(0.1 * sr), 0.02, sr, rise=0.002), sr, 80.0, 600.0, order=2)
    shards_l = np.zeros(n)
    shards_r = np.zeros(n)
    count = (45, 70, 55)[kind]
    span = min(dur - 0.05, (0.6, 1.0, 0.7)[kind])
    for i in range(count):
        dt = 0.006 + crng.gamma(1.4, span / 5.0)
        if dt >= dur - 0.03:
            continue
        s = crng.uniform(0.25, 1.0) * np.exp(-dt / (0.6 * span))
        sh = _shard(n, sr, int(dt * sr), rng, crng, bright=(1.0, 0.85, 0.95)[kind]) * s
        w = crng.uniform(0.2, 0.8)
        shards_l += sh * (1 - w)
        shards_r += sh * w
    debris_g = _grains(rng, n, 0.01 + crng.gamma(1.2, span / 4.0, 300), np.ones(300) * 0.3, sr)
    debris = bandlimit(debris_g, sr, 3000.0, 10000.0, order=4)
    base = 0.9 * _norm(crack) + 0.3 * _norm(ring) + (0.4 if kind == 2 else 0.2) * _norm(thud) + 0.45 * _norm(debris)
    m = max(np.abs(shards_l).max(), np.abs(shards_r).max(), 1e-12)
    return base + 1.4 * shards_l / m, base + 1.4 * shards_r / m


def _clock_tick(ev, sr, rng, crng, p, seed):
    """The escapement: each tick is a tiny metal click (high modes) knocking
    the case (a wooden body). Tick and tock differ a little."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    period = (1.0, 0.2, 2.0)[kind]
    hi = (crng.uniform(2600, 3400), crng.uniform(4200, 5200), crng.uniform(1800, 2300))[kind]
    case = (crng.uniform(600, 900), crng.uniform(1500, 2000), crng.uniform(220, 320))[kind]
    case_t = (0.05, 0.02, 0.18)[kind]
    left = np.zeros(n)
    tm = 0.01
    i = 0
    while tm < dur - 0.03:
        at = int(tm * sr)
        tock = i % 2 == 1
        f = hi * (0.86 if tock else 1.0)
        click = _strike(n, sr, at, (f, f * 1.53, f * 2.21), (0.012, 0.009, 0.006), (1, 0.6, 0.4), rng, 0.00015, noise=0.3)
        body = _strike(n, sr, at, (case * (0.9 if tock else 1.0), case * 1.7), (case_t, case_t * 0.6), (1, 0.4), rng, 0.0006)
        a = crng.uniform(0.9, 1.0) * (0.85 if tock else 1.0)
        left += a * (_norm(click) + (0.5, 0.25, 0.9)[kind] * _norm(body))
        tm += period / (2 if kind == 2 else 1) * (1.0 if kind != 0 else crng.uniform(0.995, 1.005))
        i += 1
    return left, left


def _bell_ring(n, sr, at, length, f, rng, crng, rate=20.0, partials=None):
    """An electromechanical bell: a clapper striking a metal gong `rate`
    times per second for `length` s."""
    partials = partials or (1.0, 2.76, 5.4, 8.9)
    t60 = (0.6, 0.35, 0.2, 0.12)
    out = np.zeros(n)
    k = int(length * rate)
    for i in range(k):
        a_ = at + int(i / rate * sr)
        if a_ >= n:
            break
        out += crng.uniform(0.85, 1.0) * _strike(n, sr, a_, f * np.array(partials), t60, (1.0, 0.6, 0.35, 0.2), rng, 0.0004, noise=0.2)
    return out


def _phone_ring(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 4
    mono = np.zeros(n)
    if kind in (0, 1):
        # Two bells side by side, hit alternately by one clapper.
        f1, f2 = crng.uniform(1050, 1250), crng.uniform(1350, 1550)
        if kind == 0:
            cadence = [(0.0, 2.0)]
            cycle = 6.0
        else:
            cadence = [(0.0, 0.4), (0.6, 0.4)]
            cycle = 3.0
        start = 0.01
        while start < dur - 0.2:
            for off, L in cadence:
                at = int((start + off) * sr)
                L = min(L, dur - start - off - 0.05)
                if L <= 0.05:
                    continue
                mono += _bell_ring(n, sr, at, L, f1, rng, crng, rate=20.0)
                mono += _bell_ring(n, sr, at + int(0.025 * sr), L, f2, rng, crng, rate=20.0)
            start += cycle
    elif kind == 2:
        # A modern ringtone: a soft marimba-like arpeggio, repeated.
        root = crng.uniform(520, 640)
        steps = [0, 4, 7, 12, 7, 4, 0, 12] if crng.random() < 0.5 else [0, 7, 12, 16, 12, 7, 12, 19]
        note = 0.14
        start = 0.01
        while start < dur - 0.2:
            for j, s in enumerate(steps):
                at = int((start + j * note) * sr)
                if at >= n - int(0.05 * sr):
                    break
                f = root * 2 ** (s / 12)
                mono += _strike(n, sr, at, (f, 4.0 * f, 9.2 * f), (0.35, 0.08, 0.03), (1.0, 0.22, 0.05), rng, 0.0015)
            start += len(steps) * note + 0.8
    else:
        # Vibrating on a table: a motor buzz (~150 Hz) rattling the surface,
        # in pulses.
        start = 0.01
        while start < dur - 0.2:
            for off in (0.0, 0.55):
                at = int((start + off) * sr)
                k = int(0.4 * sr)
                if at + k >= n:
                    break
                tt = np.arange(k) / sr
                fm = crng.uniform(140, 170)
                buzz = _harmonic(np.full(k, fm), sr, (1.0, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1))
                rattle = bandlimit(rng.standard_normal(k), sr, 300.0, 2500.0, order=2) * (0.5 + 0.5 * np.sin(TAU * fm * tt)) ** 4
                y = (buzz + 0.6 * _norm(rattle)) * np.sin(np.pi * np.clip(tt / 0.02, 0, 0.5)) * np.cos(0.5 * np.pi * np.clip((tt - 0.37) / 0.03, 0, 1))
                mono[at : at + k] += y
            start += 2.0
        mono = bandlimit(mono, sr, 80.0, 4000.0, order=2)
    return mono, _decorrelate(mono, sr, crng)


def _key(n, sr, at, rng, crng, kind, heavy=False):
    """One keystroke: the cap clicking down (high plastic modes), bottoming
    out (a duller thock), and the release a little later."""
    if kind == 2:  # typewriter: a type bar hits the platen
        f = crng.uniform(1600, 2400)
        hit = _strike(n, sr, at, (f, f * 1.8, f * 2.9, 450.0), (0.03, 0.02, 0.012, 0.05), (1, 0.6, 0.3, 0.3), rng, 0.0003, noise=0.6)
        return hit
    f = crng.uniform(2500, 4200) if kind == 1 else crng.uniform(2400, 3800)
    click = _strike(n, sr, at, (f, f * 1.6, f * 2.3), (0.012, 0.008, 0.005), (1, 0.5, 0.3), rng, 0.0002, noise=0.4)
    thock_f = crng.uniform(250, 420) * (0.75 if heavy else 1.0)
    thock = _strike(n, sr, at + int(crng.uniform(0.004, 0.009) * sr), (thock_f, thock_f * 2.2, thock_f * 3.6),
                    (0.03, 0.02, 0.012), (1, 0.6, 0.3), rng, 0.0012)
    rel_at = at + int(crng.uniform(0.07, 0.13) * sr)
    rel = _strike(n, sr, rel_at, (f * 1.1, f * 1.7), (0.008, 0.005), (1, 0.4), rng, 0.0002, noise=0.3)
    c = (1.0, 1.3)[kind]
    return c * _norm(click) + (0.7 if heavy else 0.3) * _norm(thock) + 0.35 * c * _norm(rel)


def _keyboard_typing(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    left = np.zeros(n)
    right = np.zeros(n)
    tm = 0.01
    word = int(crng.integers(3, 8))
    i = 0
    while tm < dur - 0.15:
        space = i == word
        k = _key(n, sr, int(tm * sr), rng, crng, kind, heavy=space)
        a = crng.uniform(0.6, 1.0)
        w = 0.5 + crng.uniform(-0.25, 0.25)
        left += a * k * (1 - w)
        right += a * k * w
        if kind == 2 and space:
            # carriage: a ratchet run (rarely) or a small bell
            pass
        if space:
            word = int(crng.integers(3, 8))
            i = 0
            tm += crng.uniform(0.18, 0.35)
        else:
            i += 1
            tm += float(np.clip(crng.lognormal(np.log(0.12 if kind < 2 else 0.17), 0.35), 0.06, 0.4))
        if crng.random() < 0.04:
            tm += crng.uniform(0.4, 0.9)  # a pause to think
    return left, right


def _mouse_click(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    mono = np.zeros(n)
    times = (0.005,) if seed % 2 == 0 else (0.005, 0.005 + crng.uniform(0.12, 0.17))
    for tm in times:
        at = int(tm * sr)
        f = crng.uniform(2800, 3600)
        press = _strike(n, sr, at, (f, f * 1.45, f * 2.1, 900.0), (0.008, 0.006, 0.004, 0.012), (1, 0.6, 0.3, 0.5), rng, 0.00015, noise=0.3)
        rel = _strike(n, sr, at + int(crng.uniform(0.06, 0.09) * sr), (f * 1.12, f * 1.6, 1000.0), (0.006, 0.004, 0.01), (1, 0.4, 0.4), rng, 0.00015, noise=0.3)
        mono += _norm(press) + 0.45 * _norm(rel)
    return mono, mono


def _shutter_once(n, sr, at, rng, crng, film=False):
    """Mirror up (a clack), the curtains (two quick ticks), mirror down."""
    out = np.zeros(n)
    f = crng.uniform(1500, 2400)
    for off, a, ff in ((0.0, 1.0, f), (0.018, 0.55, f * 1.7), (0.03 + crng.uniform(0.0, 0.02), 0.5, f * 1.8), (0.075, 0.9, f * 0.85)):
        out += a * _strike(n, sr, at + int(off * sr), (ff, ff * 1.55, ff * 2.4, 420.0), (0.02, 0.014, 0.008, 0.04),
                           (1, 0.5, 0.3, 0.6), rng, 0.0003, noise=0.5)
    if film:
        # wind-on lever: a ratchet of small clicks
        st = at + int(0.35 * sr)
        for j in range(14):
            out += 0.25 * _strike(n, sr, st + int(j * 0.016 * sr), (3200.0 * crng.uniform(0.95, 1.05), 5100.0), (0.006, 0.004), (1, 0.4), rng, 0.0002, noise=0.4)
    return out


def _camera_shutter(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    mono = np.zeros(n)
    if kind == 1:
        tm = 0.005
        while tm < dur - 0.12:
            mono += _shutter_once(n, sr, int(tm * sr), rng, crng)
            tm += 1 / 8.0
    else:
        mono += _shutter_once(n, sr, int(0.005 * sr), rng, crng, film=kind == 2)
    return mono, _decorrelate(mono, sr, crng, taps=3, span=0.006)


def _paper_rustle(ev, sr, rng, crng, p, seed):
    """Paper: many small crackles (the sheet buckling), denser while it is
    handled, plus a soft broadband swish as it moves."""
    n = _n(ev, sr)
    dur = n / sr
    crumple = seed % 2 == 1
    act = (0.25 + 0.75 * _smooth(crng, n, sr, 4.0 if crumple else 2.5)) * _fade_out(n, sr, 0.1)
    rate = 900.0 if crumple else 350.0

    def chan():
        cnt = int(rate * dur)
        tt = crng.uniform(0.0, dur, cnt)
        amps = crng.pareto(2.5, cnt) * np.interp(tt, np.arange(n) / sr, act)
        g = _grains(rng, n, tt, amps, sr)
        crack = _ring(np.cumsum(g) * 0.02, (crng.uniform(1500, 2500), crng.uniform(3500, 4800)), (0.004, 0.003), (1, 0.6), sr)
        crack = bandlimit(crack + 0.5 * g, sr, 600.0, 7000.0, order=4)
        swish = _wash(rng, n, sr, 500.0, 4500.0) * act ** 2
        return _norm(crack) + 0.35 * swish

    a = _attack(n, sr, 0.0, 0.03)
    return chan() * a, chan() * a


def _cork_pop(ev, sr, rng, crng, p, seed):
    """A cork leaving a bottle. While the cork slides out it squeaks against
    the glass (stick-slip, a few short rubbery chirps). When it clears the
    lip, the pressure difference equalises in one pulse (air rushing in
    behind a wine cork, gas rushing out of champagne) that sets the air in
    the neck and headspace ringing as a Helmholtz resonator. The opening
    grows as the cork leaves, so the pitch sweeps up over a few ms, and the
    open neck radiates so well that the ring dies in a couple of cycles:
    a hollow 'pop', not a struck tone. Champagne adds a sharper report, the
    hiss of the gas gushing out of the neck, and the fizz of bubbles
    bursting at the surface.
    """
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    champ = seed % 2 == 1
    t0 = min(0.3 * dur, crng.uniform(0.12, 0.2) if not champ else crng.uniform(0.03, 0.06))
    at = int(t0 * sr)
    tt = np.maximum(t - t0, 0.0)
    mono = np.zeros(n)

    if not champ:
        # the cork twisting out: two or three short rubbery squeaks
        for _ in range(int(crng.integers(2, 4))):
            s0 = crng.uniform(0.0, max(0.01, t0 - 0.06))
            L = crng.uniform(0.02, 0.05)
            k = int(L * sr)
            a0 = int(s0 * sr)
            if a0 + k >= n or k < 8:
                continue
            u = np.arange(k) / k
            rate = crng.uniform(250, 450) * (1.0 + 0.3 * u)
            ph = np.cumsum(rate) / sr
            pul = np.zeros(k)
            pul[np.nonzero(np.diff(np.floor(ph), prepend=0.0) > 0)[0]] = 1.0
            sq = np.zeros(n)
            sq[a0 : a0 + k] = pul * np.sin(np.pi * u) * crng.uniform(0.5, 1.0)
            mono += 0.12 * _norm(_fftconv(sq, _modes_ir((1400.0, 2300.0, 3300.0), (0.006, 0.004, 0.003), (1.0, 0.6, 0.3), sr, n), n))

    # the pulse: a smooth 0.5-1 ms push (a suck for wine), then the neck's
    # air ringing with Q ~3 while the pitch rises as the opening clears
    w = (0.0009, 0.0005)[champ]
    push = _pulse(n, at, max(2, int(w * sr))) * (-1.0 if not champ else 1.0)
    f0 = (crng.uniform(330, 480), crng.uniform(420, 600))[champ]
    f = f0 * (0.6 + 0.4 * (1.0 - np.exp(-tt / 0.004)))
    d = np.pi * f0 / 3.0
    ring = np.sin(TAU * np.cumsum(f) / sr) * np.exp(-d * tt) * (t >= t0) * _attack(n, sr, t0, 0.0004)
    pop = _norm(ring) + 0.5 * _norm(bandlimit(push, sr, 80.0, None, order=2))
    # the rim and the cork's edge: a short dull tick of air turbulence
    tick = bandlimit(_burst(rng, n, at, int(0.02 * sr), (0.002, 0.003)[champ], sr, rise=0.0004), sr, 300.0, 4000.0, order=2)
    mono += pop + (0.35, 0.7)[champ] * _norm(tick)

    if champ:
        # gas gushing out of the neck: a hiss that swells in 20 ms and fades
        # over half a second, darker as it slows
        g_env = (1.0 - np.exp(-tt / 0.02)) * np.exp(-tt / crng.uniform(0.3, 0.5)) * (t >= t0)
        gush = bandlimit(rng.standard_normal(n), sr, 1500.0, 7000.0, order=2) * g_env
        # the fizz: thousands of tiny bubbles bursting at the surface, random
        # clicks (each a small, very short burst) thinning out over seconds
        rate = 2500.0 * np.exp(-tt / 0.9) * (t >= t0 + 0.02)
        cnt = int(rate.sum() / sr) + 1
        times = rng.choice(n, size=cnt, p=rate / rate.sum()) / sr
        fz = _grains(rng, n, times, crng.uniform(0.1, 1.0, cnt) ** 2, sr)
        fizz = bandlimit(fz, sr, 2500.0, 8000.0, order=2)
        mono = 1.4 * mono + 0.22 * _norm(gush) + 0.18 * _norm(fizz)
    return mono, _decorrelate(mono, sr, crng)


def _coin(n, sr, at, rng, crng, a=1.0, size=1.0):
    f = crng.uniform(3400, 4400) / size
    freqs = f * np.array((1.0, 1.52, 2.13, 2.8))
    return a * _strike(n, sr, at, freqs, (0.25, 0.18, 0.12, 0.08), (1, 0.6, 0.4, 0.25), rng, 0.00015, noise=0.3)


def _coin_drop(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    mono = np.zeros(n)
    if kind == 0:
        times = 0.005 + _bounces(crng.uniform(0.16, 0.22), 0.62, 12, crng)
        for i, tm in enumerate(times):
            if tm < dur - 0.05:
                mono += _coin(n, sr, int(tm * sr), rng, crng, 0.75**i)
    elif kind == 1:
        # Euler's disk: contacts come faster and quieter as it settles
        times = [0.005]
        g = 0.09
        while times[-1] < dur - 0.1 and g > 0.004:
            times.append(times[-1] + g)
            g *= 0.93
        for i, tm in enumerate(times):
            if tm < dur - 0.03:
                mono += _coin(n, sr, int(tm * sr), rng, crng, 0.5 + 0.5 * (i % 2), 1.0)
        # the final shudder
        st = int(times[-1] * sr)
        mono += 0.3 * bandlimit(_burst(rng, n, st, int(0.08 * sr), 0.02, sr), sr, 2000.0, 7000.0, order=4)
    else:
        for _ in range(int(crng.integers(6, 11))):
            t0 = 0.005 + crng.gamma(1.5, 0.05)
            for i, tm in enumerate(t0 + _bounces(crng.uniform(0.04, 0.09), 0.55, 4, crng)):
                if tm < dur - 0.05:
                    mono += _coin(n, sr, int(tm * sr), rng, crng, crng.uniform(0.4, 1.0) * 0.6**i, crng.uniform(0.8, 1.3))
        dish = _strike(n, sr, int(0.006 * sr), (crng.uniform(700, 900), 1650.0, 2700.0), (0.4, 0.25, 0.15), (1, 0.5, 0.3), rng, 0.0005)
        mono += 0.4 * np.max(np.abs(mono)) * _norm(dish)
    return mono, _decorrelate(mono, sr, crng)


def _metal_clang(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    kind = seed % 3
    at = int(0.003 * sr)
    if kind == 0:  # a pipe: free-free bar modes (1, 2.76, 5.40, 8.93)
        f = crng.uniform(380, 620)
        freqs = f * np.array((1.0, 2.756, 5.404, 8.933, 13.34))
        t60 = np.array((2.2, 1.6, 1.0, 0.6, 0.35))
        g = (1.0, 0.8, 0.55, 0.35, 0.2)
    elif kind == 1:  # sheet / bin lid: dense, inharmonic, quicker
        f = crng.uniform(180, 300)
        freqs = f * np.sort(crng.uniform(1.0, 18.0, 14))
        t60 = crng.uniform(0.3, 0.9, 14)
        g = crng.uniform(0.3, 1.0, 14)
    else:  # anvil: a bright ping, long
        f = crng.uniform(900, 1200)
        freqs = f * np.array((1.0, 1.58, 2.33, 3.1, 4.4))
        t60 = np.array((2.8, 1.8, 1.1, 0.7, 0.4))
        g = (1.0, 0.6, 0.45, 0.3, 0.2)
    ring = _strike(n, sr, at, freqs * crng.uniform(0.99, 1.01, len(freqs)), t60, g, rng, 0.0004, noise=0.4)
    hit = bandlimit(_burst(rng, n, at, int(0.02 * sr), 0.003, sr, rise=0.0005), sr, 400.0, 8000.0, order=4)
    mono = _norm(ring) + 0.35 * _norm(hit)
    beat = 1.0 + 0.15 * np.sin(TAU * crng.uniform(1.5, 4.0) * np.arange(n) / sr)
    return mono * beat, _decorrelate(mono, sr, crng) * beat


def _wood_creak(ev, sr, rng, crng, p, seed):
    """Stick-slip friction: a train of tiny slips whose rate wanders (the
    pitch of the creak), each one ringing the wood's modes."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    rate0, spread = ((90, 140), (45, 70), (25, 60))[kind]
    t = np.arange(n) / sr
    act = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** (0.6 if kind != 1 else 1.2)
    if kind == 1:
        act = act * (0.3 + 0.7 * _smooth(crng, n, sr, 3.0))
    rate = rate0 + spread * _smooth(crng, n, sr, 1.5 if kind != 1 else 4.0)
    ph = np.cumsum(rate) / sr
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    amps = act[idx] * crng.uniform(0.6, 1.0, len(idx))
    slips = np.zeros(n)
    slips[idx] = amps
    f = crng.uniform(380, 520) * (0.7 if kind == 2 else 1.0)
    freqs = f * np.array((1.0, 1.9, 2.7, 3.8, 5.2))
    body = _fftconv(slips, _modes_ir(freqs, (0.04, 0.03, 0.025, 0.018, 0.012), (1, 0.7, 0.5, 0.35, 0.2), sr, n), n)
    grit = bandlimit(slips * rng.standard_normal(n), sr, 1500.0, 6000.0, order=2)
    mono = _norm(body) + 0.15 * _norm(grit)
    return mono, _decorrelate(mono, sr, crng)


def _heartbeat(ev, sr, rng, crng, p, seed):
    """Lub-dub: two low, soft thumps (valves closing), heard through the
    chest, the second shorter and a little higher."""
    n = _n(ev, sr)
    dur = n / sr
    bpm = (62.0, 88.0, 120.0)[seed % 3] * crng.uniform(0.97, 1.03)
    period = 60.0 / bpm
    split = 0.3 * period ** 0.5 * 0.55
    mono = np.zeros(n)
    tm = 0.02
    while tm < dur - 0.25:
        for off, f, d, a in ((0.0, crng.uniform(58, 68), 0.055, 1.0), (split, crng.uniform(78, 92), 0.04, 0.8)):
            at = int((tm + off) * sr)
            k = min(n - at, int(8 * d * sr))
            if k < 10:
                continue
            tt = np.arange(k) / sr
            # a damped thump with a slight downward pitch bend, a soft
            # attack (the chest wall filters the click), a little body
            fb = f * (1.0 + 0.25 * np.exp(-tt / 0.015))
            ph = TAU * np.cumsum(fb) / sr
            env = (1.0 - np.exp(-tt / 0.004)) * np.exp(-tt / d)
            y = (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3.1 * ph)) * env
            mono[at : at + k] += a * y
        tm += period * crng.uniform(0.98, 1.02)
    mono = bandlimit(mono, sr, 35.0, 900.0, order=2)
    return mono, mono


def _applause(ev, sr, rng, crng, p, seed):
    """Applause: N people clapping at their own rate (3-5 Hz), each clap a
    short burst whose colour depends on the hand shape; the crowd swells in
    and dies away."""
    n = _n(ev, sr)
    dur = n / sr
    people = (int(crng.integers(4, 8)), int(crng.integers(25, 40)), int(crng.integers(120, 160)))[seed % 3]
    start = min(0.6, 0.15 * dur)
    stop = max(start + 0.2, dur - min(1.5, 0.35 * dur))
    left = np.zeros(n)
    right = np.zeros(n)
    bands = ((700, 2400), (900, 3200), (1200, 4200), (500, 1800))
    clap_bank = []
    for b in bands:
        for _ in range(2):
            k = int(0.03 * sr)
            e = np.exp(-np.arange(k) / sr / crng.uniform(0.004, 0.009))
            e[: int(0.0008 * sr)] *= np.linspace(0, 1, int(0.0008 * sr))
            clap_bank.append(bandlimit(rng.standard_normal(k) * e, sr, b[0], b[1], order=2))
    for _ in range(people):
        rate = crng.uniform(3.0, 5.5)
        tm = crng.uniform(0.0, start + 0.4)
        stop_i = stop + crng.uniform(-0.5, 0.4)
        clap = clap_bank[int(crng.integers(len(clap_bank)))]
        level = crng.lognormal(0.0, 0.4) / np.sqrt(people)
        w = crng.uniform(0.0, 1.0)
        imp = np.zeros(n)
        while tm < min(stop_i, dur - 0.05):
            imp[int(tm * sr)] += level * crng.uniform(0.7, 1.0) * (1.0 if tm > 0.2 else tm / 0.2 + 0.1)
            tm += (1.0 / rate) * crng.uniform(0.9, 1.1)
        y = _fftconv(imp, clap, n)
        left += y * np.sqrt(1 - w)
        right += y * np.sqrt(w)
    fade = _fade_out(n, sr, min(1.0, 0.3 * dur))
    return left * fade, right * fade


def _pouring_water(ev, sr, rng, crng, p, seed):
    """A stream of water: turbulent splashing at the surface (bubbles at many
    sizes) and, into a glass, the air column above the water rising in pitch
    as it fills."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    glass = seed % 2 == 0
    env = _attack(n, sr, 0.0, 0.08) * _fade_out(n, sr, min(0.4, 0.2 * dur))
    bubbles = np.zeros(n)
    count = int((240 if glass else 160) * dur)
    for _ in range(count):
        tm = crng.uniform(0.0, dur - 0.03)
        fb = crng.uniform(600, 3000) if glass else crng.uniform(250, 1500)
        _bubble_into(bubbles, sr, int(tm * sr), fb, 1.2, 0.3, crng.uniform(0.2, 1.0))
    stream = _wash(rng, n, sr, 300.0, 4000.0)
    if glass:
        # the resonance of the shrinking air column: 300 Hz -> 1.5 kHz
        f = 300.0 * (5.0 ** (t / dur))
        res = np.zeros(n)
        noise = rng.standard_normal(n)
        blk = int(0.04 * sr)
        for s in range(0, n, blk):
            e = min(n, s + 2 * blk)
            c = float(f[min(n - 1, s + blk // 2)])
            res[s:e] += bandlimit(noise[s:e], sr, c * 0.9, c * 1.1, order=2) * np.hanning(e - s)
        mono = _norm(bubbles) + 0.35 * stream + 0.6 * _norm(res)
    else:
        mono = _norm(bubbles) + 0.5 * stream
    mono = mono * env * (0.85 + 0.15 * _smooth(crng, n, sr, 8.0))
    return mono, _decorrelate(mono, sr, crng)


def _can_opening(ev, sr, rng, crng, p, seed):
    """Pull tab: a click and crack of the scored lid, the burst of gas (a
    hiss), and the fizz that follows."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    beer = seed % 2 == 1
    at = int(0.02 * sr)
    tab = _strike(n, sr, at, (2200.0, 3500.0, 5200.0), (0.02, 0.012, 0.008), (1, 0.6, 0.3), rng, 0.0003, noise=0.6)
    crack_at = at + int(crng.uniform(0.05, 0.09) * sr)
    crack = _strike(n, sr, crack_at, (1800.0 * crng.uniform(0.9, 1.1), 2900.0, 4300.0, 900.0), (0.02, 0.014, 0.008, 0.03),
                    (1, 0.5, 0.3, 0.4), rng, 0.0002, noise=1.0)
    hiss_env = _attack(n, sr, crack_at / sr, 0.004) * np.exp(-np.maximum(t - crack_at / sr, 0) / (0.12 if beer else 0.07))
    hiss = _wash(rng, n, sr, 1800.0, 9000.0, order=4) * hiss_env
    cnt = int(300 * dur)
    tt = crack_at / sr + crng.exponential(dur * 0.4, cnt)
    fizz = bandlimit(_grains(rng, n, tt, np.exp(-tt / dur) * crng.uniform(0.3, 1.0, cnt), sr), sr, 2500.0, 8000.0, order=4)
    mono = 0.6 * _norm(tab) + _norm(crack) + (0.9 if beer else 0.7) * hiss + 0.25 * _norm(fizz)
    return mono, _decorrelate(mono, sr, crng, taps=3)


def _clock_alarm(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    mono = np.zeros(n)
    if seed % 2 == 0:
        # a hammer flicking between two bells, ~18 hits per second
        f1, f2 = crng.uniform(2300, 2700), crng.uniform(2800, 3200)
        rate = crng.uniform(16, 20)
        k = int(dur * rate)
        for i in range(k):
            at = int((0.005 + i / rate) * sr)
            if at >= n - int(0.02 * sr):
                break
            f = f1 if i % 2 == 0 else f2
            mono += crng.uniform(0.85, 1.0) * _strike(n, sr, at, (f, f * 2.32, f * 0.5), (0.25, 0.12, 0.3), (1.0, 0.35, 0.3), rng, 0.0003, noise=0.3)
    else:
        # digital: four short square-ish beeps at ~2.7 kHz, then a pause
        f = crng.uniform(2600, 2900)
        start = 0.005
        while start < dur - 0.1:
            for j in range(4):
                at = int((start + j * 0.12) * sr)
                k = int(0.07 * sr)
                if at + k >= n:
                    break
                tt = np.arange(k) / sr
                y = _harmonic(np.full(k, f), sr, (1.0, 0.0, 0.25, 0.0, 0.1))
                y *= np.sin(np.pi * np.clip(tt / 0.003, 0, 0.5)) * np.cos(0.5 * np.pi * np.clip((tt - 0.066) / 0.004, 0, 1))
                mono[at : at + k] += y
            start += 1.0
    return mono, _decorrelate(mono, sr, crng, taps=3)


# =============================================================================
# Action and game
# =============================================================================


def _punch_reso(x, sr, f, q):
    """`x` through one damped resonance at `f` Hz with quality `q` (a body
    cavity, a skin patch): its impulse response convolved by FFT."""
    n = len(x)
    d = np.pi * f / q  # amplitude decay rate, 1/s
    k = min(n, int(7.0 / d * sr) + 2)
    t = np.arange(k) / sr
    ir = np.exp(-d * t) * np.sin(TAU * f * t)
    return _fftconv(x, ir, n)


def _punch(ev, sr, rng, crng, p, seed):
    """A fist landing, the way film foley records it: the swish of the arm
    and sleeve coming in, the skin slap of the knuckles (a sharp, bright,
    mid-heavy crack), and the thud of the mass behind it, a low, heavily
    damped boom of the torso or skull that is over in a tenth of a second.
    A body punch is mostly thud; a face punch is mostly slap with the crunch
    of bone and cartilage; the heavy movie punch is a long whoosh into a
    bigger slap and a deep thud.
    """
    n = _n(ev, sr)
    t = np.arange(n) / sr
    kind = seed % 3  # 0 body, 1 face, 2 heavy movie punch
    t0 = (crng.uniform(0.07, 0.1), crng.uniform(0.06, 0.09), crng.uniform(0.17, 0.22))[kind]
    t0 = min(t0, 0.3 * n / sr)
    at = int(t0 * sr)

    # 1. the swish: air and cloth torn past by the arm, a noise band whose
    # centre climbs as the fist speeds up, cut off by the hit
    u = np.clip(t / t0, 0.0, 1.0)
    swish_env = np.sin(0.5 * np.pi * u) ** 3 * (t < t0)
    noise = rng.standard_normal(n)
    sw_lo = bandlimit(noise, sr, 250.0, 900.0, order=2)
    sw_hi = bandlimit(noise, sr, 900.0, 3500.0, order=2)
    swish = (sw_lo * (1.0 - u) + sw_hi * u) * swish_env

    # 2. the slap: the knuckles meeting skin, a very short broadband crack
    # coloured by the skin patch (a broad 1-2.5 kHz hump) and a lower smack
    tau_s = (0.006, 0.004, 0.007)[kind]
    crack = _burst(rng, n, at, int(0.05 * sr), tau_s, sr, rise=0.0003)
    slap = (bandlimit(crack, sr, 500.0, 7000.0, order=2)
            + 0.8 * _punch_reso(crack, sr, crng.uniform(1100, 2400), 3.0)
            + 0.5 * _punch_reso(crack, sr, crng.uniform(450, 750), 2.5))

    # 3. the thud: the force of the hit (a 4-8 ms push) into the body, a low
    # resonance with Q ~2 plus the mass's broadband low knock
    width = (0.007, 0.004, 0.009)[kind]
    push = _pulse(n, at, int(width * sr))
    f_body = (crng.uniform(85, 120), crng.uniform(150, 210), crng.uniform(60, 85))[kind]
    thud = _punch_reso(push, sr, f_body, 1.8) + 0.6 * _punch_reso(push, sr, f_body * 2.3, 2.5)
    knock = bandlimit(_burst(rng, n, at, int(0.12 * sr), (0.025, 0.015, 0.04)[kind], sr, rise=0.001), sr, 40.0, 400.0, order=2)
    thud = _norm(thud) + 0.6 * _norm(knock)

    lv = ((1.0, 0.6, 0.15), (0.7, 1.0, 0.12), (1.1, 0.9, 0.3))[kind]  # thud, slap, swish
    mono = lv[0] * thud + lv[1] * _norm(slap) + lv[2] * _norm(swish)
    if kind == 1:
        # bone and cartilage: a few tiny cracks in the first 20 ms
        cnt = int(crng.integers(5, 12))
        cr = _grains(rng, n, t0 + crng.gamma(1.5, 0.004, cnt), crng.uniform(0.3, 1.0, cnt), sr)
        mono += 0.25 * _norm(bandlimit(cr, sr, 1200.0, 5000.0, order=2))
    # the cloth and the body settle: a short soft rustle after the hit
    settle = bandlimit(_burst(rng, n, at + int(0.01 * sr), int(0.15 * sr), 0.04, sr, rise=0.01), sr, 400.0, 3000.0, order=2)
    mono += 0.06 * _norm(settle)
    mono = saturate(_norm(mono) * (1.2, 1.1, 1.6)[kind], 1.3)
    return mono, _decorrelate(mono, sr, crng, taps=3)


def _blade(n, sr, at, rng, crng):
    f = crng.uniform(1100, 1700)
    freqs = f * np.array((1.0, 1.37, 2.13, 2.71, 3.66, 4.9))
    t60 = np.array((1.2, 0.9, 0.7, 0.5, 0.35, 0.25)) * crng.uniform(0.8, 1.2)
    return _strike(n, sr, at, freqs, t60, (1.0, 0.8, 0.6, 0.45, 0.3, 0.2), rng, 0.0002, noise=0.7)


def _sword_clash(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    if kind == 1:
        # drawn from a sheath: a rising scrape ("shing") ending in a ring
        L = min(0.45, 0.6 * dur)
        u = np.clip(t / L, 0, 1)
        scrape = np.zeros(n)
        noise = rng.standard_normal(n)
        blk = int(0.03 * sr)
        for s in range(0, n, blk):
            e = min(n, s + 2 * blk)
            c = 1800 + 2600 * float(u[s])
            scrape[s:e] += bandlimit(noise[s:e], sr, c * 0.8, c * 1.2, order=2) * np.hanning(e - s)
        env = np.sin(0.5 * np.pi * u) ** 2 * np.exp(-np.maximum(t - L, 0) / 0.05)
        ring = _blade(n, sr, int(L * sr), rng, crng)
        mono = 0.8 * _norm(scrape) * env + 0.7 * _norm(ring)
    else:
        hits = (0.004,) if kind == 0 else (0.004, crng.uniform(0.18, 0.3))
        mono = np.zeros(n)
        for h in hits:
            at = int(h * sr)
            ring = _blade(n, sr, at, rng, crng)
            hit = bandlimit(_burst(rng, n, at, int(0.03 * sr), 0.004, sr, rise=0.0005), sr, 1000.0, 9000.0, order=4)
            grind = bandlimit(_burst(rng, n, at, int(0.15 * sr), 0.04, sr), sr, 2500.0, 7000.0, order=4)
            mono += _norm(ring) + 0.5 * _norm(hit) + 0.2 * _norm(grind)
    return mono, _decorrelate(mono, sr, crng)


def _gun_nwave(n, sr, at, length):
    """A pressure N-wave: a near-instant rise (0.1 ms), a linear fall through
    zero to the same negative peak over `length` s, and a 0.1 ms return."""
    out = np.zeros(n)
    k = max(4, int(length * sr))
    r = max(2, int(0.0001 * sr))
    if at + k + 2 * r >= n:
        return out
    up = 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    out[at : at + r] = up
    out[at + r : at + r + k] = np.linspace(1.0, -1.0, k)
    out[at + r + k : at + 2 * r + k] = -up[::-1]
    return out


def _gun_blast(n, sr, at, length, decay):
    """The muzzle blast: a Friedlander pulse (instant rise, decay through a
    longer, shallower negative phase) of positive phase `length` s."""
    out = np.zeros(n)
    k = min(n - at, int((length * 6 + 0.002) * sr))
    if k < 8:
        return out
    t = np.arange(k) / sr
    r = max(2, int(0.00012 * sr))
    w = (1.0 - t / length) * np.exp(-decay * t / length)
    w[:r] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    out[at : at + k] = w
    return out


def _gunshot(ev, sr, rng, crng, p, seed):
    """A real shot, recorded outdoors. The muzzle blast is a Friedlander
    pressure pulse (a 0.1 ms rise, a 1-2 ms positive phase, a longer negative
    one) with the turbulent roar of the propellant gas behind it; a rifle's
    supersonic bullet adds its own sharp N-wave crack just ahead. The ground
    answers within a few ms, then the surroundings: a handful of discrete,
    darkened echoes (buildings, a treeline) over half a second and a diffuse
    rolling tail that dies over 1-2 s. A distant shot loses its highs to the
    air, so the crack is gone, the blast is a dull thud and the echoes and
    the roll carry most of the energy."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3  # 0 pistol, 1 rifle, 2 distant
    t0 = 0.004
    at = int(t0 * sr)

    T = (crng.uniform(0.0011, 0.0016), crng.uniform(0.0016, 0.0024), crng.uniform(0.002, 0.003))[kind]
    blast = _gun_blast(n, sr, at, T, 2.2)
    # propellant gas: a roar of turbulent noise, loudest in the first 10 ms
    tau = (0.012, 0.018, 0.02)[kind]
    gas = bandlimit(_burst(rng, n, at, int(0.25 * sr), tau, sr, rise=0.0003), sr, 180.0, 6500.0, order=2)
    src = _norm(blast) + (0.55, 0.5, 0.45)[kind] * _norm(gas)
    if kind == 1:
        # the bullet's crack arrives ahead of the muzzle blast
        lead = int(crng.uniform(0.0015, 0.004) * sr)
        crack = _gun_nwave(n, sr, at, crng.uniform(0.00025, 0.0004))
        src = np.roll(src, lead) + 0.8 * crack
    if kind == 2:
        src = bandlimit(src, sr, 60.0, crng.uniform(1100.0, 1600.0), order=2)
    # the mic and the preamp take the hit: a hard, slightly asymmetric clip
    src = saturate(_norm(src) * (2.2, 2.6, 1.3)[kind], 2.0)
    # the ground reflection, a few ms behind and a little darker
    g = int(crng.uniform(0.002, 0.006) * sr)
    src = src + 0.5 * np.concatenate((np.zeros(g), bandlimit(src, sr, None, 4000.0, order=2)[: n - g]))

    # The surroundings, per ear. Discrete echoes from big reflectors (each
    # lower-passed by distance), and a diffuse roll: noise shaped by a slow
    # swell and an exponential decay, low-passed the way air takes the highs
    # off a long path.
    T60 = (0.8, 1.0, 1.6)[kind] * crng.uniform(0.9, 1.1)
    roll_lp = (2600.0, 2200.0, 1100.0)[kind]
    roll_lvl = (0.1, 0.12, 0.35)[kind]

    def space(r):
        er = np.zeros(n)
        k_echo = int(crng.integers(4, 7))
        for i in range(k_echo):
            d = crng.uniform(0.04, 0.55 if kind != 2 else 0.9)
            dd = int(d * sr)
            if dd < n:
                er[dd:] += crng.uniform(0.12, 0.3) * np.exp(-d / 0.4) * src[: n - dd]
        er = bandlimit(er, sr, 90.0, 2400.0 if kind != 2 else 1000.0, order=2)
        env = (1.0 - np.exp(-np.maximum(t - t0, 0) / (0.03 if kind != 2 else 0.08))) * np.exp(-LN1000 * np.maximum(t - t0, 0) / T60)
        roll = bandlimit(r.standard_normal(n), sr, 70.0, roll_lp, order=2) * env
        roll *= 0.7 + 0.3 * _smooth(crng, n, sr, 6.0)
        return er + roll_lvl * _norm(roll)

    direct = src * (1.0 if kind != 2 else 0.6)
    p["reverb"], p["reverb_decay"] = (0.15, 0.18, 0.3)[kind], (1.0, 1.2, 1.8)[kind]
    fade = _fade_out(n, sr, min(0.25, 0.3 * dur))
    return (direct + space(rng)) * fade, (direct + space(crng)) * fade


def _car_horn(ev, sr, rng, crng, p, seed):
    """Two diaphragm horns a third apart (around 400 and 500 Hz): a buzzy,
    harmonic-rich tone through the horn's flare."""
    n = _n(ev, sr)
    dur = n / sr
    kind = seed % 3
    f1 = crng.uniform(390, 430)
    f2 = f1 * crng.uniform(1.2, 1.26)
    honks = {0: [(0.01, min(0.45, dur - 0.05))], 1: [(0.01, 0.16), (0.27, 0.22)], 2: [(0.01, dur - 0.08)]}[kind]
    mono = np.zeros(n)
    amps = [1.0 / k ** 0.9 for k in range(1, 18)]
    for st, L in honks:
        at = int(st * sr)
        k = int(max(0.05, L) * sr)
        if at + k >= n:
            k = n - at - 1
        if k < 100:
            continue
        tt = np.arange(k) / sr
        wob = 1.0 + 0.002 * np.sin(TAU * 6 * tt)
        y = _harmonic(f1 * wob, sr, amps) + 0.9 * _harmonic(f2 * wob, sr, amps, phase=1.0)
        env = np.clip(tt / 0.012, 0, 1) * np.clip((L - tt) / 0.02, 0, 1)
        mono[at : at + k] += y * env
    # the flare: a broad resonance around 2-3 kHz, little below 300 Hz
    mono = bandlimit(mono, sr, 300.0, 4500.0, order=2) + 0.6 * bandlimit(mono, sr, 1800.0, 3200.0, order=2)
    mono = saturate(_norm(mono) * 1.2, 1.3)
    p["reverb"], p["reverb_decay"] = 0.18, 0.9
    return mono, _decorrelate(mono, sr, crng, span=0.03)


def _siren(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    if kind == 0:  # wail: slow rise and fall
        period = crng.uniform(3.5, 4.5)
        u = (t % period) / period
        f = 650 + 750 * np.where(u < 0.45, np.sin(0.5 * np.pi * u / 0.45), np.cos(0.5 * np.pi * (u - 0.45) / 0.55)) ** 1.2
    elif kind == 1:  # yelp: fast sweeps
        u = (t * crng.uniform(3.0, 3.6)) % 1.0
        f = 700 + 800 * np.sin(np.pi * u) ** 0.8
    else:  # two-tone hi-lo
        f = np.where((t * 1.0 / 0.55).astype(int) % 2 == 0, 450.0, 600.0)
        f = np.convolve(f, np.ones(int(0.01 * sr)) / int(0.01 * sr), mode="same")
    amps = (1.0, 0.0, 0.33, 0.0, 0.2, 0.0, 0.14, 0.0, 0.1)
    y = _harmonic(f, sr, amps)
    y = bandlimit(y, sr, 300.0, 5000.0, order=2)
    env = _attack(n, sr, 0.0, 0.15) * _fade_out(n, sr, min(0.5, 0.2 * dur))
    # a slow fade in level, as if passing
    mono = y * env * (0.8 + 0.2 * _smooth(crng, n, sr, 0.4))
    p["reverb"], p["reverb_decay"] = 0.25, 1.3
    return mono, _decorrelate(mono, sr, crng, span=0.04)


def _engine(ev, sr, rng, crng, p, seed):
    """A four-stroke engine: the firing rate (rpm/30 per cylinder pair) sets
    a train of low exhaust pulses through a muffler (resonant, low), with
    mechanical noise on top."""
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    if kind == 0:
        rpm = 800 * (1 + 0.03 * (_smooth(crng, n, sr, 2.0) - 0.5))
    elif kind == 1:
        u = t / dur
        rpm = 900 + 3200 * np.clip(np.sin(np.pi * np.clip(u * 1.2, 0, 1)) ** 1.5, 0, 1)
    else:
        rpm = 2200 * (1 + 0.05 * _smooth(crng, n, sr, 1.0))
    fire = rpm / 60.0 * 2.0  # 4 cylinders: 2 firings per revolution
    if kind == 2:
        # passing: doppler shift and a level swell
        tc = dur * 0.5
        v = 18.0
        dist = np.sqrt((v * (t - tc)) ** 2 + 6.0**2)
        radial = v**2 * (t - tc) / dist
        fire = fire * 343.0 / (343.0 + radial)
        level = 6.0 / dist
    else:
        level = np.ones(n)
    ph = np.cumsum(fire) / sr
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    pulses = np.zeros(n)
    pulses[idx] = crng.uniform(0.75, 1.0, len(idx))
    exhaust = _fftconv(pulses, _modes_ir((crng.uniform(90, 120), 240.0, 410.0, 680.0, 1100.0), (0.05, 0.035, 0.025, 0.018, 0.012),
                                         (1, 0.8, 0.6, 0.4, 0.25), sr, n), n)
    mech = _wash(rng, n, sr, 700.0, 4000.0) * (0.4 + 0.6 * np.clip((rpm - 700) / 3000, 0, 1))
    rumble = _wash(rng, n, sr, 30.0, 150.0)
    mono = (_norm(exhaust) + 0.2 * mech + 0.2 * rumble) * level
    mono = mono * _attack(n, sr, 0.0, 0.1) * _fade_out(n, sr, min(0.4, 0.2 * dur))
    return mono, _decorrelate(mono, sr, crng, span=0.02)


def _church_bell(ev, sr, rng, crng, p, seed):
    """A tower bell: the classic partials (hum 0.5, prime 1, tierce 1.2,
    quint 1.5, nominal 2, upper partials), the hum ringing longest, a little
    beating from the bell's asymmetry."""
    n = _n(ev, sr)
    dur = n / sr
    f = crng.uniform(380, 520)
    ratios = np.array((0.5, 1.0, 1.183, 1.506, 2.0, 2.5, 2.66, 3.01, 4.0))
    t60 = np.array((7.0, 5.0, 4.0, 3.0, 2.6, 1.6, 1.4, 1.1, 0.7))
    g = (0.4, 0.8, 0.9, 0.4, 1.0, 0.5, 0.45, 0.35, 0.25)
    strikes = [0.005] if seed % 2 == 0 else list(np.arange(0.005, max(0.01, dur - 0.5), 2.5))
    left = np.zeros(n)
    right = np.zeros(n)
    for st in strikes:
        at = int(st * sr)
        for chan in (left, right):
            fr = f * ratios * crng.uniform(0.998, 1.002, len(ratios))
            chan += _strike(n, sr, at, fr, t60, g, rng, 0.001, noise=0.2)
    clang = bandlimit(np.sum([_burst(rng, n, int(s * sr), int(0.02 * sr), 0.004, sr) for s in strikes], axis=0), sr, 600.0, 5000.0, order=2)
    c = 0.15 * _norm(clang)
    fo = _fade_out(n, sr, min(1.5, 0.3 * dur))
    return (_norm(left) + c) * fo, (_norm(right) + c) * fo


def _fireworks(ev, sr, rng, crng, p, seed):
    n = _n(ev, sr)
    dur = n / sr
    t = np.arange(n) / sr
    kind = seed % 3
    left = np.zeros(n)
    right = np.zeros(n)

    def shell(t_launch, whistle, crackle, level, far):
        nonlocal left, right
        flight = crng.uniform(0.9, 1.5)
        t_burst = t_launch + flight
        out = np.zeros(n)
        if whistle and not far:
            u = np.clip((t - t_launch) / flight, 0, 1)
            f = 900 + 1800 * u
            w = np.sin(TAU * np.cumsum(f) / sr) * np.sin(np.pi * u) ** 0.5 * (t >= t_launch) * (t < t_burst)
            out += 0.2 * w
        if t_burst < dur - 0.1:
            b = int(t_burst * sr)
            boom = bandlimit(_burst(rng, n, b, int(1.2 * sr), 0.18 if not far else 0.3, sr, rise=0.003 if not far else 0.02), sr, 35.0, 800.0 if not far else 300.0, order=2)
            crack = bandlimit(_burst(rng, n, b, int(0.05 * sr), 0.006, sr, rise=0.0006), sr, 500.0, 6000.0, order=4)
            out += _norm(boom) + (0.0 if far else 0.5) * _norm(crack)
            if crackle:
                cnt = 220
                tt = t_burst + 0.25 + crng.gamma(2.0, 0.35, cnt)
                cr = bandlimit(_grains(rng, n, tt, crng.uniform(0.3, 1.0, cnt) * np.exp(-(tt - t_burst) / 2.0), sr), sr, 1200.0, 7000.0, order=4)
                out += 0.5 * _norm(cr)
        w = crng.uniform(0.2, 0.8)
        left += level * out * np.sqrt(1 - w)
        right += level * out * np.sqrt(w)

    if kind == 2:
        tm = 0.0
        while tm < dur - 1.6:
            shell(tm, False, crng.random() < 0.3, crng.uniform(0.4, 1.0), True)
            tm += crng.uniform(0.4, 1.4)
        p["reverb"], p["reverb_decay"] = 0.35, 2.2
    else:
        shell(0.0, True, kind == 1, 1.0, False)
    fo = _fade_out(n, sr, min(0.8, 0.25 * dur))
    return left * fo, right * fo


FNS = {
    "bird_chirp": _bird_chirp,
    "crickets": _crickets,
    "wave_crash": _wave_crash,
    "wind_gust": _wind_gust,
    "splash": _splash,
    "frog": _frog,
    "owl": _owl,
    "glass_break": _glass_break,
    "clock_tick": _clock_tick,
    "phone_ring": _phone_ring,
    "keyboard_typing": _keyboard_typing,
    "mouse_click": _mouse_click,
    "camera_shutter": _camera_shutter,
    "paper_rustle": _paper_rustle,
    "cork_pop": _cork_pop,
    "coin_drop": _coin_drop,
    "metal_clang": _metal_clang,
    "wood_creak": _wood_creak,
    "heartbeat": _heartbeat,
    "applause": _applause,
    "pouring_water": _pouring_water,
    "can_opening": _can_opening,
    "clock_alarm": _clock_alarm,
    "punch": _punch,
    "sword_clash": _sword_clash,
    "gunshot": _gunshot,
    "car_horn": _car_horn,
    "siren": _siren,
    "engine": _engine,
    "church_bell": _church_bell,
    "fireworks": _fireworks,
}

assert set(FNS) == set(ROLES) == set(VARIANTS) == set(PARAMS)
