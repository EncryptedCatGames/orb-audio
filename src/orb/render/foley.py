"""Natural and foley roles: thunder, door_slam, knock, footsteps, water_drop,
explosion.

Each one is a small physical model in numpy, like the roles in `events.py`,
but built from the physics of the real sound rather than from sound-design
primitives:

* thunder: a close crack (a few stacked micro-transients plus crackle) and a
  rolling rumble of low-passed noise driven by a train of arrivals from
  different parts of the channel, decaying over seconds.
* door_slam: an air-pressure whump (40-90 Hz), the panel's thick low modes
  (70-180 Hz) and a damped mid knock, the frame thump, a quiet latch tick, and
  a room (early reflections and a short dark tail).
* knock: 2-3 knuckle raps exciting the damped modes of a wooden door.
* footsteps: heel-toe steps across `dur`, with a varied cadence.
* water_drop: a Minnaert bubble whose pitch glides up as it rises.
* explosion: a blast transient, a sub body, a rumble and a debris crackle.

The event `seed` picks the variant deterministically (documented per role in
`VARIANTS`) and also seeds the noise; the event time adds the fine noise, the
same way the other roles do. Unlike the wide roles in `events.py`, `pan` always
applies: a stereo source is narrowed as it is panned, then placed with the
same constant-power law as the mono roles.

Every clip is zero outside `[0, dur]` plus its reverb tail, starts and ends at
exactly zero, and is high-passed at 20 Hz and low-passed (flat to 9 kHz, a
smooth slope to nothing at 12 kHz) after every envelope, attack and reverb, so
it carries no DC, no subsonic offset and no fizz to Nyquist. Every noise
attack opens over at least a millisecond.
"""

from __future__ import annotations

import numpy as np

from ..score import EVENT_ROLES, Event
from .dsp import LN1000, TAU, bandlimit, db_to_gain, pan, reverb, saturate

ROLES = ("thunder", "door_slam", "knock", "footsteps", "water_drop", "explosion")

# What the seed selects. Kept as data so the docs, the MCP tool description
# and the tests all read the same table.
VARIANTS = {
    "thunder": "seed % 4 = distance: 0 close crack, 1 near, 2 mid, 3 far rumble",
    "door_slam": ("seed % 3 = door: 0 heavy, 1 medium, 2 light interior; (seed // 3) % 3 = room: "
                  "0 small room, 1 hallway, 2 large room; seed % 2 = 1 adds a latch rattle; "
                  "(seed // 9) % 2 = 1 adds the swing's air before the hit"),
    "knock": "seed % 2 = raps: 0 two, 1 three; (seed // 2) % 3 = door size",
    "footsteps": "seed % 3 = surface: 0 hard floor, 1 stone, 2 gravel; (seed // 3) % 3 = pace: slow, walk, brisk",
    "water_drop": "seed % 2 = 0 drip (small, high), 1 plop (larger bubble, splash, second bubble)",
    "explosion": "seed % 3 = 0 close and big, 1 medium, 2 distant",
}

FOLEY_PARAMS: dict[str, dict[str, float]] = {
    "thunder": dict(width=0.8, reverb=0.0),
    "door_slam": dict(f0=95.0, width=0.35, reverb=0.28, reverb_decay=0.8),
    "knock": dict(f0=185.0, width=0.25, reverb=0.16, reverb_decay=0.5),
    "footsteps": dict(f0=190.0, width=0.2, reverb=0.14, reverb_decay=0.45),
    "water_drop": dict(width=0.3, reverb=0.3, reverb_decay=1.1),
    "explosion": dict(f0=60.0, width=0.7, reverb=0.18, reverb_decay=1.4),
}

HI = 12000.0  # nothing above this in any foley clip (matches events.ATTACK_HI)
HI_PASS = 9000.0  # flat up to here, then a smooth slope to zero at HI
LO = 20.0  # high-pass: no DC, no subsonic drift


# -- helpers -----------------------------------------------------------------


def _fftconv(a: np.ndarray, b: np.ndarray, n_out: int) -> np.ndarray:
    m = len(a) + len(b) - 1
    nfft = 1 << (m - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)
    return y[:n_out]


def _modes_ir(freqs, t60s, gains, sr: int, n_max: int) -> np.ndarray:
    """Impulse response of a bank of damped modes (a struck object)."""
    n = int(min(n_max, max(16, 1.1 * max(t60s) * sr)))
    t = np.arange(n) / sr
    ir = np.zeros(n)
    for f, d, g in zip(freqs, t60s, gains):
        if 20.0 < f < 0.45 * sr:
            ir += g * np.exp(-LN1000 * t / max(d, 1e-3)) * np.sin(TAU * f * t)
    return ir


def _ring(exc: np.ndarray, freqs, t60s, gains, sr: int) -> np.ndarray:
    """Excite a mode bank with the force `exc`; same length as `exc`.

    The mode bank is a displacement response; a vibrating surface radiates
    pressure in proportion to its acceleration, so the force is differentiated
    twice (12 dB/octave below each mode, as for a real plate). Without this a
    slow force pulse (a heel, a slammed door) leaks its DC content through
    every mode as a sub-bass thud far below the object's lowest mode.
    """
    acc = np.diff(np.diff(exc, prepend=0.0), prepend=0.0)
    return _fftconv(acc, _modes_ir(freqs, t60s, gains, sr, len(exc)), len(exc))


def _burst(
    rng: np.random.Generator, n: int, at: int, length: int, tau: float, sr: int, rise: float = 0.001
) -> np.ndarray:
    """A white-noise burst starting at sample `at`, exponential decay `tau` s.

    The burst opens with a raised-cosine `rise` (1 ms by default) rather than
    a step: a step onset is a full-band click that no later band limit can
    take back out of the attack."""
    out = np.zeros(n)
    if at >= n:
        return out
    k = max(1, min(length, n - at))
    t = np.arange(k) / sr
    env = np.exp(-t / max(tau, 1e-4))
    r = min(k, max(1, int(rise * sr)))
    env[:r] *= 0.5 - 0.5 * np.cos(np.pi * (np.arange(r) + 0.5) / r)
    out[at : at + k] = rng.standard_normal(k) * env
    return out


def _pulse(n: int, at: int, width: int) -> np.ndarray:
    """A half-sine force pulse (a knuckle or a heel meeting a surface)."""
    out = np.zeros(n)
    if at >= n:
        return out
    k = max(2, min(width, n - at))
    out[at : at + k] = np.sin(np.pi * (np.arange(k) + 0.5) / k)
    return out


def _smooth(rng: np.random.Generator, n: int, sr: int, rate: float) -> np.ndarray:
    """A smooth random curve in [0, 1] with about `rate` turns per second."""
    k = max(2, int(n / sr * rate) + 2)
    pts = rng.uniform(0.0, 1.0, k)
    x = np.linspace(0.0, k - 1.0, n)
    i = np.minimum(np.floor(x).astype(int), k - 2)
    w = 0.5 - 0.5 * np.cos(np.pi * (x - i))
    return pts[i] * (1.0 - w) + pts[i + 1] * w


def _fade_out(n: int, sr: int, length: float) -> np.ndarray:
    """1 everywhere, then a raised-cosine to exactly 0 over the last `length` s."""
    env = np.ones(n)
    b = min(n, max(2, int(length * sr)))
    env[-b:] = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, b))
    return env


def _norm(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x if m < 1e-12 else x / m


def _attack(n: int, sr: int, at: float, rise: float) -> np.ndarray:
    """0 before `at`, raised-cosine rise over `rise` s, then 1."""
    t = np.arange(n) / sr
    u = np.clip((t - at) / max(rise, 1e-4), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


# -- thunder -----------------------------------------------------------------


def _thunder(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    n = max(1, int(ev.dur * sr))
    dur = n / sr
    t = np.arange(n) / sr
    # Distance: the seed's low two bits, with a little jitter inside the step.
    dist = float(np.clip((seed % 4) / 3.0 + crng.uniform(-0.06, 0.06), 0.0, 1.0))

    # Rolling rumble. Sound from different parts of the channel arrives over
    # the first part of the event; each arrival is a swell of low noise that
    # decays over seconds. High frequencies die first (air absorption), so the
    # mid band has its own, faster decays.
    onset = 0.02 + 0.30 * dist  # a far strike has no sharp front
    spread = (0.12 + 0.12 * dist) * dur
    k = int(crng.integers(6, 12))
    arrivals = np.sort(onset + crng.exponential(spread, k))
    arrivals[0] = onset
    amps = crng.lognormal(0.0, 0.5, k) * np.exp(-(arrivals - onset) / max(0.35 * dur, 1e-3))
    amps[0] = max(amps[0], float(amps.max()))
    decay = crng.uniform(0.06, 0.16, k) * dur  # e-folding time of each roll
    rise = crng.uniform(0.03, 0.15, k) + 0.3 * dist
    env_lo = np.zeros(n)
    env_mid = np.zeros(n)
    for a, at, d, r in zip(amps, arrivals, decay, rise):
        if at >= dur:
            continue
        env_lo += a * _attack(n, sr, at, r) * np.exp(-np.maximum(t - at, 0.0) / d)
        env_mid += a * _attack(n, sr, at, r * 0.6) * np.exp(-np.maximum(t - at, 0.0) / (0.5 * d))
    # the whole roll dies away over the event, with slow random swells
    overall = np.exp(-np.maximum(t - onset, 0.0) / max(0.33 * dur, 0.3))
    swell = 0.3 + 0.7 * _smooth(crng, n, sr, 1.3)
    tail = _fade_out(n, sr, min(0.3 * dur, 1.5))
    env_lo *= overall * swell * tail
    env_mid *= overall * swell * tail

    # Thunder's power peaks well below 100 Hz, but most of that is below what
    # a speaker (or an ESC-50 recording) carries; the audible rumble is 40-250
    # Hz. Far thunder keeps only the lowest of it (air absorption).
    fc_lo = 230.0 - 100.0 * dist
    fc_mid = 300.0 + 700.0 * (1.0 - dist)
    mid_gain = 0.6 * (1.0 - 0.7 * dist)

    def rumble() -> np.ndarray:
        lo = bandlimit(rng.standard_normal(n), sr, 38.0, fc_lo, order=2)
        mid = bandlimit(rng.standard_normal(n), sr, 90.0, fc_mid, order=2)
        return _norm(lo) * env_lo + mid_gain * _norm(mid) * env_mid

    left, right = rumble(), rumble()
    peak = max(float(np.abs(left).max()), float(np.abs(right).max()), 1e-12)
    left, right = left / peak, right / peak

    # The close crack: a few fast, stacked micro-transients (the clap) and a
    # tearing crackle over the first few hundred ms. Gone beyond ~mid range.
    crack_amp = float(np.clip(1.0 - dist / 0.7, 0.0, 1.0))
    if crack_amp > 0.0:
        crack = np.zeros(n)
        m = int(crng.integers(3, 8))
        times = np.sort(crng.uniform(0.0, 0.10, m))
        times[0] = 0.002
        for i, tm in enumerate(times):
            at = int(tm * sr)
            length = int(crng.uniform(0.002, 0.008) * sr)
            a = (1.0 if i == 0 else crng.uniform(0.35, 0.9))
            crack += a * _burst(rng, n, at, 6 * length, length / sr, sr)
        crack = bandlimit(crack, sr, 350.0, 9000.0, order=4)
        # crackle: sparse impulses, dense at first, thinning out
        span = min(dur, 0.35 + 0.8 * (1.0 - dist))
        rate = 320.0
        cnt = int(rng.poisson(rate * span * 0.4))
        cr = np.zeros(n)
        if cnt:
            tt = rng.exponential(span * 0.35, cnt)
            tt = tt[tt < span]
            idx = (tt * sr).astype(int)
            np.add.at(cr, idx, rng.standard_normal(len(idx)) * np.exp(-tt / (span * 0.5)))
        cr = bandlimit(cr, sr, 900.0, 7000.0, order=4)
        crack = _norm(crack) + 0.6 * _norm(cr)
        crack = _norm(crack) * crack_amp
        # the clap also kicks the low end: a short boom under the crack
        boom = bandlimit(_burst(rng, n, 0, int(0.25 * sr), 0.06, sr), sr, 45.0, 200.0, order=2)
        crack += 0.6 * crack_amp * _norm(boom)
        rumble_level = 0.5 + 0.3 * dist
        left = crack + rumble_level * left
        right = crack + rumble_level * right
    level = 1.0 - 0.35 * dist  # distance costs level
    return left * level, right * level


# -- door slam ---------------------------------------------------------------


DOOR_WHUMP, DOOR_LO, DOOR_MID, DOOR_FRAME = 0.4, 0.6, 1.4, 0.9


def _door_slam(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    """A door slammed in a room: mostly low and mid, a wooden WHUMP.

    Five parts, loudest first: the air-pressure whump (the door shoves the
    air in the gap and stops dead, a 40-90 Hz push), the panel's thick low
    modes (70-180 Hz) and a damped mid knock (250-700 Hz), the frame and wall
    taking the hit (short, below 3 kHz), a quiet latch tick 12-18 dB under
    the body, and the room: early reflections here and a short, dark diffuse
    tail from the shared reverb (set per room below). Some seeds add the
    faint air of the swing before the hit.
    """
    n = max(1, int(ev.dur * sr))
    weight = (1.0, 0.8, 0.62)[seed % 3]  # heavy, medium, light interior door
    room = (seed // 3) % 3  # 0 small room, 1 hallway, 2 large room
    swing = (seed // 9) % 2 == 1
    t0 = 0.002 + (0.11 if swing else 0.0)
    at = int(t0 * sr)
    t = np.arange(n) / sr

    # 1. Air whump: one and a half cycles of a low push under a smooth window,
    # plus a breath of low noise as the air squeezes out of the gap.
    fw = crng.uniform(46.0, 58.0) / weight**0.8
    L = min(n - at, int(1.5 / fw * sr))
    whump = np.zeros(n)
    if L > 2:
        tw = np.arange(L) / sr
        whump[at : at + L] = np.sin(TAU * fw * tw) * np.sin(np.pi * (np.arange(L) + 0.5) / L) ** 2
    puff = bandlimit(_burst(rng, n, at, int(0.15 * sr), 0.03, sr, rise=0.004), sr, 30.0, 160.0, order=2)
    whump = _norm(whump) + 0.45 * _norm(puff)

    # 2. The panel. A slow push (the slam) drives the low modes; a shorter,
    # rougher contact drives the mids. Inharmonic, damped in 100-250 ms.
    f_lo = 74.0 / weight * crng.uniform(0.94, 1.06)
    lo_f = f_lo * np.array((1.0, 1.29, 1.71, 2.18)) * crng.uniform(0.97, 1.03, 4)
    lo_t = np.array((0.25, 0.21, 0.17, 0.13)) * (0.65 + 0.35 * weight)
    lo_g = (1.0, 0.8, 0.6, 0.4)
    push = _pulse(n, at, int((0.006 + 0.004 * weight) * sr))
    body_lo = _fftconv(push, _modes_ir(lo_f, lo_t, lo_g, sr, n), n)
    k_mid = 6
    mid_f = np.sort(crng.uniform(250.0, 700.0, k_mid)) / weight**0.3
    mid_t = crng.uniform(0.05, 0.11, k_mid)
    mid_g = crng.uniform(0.5, 1.0, k_mid) * np.linspace(1.0, 0.5, k_mid)
    contact = _pulse(n, at, int(0.0018 * sr)) + 0.3 * _burst(rng, n, at, int(0.006 * sr), 0.0015, sr)
    body_mid = _fftconv(contact, _modes_ir(mid_f, mid_t, mid_g, sr, n), n)
    body = DOOR_LO * _norm(body_lo) + DOOR_MID * _norm(body_mid)

    # 3. Frame and wall: a short, dull thump-crack, mostly below 3 kHz.
    frame = bandlimit(_burst(rng, n, at, int(0.05 * sr), 0.009, sr, rise=0.0012), sr, 110.0, 2600.0, order=4)

    # 4. Latch: a small metal tick just after the hit, 12-18 dB under the body,
    # with a rattle on odd seeds.
    lf = crng.uniform(2300.0, 3600.0)
    latch_ir = _modes_ir((lf, lf * 1.47, lf * 1.93), (0.03, 0.022, 0.015), (1.0, 0.55, 0.35), sr, n)
    l_at = at + int(crng.uniform(0.004, 0.016) * sr)
    latch = _fftconv(_burst(rng, n, l_at, int(0.002 * sr), 0.0005, sr), latch_ir, n)
    if seed % 2:
        for _ in range(int(crng.integers(1, 3))):
            r_at = l_at + int(crng.uniform(0.025, 0.075) * sr)
            latch += crng.uniform(0.25, 0.45) * _fftconv(_burst(rng, n, r_at, int(0.0015 * sr), 0.0004, sr), latch_ir, n)
    latch_db = crng.uniform(-18.0, -13.0)

    dry = DOOR_WHUMP * whump + body + DOOR_FRAME * _norm(frame)
    dry = _norm(dry) + db_to_gain(latch_db) * _norm(latch)

    # 5. Optional swing: the faint air of the door moving before it lands.
    if swing:
        u = np.clip(t / t0, 0.0, 1.0)
        env = (0.5 - 0.5 * np.cos(np.pi * u)) * np.exp(-np.maximum(t - t0, 0.0) / 0.008)
        air = bandlimit(rng.standard_normal(n), sr, 150.0, 1200.0, order=2)
        dry = dry + 0.1 * _norm(air) * env

    # Early reflections: a dozen irregular, darkened copies in the first
    # 4-45 ms (walls, floor, a hallway's far end), different per ear. Many
    # weak, randomly spaced taps rather than a few strong ones, so they blur
    # into a room instead of combing.
    span = (0.018, 0.04, 0.045)[room]
    first = (0.003, 0.004, 0.008)[room]

    def early() -> np.ndarray:
        er = np.zeros(n)
        for _ in range(12):
            d = int(crng.uniform(first, first + span) * sr)
            if d < n:
                er[d:] += crng.uniform(0.08, 0.22) * np.exp(-(d / sr) / 0.035) * dry[: n - d]
        return bandlimit(er, sr, 80.0, 2800.0, order=2)

    # The room's diffuse tail is the shared dark reverb: small room, hallway,
    # large room.
    p["reverb"] = (0.3, 0.36, 0.42)[room]
    p["reverb_decay"] = (0.45, 0.7, 0.9)[room]
    return dry + early(), dry + early()


# -- knock -------------------------------------------------------------------


def _knock(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    n = max(1, int(ev.dur * sr))
    dur = n / sr
    raps = 2 + seed % 2
    size = (1.0, 0.85, 1.2)[(seed // 2) % 3]
    f0 = float(p["f0"]) / size * crng.uniform(0.95, 1.05)
    ratios = np.array((1.0, 1.59, 2.14, 2.65, 3.61, 4.43)) * crng.uniform(0.97, 1.03, 6)
    t60 = (0.10, 0.08, 0.06, 0.045, 0.035, 0.025)
    gains = (1.0, 0.85, 0.7, 0.5, 0.35, 0.25)

    # Spacing: a natural knock is 0.12-0.22 s apart; fit them in `dur`.
    gap = float(np.clip((dur - 0.15) / max(1, raps - 1), 0.07, crng.uniform(0.15, 0.21)))
    force = np.zeros(n)
    click = np.zeros(n)
    tm = 0.002
    for i in range(raps):
        at = int(tm * sr)
        if at >= n:
            break
        a = crng.uniform(0.8, 1.0) * (1.08 if i == raps - 1 else 1.0)
        width = int(crng.uniform(0.0010, 0.0016) * sr)  # knuckle contact time
        force += a * _pulse(n, at, width)
        click += a * _burst(rng, n, at, int(0.003 * sr), 0.0007, sr)
        tm += gap * crng.uniform(0.92, 1.05)
    panel = _ring(force, f0 * ratios, t60, gains, sr)
    # the whole door moves a little: one low mode under the panel
    door = _ring(force, (crng.uniform(55, 75) / size,), (0.12,), (1.0,), sr)
    click = bandlimit(click, sr, 1200.0, 6000.0, order=4)
    mono = _norm(panel) + 0.25 * _norm(door) + 0.22 * _norm(click)
    return mono, mono


# -- footsteps ---------------------------------------------------------------


def _footsteps(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    n = max(1, int(ev.dur * sr))
    dur = n / sr
    surface = seed % 3  # 0 hard floor, 1 stone, 2 gravel
    interval = (0.66, 0.54, 0.43)[(seed // 3) % 3] * crng.uniform(0.95, 1.05)
    f_body = float(p["f0"]) * (1.15, 0.8, 0.9)[surface]

    left = np.zeros(n)
    right = np.zeros(n)
    tm = 0.004
    step = 0
    while tm < dur - 0.12 or step == 0:
        foot = step % 2
        a = db_to_gain(crng.uniform(-2.0, 1.0) - (0.8 if foot else 0.0))
        heel_at = int(tm * sr)
        toe_at = heel_at + int(crng.uniform(0.055, 0.095) * sr)
        fb = f_body * (0.96 if foot else 1.03)
        sig = np.zeros(n)
        if surface == 2:
            # gravel: the weight crushes stones, a burst of tiny grains
            for start, length, g in ((heel_at, 0.11, 1.0), (toe_at, 0.08, 0.6)):
                cnt = int(crng.integers(35, 70) * length / 0.1)
                tt = start + (crng.gamma(2.0, length / 5.0, cnt) * sr).astype(int)
                tt = tt[tt < n]
                grains = np.zeros(n)
                np.add.at(grains, tt, rng.standard_normal(len(tt)) * g)
                sig += bandlimit(grains, sr, 700.0, 4800.0, order=4)
            thump = _ring(_pulse(n, heel_at, int(0.012 * sr)), (fb, fb * 1.7), (0.05, 0.03), (1.0, 0.5), sr)
            sig = _norm(sig) + 0.35 * _norm(thump)
        else:
            # heel: a soft force pulse rings the floor; the shoe scuffs
            heel = _ring(_pulse(n, heel_at, int(0.008 * sr)), (fb, fb * 1.63, fb * 2.4, fb * 3.7),
                         (0.06, 0.045, 0.03, 0.02), (1.0, 0.6, 0.4, 0.25), sr)
            scuff_lo, scuff_hi = (450.0, 4200.0) if surface == 0 else (350.0, 4000.0)
            scuff = bandlimit(_burst(rng, n, heel_at, int(0.05 * sr), 0.012, sr)
                              + 0.5 * _burst(rng, n, toe_at, int(0.04 * sr), 0.008, sr),
                              sr, scuff_lo, scuff_hi, order=4)
            toe = _ring(_pulse(n, toe_at, int(0.004 * sr)), (fb * 1.9, fb * 3.1), (0.035, 0.02), (1.0, 0.5), sr)
            click = np.zeros(n)
            if surface == 0:  # a hard heel on a hard floor clicks
                click = bandlimit(_burst(rng, n, heel_at, int(0.004 * sr), 0.001, sr), sr, 1800.0, 4800.0, order=4)
            sig = _norm(heel) + 0.35 * _norm(toe) + (0.55 if surface == 1 else 0.4) * _norm(scuff) + 0.3 * _norm(click)
        # the feet alternate a little left and right
        w = 0.5 + (0.08 if foot else -0.08)
        left += a * sig * (1.0 - w) * 2.0
        right += a * sig * w * 2.0
        step += 1
        tm += interval * crng.uniform(0.93, 1.07)
        if step > 64:
            break
    return left, right


# -- water drop --------------------------------------------------------------


def _bubble(n: int, sr: int, at: int, f0: float, damp_scale: float = 0.7, rise: float = 0.1) -> np.ndarray:
    """A Minnaert bubble (van den Doel 2005): f(t) = f0 (1 + rise * d * t),
    amplitude exp(-d t), d = 0.043 f0 + 0.0014 f0^1.5. The pitch glides up as
    the bubble nears the surface."""
    out = np.zeros(n)
    if at >= n:
        return out
    d = (0.043 * f0 + 0.0014 * f0**1.5) * damp_scale
    k = min(n - at, int(LN1000 / d * 1.2 * sr))
    t = np.arange(k) / sr
    f = f0 * (1.0 + rise * d * t)
    ph = TAU * np.cumsum(f) / sr
    # a 2 ms raised-cosine onset: a sub-millisecond window on a sine has
    # sidelobes every 1/window Hz, a comb that runs to Nyquist
    a = min(k, max(2, int(0.002 * sr)))
    env = np.exp(-d * t)
    env[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, a))
    out[at : at + k] = np.sin(ph) * env
    return out


def _water_drop(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    n = max(1, int(ev.dur * sr))
    at = int(0.003 * sr)
    if seed % 2 == 0:
        # drip: a small droplet, a high plink
        f0 = crng.uniform(1300.0, 2500.0)
        bub = _bubble(n, sr, at + int(crng.uniform(0.002, 0.006) * sr), f0, crng.uniform(0.55, 0.8), crng.uniform(0.08, 0.14))
        tick = bandlimit(_burst(rng, n, at, int(0.003 * sr), 0.0007, sr), sr, 2500.0, 10000.0, order=4)
        mono = _norm(bub) + 0.25 * _norm(tick)
    else:
        # plop: a larger drop, a splash, the main bubble and a smaller one
        f0 = crng.uniform(480.0, 900.0)
        bub = _bubble(n, sr, at + int(crng.uniform(0.006, 0.014) * sr), f0, crng.uniform(0.6, 0.85), crng.uniform(0.1, 0.16))
        bub2 = _bubble(n, sr, at + int(crng.uniform(0.02, 0.06) * sr), f0 * crng.uniform(1.8, 2.6), 0.8, 0.12)
        splash = bandlimit(_burst(rng, n, at, int(0.06 * sr), 0.012, sr, rise=0.002), sr, 300.0, 3200.0, order=4)
        mono = _norm(bub) + 0.35 * _norm(bub2) + 0.4 * _norm(splash)
    return mono, mono


# -- explosion ---------------------------------------------------------------


def _explosion(ev: Event, sr: int, rng: np.random.Generator, crng: np.random.Generator, p: dict, seed: int):
    n = max(1, int(ev.dur * sr))
    dur = n / sr
    t = np.arange(n) / sr
    dist = (0.0, 0.4, 0.85)[seed % 3]
    f0 = float(p["f0"]) * crng.uniform(0.85, 1.15)
    at = 0.002

    # Sub body: the pressure wave, a falling low sine with a fast attack.
    f_sub = f0 * (1.0 + 1.2 * np.exp(-t / 0.06)) * (0.8 + 0.2 * np.exp(-t / max(0.6, 0.3 * dur)))
    sub = np.sin(TAU * np.cumsum(f_sub) / sr) * _attack(n, sr, at, 0.004) * np.exp(-t / 0.45)

    # Blast: broadband noise whose brightness falls fast (fireball to rumble).
    bright = bandlimit(_burst(rng, n, int(at * sr), int(0.6 * sr), 0.07 + 0.03 * dist, sr), sr, 300.0, 9000.0 - 6000.0 * dist, order=2)
    mid = bandlimit(_burst(rng, n, int(at * sr), int(min(dur, 2.0) * sr), 0.35, sr), sr, 90.0, 1600.0, order=2)
    blast = (1.0 - 0.6 * dist) * _norm(bright) + 1.1 * _norm(mid)

    # Rumble: low noise that rolls on, like distant thunder.
    roll = bandlimit(rng.standard_normal(n), sr, 40.0, 260.0, order=2)
    roll_env = _attack(n, sr, at, 0.03 + 0.15 * dist) * np.exp(-t / max(0.25 * dur, 0.3)) * (0.6 + 0.4 * _smooth(crng, n, sr, 2.0))

    # Debris: falling grains and chunks, dense just after the blast.
    span = max(0.2, 0.85 * dur)
    cnt = int(crng.integers(60, 140) * span)
    tt = 0.12 + crng.exponential(span * 0.3, cnt)
    tt = tt[tt < dur - 0.02]
    grains = np.zeros(n)
    np.add.at(grains, (tt * sr).astype(int), rng.standard_normal(len(tt)) * np.exp(-(tt - 0.12) / (span * 0.45)))
    debris = _ring(grains, (crng.uniform(1800, 2600), crng.uniform(3200, 4400), crng.uniform(900, 1300)),
                   (0.012, 0.008, 0.02), (1.0, 0.6, 0.5), sr)
    debris = bandlimit(debris, sr, 600.0, 8000.0, order=4)

    tail = _fade_out(n, sr, min(0.3 * dur, 1.2))
    mono = (0.55 * _norm(sub) + blast + 0.45 * _norm(roll) * roll_env + (0.3 - 0.15 * dist) * _norm(debris)) * tail
    mono = saturate(_norm(mono) * 1.2, 1.6)
    # a second, decorrelated rumble and debris for width
    roll2 = bandlimit(rng.standard_normal(n), sr, 40.0, 260.0, order=2)
    side = 0.35 * (_norm(roll2) - _norm(roll)) * roll_env * tail
    level = 1.0 - 0.3 * dist
    return (mono + side) * level, (mono - side) * level


_FNS = {
    "thunder": _thunder,
    "door_slam": _door_slam,
    "knock": _knock,
    "footsteps": _footsteps,
    "water_drop": _water_drop,
    "explosion": _explosion,
}


def _smooth_lowpass(x: np.ndarray, sr: int, f_pass: float = HI_PASS, f_stop: float = HI, pad: int = 2048) -> np.ndarray:
    """Zero-phase low-pass: flat to `f_pass`, a raised-cosine slope to exactly
    zero at `f_stop`, nothing above. The slope is 3 kHz wide, so the impulse
    response is a fraction of a millisecond long and does not ring; a
    Butterworth of the same corner leaves -24 dB at Nyquist under a hard
    transient, which is what used to fizz up to 24 kHz on every attack."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    m = n + 2 * pad
    f = np.fft.rfftfreq(m, 1.0 / sr)
    u = np.clip((f - f_pass) / (f_stop - f_pass), 0.0, 1.0)
    mag = 0.5 + 0.5 * np.cos(np.pi * u)
    return np.fft.irfft(np.fft.rfft(np.pad(x, (pad, pad)), m) * mag, m)[pad : pad + n]


def _edges(x: np.ndarray, sr: int) -> np.ndarray:
    """Raised-cosine fades: 1.5 ms in, 12 ms out, both ends exactly zero."""
    n = len(x)
    a = min(n, max(2, int(0.0015 * sr)))
    b = min(n, max(2, int(0.012 * sr)))
    x[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, a))
    x[-b:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, b))
    return x


def _finish(x: np.ndarray, sr: int) -> np.ndarray:
    """High-pass, smooth low-pass (after every envelope and attack), remove
    DC, and fade both ends to exactly zero."""
    x = bandlimit(x, sr, LO, None, order=4)
    x = _smooth_lowpass(x, sr)
    return _edges(x - x.mean(), sr)


def render_foley(ev: Event, sr: int, rng: np.random.Generator, p: dict, seed: int, tail: bool) -> np.ndarray:
    """Render one foley event to stereo, shape (n, 2). See `render_event`."""
    idx = EVENT_ROLES.index(ev.role)
    # The variant (distance, surface, door size...) depends on the seed only,
    # not on where the event sits in the cue.
    crng = np.random.default_rng((seed * 2_654_435_761 + idx * 40_503 + 0x0F01E) & 0xFFFFFFFF)
    fn = _FNS.get(ev.role)
    if fn is None:  # the everyday / nature / action library (v0.3)
        from .natural import FNS as _NAT

        fn = _NAT[ev.role]
    left, right = fn(ev, sr, rng, crng, p, int(seed))
    left, right = _finish(left, sr), _finish(right, sr)
    peak = max(float(np.abs(left).max()), float(np.abs(right).max()), 1e-12)
    left, right = left / peak, right / peak

    wet = float(p.get("reverb", 0.0))
    if wet > 0:
        decay = float(p.get("reverb_decay", 0.5))
        s1, s2 = int(rng.integers(1 << 31)), int(rng.integers(1 << 31))
        left = reverb(left, sr, decay=decay, wet=wet, seed=s1, tail=tail)
        right = reverb(right, sr, decay=decay, wet=wet, seed=s2, tail=tail)

    # The reverb follows the band limit; band-limit and fade once more at the
    # very end, so nothing added after `_finish` can reach past HI.
    left = _edges(_smooth_lowpass(left, sr), sr)
    right = _edges(_smooth_lowpass(right, sr), sr)

    # Narrow the image as it is panned, then place the mid with the same
    # constant-power law as the mono roles (centre = 0.707 per channel).
    width = float(np.clip(p.get("width", 0.0), 0.0, 1.0)) * (1.0 - abs(float(ev.pan)))
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * width * np.sqrt(0.5)
    out_l, out_r = pan(mid, ev.pan)
    return np.stack([out_l + side, out_r - side], axis=1) * db_to_gain(ev.gain_db)
