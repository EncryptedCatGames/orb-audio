"""Deterministic score grammar.

The caption must determine the score. A small `Spec` (duration, texture, mood,
and a list of timed events) is sampled, the score is a deterministic function
of that spec, and the caption is generated from the score. The model can learn
the mapping instead of the dataset.

v1 is SFX + texture only. Tempo, key and mode are deliberately absent from the
Spec: with music off they do not affect the audio, so putting them in the Spec
would inflate exact-match without being verifiable. They return in v2 with a
version token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .score import (
    EVENT_ROLES,
    TEXTURES,
    Event,
    MusicTrack,
    Note,
    Score,
    Section,
    Texture,
)

MOODS = (
    "calm",
    "warm",
    "playful",
    "hopeful",
    "bright",
    "tense",
    "dark",
    "eerie",
    "neutral",
    "intense",
)
MOOD_ENERGY = {
    "calm": 0.35,
    "warm": 0.50,
    "playful": 0.60,
    "hopeful": 0.65,
    "bright": 0.70,
    "tense": 0.75,
    "dark": 0.80,
    "eerie": 0.70,
    "neutral": 0.55,
    "intense": 0.90,
}
SECTION_ENERGY = {"intro": 0.35, "build": 0.60, "peak": 0.90, "outro": 0.40}
SECTION_DENSITY = {"intro": 0.25, "build": 0.50, "peak": 0.75, "outro": 0.35}

# role: (character, apex fraction, default gain dB, default duration s)
EVENT_STYLE = {
    "whoosh": ("fast, airy", 0.05, -8.0, 0.80),
    "impact": ("short, deep", 0.00, -6.0, 0.70),
    "ui_pop": ("soft, rounded", 0.00, -14.0, 0.16),
    "riser": ("dark, tense", 0.92, -20.0, 2.00),
    "swell": ("wide, shimmering", 0.90, -18.0, 1.50),
    "click": ("tight, clean", 0.00, -16.0, 0.10),
    "bell": ("bright, metallic", 0.10, -14.0, 1.40),
    "sub_drop": ("deep, clean", 0.10, -12.0, 1.00),
    "glitch": ("digital, broken", 0.00, -16.0, 0.20),
    "noise_sweep": ("wide, airy", 0.50, -18.0, 1.50),
}

PAN_LABELS = ("L4", "L3", "L2", "L1", "C", "R1", "R2", "R3", "R4")
GAIN_LEVELS = 16  # -30 dB .. 0 dB in 2 dB steps
MAX_EVENTS = 8


def quantize_time(x: float) -> float:
    return round(float(x) * 100.0) / 100.0


def quantize_gain(x: float) -> float:
    return float(np.clip(round(float(x) / 2.0) * 2.0, -30.0, 0.0))


def pan_index(pan: float) -> int:
    return int(round((float(np.clip(pan, -1.0, 1.0)) + 1.0) * 0.5 * (len(PAN_LABELS) - 1)))


def pan_value(index: int) -> float:
    return float(index) / (len(PAN_LABELS) - 1) * 2.0 - 1.0


def pan_label(pan: float) -> str:
    return PAN_LABELS[pan_index(pan)]


def gain_index(gain_db: float) -> int:
    return int(np.clip(round((float(gain_db) + 30.0) / 2.0), 0, GAIN_LEVELS - 1))


def gain_value(index: int) -> float:
    return float(index) * 2.0 - 30.0


@dataclass
class SpecEvent:
    role: str
    t: float
    dur: float
    gain_db: float
    pan: float
    seed: int

    def __post_init__(self) -> None:
        if self.role not in EVENT_ROLES:
            raise ValueError(f"unknown event role {self.role!r}")
        if self.seed < 0 or self.seed > 255:
            raise ValueError("event seed must be in [0, 255]")


@dataclass
class Spec:
    duration: float
    texture: str
    mood: str
    events: list[SpecEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.texture not in TEXTURES:
            raise ValueError(f"unknown texture {self.texture!r}")
        if self.mood not in MOODS:
            raise ValueError(f"unknown mood {self.mood!r}")
        if len(self.events) > MAX_EVENTS:
            raise ValueError(f"at most {MAX_EVENTS} events")


def sample_spec(rng: np.random.Generator) -> Spec:
    # Half-second steps from 2 s to 30 s: agents ask for short UI cues as well
    # as long beds, and the planner must copy any of them.
    duration = float(rng.integers(4, 61)) / 2.0
    texture = str(rng.choice(TEXTURES))
    mood = str(rng.choice(MOODS))
    # Agents ask for a single sound as often as for a full scene, so the
    # planner must see 0 (pure ambience) through MAX_EVENTS.
    n_events = int(rng.integers(0, MAX_EVENTS + 1))
    roles = [str(x) for x in rng.choice(EVENT_ROLES, size=max(1, n_events), replace=False)][:n_events]

    events: list[SpecEvent] = []
    for role in roles:
        _, _, gain0, dur0 = EVENT_STYLE[role]
        dur = quantize_time(np.clip(dur0 * rng.uniform(0.7, 1.4), 0.02, min(9.99, duration)))
        t = quantize_time(rng.uniform(0.0, max(0.0, duration - dur)))
        if t + dur > duration:
            t = quantize_time(duration - dur)
        gain = quantize_gain(gain0 + rng.uniform(-4.0, 4.0))
        pan = pan_value(pan_index(rng.uniform(-1.0, 1.0)))
        seed = int(rng.integers(0, 256))
        events.append(SpecEvent(role, t, dur, gain, pan, seed))

    events.sort(key=lambda e: e.t)
    return Spec(duration, texture, mood, events)


def _chord(root: int, mode: str, degree: int) -> list[int]:
    scale = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return [root + 12 * ((degree + k) // 7) + scale[(degree + k) % 7] for k in (0, 2, 4)]


def score_from_spec(spec: Spec, with_music: bool = False) -> Score:
    n_sections = 4
    sec_dur = spec.duration / n_sections
    roles = ("intro", "build", "peak", "outro")
    energy_scale = MOOD_ENERGY.get(spec.mood, 0.6)
    sections = [
        Section(
            i * sec_dur,
            sec_dur,
            role,
            min(1.0, SECTION_ENERGY[role] * energy_scale / 0.7),
            SECTION_DENSITY[role],
        )
        for i, role in enumerate(roles)
    ]

    music: list[MusicTrack] = []
    if with_music:
        # v2. v1 has no tempo/key/mode in the Spec, so music uses fixed defaults.
        root = 48
        bar = 4.0 * 60.0 / 120.0
        n_bars = max(1, int(spec.duration // bar))
        pad_notes: list[Note] = []
        bass_notes: list[Note] = []
        for b in range(n_bars):
            t = b * bar
            chord = _chord(root, "minor", (0, 5, 3, 4)[b % 4])
            for midi in chord:
                pad_notes.append(Note(t, bar * 0.98, midi, 0.45))
            bass_notes.append(Note(t, bar * 0.90, chord[0] - 12, 0.60))
        music = [
            MusicTrack("pad", pad_notes, gain_db=-17.0),
            MusicTrack("bass", bass_notes, gain_db=-15.0),
        ]

    textures = [Texture(0.0, spec.duration, spec.texture, gain_db=-30.0, width=0.7)]

    events: list[Event] = []
    for se in spec.events:
        character, apex_frac, _, _ = EVENT_STYLE[se.role]
        dur = min(se.dur, spec.duration - se.t)
        if dur <= 0.02:
            continue
        apex = se.t + apex_frac * dur
        events.append(
            Event(
                se.t,
                se.role,
                dur,
                character,
                apex=apex,
                gain_db=se.gain_db,
                pan=se.pan,
                seed=se.seed,
            )
        )

    return Score(
        duration=spec.duration,
        sample_rate=48000,
        seed=0,
        tempo=120.0,
        key="C",
        mode="minor",
        mood=spec.mood,
        target_lufs=-20.0,
        sections=sections,
        music=music,
        textures=textures,
        events=events,
    )


def sample_score(rng: np.random.Generator, with_music: bool = True) -> Score:
    return score_from_spec(sample_spec(rng), with_music=with_music)


def caption(score: Score) -> str:
    """A short, agent-style description that determines the score."""
    parts = [f"{score.duration:.2f}-second {score.mood} cue"]
    tex = ", ".join(sorted({t.character.replace("_", " ") for t in score.textures}))
    if tex:
        parts.append(f"{tex} bed")
    evs = []
    for e in score.events:
        role = e.role.replace("_", " ")
        evs.append(
            f"{role} @{e.t:.2f} {e.dur:.2f}s {e.gain_db:+.0f}dB {pan_label(e.pan)} #{e.seed:02x}"
        )
    if evs:
        parts.append("with " + ", ".join(evs))
    return ", ".join(parts)


def spec_from_score(score: Score) -> Spec:
    """Inverse of `score_from_spec` for the fields the Spec carries."""
    events = [
        SpecEvent(e.role, quantize_time(e.t), quantize_time(e.dur), quantize_gain(e.gain_db), e.pan, e.seed)
        for e in score.events
    ]
    texture = score.textures[0].character if score.textures else TEXTURES[0]
    return Spec(score.duration, texture, score.mood, events)
