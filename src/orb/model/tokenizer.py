"""Byte-level tokenizer for captions and the typed score.

No BPE, no pretrained vocab. Captions are UTF-8 bytes; score fields are
special tokens with quantized values. The vocabulary is ~4.4k, small enough
that a 38M planner spends its parameters on attention, not embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..score import (
    EVENT_ROLES,
    INSTRUMENTS,
    SECTION_ROLES,
    TEXTURES as TEXTURE_NAMES,
    Event,
    MusicTrack,
    Note,
    Score,
    Section,
    Texture,
)

# -- vocabulary -------------------------------------------------------------

PAD = 256
BOS = 257
EOS = 258
UNK = 259
SEP = 260

SCORE = 261
END_SCORE = 262
META = 263
END_META = 264
SECTIONS = 265
SEC = 266
END_SEC = 267
END_SECTIONS = 268
MUSIC = 269
TRACK = 270
END_TRACK = 271
END_MUSIC = 272
TEXTURES = 273
TEX = 274
END_TEX = 275
END_TEXTURES = 276
EVENTS = 277
EVT = 278
END_EVT = 279
END_EVENTS = 280

DURATION = 281
SR = 282
TEMPO = 283
METER = 284
KEY = 285
MODE = 286
LUFS = 287
T = 288
D = 289
ROLE = 290
ENERGY = 291
DENSITY = 292
INST = 293
GAIN = 294
PAN = 295
APEX = 296
CHAR = 297
END_CHAR = 298
NOTE = 299
MIDI = 300
VEL = 301
WIDTH = 302
TCHAR = 303

ROLE_VAL = 304
INST_VAL = 314
KEY_VAL = 320
MODE_VAL = 327
METER_VAL = 329
SR_VAL = 333
SECROLE_VAL = 339
TCHAR_VAL = 344

VAL_BASE = 352
VAL_BINS = 4096
VOCAB_SIZE = VAL_BASE + VAL_BINS

KEYS = "ABCDEFG"
MODES = ("major", "minor")
METERS = ((4, 4), (3, 4), (6, 8), (5, 4))
SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000)

ROLE_TO_ID = {r: i for i, r in enumerate(EVENT_ROLES)}
INST_TO_ID = {r: i for i, r in enumerate(INSTRUMENTS)}
KEY_TO_ID = {r: i for i, r in enumerate(KEYS)}
MODE_TO_ID = {r: i for i, r in enumerate(MODES)}
METER_TO_ID = {r: i for i, r in enumerate(METERS)}
SR_TO_ID = {r: i for i, r in enumerate(SAMPLE_RATES)}
SECROLE_TO_ID = {r: i for i, r in enumerate(SECTION_ROLES)}
TCHAR_TO_ID = {r: i for i, r in enumerate(TEXTURE_NAMES)}

# -- quantization -----------------------------------------------------------

DUR_LO, DUR_HI = 0.005, 30.0
TIME_LO, TIME_HI = 0.0, 32.0


def _q(x: float, lo: float, hi: float, bins: int = VAL_BINS) -> int:
    x = min(max(float(x), lo), hi)
    return int(round((x - lo) / (hi - lo) * (bins - 1)))


def _dq(v: int, lo: float, hi: float, bins: int = VAL_BINS) -> float:
    return lo + v / (bins - 1) * (hi - lo)


def _q_dur(x: float) -> int:
    return _q(math.log(max(x, DUR_LO)), math.log(DUR_LO), math.log(DUR_HI))


def _dq_dur(v: int) -> float:
    return math.exp(_dq(v, math.log(DUR_LO), math.log(DUR_HI)))


def _q_time(x: float) -> int:
    return _q(x, TIME_LO, TIME_HI)


def _dq_time(v: int) -> float:
    return _dq(v, TIME_LO, TIME_HI)


def _val(tok: int) -> int:
    return VAL_BASE + tok


def _bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


# -- tokenizer --------------------------------------------------------------


@dataclass
class EncodedPair:
    input_ids: list[int]
    labels: list[int]
    weights: list[float]


class ScoreTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id = PAD
    bos_id = BOS
    eos_id = EOS
    sep_id = SEP

    # -- captions ------------------------------------------------------

    def encode_caption(self, text: str, add_special: bool = True) -> list[int]:
        body = _bytes(text)
        return ([BOS] + body + [EOS]) if add_special else body

    def decode_caption(self, ids: list[int]) -> str:
        body = [i for i in ids if 0 <= i < 256]
        return bytes(body).decode("utf-8", errors="replace")

    # -- scores --------------------------------------------------------

    def encode_score(self, score: Score, add_special: bool = True) -> list[int]:
        out: list[int] = []
        if add_special:
            out.append(BOS)
        out += [
            SCORE,
            META,
            DURATION,
            _val(_q_time(score.duration)),
            SR,
            SR_VAL + SR_TO_ID[score.sample_rate],
            TEMPO,
            _val(_q(score.tempo, 40, 220)),
            METER,
            METER_VAL + METER_TO_ID[score.meter],
            KEY,
            KEY_VAL + KEY_TO_ID[score.key],
            MODE,
            MODE_VAL + MODE_TO_ID[score.mode],
            LUFS,
            _val(_q(score.target_lufs, -40, -6)),
            END_META,
            SECTIONS,
        ]
        for s in score.sections:
            out += [
                SEC,
                T,
                _val(_q_time(s.t)),
                D,
                _val(_q_dur(s.dur)),
                ROLE,
                SECROLE_VAL + SECROLE_TO_ID[s.role],
                ENERGY,
                _val(_q(s.energy, 0, 1)),
                DENSITY,
                _val(_q(s.density, 0, 1)),
                END_SEC,
            ]
        out += [END_SECTIONS, MUSIC]
        for tr in score.music:
            out += [
                TRACK,
                INST,
                INST_VAL + INST_TO_ID[tr.instrument],
                GAIN,
                _val(_q(tr.gain_db, -60, 6)),
                PAN,
                _val(_q(tr.pan, -1, 1)),
            ]
            for n in tr.notes:
                out += [
                    NOTE,
                    T,
                    _val(_q_time(n.t)),
                    MIDI,
                    _val(_q(n.midi, 0, 127)),
                    VEL,
                    _val(_q(n.vel, 0, 1)),
                    D,
                    _val(_q_dur(n.dur)),
                ]
            out.append(END_TRACK)
        out += [END_MUSIC, TEXTURES]
        for tx in score.textures:
            out += [
                TEX,
                T,
                _val(_q_time(tx.t)),
                D,
                _val(_q_dur(tx.dur)),
                TCHAR,
                TCHAR_VAL + TCHAR_TO_ID[tx.character],
                GAIN,
                _val(_q(tx.gain_db, -60, 6)),
                WIDTH,
                _val(_q(tx.width, 0, 1)),
                END_TEX,
            ]
        out += [END_TEXTURES, EVENTS]
        for ev in score.events:
            out += [
                EVT,
                T,
                _val(_q_time(ev.t)),
                ROLE,
                ROLE_VAL + ROLE_TO_ID[ev.role],
                CHAR,
            ]
            out += _bytes(ev.character)
            out += [
                END_CHAR,
                D,
                _val(_q_dur(ev.dur)),
                APEX,
                _val(_q_time(ev.apex)),
                GAIN,
                _val(_q(ev.gain_db, -60, 6)),
                PAN,
                _val(_q(ev.pan, -1, 1)),
                END_EVT,
            ]
        out += [END_EVENTS, END_SCORE]
        if add_special:
            out.append(EOS)
        return out

    def decode_score(self, ids: list[int]) -> Score:
        toks = [int(i) for i in ids if int(i) not in (PAD, BOS, EOS)]
        p = 0

        def take(expected: int | None = None) -> int:
            nonlocal p
            if p >= len(toks):
                raise ValueError("unexpected end of score tokens")
            tok = toks[p]
            p += 1
            if expected is not None and tok != expected:
                raise ValueError(f"expected {expected}, got {tok} at {p - 1}")
            return tok

        def value(lo: float, hi: float) -> float:
            tok = take()
            if not (VAL_BASE <= tok < VAL_BASE + VAL_BINS):
                raise ValueError(f"expected value token, got {tok}")
            return _dq(tok - VAL_BASE, lo, hi)

        def time_value() -> float:
            return value(TIME_LO, TIME_HI)

        def dur_value() -> float:
            tok = take()
            if not (VAL_BASE <= tok < VAL_BASE + VAL_BINS):
                raise ValueError(f"expected duration token, got {tok}")
            return _dq_dur(tok - VAL_BASE)

        def categorical(base: int, table: dict) -> str:
            tok = take()
            idx = tok - base
            if not (0 <= idx < len(table)):
                raise ValueError(f"bad categorical token {tok}")
            return list(table)[idx]

        take(SCORE)
        take(META)
        take(DURATION)
        duration = max(0.1, time_value())
        take(SR)
        sr_tok = take()
        sr = SAMPLE_RATES[sr_tok - SR_VAL]
        take(TEMPO)
        tempo = value(40, 220)
        take(METER)
        meter = METERS[take() - METER_VAL]
        take(KEY)
        key = KEYS[take() - KEY_VAL]
        take(MODE)
        mode = MODES[take() - MODE_VAL]
        take(LUFS)
        lufs = value(-40, -6)
        take(END_META)

        sections: list[Section] = []
        take(SECTIONS)
        while p < len(toks) and toks[p] == SEC:
            take(SEC)
            take(T)
            t = time_value()
            take(D)
            dur = dur_value()
            take(ROLE)
            role = categorical(SECROLE_VAL, SECROLE_TO_ID)
            take(ENERGY)
            energy = value(0, 1)
            take(DENSITY)
            density = value(0, 1)
            take(END_SEC)
            t = min(max(t, 0.0), max(0.0, duration - 0.01))
            dur = min(max(dur, 0.01), duration - t)
            sections.append(Section(t, dur, role, energy, density))
        take(END_SECTIONS)

        music: list[MusicTrack] = []
        take(MUSIC)
        while p < len(toks) and toks[p] == TRACK:
            take(TRACK)
            take(INST)
            inst = categorical(INST_VAL, INST_TO_ID)
            take(GAIN)
            gain = value(-60, 6)
            take(PAN)
            pan = value(-1, 1)
            notes: list[Note] = []
            while p < len(toks) and toks[p] == NOTE:
                take(NOTE)
                take(T)
                nt = time_value()
                take(MIDI)
                midi = int(round(value(0, 127)))
                take(VEL)
                vel = value(0, 1)
                take(D)
                ndur = dur_value()
                nt = min(max(nt, 0.0), max(0.0, duration - 0.01))
                ndur = min(max(0.01, ndur), duration - nt)
                notes.append(Note(nt, ndur, midi, vel))
            take(END_TRACK)
            music.append(MusicTrack(inst, notes, gain, pan))
        take(END_MUSIC)

        textures: list[Texture] = []
        take(TEXTURES)
        while p < len(toks) and toks[p] == TEX:
            take(TEX)
            take(T)
            t = time_value()
            take(D)
            dur = dur_value()
            take(TCHAR)
            char = categorical(TCHAR_VAL, TCHAR_TO_ID)
            take(GAIN)
            gain = value(-60, 6)
            take(WIDTH)
            width = value(0, 1)
            take(END_TEX)
            t = min(max(t, 0.0), max(0.0, duration - 0.01))
            dur = min(max(dur, 0.01), duration - t)
            textures.append(Texture(t, dur, char, gain, width))
        take(END_TEXTURES)

        events: list[Event] = []
        take(EVENTS)
        while p < len(toks) and toks[p] == EVT:
            take(EVT)
            take(T)
            t = time_value()
            take(ROLE)
            role = categorical(ROLE_VAL, ROLE_TO_ID)
            take(CHAR)
            chars: list[int] = []
            while p < len(toks) and toks[p] != END_CHAR:
                chars.append(toks[p])
                p += 1
            take(END_CHAR)
            character = bytes([c for c in chars if 0 <= c < 256]).decode("utf-8", errors="replace")
            take(D)
            dur = dur_value()
            take(APEX)
            apex = time_value()
            take(GAIN)
            gain = value(-60, 6)
            take(PAN)
            pan = value(-1, 1)
            take(END_EVT)
            t = min(max(t, 0.0), max(0.0, duration - 0.01))
            dur = min(max(dur, 0.01), duration - t)
            apex = min(max(apex, t), t + dur)
            events.append(Event(t, role, dur, character, apex, {}, gain, pan))
        take(END_EVENTS)
        take(END_SCORE)

        return Score(
            duration=duration,
            sample_rate=sr,
            seed=0,
            tempo=tempo,
            meter=meter,
            key=key,
            mode=mode,
            target_lufs=lufs,
            sections=sections,
            music=music,
            textures=textures,
            events=events,
        )

    # -- training pairs -------------------------------------------------

    def _score_weights(self, body: list[int]) -> list[float]:
        """Upweight event and texture tokens so the planner cannot skip them."""
        weights = [1.0] * len(body)
        in_events = False
        in_textures = False
        for i, token in enumerate(body):
            if token == EVENTS:
                in_events = True
            elif token == END_EVENTS:
                in_events = False
            elif token == TEXTURES:
                in_textures = True
            elif token == END_TEXTURES:
                in_textures = False
            if in_events:
                weights[i] = 2.0
            elif in_textures:
                weights[i] = 1.5
        return weights

    def encode_pair(self, caption: str, score: Score, max_len: int | None = None) -> EncodedPair:
        cap = _bytes(caption)
        body = self.encode_score(score, add_special=False)
        input_ids = [BOS] + cap + [SEP] + body + [EOS]
        labels = [-100] * (1 + len(cap) + 1) + body + [EOS]
        weights = [0.0] * (1 + len(cap) + 1) + self._score_weights(body) + [1.0]
        if max_len is not None:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            weights = weights[:max_len]
        return EncodedPair(input_ids, labels, weights)

    def prompt(self, caption: str) -> list[int]:
        return [BOS] + _bytes(caption) + [SEP]
