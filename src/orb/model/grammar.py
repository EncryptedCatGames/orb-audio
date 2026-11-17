"""Grammar-constrained decoding for the score token stream.

The mask guarantees that the planner emits a parseable score. It does not
enforce numeric ranges; the tokenizer clamps those on decode. This is the
difference between "the JSON parses" and "the model is right", and it is what
makes a 38M planner usable.
"""

from __future__ import annotations

from .tokenizer import (
    APEX,
    BOS,
    CHAR,
    D,
    DENSITY,
    DURATION,
    END_CHAR,
    END_EVENTS,
    END_META,
    END_MUSIC,
    END_SCORE,
    END_SEC,
    END_SECTIONS,
    END_TEX,
    END_TEXTURES,
    END_TRACK,
    END_EVT,
    ENERGY,
    EOS,
    EVENTS,
    EVT,
    GAIN,
    INST,
    INST_VAL,
    KEY,
    KEY_VAL,
    LUFS,
    META,
    METER,
    METER_VAL,
    MIDI,
    MODE,
    MODE_VAL,
    MUSIC,
    NOTE,
    PAN,
    ROLE,
    role_tokens,
    SCORE,
    SEC,
    SECROLE_VAL,
    SECTIONS,
    SR,
    SR_VAL,
    T,
    TCHAR,
    tchar_tokens,
    TEMPO,
    TEX,
    TEXTURES,
    TRACK,
    VAL_BASE,
    VAL_BINS,
    VEL,
    WIDTH,
)

VAL = set(range(VAL_BASE, VAL_BASE + VAL_BINS))
BYTES = set(range(256)) | {END_CHAR}


class GrammarMask:
    """A state machine over the score grammar."""

    def __init__(self, start: str = "START", min_events: int = 0) -> None:
        self.state = start
        self.min_events = min_events
        self.events_emitted = 0

    def reset(self, start: str = "START") -> None:
        self.state = start
        self.events_emitted = 0

    def allowed(self) -> set[int]:
        s = self.state
        if s == "START":
            return {BOS}
        if s == "SCORE":
            return {SCORE}
        if s == "META":
            return {META}
        if s == "META_DURATION":
            return {DURATION}
        if s == "META_DUR_VAL":
            return VAL
        if s == "META_SR":
            return {SR}
        if s == "META_SR_VAL":
            return set(range(SR_VAL, SR_VAL + 6))
        if s == "META_TEMPO":
            return {TEMPO}
        if s == "META_TEMPO_VAL":
            return VAL
        if s == "META_METER":
            return {METER}
        if s == "META_METER_VAL":
            return set(range(METER_VAL, METER_VAL + 4))
        if s == "META_KEY":
            return {KEY}
        if s == "META_KEY_VAL":
            return set(range(KEY_VAL, KEY_VAL + 7))
        if s == "META_MODE":
            return {MODE}
        if s == "META_MODE_VAL":
            return set(range(MODE_VAL, MODE_VAL + 2))
        if s == "META_LUFS":
            return {LUFS}
        if s == "META_LUFS_VAL":
            return VAL
        if s == "META_END":
            return {END_META}
        if s == "SECTIONS":
            return {SECTIONS}
        if s == "SECTION_OR_END":
            return {SEC, END_SECTIONS}
        if s == "SEC_T":
            return {T}
        if s == "SEC_T_VAL":
            return VAL
        if s == "SEC_D":
            return {D}
        if s == "SEC_D_VAL":
            return VAL
        if s == "SEC_ROLE":
            return {ROLE}
        if s == "SEC_ROLE_VAL":
            return set(range(SECROLE_VAL, SECROLE_VAL + 5))
        if s == "SEC_ENERGY":
            return {ENERGY}
        if s == "SEC_ENERGY_VAL":
            return VAL
        if s == "SEC_DENSITY":
            return {DENSITY}
        if s == "SEC_DENSITY_VAL":
            return VAL
        if s == "SEC_END":
            return {END_SEC}
        if s == "AFTER_SECTIONS":
            return {MUSIC}
        if s == "MUSIC_OR_END":
            return {TRACK, END_MUSIC}
        if s == "TRACK_INST":
            return {INST}
        if s == "TRACK_INST_VAL":
            return set(range(INST_VAL, INST_VAL + 6))
        if s == "TRACK_GAIN":
            return {GAIN}
        if s == "TRACK_GAIN_VAL":
            return VAL
        if s == "TRACK_PAN":
            return {PAN}
        if s == "TRACK_PAN_VAL":
            return VAL
        if s == "TRACK_BODY":
            return {NOTE, END_TRACK}
        if s == "NOTE_T":
            return {T}
        if s == "NOTE_T_VAL":
            return VAL
        if s == "NOTE_MIDI":
            return {MIDI}
        if s == "NOTE_MIDI_VAL":
            return VAL
        if s == "NOTE_VEL":
            return {VEL}
        if s == "NOTE_VEL_VAL":
            return VAL
        if s == "NOTE_D":
            return {D}
        if s == "NOTE_D_VAL":
            return VAL
        if s == "AFTER_TRACK":
            return {TRACK, END_MUSIC}
        if s == "AFTER_MUSIC":
            return {TEXTURES}
        if s == "TEXTURE_OR_END":
            return {TEX, END_TEXTURES}
        if s == "TEX_T":
            return {T}
        if s == "TEX_T_VAL":
            return VAL
        if s == "TEX_D":
            return {D}
        if s == "TEX_D_VAL":
            return VAL
        if s == "TEX_CHAR":
            return {TCHAR}
        if s == "TEX_CHAR_VAL":
            return tchar_tokens()
        if s == "TEX_GAIN":
            return {GAIN}
        if s == "TEX_GAIN_VAL":
            return VAL
        if s == "TEX_WIDTH":
            return {WIDTH}
        if s == "TEX_WIDTH_VAL":
            return VAL
        if s == "TEX_END":
            return {END_TEX}
        if s == "AFTER_TEXTURES":
            return {EVENTS}
        if s == "EVENT_OR_END":
            return {EVT, END_EVENTS} if self.events_emitted >= self.min_events else {EVT}
        if s == "EVT_T":
            return {T}
        if s == "EVT_T_VAL":
            return VAL
        if s == "EVT_ROLE":
            return {ROLE}
        if s == "EVT_ROLE_VAL":
            return role_tokens()
        if s == "EVT_CHAR_FIELD":
            return {CHAR}
        if s == "EVT_CHAR":
            return BYTES
        if s == "EVT_CHAR_BYTES":
            return BYTES
        if s == "EVT_D":
            return {D}
        if s == "EVT_D_VAL":
            return VAL
        if s == "EVT_APEX":
            return {APEX}
        if s == "EVT_APEX_VAL":
            return VAL
        if s == "EVT_GAIN":
            return {GAIN}
        if s == "EVT_GAIN_VAL":
            return VAL
        if s == "EVT_PAN":
            return {PAN}
        if s == "EVT_PAN_VAL":
            return VAL
        if s == "EVT_END":
            return {END_EVT}
        if s == "AFTER_EVENTS":
            return {END_SCORE}
        if s == "AFTER_SCORE":
            return {EOS}
        if s == "DONE":
            return set()
        raise RuntimeError(f"unknown grammar state {self.state!r}")

    def advance(self, token: int) -> None:
        if token not in self.allowed():
            raise ValueError(f"token {token} not allowed in state {self.state!r}")
        s = self.state
        if s == "START":
            self.state = "SCORE"
        elif s == "SCORE":
            self.state = "META"
        elif s == "META":
            self.state = "META_DURATION"
        elif s == "META_DURATION":
            self.state = "META_DUR_VAL"
        elif s == "META_DUR_VAL":
            self.state = "META_SR"
        elif s == "META_SR":
            self.state = "META_SR_VAL"
        elif s == "META_SR_VAL":
            self.state = "META_TEMPO"
        elif s == "META_TEMPO":
            self.state = "META_TEMPO_VAL"
        elif s == "META_TEMPO_VAL":
            self.state = "META_METER"
        elif s == "META_METER":
            self.state = "META_METER_VAL"
        elif s == "META_METER_VAL":
            self.state = "META_KEY"
        elif s == "META_KEY":
            self.state = "META_KEY_VAL"
        elif s == "META_KEY_VAL":
            self.state = "META_MODE"
        elif s == "META_MODE":
            self.state = "META_MODE_VAL"
        elif s == "META_MODE_VAL":
            self.state = "META_LUFS"
        elif s == "META_LUFS":
            self.state = "META_LUFS_VAL"
        elif s == "META_LUFS_VAL":
            self.state = "META_END"
        elif s == "META_END":
            self.state = "SECTIONS"
        elif s == "SECTIONS":
            self.state = "SECTION_OR_END"
        elif s == "SECTION_OR_END":
            self.state = "SEC_T" if token == SEC else "AFTER_SECTIONS"
        elif s == "SEC_T":
            self.state = "SEC_T_VAL"
        elif s == "SEC_T_VAL":
            self.state = "SEC_D"
        elif s == "SEC_D":
            self.state = "SEC_D_VAL"
        elif s == "SEC_D_VAL":
            self.state = "SEC_ROLE"
        elif s == "SEC_ROLE":
            self.state = "SEC_ROLE_VAL"
        elif s == "SEC_ROLE_VAL":
            self.state = "SEC_ENERGY"
        elif s == "SEC_ENERGY":
            self.state = "SEC_ENERGY_VAL"
        elif s == "SEC_ENERGY_VAL":
            self.state = "SEC_DENSITY"
        elif s == "SEC_DENSITY":
            self.state = "SEC_DENSITY_VAL"
        elif s == "SEC_DENSITY_VAL":
            self.state = "SEC_END"
        elif s == "SEC_END":
            self.state = "SECTION_OR_END"
        elif s == "AFTER_SECTIONS":
            self.state = "MUSIC_OR_END"
        elif s == "MUSIC_OR_END":
            self.state = "TRACK_INST" if token == TRACK else "AFTER_MUSIC"
        elif s == "TRACK_INST":
            self.state = "TRACK_INST_VAL"
        elif s == "TRACK_INST_VAL":
            self.state = "TRACK_GAIN"
        elif s == "TRACK_GAIN":
            self.state = "TRACK_GAIN_VAL"
        elif s == "TRACK_GAIN_VAL":
            self.state = "TRACK_PAN"
        elif s == "TRACK_PAN":
            self.state = "TRACK_PAN_VAL"
        elif s == "TRACK_PAN_VAL":
            self.state = "TRACK_BODY"
        elif s == "TRACK_BODY":
            self.state = "NOTE_T" if token == NOTE else "AFTER_TRACK"
        elif s == "NOTE_T":
            self.state = "NOTE_T_VAL"
        elif s == "NOTE_T_VAL":
            self.state = "NOTE_MIDI"
        elif s == "NOTE_MIDI":
            self.state = "NOTE_MIDI_VAL"
        elif s == "NOTE_MIDI_VAL":
            self.state = "NOTE_VEL"
        elif s == "NOTE_VEL":
            self.state = "NOTE_VEL_VAL"
        elif s == "NOTE_VEL_VAL":
            self.state = "NOTE_D"
        elif s == "NOTE_D":
            self.state = "NOTE_D_VAL"
        elif s == "NOTE_D_VAL":
            self.state = "TRACK_BODY"
        elif s == "AFTER_TRACK":
            self.state = "TRACK_INST" if token == TRACK else "AFTER_MUSIC"
        elif s == "AFTER_MUSIC":
            self.state = "TEXTURE_OR_END"
        elif s == "TEXTURE_OR_END":
            self.state = "TEX_T" if token == TEX else "AFTER_TEXTURES"
        elif s == "TEX_T":
            self.state = "TEX_T_VAL"
        elif s == "TEX_T_VAL":
            self.state = "TEX_D"
        elif s == "TEX_D":
            self.state = "TEX_D_VAL"
        elif s == "TEX_D_VAL":
            self.state = "TEX_CHAR"
        elif s == "TEX_CHAR":
            self.state = "TEX_CHAR_VAL"
        elif s == "TEX_CHAR_VAL":
            self.state = "TEX_GAIN"
        elif s == "TEX_GAIN":
            self.state = "TEX_GAIN_VAL"
        elif s == "TEX_GAIN_VAL":
            self.state = "TEX_WIDTH"
        elif s == "TEX_WIDTH":
            self.state = "TEX_WIDTH_VAL"
        elif s == "TEX_WIDTH_VAL":
            self.state = "TEX_END"
        elif s == "TEX_END":
            self.state = "TEXTURE_OR_END"
        elif s == "AFTER_TEXTURES":
            self.state = "EVENT_OR_END"
        elif s == "EVENT_OR_END":
            if token == EVT:
                self.events_emitted += 1
                self.state = "EVT_T"
            else:
                self.state = "AFTER_EVENTS"
        elif s == "EVT_T":
            self.state = "EVT_T_VAL"
        elif s == "EVT_T_VAL":
            self.state = "EVT_ROLE"
        elif s == "EVT_ROLE":
            self.state = "EVT_ROLE_VAL"
        elif s == "EVT_ROLE_VAL":
            self.state = "EVT_CHAR_FIELD"
        elif s == "EVT_CHAR_FIELD":
            self.state = "EVT_CHAR"
        elif s == "EVT_CHAR":
            self.state = "EVT_D" if token == END_CHAR else "EVT_CHAR_BYTES"
        elif s == "EVT_CHAR_BYTES":
            self.state = "EVT_D" if token == END_CHAR else "EVT_CHAR_BYTES"
        elif s == "EVT_D":
            self.state = "EVT_D_VAL"
        elif s == "EVT_D_VAL":
            self.state = "EVT_APEX"
        elif s == "EVT_APEX":
            self.state = "EVT_APEX_VAL"
        elif s == "EVT_APEX_VAL":
            self.state = "EVT_GAIN"
        elif s == "EVT_GAIN":
            self.state = "EVT_GAIN_VAL"
        elif s == "EVT_GAIN_VAL":
            self.state = "EVT_PAN"
        elif s == "EVT_PAN":
            self.state = "EVT_PAN_VAL"
        elif s == "EVT_PAN_VAL":
            self.state = "EVT_END"
        elif s == "EVT_END":
            self.state = "EVENT_OR_END"
        elif s == "AFTER_EVENTS":
            self.state = "AFTER_SCORE"
        elif s == "AFTER_SCORE":
            self.state = "DONE"
        else:
            raise RuntimeError(f"unknown grammar state {s!r}")

    def mask(self, vocab_size: int) -> list[bool]:
        allowed = self.allowed()
        return [i in allowed for i in range(vocab_size)]
