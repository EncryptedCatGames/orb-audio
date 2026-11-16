"""Spec grammar, tokenizer and caption determinism."""

from __future__ import annotations

import numpy as np

from orb.grammar import Spec, SpecEvent, caption, sample_spec, score_from_spec, spec_from_score
from orb.model.spec_tokenizer import (
    DIGIT,
    DUR,
    EVENT,
    EVENTS,
    MOOD,
    MOODVAL,
    ROLEVAL,
    SPEC,
    T,
    TEX,
    TEXVAL,
    SpecGrammar,
    SpecTokenizer,
)


def test_spec_score_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        spec = sample_spec(rng)
        score = score_from_spec(spec, with_music=False)
        score.validate()
        back = spec_from_score(score)
        assert back.duration == spec.duration
        assert back.texture == spec.texture
        assert back.mood == spec.mood
        assert len(back.events) == len(spec.events)
        for a, b in zip(back.events, spec.events):
            assert a.role == b.role
            assert abs(a.t - b.t) < 1e-6
            assert abs(a.dur - b.dur) < 1e-6
            assert abs(a.gain_db - b.gain_db) < 1e-6
            assert abs(a.pan - b.pan) < 1e-6
            assert a.seed == b.seed


def test_spec_tokenizer_roundtrip():
    rng = np.random.default_rng(1)
    tok = SpecTokenizer()
    for _ in range(20):
        spec = sample_spec(rng)
        ids = tok.encode_spec(spec)
        assert tok.decode_spec(ids) == spec


def test_grammar_accepts_encoded_spec():
    rng = np.random.default_rng(2)
    tok = SpecTokenizer()
    grammar = SpecGrammar()
    body = tok.encode_spec(sample_spec(rng), add_special=False)
    for token in body:
        assert token in grammar.allowed()
        grammar.advance(token)
    assert grammar.state == "DONE"


def test_grammar_masks_event_times_past_duration():
    grammar = SpecGrammar()
    prefix = (
        [SPEC, DUR]
        + [DIGIT + d for d in (0, 1, 2, 0, 0)]  # 1200 cs = 12.00 s
        + [TEX, TEXVAL, MOOD, MOODVAL, EVENTS, EVENT, ROLEVAL, T]
    )
    for token in prefix:
        assert token in grammar.allowed()
        grammar.advance(token)
    assert grammar.state == "T_DIGITS"
    allowed = grammar.allowed()
    assert DIGIT + 1 in allowed  # 1xxx can still fit under 1198 cs
    assert DIGIT + 9 not in allowed  # 9xxx cannot


def test_grammar_emits_end_events_at_max_events():
    """Eight events must still close with END_EVENTS, then END_SPEC."""
    from orb.grammar import MAX_EVENTS
    from orb.model.spec_tokenizer import END_EVENTS

    spec = Spec(
        12.0,
        "wind",
        "calm",
        [SpecEvent("click", 0.1 * i, 0.05, -10.0, 0.0, i) for i in range(MAX_EVENTS)],
    )
    tok = SpecTokenizer()
    grammar = SpecGrammar()
    body = tok.encode_spec(spec, add_special=False)
    for token in body:
        assert token in grammar.allowed()
        grammar.advance(token)
    assert grammar.state == "DONE"
    # At the event limit, only END_EVENTS is offered.
    grammar = SpecGrammar()
    for token in body[: body.index(END_EVENTS)]:
        grammar.advance(token)
    assert grammar.state == "EVENT_OR_END"
    assert grammar.allowed() == {END_EVENTS}


def test_caption_contains_every_event_field():
    rng = np.random.default_rng(3)
    spec = sample_spec(rng)
    score = score_from_spec(spec, with_music=False)
    text = caption(score)
    assert f"{spec.duration:.2f}-second" in text
    assert spec.texture.replace("_", " ") in text
    assert spec.mood in text
    for ev in score.events:
        assert f"@{ev.t:.2f}" in text
        assert f"{ev.dur:.2f}s" in text
        assert f"{ev.gain_db:+.0f}dB" in text
        assert f"#{ev.seed:02x}" in text
