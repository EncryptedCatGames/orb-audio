import pytest

from orb.model.grammar import GrammarMask
from orb.model.tokenizer import END_EVENTS, EVT, SCORE, ScoreTokenizer
from orb.score import Score


def test_mask_accepts_full_demo():
    tok = ScoreTokenizer()
    ids = tok.encode_score(Score.demo())
    grammar = GrammarMask()
    for i, token in enumerate(ids):
        assert token in grammar.allowed(), f"token {token} rejected at step {i} in {grammar.state}"
        grammar.advance(token)
    assert grammar.state == "DONE"


def test_mask_rejects_wrong_first_token():
    grammar = GrammarMask()
    with pytest.raises(ValueError):
        grammar.advance(SCORE)


def test_mask_rejects_skipping_meta():
    grammar = GrammarMask()
    grammar.advance(257)  # BOS
    grammar.advance(SCORE)
    with pytest.raises(ValueError):
        grammar.advance(265)  # SECTIONS, expected META


def test_min_events_forces_an_event():
    tok = ScoreTokenizer()
    ids = tok.encode_score(Score.demo())
    grammar = GrammarMask(start="START", min_events=1)
    for token in ids:
        if grammar.state == "EVENT_OR_END":
            assert EVT in grammar.allowed()
            assert END_EVENTS not in grammar.allowed()
            return
        grammar.advance(token)
    raise AssertionError("never reached EVENT_OR_END")
