import json

import pytest

from orb.model.tokenizer import ScoreTokenizer
from orb.score import Score


def test_demo_roundtrip():
    tok = ScoreTokenizer()
    score = Score.demo()
    ids = tok.encode_score(score)
    out = tok.decode_score(ids)

    assert out.sample_rate == score.sample_rate
    assert abs(out.duration - score.duration) < 0.05
    assert len(out.sections) == len(score.sections)
    assert len(out.music) == len(score.music)
    assert len(out.textures) == len(score.textures)
    assert len(out.events) == len(score.events)

    for a, b in zip(out.events, score.events):
        assert a.role == b.role
        assert abs(a.t - b.t) < 0.02
        assert abs(a.dur - b.dur) < 0.05
        assert abs(a.apex - b.apex) < 0.02


def test_caption_roundtrip():
    tok = ScoreTokenizer()
    text = "a soft rounded pop for a friendly notification"
    ids = tok.encode_caption(text)
    assert tok.decode_caption(ids) == text


def test_pair_labels_mask_caption():
    tok = ScoreTokenizer()
    pair = tok.encode_pair("a fast airy whoosh", Score.demo(), max_len=2048)
    assert len(pair.input_ids) == len(pair.labels)
    first_score = pair.labels.index(next(x for x in pair.labels if x != -100))
    assert all(x == -100 for x in pair.labels[:first_score])
    assert pair.labels[-1] == tok.eos_id


def test_vocab_size():
    tok = ScoreTokenizer()
    assert tok.vocab_size > 4000
    assert tok.pad_id < tok.vocab_size


def test_pair_weights_upweight_events():
    tok = ScoreTokenizer()
    pair = tok.encode_pair("a test cue", Score.demo())
    assert len(pair.weights) == len(pair.input_ids)
    assert max(pair.weights) == 2.0
    assert min(pair.weights) == 0.0


def test_invalid_role_rejected_by_score():
    with pytest.raises(ValueError):
        Score.from_dict(
            {
                "duration": 1.0,
                "events": [{"t": 0.0, "role": "laser", "dur": 0.5}],
            }
        )
