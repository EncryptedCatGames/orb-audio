import json

import pytest

from orb.score import Event, Score


def test_demo_roundtrip():
    score = Score.demo()
    score.validate()
    restored = Score.from_dict(json.loads(json.dumps(score.to_dict())))
    assert restored.to_dict() == score.to_dict()


def test_apex_must_be_inside_event():
    with pytest.raises(ValueError):
        Event(1.0, "whoosh", 0.5, apex=2.0)


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        Event(0.0, "laser", 0.5)


def test_out_of_range_event_rejected():
    score = Score(duration=2.0, events=[Event(1.8, "click", 0.5)])
    with pytest.raises(ValueError):
        score.validate()


def test_default_apex():
    ev = Event(1.0, "impact", 0.4)
    assert ev.apex == pytest.approx(1.0 + 0.55 * 0.4)
