"""Destruction Cars fork: the UFO roles render, and the hover hum loops."""

import numpy as np
import pytest

from orb.render.events import render_event
from orb.score import EVENT_ROLES, GAME_ROLES, Event

SR = 24000


@pytest.mark.parametrize("role", GAME_ROLES)
def test_ufo_role_renders(role):
    ev = Event(t=0.0, role=role, dur=1.5, gain_db=0.0, seed=3)
    out = render_event(ev, SR, seed=3)
    assert out.shape == (int(1.5 * SR), 2)
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 1e-3


def test_game_roles_leave_planner_roles_alone():
    # The planner's token layout depends on EVENT_ROLES: it must not grow.
    assert len(EVENT_ROLES) == 47
    assert not set(GAME_ROLES) & set(EVENT_ROLES)


def test_hover_loops_without_a_seam():
    out = render_event(Event(t=0.0, role="ufo_hover", dur=3.0, gain_db=0.0, seed=1), SR, seed=1)
    seam = np.abs(out[0] - out[-1]).max()
    typical = np.percentile(np.abs(np.diff(out, axis=0)), 99)
    assert seam <= typical
