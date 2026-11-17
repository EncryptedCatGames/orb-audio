"""The everyday-sound library (v0.3): 31 natural roles, 5 scene beds, and
the DSP-only routing that keeps the detail stage off them."""

from __future__ import annotations

import numpy as np
import pytest
from test_foley import _hf_metrics  # same HF metrics as the first six roles

from orb.api import render_spec
from orb.grammar import (
    AMBIENCE_BED_DB,
    EVENT_STYLE,
    SCENES,
    caption,
    default_bed_db,
    parse_caption,
    sample_scene,
    score_from_spec,
    spec_from_score,
)
from orb.render import foley
from orb.render.natural import ROLES, TONAL, VARIANTS
from orb.score import AMBIENCE_TEXTURES, EVENT_ROLES, TEXTURES

SR = 48000
ONE_SHOT = {"mouse_click", "camera_shutter"}  # fixed physical length (< 0.5 s)
HF_SEEDS = (0, 1, 2, 5)


def _spec(role, pan=0.0, seed=3, gain=-8.0, t=0.5, texture="none"):
    dur = EVENT_STYLE[role][3]
    return {"duration": float(np.ceil(t + dur + 3.0)), "texture": texture, "mood": "neutral",
            "events": [{"role": role, "t": t, "dur": dur, "gain_db": gain, "pan": pan, "seed": seed}]}


def test_roles_and_textures_are_appended():
    n0 = 10 + len(foley.ROLES)
    assert EVENT_ROLES[n0:] == ROLES
    assert TEXTURES[:8] == ("room_cold", "wind", "city_hum", "vinyl", "drone", "rain", "fire", "sci_fi")
    assert TEXTURES[8:] == AMBIENCE_TEXTURES
    assert set(VARIANTS) == set(ROLES) and set(TONAL) <= set(ROLES)
    assert set(EVENT_STYLE) == set(EVENT_ROLES)


@pytest.mark.parametrize("role", ROLES)
def test_deterministic_and_seed_dependent(role):
    a, b = render_spec(_spec(role, seed=5)), render_spec(_spec(role, seed=5))
    c = render_spec(_spec(role, seed=6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("role", ROLES)
def test_silent_outside_the_event_and_ends_at_zero(role):
    t = 0.5
    x = render_spec(_spec(role, t=t))
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x[: int(t * SR) - 1])) == 0.0
    assert np.max(np.abs(x[-int(0.05 * SR):])) < 1e-6
    assert 0.05 < np.max(np.abs(x)) <= 1.0


@pytest.mark.parametrize("role", ROLES)
def test_pan_moves_energy(role):
    left = render_spec(_spec(role, pan=-0.9))
    right = render_spec(_spec(role, pan=0.9))
    e = lambda x, c: float(np.sum(x[:, c].astype(np.float64) ** 2))  # noqa: E731
    assert e(left, 0) > 2.0 * e(left, 1)
    assert e(right, 1) > 2.0 * e(right, 0)


@pytest.mark.parametrize("role", ROLES)
def test_gain_and_duration_are_honoured(role):
    from orb.render.events import render_event
    from orb.score import Event

    def clip(g, dur=EVENT_STYLE[role][3]):
        return render_event(Event(0.0, role, dur, apex=0.0, gain_db=g, pan=0.0, seed=3), SR, seed=3)

    assert float(np.abs(clip(-18.0)).max()) == pytest.approx(float(np.abs(clip(-6.0)).max()) * 10 ** (-12 / 20), rel=1e-4)
    short, long = clip(-6.0, 0.5), clip(-6.0, 2.5)
    assert short.shape[0] == int(0.5 * SR) and long.shape[0] == int(2.5 * SR)
    assert np.abs(short).max() > 0 and np.abs(long).max() > 0
    if role in ONE_SHOT:
        # A click or a shutter has a physical length; dur is the slot it must
        # fit in, not a length to stretch to. Both renders end on true zero.
        assert np.all(short[-1] == 0.0) and np.all(long[-1] == 0.0)
        return
    on = lambda x: np.nonzero(np.abs(x).max(axis=1) > 1e-4 * np.abs(x).max())[0][-1] / SR  # noqa: E731
    assert on(short) < on(long)


# -- no fizz, no buzz, no whistle above 6 kHz (thresholds as in test_foley) --


@pytest.fixture(scope="module")
def hf_table():
    return {(r, s): _hf_metrics(render_spec(_spec(r, seed=s, gain=-6.0))) for r in ROLES for s in HF_SEEDS}


@pytest.mark.parametrize("role", ROLES)
def test_no_energy_near_nyquist(role, hf_table):
    for s in HF_SEEDS:
        assert hf_table[role, s][0] < 1e-3, (role, s)


@pytest.mark.parametrize("role", [r for r in ROLES if r not in TONAL])
def test_noise_roles_do_not_buzz_or_whistle(role, hf_table):
    # Tonal sources (birdsong, bells, horns, sirens, ringing metal) are lines
    # and periodic by nature, and are exempt; everything else is held to the
    # same limits as the first six foley roles.
    for s in HF_SEEDS:
        _, hf6, buzz, tonal = hf_table[role, s]
        if hf6 > 1e-4:
            assert buzz <= 0.5, (role, s, buzz)
        assert tonal < 15.0, (role, s, tonal)


# -- the scene beds -------------------------------------------------------


def _bed(tex, seed=2, dur=8.0):
    return render_spec({"duration": dur, "texture": tex, "mood": "calm", "events": []}, seed=seed)


@pytest.mark.parametrize("tex", AMBIENCE_TEXTURES)
def test_ambience_is_deterministic_band_limited_and_fades(tex):
    a = _bed(tex)
    assert np.array_equal(a, _bed(tex))
    assert not np.array_equal(a, _bed(tex, seed=3))
    assert np.all(np.isfinite(a)) and 0.05 < np.max(np.abs(a)) <= 1.0
    assert np.max(np.abs(a[-int(0.002 * SR):])) < 1e-3 and not np.any(a[-1])
    m = a.mean(axis=1).astype(np.float64)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    assert P[f >= 14000].sum() / P.sum() < 1e-3
    # a scene, not a hiss: most of the energy sits below 6 kHz
    assert P[f >= 6000].sum() / P.sum() < 0.1


def test_ambience_beds_default_near_the_events():
    assert default_bed_db("ocean") == AMBIENCE_BED_DB > default_bed_db("rain")
    spec = sample_scene(np.random.default_rng(3), "beach")
    back = spec_from_score(score_from_spec(spec))
    assert back.texture == "ocean" and back.bed_db is None


@pytest.mark.parametrize("kind", SCENES)
def test_every_scene_round_trips_its_caption(kind):
    rng = np.random.default_rng(7)
    for _ in range(30):
        spec = sample_scene(rng, kind)
        assert spec.events
        back = parse_caption(caption(score_from_spec(spec)))
        assert back is not None and [e.role for e in back.events] == [e.role for e in spec.events]


# -- DSP-only routing ---------------------------------------------------------


def test_dsp_only_stem_is_empty_for_the_original_roles():
    from orb.render.mix import render_stems

    spec = {"duration": 3.0, "texture": "rain", "mood": "calm",
            "events": [{"role": "impact", "t": 0.2, "dur": 0.7, "gain_db": -6, "pan": 0, "seed": 1},
                       {"role": "bell", "t": 1.0, "dur": 1.4, "gain_db": -12, "pan": 0.25, "seed": 2}]}
    from orb.api import dict_to_spec

    stems = render_stems(score_from_spec(dict_to_spec(spec)))
    assert not np.any(stems["dsp_only"])
    stems = render_stems(score_from_spec(dict_to_spec(_spec("church_bell"))))
    assert np.array_equal(stems["dsp_only"], stems["mix"])


def test_detail_stage_skips_the_physical_models():
    pytest.importorskip("torch")
    from pathlib import Path

    from orb.api import load_detail
    from orb.render.detail import apply_detail

    path = Path(__file__).resolve().parents[1] / "weights" / "detail.pt"
    if not path.exists():
        pytest.skip("no detail weights")
    detail = load_detail(path)
    # a pure physical-model cue comes back exactly as the DSP render
    spec = _spec("wave_crash")
    assert np.array_equal(render_spec(spec, detail=detail), render_spec(spec))
    # a cue with only the original roles takes exactly the original path
    old = {"duration": 2.0, "texture": "rain", "mood": "calm",
           "events": [{"role": "whoosh", "t": 0.2, "dur": 0.8, "gain_db": -8, "pan": 0, "seed": 4}]}
    dry = render_spec(old)
    want = np.asarray(apply_detail(dry, detail, seed=0, strength=1.0), dtype=np.float32)
    assert np.array_equal(render_spec(old, detail=detail), want)
