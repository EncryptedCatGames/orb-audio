"""Natural and foley roles: thunder, door_slam, knock, footsteps, water_drop, explosion."""

from __future__ import annotations

import numpy as np
import pytest

from orb.api import render_spec
from orb.grammar import EVENT_STYLE, SCENES, Spec, caption, parse_caption, sample_scene, sample_spec, score_from_spec
from orb.render.foley import ROLES, VARIANTS
from orb.score import EVENT_ROLES

SR = 48000
ORIGINAL = ("whoosh", "impact", "ui_pop", "riser", "swell", "click", "bell", "sub_drop", "glitch", "noise_sweep")


def _spec(role, pan=0.0, seed=3, gain=-8.0, t=0.5, texture="none"):
    dur = EVENT_STYLE[role][3]
    return {"duration": float(np.ceil(t + dur + 1.5)), "texture": texture, "mood": "neutral",
            "events": [{"role": role, "t": t, "dur": dur, "gain_db": gain, "pan": pan, "seed": seed}]}


def test_roles_are_appended_not_inserted():
    assert EVENT_ROLES[:10] == ORIGINAL
    assert EVENT_ROLES[10 : 10 + len(ROLES)] == ROLES
    assert set(VARIANTS) == set(ROLES)


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
    assert np.max(np.abs(x)) <= 1.0
    assert np.max(np.abs(x)) > 0.05


@pytest.mark.parametrize("role", ROLES)
def test_pan_moves_energy(role):
    left = render_spec(_spec(role, pan=-0.9))
    right = render_spec(_spec(role, pan=0.9))
    e = lambda x, c: float(np.sum(x[:, c].astype(np.float64) ** 2))  # noqa: E731
    assert e(left, 0) > 2.0 * e(left, 1)
    assert e(right, 1) > 2.0 * e(right, 0)


@pytest.mark.parametrize("role", ROLES)
def test_gain_scales_level(role):
    # The mix peak-normalises a cue, so gain is checked on the event itself.
    from orb.render.events import render_event
    from orb.score import Event

    def peak(g):
        ev = Event(0.0, role, EVENT_STYLE[role][3], apex=0.0, gain_db=g, pan=0.0, seed=3)
        return float(np.max(np.abs(render_event(ev, SR, seed=3))))

    assert peak(-18.0) == pytest.approx(peak(-6.0) * 10 ** (-12 / 20), rel=1e-4)


def test_thunder_is_low_and_long():
    x = render_spec(_spec("thunder", seed=3)).mean(axis=1)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    low = spec[(f >= 25) & (f < 250)].sum() / spec.sum()
    assert low > 0.4  # far thunder is rumble
    env = np.abs(x)
    active = np.nonzero(env > 1e-3 * env.max())[0]
    assert (active[-1] - active[0]) / SR > 2.5


def test_water_drop_glides_up():
    x = render_spec(_spec("water_drop", seed=1, gain=-6)).mean(axis=1)
    on = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
    seg = lambda a, b: x[on + int(a * SR): on + int(b * SR)]  # noqa: E731

    def peak(s):
        w = np.abs(np.fft.rfft(s * np.hanning(len(s)), 8 * len(s)))
        return np.fft.rfftfreq(8 * len(s), 1 / SR)[np.argmax(w)]

    assert peak(seg(0.03, 0.06)) > peak(seg(0.0, 0.02))


def test_spec_tokenizer_round_trips_new_roles():
    pytest.importorskip("torch")
    from orb.model.spec_tokenizer import VOCAB_SIZE, SpecTokenizer

    tok = SpecTokenizer()
    from orb.score import TEXTURES

    assert tok.vocab_size == VOCAB_SIZE == 342 + len(EVENT_ROLES) + len(TEXTURES) - 8 + 1  # +1: NOBED
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(400):
        spec = sample_scene(rng)
        ids = tok.encode_spec(spec)
        assert max(ids) < VOCAB_SIZE
        back = tok.decode_spec(ids)
        assert [e.role for e in back.events] == [e.role for e in spec.events]
        seen |= {e.role for e in spec.events}
    assert set(ROLES) <= seen


def test_spec_tokenizer_rejects_texture_none_clearly():
    pytest.importorskip("torch")
    from orb.model.spec_tokenizer import SpecTokenizer

    tok = SpecTokenizer()
    with pytest.raises(ValueError, match="texture"):
        tok.encode_spec(Spec(2.0, "none", "neutral", []))


def test_legacy_tokenizer_extension_range():
    pytest.importorskip("torch")
    from orb.model import tokenizer as legacy

    assert {legacy.ROLE_VAL + i for i in range(10)} <= legacy.role_tokens()
    for role in EVENT_ROLES:
        assert legacy.role_from_token(legacy.role_token(role)) == role
    ext = [legacy.role_token(r) for r in ROLES]
    assert min(ext) >= legacy.ROLE_EXT and max(ext) < legacy.VOCAB_SIZE


@pytest.mark.parametrize("kind", SCENES)
def test_scenes_sample_and_caption_round_trip(kind):
    rng = np.random.default_rng(11)
    for _ in range(50):
        spec = sample_scene(rng, kind)
        assert spec.events and all(e.t + e.dur <= spec.duration + 1e-6 for e in spec.events)
        back = parse_caption(caption(score_from_spec(spec)))
        assert back is not None and [e.role for e in back.events] == [e.role for e in spec.events]


def test_sample_spec_mixes_scenes_and_new_roles():
    rng = np.random.default_rng(0)
    roles = [e.role for _ in range(2000) for e in sample_spec(rng).events]
    for r in EVENT_ROLES:
        assert r in roles


# -- no fizz, no buzz, no whistle above 6 kHz ----------------------------------
# Measured on the DSP render of one event. Each check names the artefact a
# listener reported: energy to Nyquist on the attacks ("electric" fizz), a
# periodic high-band envelope (buzz), and narrow lines in the 4-16 kHz band
# (whistle, comb). Thresholds were set against the original roles: their
# local tonality is <= 9 dB (the bell, tonal by design, sits at 33-37 dB).
HF_SEEDS = (0, 1, 2, 5)


def _hf_metrics(x):
    from orb.render.dsp import bandlimit

    m = x.mean(axis=1).astype(np.float64)
    P = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    hf14 = P[f >= 14000].sum() / P.sum()
    hf6 = P[f >= 6000].sum() / P.sum()

    # periodicity of the >6 kHz envelope inside the event: the highest local
    # autocorrelation peak at a 2-30 ms lag, after removing the 40 ms trend
    env = np.abs(bandlimit(m, SR, 6000.0, None, order=8))
    k = int(0.0005 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    act = np.nonzero(env > 0.05 * env.max())[0]
    buzz = 0.0
    seg = env[act[0] : act[-1] + 1]
    w = int(0.04 * SR)
    if len(seg) > w + int(0.03 * SR):
        r = seg - np.convolve(seg, np.ones(w) / w, mode="same")
        r = r - r.mean()
        F = np.fft.rfft(r, 2 * len(r))  # autocorrelation by FFT (np.correlate is O(n^2) on long events)
        ac = np.fft.irfft(F * np.conj(F), 2 * len(r))[: len(r)]
        ac = ac / max(ac[0], 1e-30)
        a = ac[int(0.002 * SR) - 1 : int(0.030 * SR) + 1]
        peaks = a[1:-1][(a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])]
        buzz = float(peaks.max()) if len(peaks) else 0.0

    # the strongest narrow line in 4-16 kHz over its +-400 Hz neighbourhood
    nf, hop = 4096, 1024
    S = np.mean([np.abs(np.fft.rfft(m[i : i + nf] * np.hanning(nf))) ** 2 for i in range(0, len(m) - nf, hop)], axis=0)
    ff = np.fft.rfftfreq(nf, 1 / SR)
    band = (ff >= 4000) & (ff <= 16000)
    wl = int(800 / (SR / nf)) | 1
    loc = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(S, wl // 2, mode="edge"), wl), axis=1)
    sel = band & (S > S[band].max() * 1e-5)
    tonal = float(10 * np.log10((S[sel] / np.maximum(loc[sel], 1e-30)).max()))
    return hf14, hf6, buzz, tonal


@pytest.fixture(scope="module")
def hf_table():
    return {(r, s): _hf_metrics(render_spec(_spec(r, seed=s, gain=-6.0))) for r in ROLES for s in HF_SEEDS}


@pytest.mark.parametrize("role", ROLES)
def test_no_energy_near_nyquist(role, hf_table):
    for s in HF_SEEDS:
        assert hf_table[role, s][0] < 1e-3, (role, s)  # < 0.1 % above 14 kHz


@pytest.mark.parametrize("role", ROLES)
def test_high_band_does_not_buzz(role, hf_table):
    for s in HF_SEEDS:
        hf14, hf6, buzz, _ = hf_table[role, s]
        # an envelope 40 dB under the sound is inaudible, buzz or not
        if hf6 > 1e-4:
            assert buzz <= 0.5, (role, s, buzz)


@pytest.mark.parametrize("role", ROLES)
def test_no_whistle_lines_in_the_high_band(role, hf_table):
    for s in HF_SEEDS:
        assert hf_table[role, s][3] < 15.0, (role, s)


def test_door_slam_is_a_low_wooden_slam():
    for s in range(6):
        x = render_spec(_spec("door_slam", seed=s, gain=-6.0)).mean(axis=1)
        P = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        centroid = (f * P).sum() / P.sum()
        assert 250 < centroid < 650, (s, centroid)
        assert P[f >= 6000].sum() / P.sum() < 0.02
        assert P[f < 200].sum() / P.sum() > 0.2  # the whump and the panel
