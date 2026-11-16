import numpy as np

from orb.render.events import render_event
from orb.render.mix import render
from orb.score import Event, Score


def test_render_is_deterministic():
    score = Score.demo()
    a = render(score)
    b = render(score)
    assert np.array_equal(a, b)


def test_peak_is_bounded():
    audio = render(Score.demo())
    assert np.max(np.abs(audio)) <= 0.99


def test_whoosh_apex_lands():
    ev = Event(0.5, "whoosh", 0.8, apex=0.737, gain_db=0.0)
    audio = render_event(ev, 24000, seed=1)
    env = np.abs(audio.mean(axis=1))
    k = 120
    env = np.convolve(env, np.ones(k) / k, mode="same")
    peak_t = float(np.argmax(env) / 24000)
    assert abs(peak_t - (ev.apex - ev.t)) < 0.02


def test_stereo_shape():
    score = Score.demo()
    audio = render(score)
    assert audio.ndim == 2 and audio.shape[1] == 2
    assert audio.shape[0] == int(12.0 * score.sample_rate)


def test_stems_sum_to_mix():
    from orb.render.mix import render_stems

    stems = render_stems(Score.demo())
    total = stems["music"] + stems["texture"] + stems["events"]
    assert np.allclose(total, stems["mix"], atol=1e-5)
