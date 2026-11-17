import numpy as np

from orb.grammar import EVENT_STYLE
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


# -- noise: band-limited attacks, dark reverb, clean bed edges ---------------

SR = 48000


def _hf_share(x: np.ndarray, cutoff: float, sr: int = SR) -> float:
    mono = x.mean(axis=1) if x.ndim > 1 else x
    p = np.abs(np.fft.rfft(mono)) ** 2
    f = np.fft.rfftfreq(len(mono), 1.0 / sr)
    return float(p[f > cutoff].sum() / (p.sum() + 1e-30))


def test_noise_attacks_do_not_spray_to_nyquist():
    # The raw white-noise bursts these replaced had ~25% (click) of their
    # energy above 12 kHz; the band-limited ones stay well under that.
    limits = {"click": 0.08, "impact": 0.001, "ui_pop": 0.001, "glitch": 0.001}
    for role, limit in limits.items():
        ev = Event(0.0, role, 0.3, gain_db=0.0, seed=5)
        audio = render_event(ev, SR, seed=5)
        assert _hf_share(audio, 12000.0) < limit, role
        assert _hf_share(audio, 18000.0) < limit / 10, role


def test_event_render_is_deterministic_per_seed():
    for role in ("click", "impact", "ui_pop", "glitch", "whoosh"):
        ev = Event(0.2, role, 0.4, gain_db=0.0)
        a = render_event(ev, SR, seed=9)
        b = render_event(ev, SR, seed=9)
        c = render_event(ev, SR, seed=10)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


def test_reverb_ir_is_dark():
    from orb.render.dsp import reverb

    x = np.zeros(SR)
    x[0] = 1.0
    wet = reverb(x, SR, decay=0.5, wet=1.0, seed=3)
    # A white-noise IR has ~67% of its energy above 8 kHz and ~50% above 12.
    assert _hf_share(wet, 8000.0) < 0.03
    assert _hf_share(wet, 12000.0) < 0.002


def test_reverb_tail_rings_out_and_ends_silent():
    ev = Event(0.0, "bell", 0.5, gain_db=0.0, seed=1)
    short = render_event(ev, SR, seed=1)
    long = render_event(ev, SR, seed=1, tail=True)
    assert len(short) == int(0.5 * SR)
    assert len(long) > len(short)
    # The tail decays to nothing, and the cut clip fades to zero.
    assert np.abs(long[-100:]).max() < 1e-6 * np.abs(long).max()
    assert np.abs(short[-1]).max() < 1e-3 * np.abs(short).max()


def test_bed_starts_and_ends_at_zero_without_dc():
    from orb.render.texture import PRESETS, render_texture
    from orb.score import Texture

    for name in PRESETS:
        x = render_texture(Texture(0.0, 2.0, name, gain_db=0.0), SR, seed=4)
        assert np.abs(x[0]).max() < 1e-9 and np.abs(x[-1]).max() < 1e-6, name
        assert np.abs(x.mean(axis=0)).max() < 1e-3, name
        # The first 2 ms are well under the bed level: a fade, not a step.
        assert np.abs(x[:96]).max() < 0.2 * np.abs(x).max(), name


def test_beds_are_ambience_not_hiss():
    """No bed is a flat broadband floor above 10 kHz."""
    from orb.render.texture import PRESETS, render_texture
    from orb.score import Texture

    for name in PRESETS:
        x = render_texture(Texture(0.0, 4.0, name, gain_db=0.0), SR, seed=0).mean(axis=1)
        fr = x[: len(x) // 2048 * 2048].reshape(-1, 2048) * np.hanning(2048)
        p = np.mean(np.abs(np.fft.rfft(fr, axis=1)) ** 2, axis=0)
        f = np.fft.rfftfreq(2048, 1.0 / SR)
        hi = p[f > 10000.0]
        flatness = float(np.exp(np.mean(np.log(hi + 1e-30))) / hi.mean())
        assert flatness < 0.3, name
        assert _hf_share(x, 10000.0) < 0.02, name


def test_cue_end_is_faded():
    from orb.render.mix import render_stems
    from orb.score import Score

    # A bell whose reverb would run past the end of the cue.
    score = Score(duration=1.0, sample_rate=SR, events=[Event(0.4, "bell", 0.6, gain_db=0.0)])
    stems = render_stems(score)
    assert np.abs(stems["mix"][-1]).max() < 1e-6


def test_continuous_roles_start_and_end_without_a_step():
    # whoosh is aligned to its apex by shifting the clip up to 50 ms left;
    # when the fade ran before the shift the clip began mid-waveform, a step
    # of up to -15 dBFS (an audible click on silence).
    for role, dur in (("whoosh", 0.9), ("riser", 2.0), ("swell", 1.5), ("noise_sweep", 2.0)):
        for seed in range(0, 256, 16):
            ev = Event(0.0, role, dur, apex=EVENT_STYLE[role][1] * dur, gain_db=0.0, seed=seed)
            x = render_event(ev, SR, seed=seed, tail=True).mean(axis=1)
            # The first 0.5 ms stays 40 dB below the clip's peak.
            assert np.abs(x[:24]).max() < np.abs(x).max() * 0.01, (role, seed)


def test_continuous_roles_fade_out_smoothly():
    # A noise_sweep ends at ~13% of its peak; a 10 ms linear ramp from there
    # was a small tick. Continuous roles now get a 20-80 ms raised cosine.
    from orb.render.events import _edges

    for dur in (0.3, 1.5, 4.0):
        x = _edges(np.ones(int(dur * SR)), SR, "noise_sweep")
        assert x[0] == 0.0 and x[-1] == 0.0
        assert np.count_nonzero(x < 0.5) >= int(0.01 * SR)  # >= 10 ms each side of the half point
        assert np.abs(np.diff(x[len(x) // 2 :])).max() < 2e-3  # fade-out: >= 20 ms raised cosine
    # Transient roles keep their short attack.
    assert _edges(np.ones(SR), SR, "click")[int(0.002 * SR)] == 1.0
