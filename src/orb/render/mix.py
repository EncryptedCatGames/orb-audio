"""Mix the three decoders into one stereo cue, with stems."""

from __future__ import annotations

import numpy as np

from ..score import Score
from .dsp import db_to_gain, rms
from .events import render_event
from .music import render_track
from .texture import render_texture


def _add_at(out: np.ndarray, sig: np.ndarray, t: float, sr: int) -> None:
    start = max(0, int(round(t * sr)))
    end = min(len(out), start + len(sig))
    if end > start:
        out[start:end] += sig[: end - start]


def _active_ref(x: np.ndarray, sr: int, frame: float = 0.05) -> float:
    """Loudness reference: 95th percentile of active-frame RMS.

    Whole-cue RMS depends on duty cycle, so a sparse cue gets boosted far more
    than a dense one: the same -30 dB bed lands 26 dB louder in a one-pop cue
    than in a five-event one. Measuring only the frames that carry signal keeps
    the foreground loudness stable across cue densities.
    """
    mono = x.mean(axis=1) if x.ndim > 1 else x
    win = max(1, int(frame * sr))
    hop = max(1, win // 2)
    if len(mono) < win:
        return float(np.sqrt(np.mean(mono**2) + 1e-12))
    frames = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop]
    fr = np.sqrt(np.mean(frames**2, axis=1) + 1e-18)
    peak = float(fr.max())
    if peak <= 1e-9:
        return 0.0
    active = fr[fr > 0.05 * peak]
    return float(np.percentile(active, 95)) if active.size else 0.0


def render_stems(score: Score, normalize: bool = True) -> dict[str, np.ndarray]:
    """Render a score to separate stems plus a mix.

    Returns a dict with `music`, `texture`, `events`, and `mix`, each (n, 2)
    float32. The stems sum to the mix, and every stem receives the same safety
    gain, so the stems always reconstruct it.
    """
    score.validate()
    sr = score.sample_rate
    n = int(score.duration * sr)
    music = np.zeros((n, 2), dtype=np.float64)
    texture = np.zeros((n, 2), dtype=np.float64)
    events = np.zeros((n, 2), dtype=np.float64)

    for i, track in enumerate(score.music):
        music += render_track(track, sr, n, seed=score.seed + i)
    for i, tex in enumerate(score.textures):
        _add_at(texture, render_texture(tex, sr, seed=score.seed + 100 + i), tex.t, sr)
    for i, event in enumerate(score.events):
        seed = event.seed if event.seed else score.seed + 200 + i
        _add_at(events, render_event(event, sr, seed=seed), event.t, sr)

    # A texture is continuous, so removing its mean cannot open a hole in
    # silence. Slow LFOs otherwise leave a DC drift over a long cue.
    texture -= texture.mean(axis=0, keepdims=True)
    music -= music.mean(axis=0, keepdims=True)

    if normalize:
        # The foreground sets the loudness; the bed is placed relative to it.
        # Normalising the whole mix instead lets the bed absorb the gain the
        # foreground did not use, which is what made sparse cues hiss.
        fg = music + events
        target = db_to_gain(score.target_lufs)
        fg_peak = float(np.abs(fg).max()) if fg.size else 0.0

        if fg_peak > 1e-6:
            # SFX convention: the foreground peaks just under full scale.
            g_fg = 0.89 / fg_peak
            # The bed sits `gain_db` below the foreground's *active* loudness,
            # so its level does not depend on how sparse the events are.
            fg_ref = _active_ref(fg, sr) * g_fg
            bed_db = score.textures[0].gain_db if score.textures else -30.0
            bed_target = fg_ref * db_to_gain(bed_db)
            tex_rms = max(rms(texture), 1e-9)
            g_tex = min(bed_target / tex_rms, fg_ref / tex_rms)
        else:
            # Ambience-only cue: the bed is the sound, so it takes the target.
            g_fg = 0.0
            tex_rms = max(rms(texture), 1e-9)
            g_tex = target / tex_rms
            tex_peak = float(np.abs(texture).max())
            if tex_peak > 1e-9:
                g_tex = min(g_tex, 0.89 / tex_peak)

        music *= g_fg
        events *= g_fg
        texture *= g_tex

    mix = music + texture + events

    # One safety gain for every stem, so they still sum to the mix.
    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 0.98:
        s = 0.98 / peak
        music *= s
        texture *= s
        events *= s
        mix *= s

    return {
        "music": music.astype(np.float32),
        "texture": texture.astype(np.float32),
        "events": events.astype(np.float32),
        "mix": mix.astype(np.float32),
    }


def render(
    score: Score,
    normalize: bool = True,
    detail=None,
    detail_strength: float = 1.0,
) -> np.ndarray:
    """Render a score to stereo float32, shape (n, 2).

    `detail` is an optional `DetailUNet`; when given, the neural detail stage
    runs after the DSP mix and before the result is returned.
    """
    audio = render_stems(score, normalize)["mix"]
    if detail is not None:
        # Imported here so the DSP path stays numpy-only and never loads torch.
        from .detail import apply_detail

        audio = apply_detail(
            audio,
            detail,
            seed=score.seed,
            strength=detail_strength,
        ).astype(np.float32)
    return audio
