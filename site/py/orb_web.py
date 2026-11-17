"""Browser entry point for ORB.

Runs inside Pyodide (see ``site/worker.js``) and also inside
``scripts/build_site.py``, so the gallery's pre-rendered visuals and the live
in-browser renders are computed by exactly the same code.

numpy only: it uses the public ``orb.api.render_spec`` DSP path, never torch.
"""

from __future__ import annotations

import base64
import time

import numpy as np

import sys


def _install_wasm32_window_shim() -> None:
    """Make ``sliding_window_view(x, w)[::hop]`` work on 32-bit WebAssembly.

    The renderer builds windowed views as ``sliding_window_view(x, w)[::hop]``
    (``orb.render.mix._active_ref`` and ``orb.render.dsp.stft``). The full view
    is never materialised, but numpy still checks its *nominal* byte size,
    which for a 12 s cue is ~11 GB and overflows the 2 GB limit of a 32-bit
    address space, so Pyodide raises "array is too big". This shim returns a
    lazy object whose ``[::hop]`` is the exact same strided view, built
    directly at its real (small) size. Results are bit-identical; native
    platforms are left alone.
    """
    if sys.platform != "emscripten":
        return
    from numpy.lib import stride_tricks

    original = stride_tricks.sliding_window_view
    if getattr(original, "_orb_shim", False):
        return

    class _Windows:
        def __init__(self, x: np.ndarray, w: int) -> None:
            self.x, self.w = x, w

        def __getitem__(self, key):
            if isinstance(key, slice) and key.start is None and key.stop is None:
                step = key.step or 1
                n = (len(self.x) - self.w) // step + 1
                s = self.x.strides[0]
                return stride_tricks.as_strided(
                    self.x, shape=(n, self.w), strides=(s * step, s), writeable=False
                )
            return original(self.x, self.w)[key]

    def shim(x, window_shape, axis=None, *, subok=False, writeable=False):
        x = np.asarray(x)
        if (
            x.ndim == 1
            and axis is None
            and isinstance(window_shape, (int, np.integer))
            and not writeable
            and (len(x) - int(window_shape) + 1) * int(window_shape) * x.itemsize > (1 << 30)
        ):
            return _Windows(x, int(window_shape))
        return original(x, window_shape, axis=axis, subok=subok, writeable=writeable)

    shim._orb_shim = True
    stride_tricks.sliding_window_view = shim
    np.lib.stride_tricks.sliding_window_view = shim
    # orb.render.dsp imports the name directly, so patch it if already loaded.
    dsp = sys.modules.get("orb.render.dsp")
    if dsp is not None:
        dsp.sliding_window_view = shim


_install_wasm32_window_shim()

SAMPLE_RATE = 48000
PEAK_COLUMNS = 240
SPEC_BINS = 48
SPEC_FRAMES = 240


# Loudness cap. render_spec peak-normalises every cue, so a dense cue (a long
# vent, a stacked hit) plays several dB louder than a sparse one. Capping the
# momentary loudness (BS.1770 K-weighting, 400 ms windows) at one target keeps
# the playground's presets, and every edit of them, within about 1 dB of each
# other. It only ever attenuates, so the peak ceiling still holds.
LOUDNESS_CAP_LUFS_M = -13.0


def _k_weight_gain(freqs: np.ndarray) -> np.ndarray:
    """|H(f)| of the BS.1770 K-weighting pair (48 kHz coefficients)."""
    z = np.exp(-2j * np.pi * freqs / SAMPLE_RATE)

    def mag(b, a):
        return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z))

    shelf = mag((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
    hpf = mag((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))
    return shelf * hpf


def momentary_max(audio: np.ndarray) -> float:
    """Max momentary loudness (LUFS-M, 400 ms window, 100 ms hop)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(n + SAMPLE_RATE // 2)))
    spec = np.fft.rfft(x, nfft, axis=0) * _k_weight_gain(np.fft.rfftfreq(nfft, 1.0 / SAMPLE_RATE))[:, None]
    k = np.fft.irfft(spec, nfft, axis=0)[:n]
    power = np.concatenate([[0.0], np.cumsum((k * k).sum(axis=1))])
    win = min(n, int(0.4 * SAMPLE_RATE))
    starts = np.arange(0, max(1, n - win + 1), int(0.1 * SAMPLE_RATE))
    ms = (power[starts + win] - power[starts]) / max(1, win)
    return float(-0.691 + 10.0 * np.log10(ms.max() + 1e-20))


def level(audio: np.ndarray, cap: float = LOUDNESS_CAP_LUFS_M) -> np.ndarray:
    """Attenuate ``audio`` so its momentary loudness never exceeds ``cap``."""
    over = momentary_max(audio) - cap
    if over <= 0.0:
        return audio
    return (audio * 10.0 ** (-over / 20.0)).astype(audio.dtype, copy=False)


def peaks(audio: np.ndarray, columns: int = PEAK_COLUMNS) -> list[list[float]]:
    """Per-column (min, max) of the mono mix, rounded for a small JSON."""
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    n = len(mono)
    out: list[list[float]] = []
    for i in range(columns):
        a = (i * n) // columns
        b = max(a + 1, ((i + 1) * n) // columns)
        chunk = mono[a:b]
        out.append([round(float(chunk.min()), 3), round(float(chunk.max()), 3)])
    return out


def spectrogram(audio: np.ndarray, bins: int = SPEC_BINS, frames: int = SPEC_FRAMES) -> str:
    """A log-frequency magnitude spectrogram, quantised to uint8, base64.

    Layout is ``frames`` rows of ``bins`` bytes (low frequency first), 0 is
    -90 dB and 255 is the loudest cell. Small enough to ship in JSON.
    """
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    mono = np.asarray(mono, dtype=np.float64)
    n_fft = 2048
    n = len(mono)
    hop = max(1, (n - n_fft) // max(1, frames - 1))
    padded = np.pad(mono, (0, n_fft + hop * frames))
    idx = np.arange(frames)[:, None] * hop + np.arange(n_fft)[None, :]
    win = np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(padded[idx] * win, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(40.0, 20000.0, bins + 1)
    band = np.zeros((frames, bins))
    for k in range(bins):
        sel = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if not sel.any():
            sel = np.argmin(np.abs(freqs - edges[k]))
            band[:, k] = mag[:, sel]
        else:
            band[:, k] = mag[:, sel].mean(axis=1)
    db = 20.0 * np.log10(band + 1e-9)
    db -= db.max()
    q = np.clip((db + 90.0) / 90.0, 0.0, 1.0) * 255.0
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode("ascii")


def render(spec: dict, seed: int = 0) -> dict:
    """Spec dict -> interleaved float32 PCM plus the visuals and the audit."""
    from orb.api import render_spec
    from orb.eval.audit import audit_waveform, passes

    t0 = time.perf_counter()
    audio = level(render_spec(spec, seed=seed))
    ms = (time.perf_counter() - t0) * 1000.0
    audit = audit_waveform(audio, SAMPLE_RATE)
    return {
        "pcm": np.ascontiguousarray(audio, dtype=np.float32).reshape(-1),
        "channels": int(audio.shape[1]),
        "sample_rate": SAMPLE_RATE,
        "ms": round(ms, 1),
        "peaks": peaks(audio),
        "spectrogram": spectrogram(audio),
        "audit": {k: float(v) for k, v in audit.items()},
        "passes": bool(passes(audit)),
    }


def render_js(spec_json: str, seed: int = 0):
    """JSON in, a dict whose ``pcm`` is a numpy buffer, for the worker."""
    import json

    return render(json.loads(spec_json), seed=seed)
