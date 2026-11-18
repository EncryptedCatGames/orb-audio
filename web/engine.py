"""Shared generation engine for the ORB web demo.

The build script and the HTTP server both go through this module so the
example set and live generation can never drift apart.

Caption -> planner -> Spec -> deterministic compiler -> DSP render ->
optional neural detail -> stereo 48 kHz WAV + audit metrics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from orb.api import load_detail, load_planner  # noqa: E402
from orb.eval.audit import audit_waveform, passes  # noqa: E402
from orb.grammar import NO_BED, Spec, parse_caption, score_from_spec  # noqa: E402
from orb.io import read_wav, write_wav  # noqa: E402
from orb.model.detail import DetailUNet  # noqa: E402
from orb.model.planner import Planner  # noqa: E402
from orb.model.spec_tokenizer import SpecGrammar, SpecTokenizer  # noqa: E402
from orb.render.mix import render  # noqa: E402

PLANNER_PATH = os.environ.get("ORB_PLANNER", "weights/planner.pt")
DETAIL_PATH = os.environ.get("ORB_DETAIL", "weights/detail.pt")
SAMPLE_RATE = 48000
PEAK_BUCKETS = 360
MAX_CAPTION_CHARS = 400


_ORB_WEB = None


def _orb_web():
    """The site's renderer module (site/py/orb_web.py), loaded once."""
    global _ORB_WEB
    if _ORB_WEB is None:
        import importlib.util

        path = ROOT / "site" / "py" / "orb_web.py"
        spec = importlib.util.spec_from_file_location("orb_web", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ORB_WEB = mod
    return _ORB_WEB


def resolve_model_path(path: str) -> Path:
    """Accept either the .pt checkpoint or a .safetensors sibling."""
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    alt = candidate.with_suffix(".safetensors")
    return alt if alt.exists() else candidate


class BadInput(ValueError):
    """A request problem that should be reported as HTTP 400."""


def normalize_caption(text: str) -> str:
    """Strip, collapse whitespace, and reject empty or oversized captions."""
    if not isinstance(text, str):
        raise BadInput("caption must be a string")
    caption = " ".join(text.split())
    if not caption:
        raise BadInput("caption is empty")
    if len(caption) > MAX_CAPTION_CHARS:
        raise BadInput(
            f"caption is {len(caption)} characters; the limit is {MAX_CAPTION_CHARS}"
        )
    return caption


def spec_to_dict(spec: Spec) -> dict:
    return {
        "duration": float(spec.duration),
        "texture": spec.texture,
        "mood": spec.mood,
        "events": [
            {
                "role": e.role,
                "t": float(e.t),
                "dur": float(e.dur),
                "gain_db": float(e.gain_db),
                "pan": float(e.pan),
                "seed": int(e.seed),
            }
            for e in spec.events
        ],
    }


def compute_peaks(audio: np.ndarray, buckets: int = PEAK_BUCKETS) -> list[list[float]]:
    """Min/max envelope of the mono mix, normalised so the global max abs is 1."""
    a = np.asarray(audio, dtype=np.float64)
    mono = a.mean(axis=1) if a.ndim > 1 else a
    n = int(mono.shape[0])
    if n == 0:
        return []
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    peaks: list[list[float]] = []
    for i in range(buckets):
        seg = mono[edges[i] : edges[i + 1]]
        if seg.size == 0:
            peaks.append([0.0, 0.0])
        else:
            peaks.append([float(seg.min()), float(seg.max())])
    scale = max((max(abs(lo), abs(hi)) for lo, hi in peaks), default=0.0)
    if scale <= 0.0:
        scale = 1.0
    return [[lo / scale, hi / scale] for lo, hi in peaks]


class Engine:
    """Lazy-loaded planner/detail singletons plus a small on-disk cache."""

    def __init__(self, out_dir: str | Path | None = None, device: str = "cpu"):
        self.out_dir = Path(out_dir) if out_dir is not None else ROOT / "web" / "out"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.planner_path = PLANNER_PATH
        self.detail_path = DETAIL_PATH
        self._lock = threading.Lock()
        self._tok: SpecTokenizer | None = None
        self._planner: Planner | None = None
        self._detail: DetailUNet | None = None

    # -- model loading -----------------------------------------------------

    def warmup(self) -> "Engine":
        """Load both checkpoints up front so the first request is not a surprise."""
        with self._lock:
            self._ensure_planner()
            self._ensure_detail()
        return self

    def _ensure_planner(self) -> tuple[SpecTokenizer, Planner]:
        if self._planner is None:
            tok, model = load_planner(
                resolve_model_path(self.planner_path), device=str(self.device)
            )
            self._tok, self._planner = tok, model
        assert self._tok is not None
        return self._tok, self._planner

    def _ensure_detail(self) -> DetailUNet | None:
        if self._detail is None:
            path = resolve_model_path(self.detail_path)
            if not path.exists():
                # The detail weights are CC BY-NC-4.0 and ship as a separate
                # download; the DSP path is a complete cue without them.
                return None
            self._detail = load_detail(path, device=str(self.device))
        return self._detail

    @property
    def detail_available(self) -> bool:
        return resolve_model_path(self.detail_path).exists()

    # -- generation --------------------------------------------------------

    def _plan(self, tok: SpecTokenizer, planner: Planner, caption: str) -> Spec:
        prompt = tok.prompt(caption)
        out = planner.generate(
            torch.tensor([prompt], device=self.device),
            max_new_tokens=1024,
            temperature=1e-6,
            top_p=1.0,
            eos_id=tok.eos_id,
            grammar=SpecGrammar(),
        )
        return tok.decode_spec(out[0, len(prompt) :].tolist())

    def plan_spec(self, caption: str) -> dict:
        """Caption -> Spec dict, the same routing as ``generate`` (a ``no bed``
        grammar caption is read exactly). Used by the site's fast server path."""
        cap = normalize_caption(caption)
        started = time.perf_counter()
        try:
            parsed = parse_caption(cap)
        except ValueError:
            parsed = None
        with self._lock:
            if parsed is not None and parsed.texture == NO_BED:
                return {"spec": spec_to_dict(parsed), "planned": False, "ms": 0.0, "tokens": 0}
            tok, planner = self._ensure_planner()
            prompt = tok.prompt(cap)
            out = planner.generate(
                torch.tensor([prompt], device=self.device),
                max_new_tokens=1024,
                temperature=1e-6,
                top_p=1.0,
                eos_id=tok.eos_id,
                grammar=SpecGrammar(),
            )
            ids = out[0, len(prompt) :].tolist()
            spec = tok.decode_spec(ids)
        ms = (time.perf_counter() - started) * 1000.0
        return {"spec": spec_to_dict(spec), "planned": True, "ms": round(ms, 1), "tokens": len(ids)}

    def render_spec(self, spec: dict, seed: int = 0) -> dict:
        """Spec dict -> the exact output of the in-browser renderer
        (``site/py/orb_web.render``): same loudness cap, peaks and audit."""
        orb_web = _orb_web()
        try:
            with self._lock:
                return orb_web.render(spec, seed=int(seed) & 0xFFFFFFFF)
        except (ValueError, KeyError, TypeError) as err:
            raise BadInput(f"invalid spec: {err}") from None

    @staticmethod
    def cache_id(caption: str, detail: bool, strength: float) -> str:
        key = f"{caption}|{bool(detail)}|{float(strength):.3f}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    def generate(
        self,
        caption: str,
        detail: bool = False,
        strength: float = 1.0,
        *,
        force: bool = False,
    ) -> dict:
        cap = normalize_caption(caption)
        detail = bool(detail)
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            raise BadInput("strength must be a number between 0 and 1") from None
        if not (0.0 <= strength <= 1.0):
            raise BadInput("strength must be between 0 and 1")
        strength = round(strength, 3)

        cache_id = self.cache_id(cap, detail, strength)
        wav_path = self.out_dir / f"{cache_id}.wav"
        meta_path = self.out_dir / f"{cache_id}.json"
        started = time.perf_counter()

        # One torch run at a time: the server is threaded and concurrent
        # inference just thrashes the CPU.
        with self._lock:
            if force:
                for path in (wav_path, meta_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

            if wav_path.exists():
                data = self._load_cached(cache_id, wav_path, meta_path, cap, detail, strength)
                data["ms"] = int((time.perf_counter() - started) * 1000)
                data["cached"] = True
                return data

            # A bed-less caption ("no bed") is outside the planner's training
            # distribution; render it straight from the parsed Spec instead.
            try:
                parsed = parse_caption(cap)
            except ValueError as err:
                raise BadInput(str(err)) from None
            if parsed is not None and parsed.texture == NO_BED:
                spec = parsed
            else:
                tok, planner = self._ensure_planner()
                spec = self._plan(tok, planner, cap)
            score = score_from_spec(spec, with_music=False)
            score.validate()

            detail_model = self._ensure_detail() if detail else None
            # render(..., detail=) routes the physical-model roles and scene
            # beds around the detail stage (render.mix.apply_detail_routed).
            audio = render(score, detail=detail_model, detail_strength=strength)
            if not np.isfinite(audio).all():
                raise RuntimeError("non-finite samples in output")

            write_wav(wav_path, audio, score.sample_rate)
            audit = audit_waveform(audio, score.sample_rate)
            ok = passes(audit)
            peaks = compute_peaks(audio)
            ms = int((time.perf_counter() - started) * 1000)

            data = {
                "id": cache_id,
                "caption": cap,
                "duration_s": float(score.duration),
                "spec": spec_to_dict(spec),
                "audit": audit,
                "passes": bool(ok),
                "peaks": peaks,
                "ms": ms,
                "detail": detail,
                "detail_applied": detail_model is not None,
                "strength": strength,
                "url": f"/audio/{cache_id}.wav",
            }
            meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            data["cached"] = False
            return data

    def _load_cached(
        self,
        cache_id: str,
        wav_path: Path,
        meta_path: Path,
        caption: str,
        detail: bool,
        strength: float,
    ) -> dict:
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                if data.get("id") == cache_id and "peaks" in data and "spec" in data:
                    return data
            except (OSError, json.JSONDecodeError):
                pass

        # Sidecar missing or unreadable: rebuild the metrics from the WAV and
        # re-plan only to recover the spec that produced it.
        audio, sr = read_wav(wav_path)
        audit = audit_waveform(audio, sr)
        tok, planner = self._ensure_planner()
        spec = self._plan(tok, planner, caption)
        return {
            "id": cache_id,
            "caption": caption,
            "duration_s": float(spec.duration),
            "spec": spec_to_dict(spec),
            "audit": audit,
            "passes": bool(passes(audit)),
            "peaks": compute_peaks(audio),
            "ms": 0,
            "detail": detail,
            "strength": strength,
            "url": f"/audio/{cache_id}.wav",
        }
