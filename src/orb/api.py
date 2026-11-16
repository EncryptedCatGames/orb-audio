"""Public API for ORB.

Two entry points, one for each path through the engine.

Model-free DSP (needs only numpy)::

    from orb.api import render_spec
    audio = render_spec({
        "duration": 6.0, "texture": "wind", "mood": "bright",
        "events": [{"role": "whoosh", "t": 1.0, "dur": 0.9,
                    "gain_db": -6.0, "pan": -0.5, "seed": 42}],
    })

Full pipeline (needs torch and the planner weights)::

    from orb.api import generate
    out = generate("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a")
    out["audio"]   # (n, 2) float32, 48 kHz
    out["spec"]    # the planned Spec as a dict

Every function is deterministic for a given input and seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .eval.audit import audit_waveform, passes
from .grammar import Spec, SpecEvent, score_from_spec
from .io import write_wav
from .render.mix import render as _render_score

SAMPLE_RATE = 48000
DEFAULT_PLANNER = "weights/planner.pt"
DEFAULT_DETAIL = "weights/detail.pt"


# -- spec <-> dict ---------------------------------------------------------


def spec_to_dict(spec: Spec) -> dict[str, Any]:
    """A JSON-serialisable view of a Spec."""
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


def dict_to_spec(data: dict[str, Any]) -> Spec:
    """Inverse of :func:`spec_to_dict`."""
    events = [
        SpecEvent(
            role=e["role"],
            t=float(e["t"]),
            dur=float(e["dur"]),
            gain_db=float(e.get("gain_db", -12.0)),
            pan=float(e.get("pan", 0.0)),
            seed=int(e.get("seed", 0)),
        )
        for e in data.get("events", [])
    ]
    return Spec(
        duration=float(data["duration"]),
        texture=str(data["texture"]),
        mood=str(data["mood"]),
        events=events,
    )


# -- models ----------------------------------------------------------------


def load_planner(path: str | Path = DEFAULT_PLANNER, device: str = "cpu"):
    """Load the planner from a ``.pt`` checkpoint or a ``.safetensors`` file.

    Returns ``(tokenizer, model)``.
    """
    import torch

    from .model.planner import Planner
    from .model.spec_tokenizer import SpecTokenizer

    path = Path(path)
    tokenizer = SpecTokenizer()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        cfg = json.loads((path.parent / "config.json").read_text())
        model = Planner(cfg.get("vocab_size", tokenizer.vocab_size), **cfg["model"])
        model.load_state_dict(load_file(str(path), device="cpu"))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = Planner(tokenizer.vocab_size, **ckpt["config"]["model"])
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return tokenizer, model


def load_detail(path: str | Path = DEFAULT_DETAIL, device: str = "cpu"):
    """Load the optional detail model from ``.pt`` or ``.safetensors``.

    The detail weights are CC BY-NC-4.0 and are not part of the open-source
    release; see ``weights/README.md``.
    """
    import torch

    from .model.detail import DetailUNet

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        cfg = json.loads((path.parent / "config.json").read_text())
        model = DetailUNet(**cfg["model"])
        model.load_state_dict(load_file(str(path), device="cpu"))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = DetailUNet(**ckpt["config"]["model"])
        model.load_state_dict(ckpt["generator"])
    model.to(device).eval()
    return model


# -- generation ------------------------------------------------------------


def plan(caption: str, tokenizer, planner, device: str = "cpu") -> Spec:
    """Caption -> Spec with constrained decoding, so the result is always valid."""
    import torch

    from .model.spec_tokenizer import SpecGrammar

    prompt = tokenizer.prompt(caption)
    out = planner.generate(
        torch.tensor([prompt], device=device),
        max_new_tokens=1024,
        temperature=1e-6,
        top_p=1.0,
        eos_id=tokenizer.eos_id,
        grammar=SpecGrammar(),
    )
    return tokenizer.decode_spec(out[0, len(prompt) :].tolist())


def render_spec(
    spec: Spec | dict[str, Any],
    detail=None,
    strength: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Spec -> stereo float32 audio at 48 kHz.

    ``detail`` is an optional model from :func:`load_detail`. Without it this
    is pure numpy and needs no torch.
    """
    if isinstance(spec, dict):
        spec = dict_to_spec(spec)
    score = score_from_spec(spec, with_music=False)
    score.seed = int(seed)
    score.validate()
    audio = _render_score(score)
    if detail is not None:
        from .render.detail import apply_detail

        audio = apply_detail(audio, detail, seed=int(seed), strength=float(strength))
    return np.asarray(audio, dtype=np.float32)


def generate(
    caption: str,
    planner_path: str | Path = DEFAULT_PLANNER,
    detail_path: str | Path | None = None,
    device: str = "cpu",
    strength: float = 1.0,
    seed: int = 0,
    planner=None,
    detail=None,
) -> dict[str, Any]:
    """Caption -> audio, spec, and audit metrics in one call.

    Pass ``planner``/``detail`` to reuse already-loaded models, or leave them
    out and the weights are loaded from disk.
    """
    if planner is None:
        tokenizer, model = load_planner(planner_path, device)
    else:
        tokenizer, model = planner
    if detail is None and detail_path is not None and Path(detail_path).exists():
        detail = load_detail(detail_path, device)

    spec = plan(caption, tokenizer, model, device)
    audio = render_spec(spec, detail=detail, strength=strength, seed=seed)
    audit = audit_waveform(audio, SAMPLE_RATE)
    audit["passes"] = passes(audit)
    return {
        "audio": audio,
        "sample_rate": SAMPLE_RATE,
        "spec": spec_to_dict(spec),
        "audit": audit,
    }


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a float array to a 16-bit PCM WAV."""
    write_wav(Path(path), audio, sample_rate)
