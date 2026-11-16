"""Public API: spec round-trip, deterministic render, optional weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orb.api import dict_to_spec, render_spec, spec_to_dict
from orb.grammar import Spec, SpecEvent

ROOT = Path(__file__).resolve().parents[1]

SPEC = {
    "duration": 4.0,
    "texture": "wind",
    "mood": "bright",
    "events": [
        {"role": "whoosh", "t": 1.0, "dur": 0.9, "gain_db": -6.0, "pan": -0.5, "seed": 42},
    ],
}


def test_spec_dict_roundtrip():
    spec = Spec(6.0, "wind", "bright", [SpecEvent("whoosh", 1.0, 0.9, -6.0, -0.5, 42)])
    assert dict_to_spec(spec_to_dict(spec)) == spec


def test_render_spec_shape_and_dtype():
    audio = render_spec(SPEC)
    assert audio.shape == (4 * 48000, 2)
    assert audio.dtype == np.float32
    assert np.isfinite(audio).all()


def test_render_spec_is_deterministic():
    a = render_spec(SPEC, seed=3)
    b = render_spec(SPEC, seed=3)
    assert np.array_equal(a, b)


def test_seed_changes_the_audio():
    a = render_spec(SPEC, seed=0)
    b = render_spec(SPEC, seed=1)
    assert not np.array_equal(a, b)


def test_render_spec_accepts_a_spec_object():
    spec = dict_to_spec(SPEC)
    assert np.array_equal(render_spec(spec), render_spec(SPEC))


def test_ambience_only_renders():
    audio = render_spec({"duration": 2.0, "texture": "rain", "mood": "calm", "events": []})
    assert audio.shape == (2 * 48000, 2)
    assert np.isfinite(audio).all()


def test_render_spec_rejects_unknown_texture():
    with pytest.raises(ValueError):
        render_spec({**SPEC, "texture": "not_a_texture"})


def test_generate_with_planner():
    pytest.importorskip("torch")
    planner = ROOT / "weights/planner.pt"
    if not planner.exists():
        pytest.skip("planner weights not present")
    from orb.api import generate

    out = generate(
        "6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a",
        planner_path=planner,
    )
    assert out["spec"]["texture"] == "wind"
    assert out["spec"]["events"][0]["role"] == "whoosh"
    assert out["audit"]["passes"]
    assert out["audio"].shape[1] == 2


def test_safetensors_loader_matches_pt():
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    st = ROOT / "hf/model-planner/planner.safetensors"
    pt = ROOT / "weights/planner.pt"
    if not st.exists() or not pt.exists():
        pytest.skip("weights not present")
    from orb.api import load_planner

    tok_a, model_a = load_planner(pt)
    tok_b, model_b = load_planner(st)
    for (ka, va), (kb, vb) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert ka == kb
        assert va.shape == vb.shape
        assert np.allclose(va.numpy(), vb.numpy(), atol=1e-6)
