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


# -- weight resolution (offline: the Hub download is always monkeypatched) ---


@pytest.fixture
def no_weights(tmp_path, monkeypatch):
    """A cwd and checkout root with no weights, no env vars, no Hub access."""
    import orb.api as api

    cwd = tmp_path / "cwd"
    repo = tmp_path / "repo"
    cwd.mkdir()
    repo.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(api, "_REPO_ROOT", repo)
    monkeypatch.delenv("ORB_PLANNER", raising=False)
    monkeypatch.delenv("ORB_DETAIL", raising=False)

    def no_network(repo_id, name):
        raise AssertionError("tests must not hit the network")

    monkeypatch.setattr(api, "_hub_snapshot", no_network)
    return cwd, repo


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_resolve_planner_explicit_and_env(no_weights, monkeypatch):
    from orb.api import resolve_planner_path

    cwd, _ = no_weights
    local = _touch(cwd / "weights/planner.pt")
    other = _touch(cwd / "elsewhere/planner.pt")
    assert resolve_planner_path(other) == other
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_planner_path(cwd / "missing.pt")
    monkeypatch.setenv("ORB_PLANNER", str(other))
    assert resolve_planner_path() == other
    monkeypatch.setenv("ORB_PLANNER", str(cwd / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="ORB_PLANNER"):
        resolve_planner_path()
    monkeypatch.delenv("ORB_PLANNER")
    assert resolve_planner_path().resolve() == local.resolve()


def test_resolve_planner_local_order(no_weights):
    from orb.api import resolve_planner_path

    cwd, repo = no_weights
    # safetensors without its config.json does not count
    _touch(repo / "weights/planner.safetensors")
    st_cwd = _touch(cwd / "weights/planner.safetensors")
    _touch(cwd / "weights/config.json")
    assert resolve_planner_path().resolve() == st_cwd.resolve()
    # a .pt anywhere beats a safetensors pair
    pt_repo = _touch(repo / "weights/planner.pt")
    assert resolve_planner_path().resolve() == pt_repo.resolve()
    # ./weights beats the checkout
    pt_cwd = _touch(cwd / "weights/planner.pt")
    assert resolve_planner_path().resolve() == pt_cwd.resolve()
    # an explicit folder is searched too
    assert resolve_planner_path(cwd / "weights") == cwd / "weights/planner.pt"


def test_resolve_planner_hub_fallback(no_weights, tmp_path, monkeypatch):
    import sys
    import types

    import orb.api as api

    snap = tmp_path / "snapshot"
    st = _touch(snap / "planner.safetensors")
    _touch(snap / "config.json")
    calls = []

    def fake_snapshot(repo_id, name):
        calls.append((repo_id, name))
        return snap

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    monkeypatch.setattr(api, "_hub_snapshot", fake_snapshot)
    assert api.resolve_planner_path() == st
    assert calls == [("marcosmaceo/orb-planner", "planner")]

    def offline(repo_id, name):
        raise OSError("offline")

    monkeypatch.setattr(api, "_hub_snapshot", offline)
    with pytest.raises(FileNotFoundError, match="download from marcosmaceo/orb-planner failed"):
        api.resolve_planner_path()


def test_resolve_planner_without_hub_explains(no_weights, monkeypatch):
    import sys

    from orb.api import resolve_planner_path

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import -> ImportError
    with pytest.raises(FileNotFoundError) as e:
        resolve_planner_path()
    msg = str(e.value)
    assert "huggingface_hub is not installed" in msg
    assert "orb-audio[hub]" in msg
    assert "ORB_PLANNER" in msg


def test_resolve_detail_is_local_only(no_weights, monkeypatch):
    import sys
    import types

    from orb.api import resolve_detail_path

    cwd, repo = no_weights
    # even with huggingface_hub importable, detail never downloads
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.ModuleType("huggingface_hub"))
    with pytest.raises(FileNotFoundError, match="CC BY-NC"):
        resolve_detail_path()
    st = _touch(repo / "weights/detail.safetensors")
    _touch(repo / "weights/config.json")
    assert resolve_detail_path() == st
    pt = _touch(cwd / "weights/detail.pt")
    assert resolve_detail_path().resolve() == pt.resolve()
    monkeypatch.setenv("ORB_DETAIL", str(cwd / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="ORB_DETAIL"):
        resolve_detail_path()


def test_generate_missing_detail_path_raises(no_weights):
    pytest.importorskip("torch")
    from orb.api import generate

    cwd, _ = no_weights
    with pytest.raises(FileNotFoundError, match="detail path"):
        generate("2.00-second calm cue, wind bed", planner=(None, None), detail_path=cwd / "x.pt")


# -- the bed: texture "none" and bed_db -------------------------------------

UI = {
    "duration": 3.0,
    "texture": "none",
    "mood": "playful",
    "events": [
        {"role": "ui_pop", "t": 0.5, "dur": 0.16, "gain_db": -12.0, "pan": 0.0, "seed": 0x73},
        {"role": "click", "t": 1.5, "dur": 0.10, "gain_db": -16.0, "pan": 0.0, "seed": 0x22},
    ],
}


def test_texture_none_is_digital_silence_around_events():
    audio = render_spec(UI)
    sr = 48000
    assert np.all(audio[: int(0.5 * sr)] == 0.0)  # before the first event
    # Between the pop's reverb tail and the click, and after the click's tail.
    gap = audio[int(1.40 * sr) : int(1.5 * sr)]
    assert 20 * np.log10(np.abs(gap).max() + 1e-30) < -90
    end = audio[int(2.8 * sr) :]
    assert 20 * np.log10(np.abs(end).max() + 1e-30) < -90


def test_texture_none_roundtrips_and_bed_db_is_optional():
    spec = dict_to_spec(UI)
    assert spec.texture == "none" and spec.bed_db is None
    assert spec_to_dict(spec) == UI
    quiet = {**SPEC, "bed_db": -45.0}
    assert spec_to_dict(dict_to_spec(quiet)) == quiet
    # A planner Spec never carries bed_db, so its dict is unchanged.
    assert "bed_db" not in spec_to_dict(dict_to_spec(SPEC))


@pytest.mark.parametrize(
    "bad",
    [
        {**SPEC, "bed_db": -80.0},
        {**SPEC, "bed_db": 0.0},
        {**SPEC, "bed_db": "loud"},
        {**SPEC, "bed_db": float("nan")},
        {**UI, "bed_db": -40.0},
        {**UI, "events": []},
    ],
)
def test_bed_options_are_validated(bad):
    with pytest.raises(ValueError):
        render_spec(bad)


def test_bed_db_sets_the_bed_level():
    from orb.grammar import score_from_spec
    from orb.render.mix import render_stems

    def bed_rms(db):
        spec = dict_to_spec({**SPEC, "bed_db": db})
        stems = render_stems(score_from_spec(spec))
        return float(np.sqrt(np.mean(stems["texture"].astype(np.float64) ** 2)))

    assert 20 * np.log10(bed_rms(-30.0) / bed_rms(-42.0)) == pytest.approx(12.0, abs=0.1)
    # The default is -40 dB, so explicit -40 renders bit-identically to no bed_db.
    assert np.array_equal(render_spec({**SPEC, "bed_db": -40.0}), render_spec(SPEC))


def test_default_bed_is_quiet_under_a_sparse_ui_cue():
    # The planner always emits a bed; at the old -30 dB default it sat at
    # about -46 dBFS before the first event of a one-pop cue.
    spec = {"duration": 2.0, "texture": "room_cold", "mood": "neutral",
            "events": [{"role": "ui_pop", "t": 0.6, "dur": 0.16, "gain_db": -12.0, "pan": 0.0, "seed": 1}]}
    pre = render_spec(spec)[: int(0.55 * 48000)].astype(np.float64)
    assert 20 * np.log10(np.sqrt(np.mean(pre**2))) < -52


def test_spec_from_score_reads_the_training_bed_level_as_the_default():
    from orb.grammar import TRAIN_BED_DB, score_from_spec, spec_from_score

    score = score_from_spec(dict_to_spec(SPEC))
    score.textures[0].gain_db = TRAIN_BED_DB
    assert spec_from_score(score).bed_db is None
    score.textures[0].gain_db = -35.0
    assert spec_from_score(score).bed_db == -35.0


# -- grammar captions without a bed ------------------------------------------


def test_parse_caption_inverts_caption():
    from orb.grammar import caption, parse_caption, sample_spec, score_from_spec, spec_from_score

    rng = np.random.default_rng(0)
    for _ in range(300):
        score = score_from_spec(sample_spec(rng))
        assert parse_caption(caption(score)) == spec_from_score(score)
    ui = dict_to_spec(UI)
    cap = caption(score_from_spec(ui))
    assert ", no bed, with ui pop @0.50 0.16s -12dB C #73" in cap
    assert parse_caption(cap) == ui
    # The bed clause may also be left out entirely.
    assert parse_caption(cap.replace(" no bed,", "")) == ui
    for text in ("a soft ui pop over vinyl crackle", cap + " please", cap.replace("ui pop", "ui_pop")):
        assert parse_caption(text) is None
    with pytest.raises(ValueError):
        parse_caption("1.00-second calm cue, no bed, with click @2.00 0.06s -10dB C #05")


def test_generate_renders_a_no_bed_caption_without_the_planner(tmp_path):
    from orb.api import generate

    cap = "2.00-second neutral cue, no bed, with click @0.50 0.06s -10dB C #05"
    # A planner path that does not exist: loading it would raise.
    out = generate(cap, planner_path=tmp_path / "missing.pt")
    assert out["planned"] is False and out["audit"]["passes"]
    assert out["spec"]["texture"] == "none"
    assert np.all(out["audio"][:24000] == 0.0)


def test_texture_none_keeps_event_timing():
    sr = 48000
    for t in (0.25, 0.5, 1.237):
        spec = {**UI, "events": [{**UI["events"][1], "t": t}]}
        audio = render_spec(spec)
        start = int(round(t * sr))
        # Exact silence up to the onset sample, and the attack right after it.
        assert np.all(audio[:start] == 0.0)
        audible = int(np.argmax(np.abs(audio).max(axis=1) > 1e-4))
        assert start <= audible <= start + int(0.002 * sr)


def test_save_wav_rounds_to_the_nearest_step(tmp_path):
    from orb.api import save_wav
    from orb.io import read_wav

    x = np.array([0.0, 0.99 / 32767, -0.99 / 32767, 0.49 / 32767, 1.5, -1.5])
    save_wav(np.stack([x, x], axis=1), tmp_path / "r.wav")
    y, sr = read_wav(tmp_path / "r.wav")
    assert sr == 48000
    assert np.array_equal(np.rint(y[:, 0] * 32767), [0, 1, -1, 0, 32767, -32767])
