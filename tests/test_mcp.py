"""Agent integration: orb.tools dispatch, Spec validation, and the MCP stdio server.

Everything except the caption test runs without torch. The SDK round trip is
skipped when the optional `mcp` package is not installed; the built-in stdio
server is always tested end to end.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from orb import tools
from orb.grammar import DEFAULT_BED_DB
from orb.io import read_wav
from orb.mcp_server import handle_message

ROOT = Path(__file__).resolve().parents[1]
HAS_MCP = importlib.util.find_spec("mcp") is not None


@pytest.fixture(autouse=True)
def _output_root(tmp_path, monkeypatch):
    """Every test writes under its own tmp_path, which is the configured output root."""
    monkeypatch.setenv("ORB_OUTPUT_DIR", str(tmp_path))

SPEC = {
    "duration": 2.0,
    "texture": "vinyl",
    "mood": "calm",
    "events": [
        {"role": "ui_pop", "t": 0.10, "dur": 0.16, "gain_db": -14.0, "pan": -0.25, "seed": 3},
        {"role": "click", "t": 0.60, "dur": 0.10, "gain_db": -16.0, "pan": 0.25, "seed": 9},
    ],
}


# -- tool definitions --------------------------------------------------------


def test_tool_definitions_cover_every_handler():
    names = {t["name"] for t in tools.TOOLS}
    assert names == set(tools._HANDLERS) == {
        "orb_generate",
        "orb_render_spec",
        "orb_variants",
        "orb_describe_grammar",
        "orb_validate_spec",
    }
    for t in tools.anthropic_tools():
        assert set(t) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"
    for t in tools.openai_tools():
        assert t["type"] == "function" and set(t["function"]) == {"name", "description", "parameters"}
    json.dumps(tools.TOOLS)  # must be plain JSON


def test_spec_schema_enums_match_grammar():
    props = tools.SPEC_SCHEMA["properties"]
    assert set(props["texture"]["enum"]) == set(tools.TEXTURES) | {"none"}
    assert set(props["mood"]["enum"]) == set(tools.MOODS)
    assert set(tools.EVENT_SCHEMA["properties"]["role"]["enum"]) == set(tools.EVENT_ROLES)


def test_describe_grammar_is_self_consistent():
    g = tools.dispatch("orb_describe_grammar")
    assert set(g["roles"]) == set(tools.EVENT_ROLES)
    assert g["limits"]["max_events"] == 8
    assert g["spec_schema"] == tools.SPEC_SCHEMA
    assert tools.validate_spec(g["example_spec"])["valid"]
    assert g["example_caption"].startswith("6.00-second bright cue, sci fi bed")
    json.dumps(g)


# -- validation --------------------------------------------------------------


def test_validate_fills_defaults_and_sorts():
    r = tools.validate_spec(
        {"duration": 3.0, "texture": "wind", "mood": "bright", "events": [{"role": "impact", "t": 2.0}, {"role": "whoosh", "t": 0.5}]}
    )
    assert r["valid"] and r["errors"] == []
    evs = r["spec"]["events"]
    assert [e["role"] for e in evs] == ["whoosh", "impact"]
    assert evs[1]["dur"] == 0.7 and evs[1]["gain_db"] == -6.0  # impact defaults
    assert all(0 <= e["seed"] <= 255 and e["pan"] == 0.0 for e in evs)
    assert r["caption"].startswith("3.00-second bright cue, wind bed, with whoosh @0.50")


def test_validate_trims_default_dur_but_rejects_explicit_overrun():
    ok = tools.validate_spec({"duration": 2.0, "texture": "wind", "mood": "calm", "events": [{"role": "bell", "t": 1.5}]})
    assert ok["valid"] and ok["spec"]["events"][0]["dur"] == 0.5 and ok["warnings"]
    bad = tools.validate_spec(
        {"duration": 2.0, "texture": "wind", "mood": "calm", "events": [{"role": "bell", "t": 1.5, "dur": 1.0}]}
    )
    assert not bad["valid"] and bad["errors"][0].startswith("events[0]: ends at 2.500 s")


@pytest.mark.parametrize(
    "patch, path",
    [
        ({"texture": "jungle"}, "texture"),
        ({"mood": "angry"}, "mood"),
        ({"duration": 0.0}, "duration"),
        ({"duration": 120.0}, "duration"),
        ({"duration": "4"}, "duration"),
        ({"loudness": -14}, "loudness"),
        ({"events": [{"role": "laser", "t": 0.1}]}, "events[0].role"),
        ({"events": [{"role": "click", "t": -1}]}, "events[0].t"),
        ({"events": [{"role": "click", "t": 0.1, "pan": 2}]}, "events[0].pan"),
        ({"events": [{"role": "click", "t": 0.1, "seed": 256}]}, "events[0].seed"),
        ({"events": [{"role": "click", "t": 0.1, "seed": True}]}, "events[0].seed"),
        ({"events": [{"role": "click", "t": 0.1, "gain_db": 6}]}, "events[0].gain_db"),
        ({"events": [{"role": "click", "t": 0.1, "volume": 1}]}, "events[0].volume"),
        ({"events": [{"role": "click", "t": 0.1 * i} for i in range(9)]}, "events"),
    ],
)
def test_validate_reports_error_paths(patch, path):
    r = tools.validate_spec({**SPEC, **patch})
    assert not r["valid"] and r["spec"] is None
    assert any(e.startswith(path) for e in r["errors"]), r["errors"]


def test_validate_rejects_non_object():
    assert not tools.validate_spec([1, 2])["valid"]
    assert tools.dispatch("orb_validate_spec", {"spec": "nope"})["valid"] is False


def test_off_planner_duration_is_a_warning_not_an_error():
    r = tools.validate_spec({**SPEC, "duration": 0.5, "events": [{"role": "click", "t": 0.0}]})
    assert r["valid"] and any("planner" in w for w in r["warnings"])


# -- dispatch ----------------------------------------------------------------


def test_render_spec_writes_a_valid_wav(tmp_path):
    r = tools.dispatch("orb_render_spec", {"spec": SPEC, "output_dir": str(tmp_path)})
    path = Path(r["path"])
    assert path.parent == tmp_path.resolve() and path.suffix == ".wav"
    audio, sr = read_wav(path)
    assert sr == 48000 and audio.shape == (2 * 48000, 2)
    assert r["audit"]["passes"] is True
    assert r["spec"] == tools.validate_spec(SPEC)["spec"]
    json.dumps(r)


def test_render_spec_is_content_addressed_and_deterministic(tmp_path):
    a = tools.dispatch("orb_render_spec", {"spec": SPEC, "output_dir": str(tmp_path / "a")})
    b = tools.dispatch("orb_render_spec", {"spec": SPEC, "output_dir": str(tmp_path / "b")})
    assert Path(a["path"]).name == Path(b["path"]).name
    assert Path(a["path"]).read_bytes() == Path(b["path"]).read_bytes()
    c = tools.dispatch("orb_render_spec", {"spec": SPEC, "seed": 1, "output_dir": str(tmp_path / "a")})
    assert Path(c["path"]).name != Path(a["path"]).name


def test_name_is_sanitised(tmp_path):
    r = tools.dispatch("orb_render_spec", {"spec": SPEC, "output_dir": str(tmp_path), "name": "../../etc/evil name.wav"})
    assert Path(r["path"]).parent == tmp_path.resolve()
    assert Path(r["path"]).name == "evil_name.wav"


def test_output_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ORB_OUTPUT_DIR", str(tmp_path / "env"))
    r = tools.dispatch("orb_render_spec", {"spec": SPEC})
    assert Path(r["path"]).parent == (tmp_path / "env").resolve()


def test_relative_output_dir_lands_under_the_root(tmp_path):
    r = tools.dispatch("orb_render_spec", {"spec": SPEC, "output_dir": "ui/taps"})
    assert Path(r["path"]).parent == (tmp_path / "ui" / "taps").resolve()


@pytest.mark.parametrize("escape", ["../elsewhere", "a/../../elsewhere", "/"])
def test_output_dir_cannot_escape_the_root(tmp_path, escape):
    text, is_error = tools.run_tool("orb_render_spec", {"spec": SPEC, "output_dir": escape, "name": "../../evil"})
    assert is_error and "outside the output root" in json.loads(text)["error"]
    assert not (tmp_path.parent / "elsewhere").exists()
    assert not list(tmp_path.parent.glob("**/evil.wav"))


def test_output_dir_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    text, is_error = tools.run_tool("orb_render_spec", {"spec": SPEC, "output_dir": "link"})
    assert is_error and not list(outside.iterdir())


def test_detail_flag_gets_its_own_file_name():
    plain = tools._file_name(None, SPEC, 0)
    detail = tools._file_name(None, SPEC, 0, detail=True)
    assert plain != detail and detail.endswith("_detail.wav")
    assert tools._file_name("hit", SPEC, 0, detail=True) == "hit_detail.wav"


def test_variants_are_distinct(tmp_path):
    r = tools.dispatch("orb_variants", {"spec": SPEC, "n": 3, "output_dir": str(tmp_path), "name": "pop"})
    paths = [Path(v["path"]) for v in r["variants"]]
    assert [p.name for p in paths] == ["pop_v00.wav", "pop_v01.wav", "pop_v02.wav"]
    audio = [read_wav(p)[0] for p in paths]
    assert not np.array_equal(audio[0], audio[1]) and not np.array_equal(audio[1], audio[2])
    assert all(v["audit"]["passes"] for v in r["variants"])


def test_generate_with_spec_needs_no_model(tmp_path):
    r = tools.dispatch("orb_generate", {"spec": SPEC, "output_dir": str(tmp_path)})
    assert r["source"] == "spec" and Path(r["path"]).exists()


@pytest.mark.parametrize(
    "name, args, needle",
    [
        ("orb_generate", {}, "exactly one"),
        ("orb_generate", {"spec": SPEC, "caption": "x"}, "exactly one"),
        ("orb_render_spec", {}, "invalid spec"),
        ("orb_render_spec", {"spec": SPEC, "bogus": 1}, "unknown argument"),
        ("orb_variants", {"spec": SPEC, "n": 0}, "n:"),
        ("orb_variants", {"spec": SPEC, "n": 1000}, "n:"),
        ("orb_render_spec", {"spec": SPEC, "seed": -1}, "seed"),
        ("orb_nope", {}, "unknown tool"),
    ],
)
def test_run_tool_reports_errors_without_raising(name, args, needle):
    text, is_error = tools.run_tool(name, args)
    assert is_error
    assert needle in json.loads(text)["error"]


def test_spec_tools_never_import_torch(tmp_path):
    code = (
        "import sys, json; from orb.tools import run_tool;"
        f"run_tool('orb_render_spec', {{'spec': {SPEC!r}, 'output_dir': {str(tmp_path)!r}}});"
        f"run_tool('orb_variants', {{'spec': {SPEC!r}, 'n': 2, 'output_dir': {str(tmp_path)!r}}});"
        "run_tool('orb_describe_grammar'); run_tool('orb_validate_spec', {'spec': {}});"
        "import orb.mcp_server;"
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ORB_OUTPUT_DIR": str(tmp_path)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or not (ROOT / "weights/planner.pt").exists(),
    reason="caption mode needs torch and weights/planner.pt",
)
def test_generate_from_caption_uses_the_planner(tmp_path):
    caption = "6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a"
    r = tools.dispatch("orb_generate", {"caption": caption, "output_dir": str(tmp_path)})
    assert r["source"] == "planner"
    assert r["spec"]["events"][0]["role"] == "whoosh"
    assert r["caption"] == caption  # the planner reproduced the Spec exactly


# -- built-in MCP server (no SDK) -------------------------------------------


def test_builtin_handle_message():
    init = handle_message(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
    )
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["serverInfo"]["name"] == "orb"
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert [t["name"] for t in listed["result"]["tools"]] == list(tools.TOOL_NAMES)
    assert "inputSchema" in listed["result"]["tools"][0]
    bad = handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope"}})
    assert bad["error"]["code"] == -32602
    assert handle_message({"jsonrpc": "2.0", "id": 4, "method": "resources/list"})["error"]["code"] == -32601
    err = handle_message(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "orb_render_spec", "arguments": {}}}
    )
    assert err["result"]["isError"] is True


# -- real stdio round trips --------------------------------------------------


class StdioClient:
    """A minimal MCP client: newline-delimited JSON-RPC over a subprocess's stdio."""

    def __init__(self, args: list[str], out_dir: Path):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ORB_OUTPUT_DIR": str(out_dir)}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "orb.mcp_server", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(out_dir),
        )
        self.next_id = 0

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed stdout: " + self.proc.stderr.read()[-2000:])
            resp = json.loads(line)
            if resp.get("id") == self.next_id:
                return resp
            # Anything else (server notifications, logs) is skipped.

    def close(self) -> int:
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=20)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.stdout.close()
            self.proc.stderr.close()


def _round_trip(args: list[str], out_dir: Path) -> None:
    client = StdioClient(args, out_dir)
    try:
        init = client.request(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        )
        assert "result" in init, init
        assert init["result"]["serverInfo"]["name"] == "orb"
        assert "tools" in init["result"]["capabilities"]
        client.notify("notifications/initialized")

        listed = client.request("tools/list")["result"]["tools"]
        by_name = {t["name"]: t for t in listed}
        assert set(by_name) == set(tools.TOOL_NAMES)
        # The rich schema from orb.tools is what clients see, enums included.
        spec_schema = by_name["orb_render_spec"]["inputSchema"]["properties"]["spec"]
        assert spec_schema["properties"]["texture"]["enum"] == list(tools.TEXTURES) + ["none"]
        assert "bed_db" in spec_schema["properties"]

        res = client.request("tools/call", {"name": "orb_render_spec", "arguments": {"spec": SPEC, "name": "rt"}})
        result = res["result"]
        assert not result.get("isError"), result
        payload = json.loads(result["content"][0]["text"])
        assert Path(payload["path"]) == (out_dir / "rt.wav").resolve()
        assert Path(payload["path"]).exists() and payload["audit"]["passes"]

        res = client.request("tools/call", {"name": "orb_validate_spec", "arguments": {"spec": {"duration": 2}}})
        report = json.loads(res["result"]["content"][0]["text"])
        assert report["valid"] is False and report["errors"]

        res = client.request("tools/call", {"name": "orb_render_spec", "arguments": {"spec": {**SPEC, "mood": "x"}}})
        assert res["result"]["isError"] is True
        assert "mood" in res["result"]["content"][0]["text"]
    finally:
        assert client.close() == 0


def test_stdio_round_trip_builtin(tmp_path):
    _round_trip(["--builtin"], tmp_path)


@pytest.mark.skipif(not HAS_MCP, reason="optional 'mcp' SDK not installed")
def test_stdio_round_trip_sdk(tmp_path):
    from orb.mcp_server import _load_sdk

    assert _load_sdk() is not None
    _round_trip([], tmp_path)


# -- the bed: texture "none" and bed_db -------------------------------------


def test_validate_accepts_texture_none_with_a_no_bed_caption():
    spec = {"duration": 2.0, "texture": "none", "mood": "neutral", "events": [{"role": "click", "t": 0.5}]}
    r = tools.validate_spec(spec)
    assert r["valid"], r
    assert r["spec"]["texture"] == "none" and "bed_db" not in r["spec"]
    assert not r["warnings"]
    assert ", no bed, " in r["caption"]
    assert any("round-trip" in w for w in tools.validate_spec({**SPEC, "bed_db": -42})["warnings"])


def test_generate_from_a_no_bed_caption_skips_the_planner(tmp_path, monkeypatch):
    def no_planner():
        raise AssertionError("the planner must not run for a no-bed caption")

    monkeypatch.setattr(tools, "_planner", no_planner)
    caption = "2.00-second neutral cue, no bed, with click @0.50 0.06s -10dB C #05"
    r = tools.dispatch("orb_generate", {"caption": caption, "output_dir": str(tmp_path)})
    assert r["source"] == "caption" and r["caption"] == caption
    assert r["spec"]["texture"] == "none"
    assert r["spec"]["events"] == [
        {"role": "click", "t": 0.5, "dur": 0.06, "gain_db": -10.0, "pan": 0.0, "seed": 5}
    ]
    audio, sr = read_wav(r["path"])
    assert np.all(audio[: int(0.5 * sr)] == 0)


def test_validate_bed_db():
    ok = tools.validate_spec({**SPEC, "bed_db": -42})
    assert ok["valid"] and ok["spec"]["bed_db"] == -42.0
    for bad in (-70, 0, "quiet", True):
        r = tools.validate_spec({**SPEC, "bed_db": bad})
        assert not r["valid"] and any(e.startswith("bed_db") for e in r["errors"]), bad
    r = tools.validate_spec({**SPEC, "texture": "none", "bed_db": -40})
    assert not r["valid"] and any("no effect" in e for e in r["errors"])


def test_validate_rejects_silent_none_cue():
    r = tools.validate_spec({"duration": 2.0, "texture": "none", "mood": "calm", "events": []})
    assert not r["valid"] and any("at least one event" in e for e in r["errors"])


def test_render_spec_texture_none_is_silent_between_events(tmp_path):
    spec = {"duration": 2.0, "texture": "none", "mood": "neutral",
            "events": [{"role": "ui_pop", "t": 0.5, "dur": 0.16, "gain_db": -12.0, "pan": 0.0, "seed": 3}]}
    r = tools.dispatch("orb_render_spec", {"spec": spec, "output_dir": str(tmp_path)})
    assert r["audit"]["passes"]
    audio, sr = read_wav(r["path"])
    assert np.all(audio[: int(0.5 * sr)] == 0)


def test_describe_grammar_documents_the_bed():
    g = tools.describe_grammar()
    assert "none" in g["bed"]["none"]
    assert g["bed"]["bed_db"]["default"] == DEFAULT_BED_DB == -40.0
    assert "no bed" in g["caption_grammar"]
