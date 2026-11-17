#!/usr/bin/env python3
"""An agent scores a UI animation, using ORB as a tool. Offline, no LLM API.

This is the loop an LLM agent runs when ORB is one of its tools (over MCP or
any tool-use API), with the model's turns replaced by a small deterministic
function so it runs anywhere. It shows the point of ORB for agents: sound is
authored as data. The agent reads a motion timeline (the keyframes a motion
designer or a UI framework already has), writes a Spec whose events sit
exactly on those keyframes, checks it, fixes it, and renders it.

    python examples/05_agent_tool.py

Writes into out/agent_tool/:
    onboarding.wav          the cue, sample-locked to the timeline
    tap_v00..v03.wav        four non-repeating takes of the button tap
    onboarding.sound.json   timeline + Spec + audit: the "sound design" as a file

Every call below goes through `orb.tools.run_tool`, the same function the MCP
server (`orb-mcp`) and a real tool-use loop would call. torch is never imported.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orb.tools import anthropic_tools, run_tool  # noqa: E402

OUT = ROOT / "out" / "agent_tool"
# The output root. Tool calls may only write under it; this is the same setting
# as `orb-mcp --output-dir`, and the model cannot override it.
os.environ["ORB_OUTPUT_DIR"] = str(OUT)

# A motion timeline, as a motion tool or a UI framework would export it:
# what moves, when (seconds), for how long, and where on screen (x in -1..1).
TIMELINE = {
    "name": "onboarding",
    "duration": 4.0,
    "keyframes": [
        {"id": "logo_build", "kind": "anticipation", "t": 0.20, "dur": 1.00, "x": 0.0},
        {"id": "logo_land", "kind": "land", "t": 1.20, "dur": 0.50, "x": 0.0},
        {"id": "card_1", "kind": "slide_in", "t": 1.60, "dur": 0.45, "x": -0.6},
        {"id": "card_2", "kind": "slide_in", "t": 1.75, "dur": 0.45, "x": 0.0},
        {"id": "card_3", "kind": "slide_in", "t": 1.90, "dur": 0.45, "x": 0.6},
        {"id": "cta_tap", "kind": "tap", "t": 2.80, "dur": 0.12, "x": 0.4},
        {"id": "success", "kind": "confirm", "t": 3.00, "dur": 0.90, "x": 0.4},
    ],
}

# The agent's "taste": which sound role fits which kind of motion.
ROLE_FOR = {
    "anticipation": ("riser", -20.0),
    "land": ("impact", -6.0),
    "slide_in": ("whoosh", -14.0),
    "tap": ("click", -12.0),
    "confirm": ("bell", -12.0),
}


def call(name: str, args: dict | None = None) -> dict:
    """One tool call, logged like an agent transcript."""
    shown = json.dumps(args or {})
    print(f"\n>> tool_use {name}({shown[:110]}{'...' if len(shown) > 110 else ''})")
    text, is_error = run_tool(name, args or {})
    result = json.loads(text)
    tag = "tool_error" if is_error else "tool_result"
    print(f"<< {tag} {text[:160]}{'...' if len(text) > 160 else ''}")
    return {"error": result["error"]} if is_error else result


def agent_write_spec(timeline: dict, grammar: dict) -> dict:
    """Stand-in for the model's turn: timeline + grammar -> a Spec.

    It uses the grammar's `apex_fraction` so a riser *peaks* on the keyframe it
    builds into, instead of starting on it.
    """
    events = []
    for i, kf in enumerate(timeline["keyframes"]):
        role, gain = ROLE_FOR[kf["kind"]]
        dur = kf["dur"]
        t = kf["t"]
        if role == "riser":
            # Land the riser's apex exactly on the next keyframe.
            apex = grammar["roles"]["riser"]["apex_fraction"]
            dur = 1.2
            t = round(timeline["keyframes"][i + 1]["t"] - apex * dur, 3)
        events.append({"role": role, "t": t, "dur": dur, "gain_db": gain, "pan": kf["x"], "seed": 16 * i + 5})
    # First draft is deliberately sloppy, like a real first draft: the bell is too long.
    events[-1]["dur"] = 1.40
    return {"duration": timeline["duration"], "texture": "room_cold", "mood": "bright", "events": events}


def agent_fix(spec: dict, errors: list[str]) -> dict:
    """Stand-in for the model reading validation errors and fixing the Spec."""
    for err in errors:
        if err.startswith("events[") and "after the cue ends" in err:
            i = int(err[len("events[") : err.index("]")])
            ev = spec["events"][i]
            ev["dur"] = round(spec["duration"] - ev["t"], 2)
            print(f"   agent: events[{i}] ({ev['role']}) ran past the end; trimmed dur to {ev['dur']} s")
    return spec


def main() -> None:
    print(f"ORB exposes {len(anthropic_tools())} tools: " + ", ".join(t["name"] for t in anthropic_tools()))

    # 1. Learn the vocabulary once.
    grammar = call("orb_describe_grammar")
    print(f"   agent: {len(grammar['roles'])} roles, {len(grammar['textures'])} textures, "
          f"riser apex at {grammar['roles']['riser']['apex_fraction']:.0%} of its length")

    # 2. Draft a Spec from the motion timeline, validate, fix, validate again.
    spec = agent_write_spec(TIMELINE, grammar)
    report = call("orb_validate_spec", {"spec": spec})
    if not report["valid"]:
        spec = agent_fix(spec, report["errors"])
        report = call("orb_validate_spec", {"spec": spec})
    assert report["valid"], report["errors"]
    print(f"   agent: canonical caption -> {report['caption'][:90]}...")

    # 3. Render the cue. The Spec is exact, so no planner and no torch.
    cue = call("orb_render_spec", {"spec": report["spec"], "name": TIMELINE["name"]})
    assert cue["audit"]["passes"], cue["audit"]

    # 4. A tap is heard many times; give the UI a pool of takes, not one file.
    tap = {
        "duration": 0.5,
        "texture": "room_cold",
        "mood": "neutral",
        "events": [{"role": "click", "t": 0.0, "dur": 0.12, "gain_db": -12.0, "pan": 0.0, "seed": 9}],
    }
    pool = call("orb_variants", {"spec": tap, "n": 4, "name": "tap"})

    # 5. The deliverable is a file a motion tool can read: every sound is tied
    #    to the keyframe that caused it, at the exact same time.
    sound_design = {
        "timeline": TIMELINE,
        "spec": cue["spec"],
        "audio": Path(cue["path"]).name,
        "audit": cue["audit"],
        "bindings": [
            {"keyframe": kf["id"], "role": ev["role"], "t": ev["t"]}
            for kf, ev in zip(TIMELINE["keyframes"], spec["events"])
        ],
        "tap_pool": [Path(v["path"]).name for v in pool["variants"]],
    }
    (OUT / f"{TIMELINE['name']}.sound.json").write_text(json.dumps(sound_design, indent=2) + "\n")

    assert "torch" not in sys.modules, "the spec path must not need torch"
    print("\nwrote")
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print("\nMove a keyframe, re-run, and the sound moves with it. That is the whole idea.")


if __name__ == "__main__":
    main()
