"""ORB as an MCP (Model Context Protocol) server, over stdio.

Exposes five tools to any MCP client (Claude Code, Claude Desktop, IDEs, agent
frameworks):

* ``orb_generate``         caption or Spec -> WAV path + Spec + audit
* ``orb_render_spec``      Spec -> WAV, numpy-only
* ``orb_variants``         Spec + n -> n non-repeating takes
* ``orb_describe_grammar`` roles, textures, moods, limits and the Spec JSON Schema
* ``orb_validate_spec``    check a Spec without rendering it

Run it::

    orb-mcp                          # after `pip install "orb-audio[mcp] @ git+https://github.com/stdevMac/orb-audio"`
    python -m orb.mcp_server         # same thing
    orb-mcp --output-dir ~/sfx       # output root (default $ORB_OUTPUT_DIR or ./orb_out); tool calls cannot write outside it

The tool definitions and their behaviour live in :mod:`orb.tools`; this file
is only transport. With the official ``mcp`` Python SDK installed (v1
``FastMCP`` or v2 ``MCPServer``) it is used. Without it, a small built-in
JSON-RPC stdio loop serves the same tools, so the numpy-only install still
works as an MCP server. Torch is imported only when a caption needs the
planner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import tools as orb_tools

SERVER_NAME = "orb"
INSTRUCTIONS = (
    "ORB renders sound cues (48 kHz stereo WAV) from a typed Spec: a duration, an ambience "
    "texture (or \"none\" for silence between events, best for UI sounds), a mood and up to "
    "8 timed events (whoosh, impact, ui_pop, riser, swell, click, bell, sub_drop, glitch, "
    "noise_sweep). Timing is exact and the output is deterministic. "
    "Call orb_describe_grammar once, then author Specs directly and render them with "
    "orb_render_spec (no model needed). Use orb_variants for several takes. orb_generate also "
    "accepts a caption in the ORB caption grammar, which runs the small planner model. Every "
    "result includes the file path, the exact Spec and an audio audit."
)

# Protocol versions the built-in fallback speaks; it echoes the client's if known.
_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("orb-audio")
    except Exception:
        return "0.1.0"


# -- official SDK ------------------------------------------------------------


def _load_sdk():
    """Return ``(ServerClass, ToolError, flavour)`` or ``None`` if the SDK is missing."""
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError

        return MCPServer, ToolError, "mcp-v2"
    except ImportError:
        pass
    try:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError

        return FastMCP, ToolError, "mcp-v1"
    except ImportError:
        return None


def build_server():
    """Build the MCP server with the official SDK. Raises ImportError if it is not installed."""
    sdk = _load_sdk()
    if sdk is None:
        raise ImportError("the 'mcp' package is not installed (pip install 'orb-audio[mcp]')")
    server_cls, tool_error, flavour = sdk
    kwargs: dict[str, Any] = {"instructions": INSTRUCTIONS}
    if flavour == "mcp-v2":
        kwargs["version"] = _version()
    server = server_cls(SERVER_NAME, **kwargs)

    def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = {k: v for k, v in args.items() if v is not None}
        try:
            return orb_tools.dispatch(name, args)
        except (orb_tools.OrbToolError, KeyError) as e:
            raise tool_error(str(e)) from e

    # Explicit signatures, because the SDK builds argument parsing from them.
    # The advertised JSON Schema is then replaced by the richer one in orb.tools.
    def orb_generate(
        caption: str | None = None,
        spec: dict[str, Any] | None = None,
        seed: int = 0,
        detail: bool = False,
        output_dir: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        return call(
            "orb_generate",
            dict(caption=caption, spec=spec, seed=seed, detail=detail, output_dir=output_dir, name=name),
        )

    def orb_render_spec(
        spec: dict[str, Any],
        seed: int = 0,
        output_dir: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        return call("orb_render_spec", dict(spec=spec, seed=seed, output_dir=output_dir, name=name))

    def orb_variants(
        spec: dict[str, Any],
        n: int = 4,
        base_seed: int = 0,
        vary_event_seeds: bool = True,
        output_dir: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        return call(
            "orb_variants",
            dict(
                spec=spec,
                n=n,
                base_seed=base_seed,
                vary_event_seeds=vary_event_seeds,
                output_dir=output_dir,
                name=name,
            ),
        )

    def orb_describe_grammar() -> dict[str, Any]:
        return call("orb_describe_grammar", {})

    def orb_validate_spec(spec: Any) -> dict[str, Any]:
        return call("orb_validate_spec", {"spec": spec})

    fns = {
        "orb_generate": orb_generate,
        "orb_render_spec": orb_render_spec,
        "orb_variants": orb_variants,
        "orb_describe_grammar": orb_describe_grammar,
        "orb_validate_spec": orb_validate_spec,
    }
    for t in orb_tools.TOOLS:
        server.add_tool(fns[t["name"]], name=t["name"], description=t["description"])
        try:
            server._tool_manager.get_tool(t["name"]).parameters = t["input_schema"]
        except Exception:  # pragma: no cover - SDK internals moved; keep the generated schema
            pass
    return server


# -- built-in fallback -------------------------------------------------------


def _result(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def handle_message(msg: Any) -> dict[str, Any] | None:
    """Handle one JSON-RPC message for the built-in server. Returns the response, or None."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or "method" not in msg:
        return _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
    method, req_id, params = msg["method"], msg.get("id"), msg.get("params") or {}
    is_notification = "id" not in msg

    if method == "initialize":
        asked = params.get("protocolVersion")
        resp = _result(
            req_id,
            {
                "protocolVersion": asked if asked in _PROTOCOL_VERSIONS else _PROTOCOL_VERSIONS[-1],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": _version()},
                "instructions": INSTRUCTIONS,
            },
        )
    elif method == "ping":
        resp = _result(req_id, {})
    elif method == "tools/list":
        resp = _result(
            req_id,
            {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]} for t in orb_tools.TOOLS]},
        )
    elif method == "tools/call":
        name = params.get("name")
        if name not in orb_tools.TOOL_NAMES:
            resp = _error(req_id, -32602, f"unknown tool {name!r}")
        else:
            text, is_error = orb_tools.run_tool(name, params.get("arguments") or {})
            result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": is_error}
            if not is_error:
                result["structuredContent"] = json.loads(text)
            resp = _result(req_id, result)
    elif method.startswith("notifications/"):
        return None
    else:
        resp = _error(req_id, -32601, f"method not found: {method}")
    return None if is_notification else resp


def serve_builtin(stdin=None, stdout=None) -> None:
    """Serve MCP over newline-delimited JSON-RPC on stdio, with no dependencies."""
    stdin = stdin or sys.stdin
    real_stdout = sys.stdout
    out = stdout or real_stdout
    # Anything printed by library code must not corrupt the protocol stream.
    sys.stdout = sys.stderr
    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                resp: dict[str, Any] | None = _error(None, -32700, "parse error")
            else:
                if isinstance(msg, list):  # JSON-RPC batch (2024-11-05 / 2025-03-26)
                    resps = [r for r in (handle_message(m) for m in msg) if r is not None]
                    if resps:
                        out.write(json.dumps(resps) + "\n")
                        out.flush()
                    continue
                resp = handle_message(msg)
            if resp is not None:
                out.write(json.dumps(resp) + "\n")
                out.flush()
    finally:
        sys.stdout = real_stdout


# -- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="orb-mcp", description="ORB sound tools as an MCP stdio server.")
    ap.add_argument("--output-dir", help="output root for WAVs (default $ORB_OUTPUT_DIR or ./orb_out); per-call output_dir must stay inside it")
    ap.add_argument("--planner", help="planner weights for caption mode (default: $ORB_PLANNER, weights/planner.pt, planner.safetensors, then the Hub)")
    ap.add_argument(
        "--builtin",
        action="store_true",
        help="use the dependency-free built-in stdio server even if the mcp SDK is installed",
    )
    args = ap.parse_args(argv)
    if args.output_dir:
        os.environ["ORB_OUTPUT_DIR"] = args.output_dir
    if args.planner:
        os.environ["ORB_PLANNER"] = args.planner

    if not args.builtin and _load_sdk() is not None:
        build_server().run("stdio")
    else:
        if not args.builtin:
            print("orb-mcp: 'mcp' SDK not installed, using the built-in stdio server", file=sys.stderr)
        serve_builtin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
