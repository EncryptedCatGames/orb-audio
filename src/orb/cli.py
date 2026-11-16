"""Command line entry point: ``orb-generate``."""

from __future__ import annotations

import argparse
import json
import sys

from .api import DEFAULT_PLANNER, generate, save_wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="orb-generate",
        description="Caption -> 48 kHz WAV through the ORB planner and DSP renderer.",
    )
    ap.add_argument("caption", help="a caption in the ORB grammar")
    ap.add_argument("--out", default="cue.wav")
    ap.add_argument("--planner", default=DEFAULT_PLANNER)
    ap.add_argument("--detail", default=None, help="optional CC BY-NC detail weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print the spec and audit as JSON")
    args = ap.parse_args(argv)

    out = generate(
        args.caption,
        planner_path=args.planner,
        detail_path=args.detail,
        seed=args.seed,
        strength=args.strength,
    )
    save_wav(out["audio"], args.out)

    if args.json:
        print(json.dumps({"spec": out["spec"], "audit": out["audit"]}, indent=2))
    else:
        audit = out["audit"]
        print(f"wrote {args.out}")
        print(
            f"peak={audit['peak']:.3f} rms={audit['rms_dbfs']:.1f} dBFS "
            f"pass={audit['passes']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
