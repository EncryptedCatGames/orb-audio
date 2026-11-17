"""Existing sounds must not drift.

``tests/data/render_baseline.json`` holds 150 renders frozen before the
natural/foley roles were added: every site library cue, building block and
playground preset, the noise-eval suite, every original role and texture, and
single events straight through ``render_event``. On the platform that recorded
it (macOS arm64) each render must be bit-identical (SHA-256 of the float32
samples). Elsewhere FFT rounding may differ in the last bit, so the 50 ms RMS
fingerprint must match to 1e-6 relative instead.

Regenerate only for an intentional change to an existing sound:
``python scripts/record_render_baseline.py``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from record_render_baseline import fingerprint, render_event_case, render_spec_case  # noqa: E402

DATA = json.loads((ROOT / "tests" / "data" / "render_baseline.json").read_text())
SAME_PLATFORM = (
    DATA["platform"]["system"] == platform.system() and DATA["platform"]["machine"] == platform.machine()
)


def _render(case: dict) -> np.ndarray:
    return render_spec_case(case["spec"]) if case["kind"] == "spec" else render_event_case(case["event"])


@pytest.mark.parametrize("case", DATA["cases"], ids=[c["name"] for c in DATA["cases"]])
def test_existing_render_is_unchanged(case):
    x = _render(case)
    if SAME_PLATFORM:
        assert hashlib.sha256(x.tobytes()).hexdigest() == case["sha256"], f"{case['name']} changed"
    else:
        np.testing.assert_allclose(np.array(fingerprint(x)), np.array(case["fp"]), rtol=1e-6, atol=1e-9)


def test_baseline_covers_the_site():
    names = {c["name"] for c in DATA["cases"]}
    assert sum(n.startswith("site/library/") for n in names) == 17
    assert sum(n.startswith("site/playground/") for n in names) == 6
