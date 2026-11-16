# Contributing to ORB

Thanks for taking a look. ORB is small on purpose: the whole engine is a
deterministic renderer plus two small models, and most changes should stay
that way.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The DSP path needs only NumPy. The planner needs PyTorch:

```bash
pip install -e ".[train,dev]"
```

## Before you open a PR

1. `pytest -q` passes (36 tests).
2. `python examples/04_no_torch.py` still runs — the DSP path must never
   import torch.
3. If you touched the renderer, run `python examples/01_generate.py` and check
   that the audit still passes.
4. Keep the diff focused. Formatting-only changes should be separate.

## Adding a sound role

1. Add the name to `EVENT_ROLES` in `src/orb/score.py`.
2. Add a renderer in `src/orb/render/events.py` and register it in
   `_RENDERERS` / `ROLE_PARAMS`.
3. Add the style to `EVENT_STYLE` in `src/orb/grammar.py` (character, apex
   fraction, default gain, default duration).
4. Regenerate the planner data and retrain:

```bash
python scripts/gen_scores.py --n 100000 --out data/scores_100k.jsonl --no-music
python scripts/train_spec.py --config configs/spec_v5.json
python scripts/eval_spec.py --checkpoint runs/spec_v5/best.pt --data data/heldout.jsonl --n 500
```

The planner retrains in minutes on one GPU. Do not hand-edit the tokenizer
ranges; `spec_tokenizer.py` derives them from `EVENT_ROLES`.

## Adding a texture

Add a preset to `PRESETS` in `src/orb/render/texture.py` (spectral points, LFO,
grain rate/length) and the name to `TEXTURES` in `src/orb/score.py`. Keep the
spectral points inside 20 Hz–12 kHz; `_shape_curve` rolls off above 8 kHz and
below 30 Hz.

## Style

- Python 3.11+, `from __future__ import annotations`, type hints on public
  functions.
- No new runtime dependencies in the DSP path. NumPy only.
- Comments explain *why*, not *what*.
- Determinism matters: any new randomness must be seeded from the score or the
  event seed.

## Reporting bugs

Include the caption or Spec, the command, the Python/torch version, and the
audit output. If it is an audio problem, say which cue and what you heard.

## License

By contributing you agree that your contribution is licensed under
Apache-2.0, the same as the project.
