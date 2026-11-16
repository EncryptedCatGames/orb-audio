# Architecture

This describes what is in `src/orb` today, not what was planned. For the
numbers, see [`RESULTS.md`](../RESULTS.md); for the weights, see
[`MODEL_CARD.md`](../MODEL_CARD.md).

```
caption ──► planner (5.08M) ──► Spec ──► compiler ──► DSP render ──► [detail (10.27M)] ──► audit
```

The design bet is that a small model should produce a **description** of the
sound, not the waveform. The description is typed, so it can be validated,
edited, diffed and compiled deterministically. Almost all of the audio quality
therefore lives in code that can be changed without retraining.

## 1. The Spec

`orb.grammar.Spec` is the contract between the model and the renderer.

```python
Spec(duration, texture, mood, events=[SpecEvent(role, t, dur, gain_db, pan, seed)])
```

- `duration`: 2.0–30.0 s in 0.5 s steps
- `texture`: one of 8 beds
- `mood`: one of 10
- `events`: 0–8, each with `t`/`dur` on a 0.01 s grid, `gain_db` on a 2 dB grid
  from −30 to 0, `pan` on a 9-point scale, `seed` 0–255

`Spec.__post_init__` rejects anything off the grid, so an invalid Spec cannot
reach the renderer.

## 2. Planner

A decoder-only transformer that copies the caption into a Spec.

| | |
|---|---|
| parameters | 5,085,184 |
| layers / d / heads | 6 / 256 / 4 |
| context | 1024 |
| tokenizer | byte-level, vocab 352 |

The vocabulary is split into byte tokens (`0..255`) and structural tokens:

| range | meaning |
|---|---|
| 260–272 | `SPEC END_SPEC DUR TEX MOOD EVENTS END_EVENTS EVENT T D G P S` |
| 273–282 | decimal digits |
| 283–298 | 16 gain levels |
| 299–307 | 9 pan labels |
| 308–323 | hex digits (seed) |
| 324–331 | 8 texture values |
| 332–341 | 10 mood values |
| 342–351 | 10 role values |

Numbers in the caption are encoded as **decimal digits**, not bins. A bin would
make the model learn decimal-to-bin arithmetic; digits make it a copy.

**Constrained decoding.** `SpecGrammar` is a state machine that masks the
logits at every step: only structural tokens in the right order, only valid
categorical values, only digits that keep `t + dur <= duration`, and
`END_EVENTS` is forced at the event limit. The model can be wrong about *which*
whoosh; it cannot produce an invalid Spec.

## 3. Compiler

`orb.grammar.score_from_spec` is a pure function from Spec to
`orb.score.Score`. It expands the mood into four sections (intro, build, peak,
outro), places the texture, and converts each event into a `Score` event with
an apex time derived from the role's style table. Nothing here is learned.

## 4. DSP renderers

Pure NumPy, deterministic, seeded from the score and the event seed.

**Events** (`render/events.py`) — 10 roles: whoosh, impact, ui_pop, riser,
swell, click, bell, sub_drop, glitch, noise_sweep. Each is built from modal
synthesis, filtered noise, sweeps and envelopes, then zero-meaned before the
fade so one-sided transients do not leave DC.

**Textures** (`render/texture.py`) — white noise shaped by a spectral curve,
a three-sine LFO, and band-passed grains. `_shape_curve` rolls off −24 dB per
octave above 8 kHz and high-passes at 30 Hz, because `np.interp` otherwise
leaves a flat white-noise tail to Nyquist.

**Mix** (`render/mix.py`) — the foreground (music + events) is peak-normalised
to 0.89, and the bed is placed `gain_db` below the foreground's *active*
loudness (95th percentile of 50 ms frame RMS). This keeps the bed's level
independent of how sparse the events are. A single safety gain is applied to
every stem, so the stems always sum to the mix.

## 5. Detail model

An optional residual on top of the DSP render.

| | |
|---|---|
| parameters | 10,269,187 |
| input | `(2, 513, T)` real/imag STFT + 1 noise channel |
| architecture | complex-STFT U-Net, base 64, 4 levels, GroupNorm, SiLU, FiLM |
| residual | above bin 43 (≈2 kHz), energy-gated, learnable noise floor |

The residual is restricted to the high band and gated by input frame energy, so
it cannot move the onset or the low band. Inference runs 256-frame windows with
50% overlap and a Hann crossfade, matching the training crop size.

It is a texture stage, not a restoration model: it does not meet its own gate.
See [`RESULTS.md`](../RESULTS.md).

## 6. Audit

`orb.eval.audit` is model-free and runs on every generated cue: peak ≤ 0.99,
|DC| < 0.001, no clipping, plus informational click score, onset and spectral
centroid. It is the only gate the shipped path must pass.

## What is not in v1

- **Music.** The `Score` schema and `render/music.py` exist; the planner does
  not emit tempo, key or mode, so music is off.
- **Reward model, DPO, best-of-N critic.** Implemented in `src/orb/model` and
  `src/orb/train` but not used by the shipped path.
- **34 roles.** v1 ships 10 roles and 8 textures.
- **A CC0 detail dataset.** The detail model is trained on ESC-50 and is
  therefore non-commercial.
