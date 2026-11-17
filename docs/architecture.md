# Architecture

This describes what is in `src/orb` today, not what was planned. For the
numbers, see [`RESULTS.md`](../RESULTS.md); for the weights, see
[`MODEL_CARD.md`](../MODEL_CARD.md).

```
caption or plain text ──► planner (5.10M) ──► Spec ──► compiler ──► DSP render ──► [detail (9.06M)] ──► audit
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
- `texture`: one of 13 beds (8 noise beds + 5 scene beds), or `none`
- `mood`: one of 10
- `events`: 0–8, each with `t`/`dur` on a 0.01 s grid, `gain_db` on a 2 dB grid
  from −30 to 0, `pan` on a 9-point scale, `seed` 0–255

`Spec.__post_init__` rejects anything off the grid, so an invalid Spec cannot
reach the renderer.

## 2. Planner

A decoder-only transformer that copies a grammar caption into a Spec. Since v8 it also reads plain words ("rain on a tin roof with distant thunder"): it was trained on 150k synthetic free-text pairs from `orb.freetext` as well as the grammar captions.

| | |
|---|---|
| parameters | 5,096,192 (v8) |
| layers / d / heads | 6 / 256 / 4 |
| context | 1024 |
| tokenizer | byte-level, vocab 395 (342 + 47 role tokens + 5 extra texture tokens + `NOBED`) |

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
| 342–388 | 47 role values (the first 10 core roles, then foley and everyday roles, appended) |
| 389–393 | 5 scene-bed textures (ocean, stream, forest, night, traffic) |
| 394 | `NOBED`: texture `none` (v8; `SpecGrammar(no_bed=False)` masks it for older planners) |

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

**Events** (`render/events.py`) — 10 sound-design roles: whoosh, impact, ui_pop, riser,
swell, click, bell, sub_drop, glitch, noise_sweep. Each is built from modal
synthesis, filtered noise, sweeps and envelopes, then zero-meaned before the
fade so one-sided transients do not leave DC.

**Natural and foley** (`render/foley.py`) — 6 physically modelled roles,
appended after the original ten so existing seeds and renders are unchanged
(a regression test, `tests/test_render_baseline.py`, pins 150 renders bit for bit):

- `thunder`: a close crack (stacked micro-transients plus crackle) and a
  rolling rumble of low-passed noise driven by a train of arrivals from
  different parts of the channel, decaying over seconds; `seed % 4` is
  distance, 0 close crack to 3 far rumble (far loses the crack and the highs).
- `door_slam`: a wooden panel's low modes (80–200 Hz) as a damped mode bank,
  the frame crack, a metal latch click, and a short room tail; `seed % 3` is
  the door weight, an odd seed adds a latch rebound.
- `knock`: two or three knuckle raps (half-sine force pulses) exciting the
  damped modes of a wooden door, small to large door.
- `footsteps`: heel-toe steps across `dur` with a varied cadence and a
  floor-dependent body (hard floor, stone, gravel grains); `seed % 3` is the
  floor, `(seed // 3) % 3` the pace.
- `water_drop`: a Minnaert bubble (van den Doel 2005) whose pitch glides up
  as it rises; `seed % 2` is a small drip or a larger plop with a splash.
- `explosion`: a blast transient, a sub body, a rumble and a debris crackle;
  `seed % 3` is distance.

All are seeded, deterministic, zero outside the event plus its tail,
high-passed at 20 Hz, band-limited at 12 kHz, and they honour gain and pan.
The grammar renders them as `thunder`, `door slam`, `knock`, `footsteps`,
`water drop` and `explosion`; 20% of the training captions are small scenes
(storm, arrival, cave, battle) that combine them.

**Everyday library** (`render/natural.py`, v0.3) — 31 more physically
modelled roles, appended after the foley six, through the same
`render_foley` wrapper (20 Hz high-pass, smooth 12 kHz low-pass after every
attack, edge fades, gain and pan). Each is a small physical or
signal model, and the seed picks a variant (`natural.VARIANTS`):

- nature: `bird_chirp` (swept tonal syllables: chips, trills, warbles, two-note
  whistles, 2.4–5.2 kHz), `crickets` (a ~4.5 kHz carrier gated into
  wing-closure pulses, one cricket to a field chorus), `wave_crash` (the swell, the break with a low
  thump, the foam hissing as it runs up and drains back), `wind_gust`, `splash` (slap plus a
  cloud of Minnaert bubbles), `frog`, `owl`;
- home and office: `glass_break` (shard modes plus tinkling fragments),
  `clock_tick`, `clock_alarm`, `phone_ring` (bell, UK double ring,
  ringtone, vibrate), `keyboard_typing`, `mouse_click`, `camera_shutter`,
  `paper_rustle`, `cork_pop`, `coin_drop`, `pouring_water` (pitch rises as
  the glass fills), `can_opening`, `wood_creak` (stick-slip), `metal_clang`;
- people and action: `heartbeat` (lub-dub at 62/88/120 bpm), `applause`
  (independent claps), `punch`, `sword_clash`, `gunshot`, `fireworks`;
- street: `car_horn`, `siren` (wail, yelp, hi-lo), `engine` (idle, rev,
  pass-by with Doppler), `church_bell` (inharmonic partials: hum, prime,
  tierce, quint, nominal).

Tonal sources (birds, bells, horns, sirens, ringing metal, crickets, frogs,
clock alarm, phone, coins, glass, sword) are exempt from the no-buzz
periodicity test, because they are periodic by nature; they still have to
stay silent near Nyquist.

**Scene beds** (`render/ambience.py`, v0.3) — `ocean`, `stream`, `forest`,
`night`, `traffic`, appended after the eight noise textures. They are built
from the physical models placed at seeded random times, places and takes
(waves over a surf wash, bubbles over a soft wash, birds over leaves, crickets
with frogs and owls, cars passing over a low rumble), then go through the same
width, level, DC and edge-fade path as the other textures. Their default level
is −14 dB under the events (`AMBIENCE_BED_DB`), not −40, because they are the
scene, not a noise floor.

**DSP-only routing** (`render/mix.py`) — `render_stems` returns a
`dsp_only` stem with the physically modelled roles and the scene beds;
`apply_detail_routed` runs the detail model on the rest and adds that stem
back unchanged. With no DSP-only content it is exactly the old call.

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
| parameters | 9,064,387 (detail v6, `configs/detail_v6.json`) |
| input | `(2, 513, T)` real/imag STFT, no noise channel |
| architecture | complex-STFT U-Net, base 64, 4 levels, GroupNorm, SiLU, FiLM, bilinear resize-conv upsampling |
| residual | smoothed mask (kernel 9×3) above bin 43 (≈2 kHz), tapered off between 8 and 9.5 kHz, energy gate + quiet-frame gate (30 dB, 10 dB ramp) |

The residual is restricted to the high band and gated by input frame energy, so
it cannot move the onset or the low band. Resize-conv upsampling replaces v5's
`ConvTranspose2d`, which left checkerboard tonal stripes around 10–13 kHz and a
~4 kHz line. Inference runs 256-frame windows with
50% overlap and a Hann crossfade, matching the training crop size.

It is a texture stage, not a restoration model: it passes the no-noise artifact
gate (`scripts/eval_detail_artifacts.py`) but does not meet its 10% MR-STFT
restoration gate. See [`RESULTS.md`](../RESULTS.md).

## 6. Audit

`orb.eval.audit` is model-free and runs on every generated cue: peak ≤ 0.99,
|DC| < 0.001, no clipping, plus informational click score, onset and spectral
centroid. It is the only gate the shipped path must pass.

## What is not in v1

- **Music.** The `Score` schema and `render/music.py` exist; the planner does
  not emit tempo, key or mode, so music is off.
- **Reward model, DPO, best-of-N critic.** Implemented in `src/orb/model` and
  `src/orb/train` but not used by the shipped path.
- **Voices and vocal animals.** ORB ships 47 roles (10 sound-design + 37 physical models) and 13 textures; speech, laughter, coughs and vocal animals need recordings or a learned model.
- **A CC0 detail dataset.** The detail model is trained on ESC-50 and is
  therefore non-commercial.
