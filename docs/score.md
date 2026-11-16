# The score

The score is the semantic codec. It is the only interface between the planner and the decoders, and the only thing the agent has to understand.

## Why a score

A waveform model has to learn timing, timbre, loudness, and structure from the same latent. A score separates them:

- **time** is explicit (absolute seconds)
- **timbre** is a role plus a few character parameters
- **structure** is sections and energy
- **loudness** is a target plus per-track gain

That separation is what lets a 30M planner be sample-accurate. It also means every output is editable: the agent can move one event, change one chord, or ask for a different texture without regenerating the whole cue.

## Schema (v1)

```json
{
  "duration": 12.0,
  "sample_rate": 48000,
  "seed": 7,
  "tempo": 92.0,
  "meter": [4, 4],
  "key": "D",
  "mode": "minor",
  "target_lufs": -20.0,
  "sections": [
    {"t": 0.0, "dur": 3.0, "role": "intro", "energy": 0.25, "density": 0.2},
    {"t": 3.0, "dur": 3.0, "role": "build", "energy": 0.55, "density": 0.45},
    {"t": 6.0, "dur": 3.0, "role": "peak",  "energy": 0.9,  "density": 0.7},
    {"t": 9.0, "dur": 3.0, "role": "outro", "energy": 0.35, "density": 0.3}
  ],
  "music": [
    {"instrument": "pad", "gain_db": -16, "pan": 0.0, "articulation": "sustain",
     "notes": [{"t": 0.0, "dur": 3.0, "midi": 50, "vel": 0.45}]}
  ],
  "textures": [
    {"t": 0.0, "dur": 12.0, "character": "room_cold", "gain_db": -34, "width": 0.7}
  ],
  "events": [
    {"t": 4.2, "role": "whoosh", "dur": 0.8, "character": "fast, airy",
     "apex": 4.237, "params": {"f_peak": 2600}, "gain_db": -10, "pan": 0.0}
  ]
}
```

### Fields

| field | type | notes |
|---|---|---|
| `duration` | seconds | 1–120 |
| `sample_rate` | int | 16k, 22.05k, 24k, 32k, 44.1k, 48k; 48k default |
| `seed` | int | all renderers are deterministic given this |
| `tempo` | BPM | 40–220 |
| `meter` | `[n, d]` | 4/4 default |
| `key`, `mode` | string | `C`…`B`, `major`/`minor` |
| `target_lufs` | float | integrated target, default −20 |
| `sections` | list | `role` ∈ intro, build, peak, break, outro; `energy`/`density` ∈ [0,1] |
| `music` | list | `instrument` ∈ pad, pluck, bass, bell, perc, strings |
| `textures` | list | `character` ∈ 8 presets; optional 32-band override |
| `events` | list | `role` ∈ 10 now, 34 at S1; `apex` must lie inside `[t, t+dur]` |

### Validation

`Score.validate()` enforces:

- every event, note, section, texture lies inside `[0, duration]`
- `apex` inside the event
- `pan` in [−1, 1], `vel` in [0, 1], `energy`/`density` in [0, 1]
- `bands` has exactly 32 values if present
- sample rate is in the supported set

The planner will decode with a grammar mask, so invalid scores are unrepresentable at inference. Validation still runs on human-written and loaded scores.

## Token grammar

The planner emits a flat token stream. The grammar is designed so that a 30 s cue fits in <1k tokens.

```
<bos>
<meta dur=12.0 sr=24k lufs=-20 tempo=92 meter=4/4 key=Dm>
<sec t=0.0 dur=3.0 role=intro e=0.25 d=0.20>
<sec t=3.0 dur=3.0 role=build e=0.55 d=0.45>
...
<mus inst=pad gain=-16 pan=0.0 art=sustain>
  <bar i=0 chord=50,53,57 pat=sus>
  <bar i=1 chord=46,50,53 pat=sus>
  ...
</mus>
<mus inst=pluck gain=-20 pan=0.25 art=pluck>
  <bar i=0 pat=arp_up vel=0.35>
  ...
</mus>
<tex t=0.0 dur=12.0 char=room_cold gain=-34 width=0.7>
<evt t=4.200 role=whoosh char=fast_airy dur=0.800 apex=4.237 gain=-10 pan=0.0>
<evt t=4.237 role=impact char=short_deep dur=0.700 apex=4.300 gain=-8 pan=0.0>
...
<eos>
```

Token budget for a 30 s cue at 90 BPM (≈11 bars):

| block | tokens |
|---|---|
| meta | 8 |
| sections (5) | 30 |
| music (4 tracks × 11 bars) | 192 |
| textures (3) | 24 |
| events (12) | 120 |
| **total** | **~374** |

Bar-level patterns are the compression trick: a bar is a pattern id plus a chord, not four notes. The pattern codebook (256 rhythm/voicing patterns) is learned from the synthetic corpus in S3 and frozen. Note-level tokens remain available for the agent to edit a score directly.

## Conditioning for the detail model

The detail model needs to know what it is correcting, not just what the DSP render looks like. The score provides a compact conditioning vector:

```
c = [role_embedding, duration, apex_offset, gain,
     section_energy, section_density, instrument_embedding,
     texture_character, tempo/120, key_pc/12]
```

At inference, `c` is read directly from the score. No extra model is needed to produce it. This is why the detail model can be small: the score already carries the semantics, and the DSP render already carries the timing.

## Renderer contract

```python
def render(score: Score) -> np.ndarray:  # shape (n, 2), float32, [-1, 1]
    ...
```

- deterministic given `score.seed`
- no network, no model weights
- peak ≤ 0.99, no DC offset, no clicks at boundaries
- event `apex` lands within 10 ms of the requested time
- the optional detail model is a second stage: `detail(render(score), score)`

The DSP render is the contract. The detail model may improve timbre, but it may not move a hit, change a duration, or alter the score.

## What the score deliberately cannot express

- a specific recording ("the creak from that door")
- lyrics or vocals
- free-form timbre outside the role vocabulary
- automation curves finer than the event envelope

Those are the jobs of a big waveform model. The score is for the 90% of agent audio that is functional.
