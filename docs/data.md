# Data engine

The model is small, so the data has to be precise. Four loops, in order of importance.

## 1. Synthetic pretraining

**1M (caption, score) pairs.** Scores are ~9 KB of JSON; audio is rendered only for the detail and RL stages, never for planner SFT.

Pipeline:

```
grammar sampler → score → DSP render → captioner → (caption, score)
```

The grammar sampler (`src/orb/grammar.py`) draws:

- duration 8–32 s, tempo 70–140, key/mode, 2–4 sections
- a chord progression from a small set, voiced into 3–4 tracks
- 2–5 events at musically plausible times (cuts, downbeats, section boundaries)
- 1–2 textures with a character and a width

The renderer is the same one that ships. This is critical: the model learns the distribution it will be asked to produce, including the decoder's weaknesses.

### Captions

The naive approach — caption the score from its own fields — teaches the model to invert a template. It will score well on held-out synthetic captions and fail on agent language. Mitigations, in order:

1. **2k human-written seeds**, 20 per role, written as an agent would ask ("a soft rounded pop for a friendly notification", not "ui_pop f0=880 t60=0.07").
2. **Template grammar with held-out splits.** 10k phrasing templates; 10% never used in training. Every slot has 5–20 synonyms.
3. **50k LLM paraphrases** of the seeds, with the score as context, not as the source of truth.
4. **Caption the render, not the score.** Train a small audio captioner on real recordings + renders, then use it to caption 500k renders. This breaks the score→caption shortcut.
5. **Adversarial captions.** 10% of captions are deliberately underspecified ("a transition sound") so the planner learns to choose, not just copy.

Caption length: 8–40 tokens. Longer captions are mostly noise at this scale.

### What needs audio, and what does not

- **Planner SFT needs scores only.** The planner learns caption → score; no render is required. 1M pairs cost ~9 GB and zero audio compute.
- **Detail pairs need audio once.** 100k pairs at 1–8 s per cue on one core is ~1–3 hours on a 64-core box. Cache the renders.
- **RL and evaluation need audio repeatedly.** 500k prompts × 8 samples = 4M renders. At ~3 s per cue on 64 cores that is ~2–4 days, run once per RL stage, not per epoch.
- **The critic needs audio at inference**, which is the point.

Do not put the renderer in the planner's training loop. It is not needed and it will starve the GPU.

## 2. Detail-model pairs

This is the dataset that decides the quality thesis.

**100k real CC0 recordings**, each paired with a DSP render of a fitted score.

| step | tool | output |
|---|---|---|
| 1. select real clip | role classifier + manual spot check | 100k clips, 0.1–3 s |
| 2. fit a score | earshot `fit` + parameter search | role, dur, onset, apex, tilt, pitch |
| 3. render DSP | `src/orb/render` | paired synthetic clip |
| 4. align by onset | cross-correlation | same length, same hit |
| 5. split | by source recording | train / dev / held-out |

Pairing rules:

- the DSP render must have the same role and duration as the real clip
- onset alignment within 5 ms
- loudness-matched before the loss
- no pair where the real clip contains speech or music under the effect
- 10% of pairs are deliberately "hard": quiet, noisy, or unusual timbre

The model learns `real − DSP` as a residual. It does not learn to generate from noise, so it cannot invent a hit that is not in the score.

## 3. Real anchoring

Beyond the detail pairs, 20k clips are analyzed once:

| source | count | transform | output |
|---|---|---|---|
| one-shot SFX (Freesound CC0) | 12k | earshot `fit` | event params + role label |
| loops / beds (CC0) | 5k | basic-pitch + percussion transcription | symbolic tracks |
| textures / ambience (CC0) | 3k | STFT → 32-band trajectory | texture head targets |

Rules:

- CC0 only for anything that ships in the repo or the model card.
- CC-BY allowed for training with attribution, never for shipped reference audio.
- No scraped music. No "free download" sites with unclear provenance.
- Per-clip provenance in `data/provenance.jsonl`; the license audit at S9 reads it.

Fitted params are mixed into the synthetic distribution at 10–20%, not used as the whole dataset. Real recordings are sparse and biased; synthetic data is dense and controllable.

## 4. Teacher distillation (optional)

A big open model is a good teacher and a bad runtime. Use it offline:

1. Generate 500k clips from captions sampled from our grammar.
2. Fit a score to each clip by analysis-by-synthesis.
3. Add (caption, fitted score) to the planner data.
4. Add (DSP render, teacher clip) to the detail pairs.

This gives the small model access to a broader timbre distribution than 100k CC0 clips alone.

Licensing: check the teacher's license before generating. Some community licenses restrict training derivative models. If unclear, skip this loop; the CC0 + DSP path must work on its own. Disclose any teacher use in the model card.

## 5. Critic-in-the-loop

This is where a 66M model punches above its weight.

```
500k prompts
  → planner samples 8 scores each
  → render + detail
  → critic: reward model + measurements
  → keep best / worst pairs
  → DPO
  → (optional) GRPO with KL anchor
```

Prompt distribution: 60% agent-style requests, 25% film-context requests ("a 2 s riser into a logo at 0:08"), 15% adversarial ("make it warmer but keep the timing").

Preference pair construction:

- same prompt, two scores, critic score gap > 0.15
- no pairs where the gap comes only from loudness
- 20% of pairs are human-labelled (2k prompts, 3 annotators)

DPO: β=0.1, lr 5e-6, 1 epoch, SFT checkpoint frozen as reference. GRPO only if DPO is stable and the KL stays under 0.05.

## 6. Sizes and storage

| item | size |
|---|---|
| synthetic scores (1M) | 9 GB JSONL |
| captions (1M) | 0.2 GB |
| real recordings (100k) | ~100 GB |
| detail pairs (100k) | ~40 GB (renders cached) |
| teacher clips (500k, optional) | ~500 GB |
| fitted params | 250 MB |
| preference pairs (500k) | 400 MB |
| checkpoints (all stages) | ~150 GB |
| eval renders | ~50 GB |
| **total** | **~1.5 TB** (2 TB with teacher) |

## 7. Split policy

- **Train**: 90% of synthetic scores, 90% of real clips.
- **Dev**: 5%, used for early stopping and gate tuning.
- **Held-out**: 5%, opened once per gate. Captions use held-out templates.
- **Human eval**: 200 prompts, never seen by any model or tuner.
- **Film eval**: 12 real cuts from CC0 footage, scored with earshot `critique`.

No prompt in the human or film set may appear in any training corpus. Checked by exact match and by embedding similarity >0.95.

## 8. Failure modes to watch

| symptom | likely cause | fix |
|---|---|---|
| great on synthetic captions, bad on agent prompts | template inversion | more human seeds, caption-the-render, adversarial captions |
| planner always picks the same whoosh | mode collapse from CE | temperature, DPO, role-balanced sampling |
| detail model adds shimmer but no body | residual too high-frequency | multi-resolution loss, adversarial, more low-band weight |
| detail model moves the hit | phase-anchor loss too weak | increase weight, check alignment |
| music sounds like a MIDI file | weak instrument voices | physical modeling, humanization, mixing; accept ceiling |
| critic loves outputs humans hate | reward hacking | model-free measurements, human A/B, held-out critics |
| texture head produces noise | trajectory too coarse | more bands, grains, real anchoring, sample-bank fallback |
