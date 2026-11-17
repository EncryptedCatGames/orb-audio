# Adversarial review

Red-team of the ORB plan, before execution. The goal is to find what kills the program while it is still cheap to change.

## Verdict

The thesis survives. The plan as written does not.

Three things can kill it:

1. **The detail model is trained on a signal it cannot trust.** Fitting a score to a real clip is lossy; the residual contains fitting error, not just timbre.
2. **The data for designed SFX may not exist under CC0.** Freesound CC0 is mostly field recordings, not whooshes and impacts.
3. **The plan tries to ship SFX, music, and texture at once.** Music is the weakest target, the hardest to evaluate, and the least differentiated.

Everything else is fixable. The revised plan below cuts scope, reorders the build, and adds the missing capabilities.

## Result (2026-11-16)

The detail-model risk was tested first. A deterministic regression U-Net, trained on controlled degraded-real pairs, reached **16.1%** MR-STFT improvement over identity on held-out ESC-50 (gate 25%), with onset F1 **0.740** and high-band ratio **0.172**. A relative-magnitude loss removed the hallucinated high-frequency energy but collapsed the high band to **0.013** and improved MR-STFT by **2.2%**. The regression objective is the failure: it converges to the conditional mean. The next attempt needs a noise input and a generative objective before the 20k CC0 collection.

The planner risk was also tested. The compact Spec planner reaches **96.6% free-running exact match** and held-out CE **0.0021**, so the planner is no longer the bottleneck.

## What survives

- **Score-first.** Timing as a score property is the right abstraction. Keep it.
- **DSP as the native representation.** Exact duration, pitch, apex, determinism. Keep it.
- **The critic loop.** Measurement-first evaluation is the differentiator. Keep it.
- **The parameter cap.** 100M shipped is the discipline. Keep it, but allocate by measurement, not by guess.
- **From scratch.** No pretrained weights in the artifact. Keep it, but allow teachers offline.

## Attack 1 — the detail model's training signal is untrustworthy

The plan pairs each real clip with a DSP render of a *fitted* score. Fitting is ambiguous: many scores render to similar audio, and the fit will be wrong in ways the residual model then learns to correct. The model may become a fit-error inverter, not a timbre model.

Worse, the residual `real − DSP` can be large and structured. A U-Net with 20M params can learn to copy the real clip's identity, which looks great on the training loss and fails on new scores.

**Fix:**

- Score every pair before training: onset error, duration error, spectral distance, pitch error. Keep the top 30%. A smaller, cleaner dataset beats 100k noisy pairs.
- Add a **residual energy cap** and a **low-rank bottleneck** so the model cannot reconstruct the target from scratch.
- Train on two objectives: (a) paired residual, (b) **controlled degradation** — take a real clip, apply a known DSP-like degradation, learn to invert it. The second gives a clean signal and teaches the model what "synthetic" means.
- Evaluate on held-out real clips with human A/B, never on the training loss.
- Report the **delta over DSP-only**, not the absolute score. The delta is the only number that matters.

**Kill criterion:** if DSP+detail does not beat DSP-only by 15 points in a 5-listener A/B, stop.

## Attack 2 — the data may not exist

100k CC0 designed SFX is optimistic. Freesound CC0 is dominated by field recordings, foley, and ambience. Designed whooshes, impacts, risers, and UI sounds are usually licensed libraries, not CC0.

If the detail model trains on field recordings, it will learn to make field recordings. That does not help a designed whoosh.

**Fix:**

- Split the data by role family:
  - **texture/ambience**: CC0 is abundant. Train here first.
  - **designed SFX**: use teacher-generated clips (Stable Audio Open or equivalent) and a small licensed library. CC0 alone will not cover it.
- If the teacher license blocks derivative training, ship v1 as **DSP SFX + neural texture** and defer neural SFX. That is still a useful product.
- Audit the role distribution before writing the training script. If a role has fewer than 500 clean clips, it does not get a detail model.

## Attack 3 — the planner may be unnecessary for v1

A 32M planner is the most expensive component to train and the easiest to replace with retrieval. Ten thousand pre-rendered scores, embedded by caption, plus local search, may already beat a small planner. If it does, the planner is a later upgrade, not a prerequisite.

**Fix:** build the retrieval baseline first. Measure it on `agent-100`. Train the planner only if it beats retrieval by 10 points. This de-risks the program and gives every later model a real bar.

## Attack 4 — music is a tar pit

Six symbolic voices will not reach "library-grade" without a major synthesis effort. Music is also the hardest to evaluate, the least differentiated (Suno, Udio, Stable Audio exist), and the least tied to the timing advantage.

**Fix:** cut music from v1. Ship **SFX + texture**. Add music in v2 only if the detail model works and the SFX product is shipped. This removes ~30% of the scope and the weakest quality target.

## Attack 5 — the evaluation is underpowered and gameable

Three listeners × 200 pairs gives a 95% CI of roughly ±7 points. A 40% target is within noise of 50%. The reward model can be gamed. The "real vs synthetic" question is also wrong for designed SFX, where the reference is itself designed.

**Fix:**

- 5 listeners × 400 pairs for the go/no-go.
- Forced choice with a **"can't tell"** option; report the proportion of "can't tell" separately.
- Primary metric: **"would you ship this in a cut?"** not "is it real?"
- Model-free audit as a hard gate: clicks, DC, clipping, comb ring, timing.
- Pre-register the prompts and the analysis before listening.
- Keep a human set that no model, tuner, or reward model ever sees.

## Attack 6 — no reference-audio path

Agents often have a reference: "like this, but warmer." The plan cannot use one. That is a major capability gap and a missed differentiator.

**Fix:** add a reference path.

- v1: retrieval + fit. Embed the reference, retrieve the nearest score, fit the event params, render.
- v2: a 4M reference encoder that conditions the planner and the detail model.
- Add `reference` to the score schema as an optional 64-d embedding.

## Attack 7 — the agent interface is missing

The score is an implementation detail, not an API. Agents should not have to author JSON with 40 fields. The planner solves text → score, but the edit loop is undefined.

**Fix:** define the API before the model.

```
make_cue(brief, duration, cues, references=None) -> Cue
edit_cue(cue, diff) -> Cue          # move, replace, mute, gain, swap role
render_stems(cue) -> {music, texture, events, mix}
audit(cue) -> Report
```

The score stays internal. The agent sees a cue, a diff, and stems.

## Attack 8 — the parameter budget is arbitrary

66M is a guess. The planner may need 10M; the detail model may need 40M. Fixing the split before measuring is how you waste a month.

**Fix:** run scaling curves.

- planner: 1M, 5M, 15M, 38M on 100k / 1M / 10M scores
- detail: 5M, 10M, 20M on 20k pairs
- reward: 2M, 6M

Allocate the budget to the component with the best quality-per-param slope. Report a Pareto curve, not a single number.

## Attack 9 — "from scratch" is a purity tax

A pretrained text encoder would save 8M params and improve generalization. A pretrained vocoder would save years. The user wants from scratch; that is a legitimate product constraint, but it has a cost.

**Fix:** keep the artifact from scratch, allow teachers offline, and document the trade. If quality stalls, revisit the constraint with data, not dogma.

## Attack 10 — no memorization or provenance check

A generative model trained on 100k clips can reproduce them. For a commercial tool, that is a legal and reputational risk.

**Fix:**

- Nearest-neighbor check against the training set for every shipped render.
- Reject if the mel/CLAP distance is below a threshold calibrated on held-out clips.
- Per-clip provenance log.
- Model card with data sources and known limitations.

## Attack 11 — the timeline is fiction

24 weeks for one person, including a research-grade detail model, is optimistic. The detail model alone is a 2–4 week research sprint with an uncertain outcome.

**Fix:** 2-week detail sprint with a kill criterion, then a 12-week v1, then music v2. Re-plan after the sprint with real numbers.

## Attack 12 — the product is not defined

Who pays? Agents making videos. What do they need? SFX and room tone that sync, in a format they can edit. The score format + MCP server may be the product; the model is an implementation detail.

**Fix:** define the workflow and the API first. Build the model to fit the workflow, not the other way around.

## Attack 13 — the detail model may not generalize across roles

One U-Net conditioned on role may blur. A whoosh and a bell have different residual statistics.

**Fix:** per-role FiLM adapters or a small mixture of experts. Test on held-out roles. If one model fails, train two: transient and tonal.

## Attack 14 — no stems or mix integration

Agents need to mix under dialogue. A single stereo cue is not enough.

**Fix:** render stems (events, texture, music) and a mix bus. Add ducking and loudness targets. This is cheap and high-value.

## Attack 15 — no edit-consistency test

The score supports edits, but no gate measures whether an edit preserves the rest.

**Fix:** edit a cue (move one event, change one gain), render, and check that only the intended stem changes and that timing is preserved.

## Revised plan (v2)

### Scope

**v1: SFX + texture.** Music deferred. 34 event roles, 8 texture characters. No beds, no songs.

### Build order

1. **Retrieval baseline** (week 1). 10k pre-rendered scores, caption embeddings, nearest neighbor, local search. This is the bar.
2. **Detail sprint** (weeks 2–4). 20k clean pairs, 5M residual model, 5-listener A/B. **Go/no-go.**
3. **Detail v2** (weeks 5–7). 20M, all roles, adversarial, per-role adapters.
4. **Planner** (weeks 8–12). Only if it beats retrieval by 10 points. 38M, grammar-masked, SFT + DPO.
5. **Reward + search** (weeks 13–15). 6M reward, best-of-8, local search.
6. **Ship** (weeks 16–18). Quantize, stems, MCP, model card, license audit.

### Data

- **Texture**: 30k CC0 ambience/field recordings. Abundant.
- **Designed SFX**: 20k teacher-generated + 5k licensed (if available). CC0 only if the role audit passes.
- **Pairs**: filtered to the top 30% by fit quality.
- **Captions**: 2k human seeds, 10k templates, 50k paraphrases. Held-out templates for eval.

### Evaluation

- 5 listeners × 400 pairs for the go/no-go.
- Primary: "would you ship this in a cut?"
- Hard gates: model-free audit, timing, memorization.
- Retrieval baseline on every prompt set.
- DSP-only baseline on every detail gate.

### Parameters

- Planner: 38M (scale down if the curve says so).
- Detail: 20M (scale up to 30M if the curve says so).
- Reward: 6M.
- Cap: 100M. Target: 60M.

### Timeline

18 weeks to v1, with a kill criterion at week 4. Music v2 after.

## Decision log

| decision | reason |
|---|---|
| cut music from v1 | weakest target, hardest eval, least differentiation |
| retrieval before planner | cheaper, de-risks, gives a real bar |
| detail model first | highest risk, decides the quality thesis |
| 5 listeners, 400 pairs | 3 × 200 is within noise of chance |
| "would you ship" primary | "real vs synthetic" is the wrong question for designed SFX |
| filter pairs to top 30% | fitting error is the dominant noise source |
| residual cap + bottleneck | prevents copying the target |
| add reference path | agents have references; it is a differentiator |
| add stems | agents need to mix under dialogue |
| teacher data allowed, weights not | keeps the artifact from scratch without starving the detail model |
| 18 weeks, not 24 | scope cut and a hard kill criterion |
