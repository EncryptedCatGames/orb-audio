# Evaluation

Every gate is a number and a command. No gate is "it sounds good". Latency is not a gate; a 5-minute render is acceptable.

## Baselines

Run all four on every prompt set:

1. **Retrieval**: nearest of 10k pre-rendered scores by caption embedding. The bar. If the planner cannot beat retrieval, it is not earning its parameters.
2. **Hand-written**: a human sound designer writes the score. The ceiling of the score representation.
3. **DSP-only**: ORB without the detail model. Isolates the detail model's contribution.
4. **Stable Audio Open Small**: open-domain reference. Wins on timbre, loses on timing and editability.

## Prompt sets

| set | n | purpose |
|---|---|---|
| `agent-100` | 100 | realistic agent requests, held out |
| `film-12` | 12 | 15–60 s cuts with cue sheets |
| `roles-340` | 340 | 10 prompts × 34 roles |
| `beds-60` | 60 | 8–60 s functional music |
| `textures-40` | 40 | 8 characters × 5 |
| `adversarial-50` | 50 | ambiguous, contradictory, timing-critical |
| `detail-2000` | 2000 | paired (DSP, real) clips for the detail model |
| `human-200` | 200 | never seen by any model or tuner |

## Gates

### G0 — renderer contract (week 1)

- a hand-written 12 s score renders without error
- peak ≤ 0.99, no DC > 0.001, no click at boundaries
- event apex error < 10 ms
- deterministic: same score + seed → identical bytes
- earshot `critique` on the cue: no "major" fixes

### G1 — events (week 2)

- 95% of 340 renders pass `earshot sfx check`
- apex error p95 < 10 ms, duration error p95 < 5 ms
- pitch accuracy < 5 cents for tonal roles
- blind A/B vs earshot forge v2: ORB ≥ 40%

### G2 — music (week 2)

- 70% of 60 beds rated "usable under voice" by 3 listeners
- no masking: dialogue at −18 LUFS over the bed at −24 LUFS, STOI drop < 0.02
- key/tempo accuracy 100% (they are in the score)
- no seams at loop points

### G3 — detail model v1 (week 7, go/no-go)

Trained on whoosh and impact only, 5M params.

- blind A/B vs real designed SFX: **≥ 35%** (chance is 50%; 35% means "close, not fooling")
- earshot realness score improves by ≥ 0.15 over DSP-only
- apex error does not regress (p95 < 10 ms)
- no new artifacts: clicks, comb ring, cut tails

If this fails, the quality thesis fails. Pivot before spending the next 15 weeks.

### G4 — detail model v2 (week 10)

20M params, all roles.

- blind A/B vs real: **≥ 40%** on the 10 core roles
- 34-role earshot audit pass ≥ 95%
- no timing regression
- spectral balance within 2 dB of the reference profile in 1/3 octaves

### G5 — planner SFT (weeks 11–14)

- 95% schema validity on `agent-100` (grammar mask should make this 100%)
- 60% win vs retrieval on `agent-100`
- 80% of requested roles present
- 90% of events within 50 ms of a requested cue
- best-of-8 improves win rate by ≥ 5% over greedy

### G6 — reward model (week 15)

- held-out match AUC > 0.85
- realness AUC > 0.80
- sync head MAE < 20 ms
- rank correlation with human preference > 0.6 on 200 pairs

### G7 — DPO (week 17)

- 60% win vs SFT on `agent-100`, 3 listeners, 200 pairs
- no regression on `roles-340` (within 2%)
- KL to SFT < 0.05

### G8 — GRPO + search (week 20)

- +10% over DPO on `agent-100`
- local search adds +5% over best-of-8 alone
- no reward hacking: model-free measurements do not degrade
- human A/B confirms the gain on 100 fresh prompts

### G9 — ship (week 22)

- shipped params < 100M
- fp16 artifact < 400 MB
- no network calls at inference
- ≤5 min per 30 s cue on one GPU
- license audit: every shipped weight and sample traced to a compatible license

## Human protocol

- 3 listeners, blind, loudness-matched, randomized order.
- 200 pairs per gate, 20% repeated for consistency.
- Listeners hear the prompt and the two clips, not the model names.
- Report win rate with a 95% bootstrap interval. Do not ship on a point estimate.
- One listener is a smoke test, not evidence. The earshot README says the same; take it seriously.

## Film protocol

For each of the 12 cuts:

1. Agent gets the cut, the cue sheet, and the brief.
2. Agent produces a score, renders, runs the critic, revises up to 3 times.
3. Earshot `critique` scores the result.
4. A human rates sync, mix, and "would ship".

Metrics: earshot score, human ship rate, number of revisions, wall-clock time.

## Detail-model protocol

The detail model is evaluated as a paired transformation, not in isolation:

1. Take 500 held-out real clips.
2. Fit a score to each, render DSP-only and DSP+detail.
3. Blind A/B: real vs DSP-only, real vs DSP+detail.
4. Report both win rates and the delta. The delta is the detail model's contribution.
5. Check timing: onset, apex, tail must not move.

## What we do not measure

- FAD. Dominated by dataset statistics at this scale and rewards the wrong things.
- CLAP alone. Useful auxiliary, but it may be the thing we train against. Never the only gate.
- "Looks good in a notebook." Every gate runs from a script in `eval/`.

## Logging

Every run writes:

```
eval/runs/<date>-<stage>/
  config.json
  prompts.jsonl
  scores.jsonl
  renders/            # only for the human set
  report.md           # numbers + the exact commands
```

The report is the artifact. If a number cannot be reproduced from the report, it does not exist.
