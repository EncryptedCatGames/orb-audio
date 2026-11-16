# ORB — plan v2

**Status (2026-11-16):** the compact Spec planner is trained and verified (**96.6% free-running exact** on 500 unseen captions, 0 decode failures, 2–30 s and 0–8 events). The detail harness went no-go for a deterministic regression model; the current detail model is a complex-STFT U-Net with a noise channel, an energy gate, a high-frequency-only residual, and a weighted band-energy loss. It is still in the detail sprint. The red-team is in [`REVIEW.md`](REVIEW.md). This document is the revised contract.

**Correction to the earlier diagnosis.** The flat planner's rising validation loss was not memorization. `evaluate()` compared `logits[t]` to `labels[t]` while training used the shifted next-token target, so validation measured a different task. With the shift fixed, the compact Spec planner scores held-out token CE **0.0021** (ppl 1.002) and **96.6% free-running exact match** on 500 unseen captions. The flat 200-token planner is retired.

**Detail harness result (2026-11-16): no-go for the deterministic regression detail model.** On 400 held-out ESC-50 clips, the best run reached **16.1% MR-STFT improvement** over identity (gate 25%), onset F1 **0.740** (identity 0.749), and high-band ratio **0.172** (gate > 1.0). A relative-magnitude loss removed the hallucinated high-frequency energy but collapsed the high band to near-silence (ratio 0.013) and improved MR-STFT by only 2.2%. The degradation destroys the high-frequency information, so a regression model converges to the conditional mean. The next detail model needs a noise input and a generative objective (adversarial or flow matching), or a milder degradation that leaves more information.

## 1. What changed from v1

The thesis survives. The scope and build order do not.

| v1 | v2 |
|---|---|
| SFX + music + texture | **SFX + texture**; music deferred |
| planner first | **retrieval baseline first**, planner only if it wins |
| detail model as a component | **detail model first**, 2-week kill criterion |
| 3 listeners × 200 pairs | **5 listeners × 400 pairs**, "would you ship" primary |
| 66M split by guess | **scaling curves**, allocate by quality per param |
| no reference path | **reference audio** via retrieval + fit |
| single stereo cue | **stems + mix bus** |
| 24 weeks | **18 weeks to v1**, music v2 |

## 2. The bet

Minimize learned parameters subject to a quality target, with inference compute treated as free. "Small" means parameter count and artifact size, not latency.

The score is the semantic codec. DSP is the native renderer. A small neural detail model adds the recorded texture. The critic and test-time search close the loop.

```
intent → planner (4M) → Spec (duration, texture, mood, timed events) → compiler → DSP render → detail model (10–23M) → critic → best cue
```

## 3. v1 scope

**In:** 34 designed SFX roles, 8 texture characters, stems, reference audio, edit API.

**Out:** music beds, songs, vocals, foley, real places, realtime.

The reason is in [`REVIEW.md`](REVIEW.md) §4: music is the weakest target, the hardest to evaluate, and the least differentiated. It comes back in v2 if the detail model works.

## 4. Build order

| week | deliverable | gate |
|---|---|---|
| 1 | compact Spec v2 + 4M planner | **96.6% free-running exact match** |
| 2 | detail degradation harness | **no-go: 16.1% vs 25% gate; regression detail model insufficient** |
| 3–4 | **detail sprint**: 20k CC0 pairs, 10–23M residual, 5-listener A/B | **go/no-go: +15 pts over DSP-only** |
| 5–7 | detail v2: 23M, all roles, adversarial, per-role adapters | +20 pts over DSP-only |
| 8–10 | reward model + best-of-N, only if the audit says it helps | +10 pts over SFT |
| 11–12 | stems, MCP, quantization, model card, license audit | ship |

**Kill criterion at week 4.** If DSP+detail does not beat DSP-only by 15 points in a 5-listener A/B, stop and pivot to a vocoder, a sample-grain layer, or a narrower product.

## 5. Artifact

| component | params | shipped |
|---|---|---|
| planner (byte-level tokenizer, 6 layers, d=256, 20–170 token Spec) | 4M | yes |
| detail model (complex-STFT U-Net; base 64 harness, base 96 final) | 10M / 23M | yes |
| reward model | deferred to v2 | no |
| discriminator | 0.3M | no |
| **shipped total** | **~27M** | |

Cap: 100M. The 69M target is a ceiling, not a floor. The planner is a parser over a deterministic Spec, so extra planner parameters buy nothing; the detail model is the only component where parameters buy audible quality.

## 6. Capability targets

| capability | target | measurement |
|---|---|---|
| designed SFX, 10 core roles | +20 pts over DSP-only | 5 listeners × 400 pairs |
| designed SFX, 34 roles | 95% audit pass | model-free audit |
| texture, 8 characters | 80% character ID, no seam | 5 listeners × 200 pairs |
| film cue | 10/12 pass earshot; 60% "would ship" | 12 CC0 cuts |
| timing | apex p95 < 10 ms | 340 renders |
| edit consistency | only the intended stem changes | 50 edits |
| memorization | no render within threshold of train | NN check |
| artifact | <100M, <400 MB fp16, no network | `doctor` |

## 7. Data

| source | size | use |
|---|---|---|
| texture / ambience CC0 | 30k | detail pairs, anchoring |
| designed SFX, teacher-generated | 20k | detail pairs |
| designed SFX, licensed | 5k | detail pairs, eval |
| synthetic scores | 1M | planner SFT, retrieval |
| preference pairs | 100k | DPO |
| human seeds | 2k | captions |
| human preferences | 2k | reward calibration |

Pairs are filtered to the **top 30%** by fit quality (onset, duration, spectral distance). A smaller clean set beats 100k noisy pairs.

Teacher-generated data is allowed; teacher weights are not shipped. If the teacher license blocks derivative training, ship DSP SFX + neural texture and defer neural SFX.

## 8. Evaluation

- 5 listeners × 400 pairs for the go/no-go.
- Forced choice with "can't tell"; report it separately.
- Primary: **"would you ship this in a cut?"**
- Hard gates: model-free audit, timing, memorization.
- Retrieval and DSP-only baselines on every set.
- Human set never seen by any model, tuner, or reward model.

## 9. Compute

| item | GPU-hours |
|---|---|
| detail sprint (5M) | 80 |
| detail v2 (20M) + adversarial | 300 |
| planner SFT (1M scores) | 120 |
| reward model | 60 |
| DPO | 100 |
| search / eval | 200 |
| teacher generation | 300 |
| ablations / failures | 300 |
| **total** | **~1,460** |

4× RTX 4000 Ada (20 GB), ~3 weeks wall clock, ~$2k–4k. One card works; double the schedule.

## 10. Risks

| risk | mitigation |
|---|---|
| detail model fails | **confirmed at the harness**: switch to a generative objective (noise input, adversarial/flow matching) before collecting 20k CC0 pairs |
| designed SFX data missing | teacher data; licensed set; ship texture first |
| planner does not beat retrieval | ship retrieval + search; planner v2 |
| reward hacking | model-free gates, human holdout, KL anchor |
| music scope creep | it is out of v1 |
| memorization | NN check, provenance log, model card |

## 11. Week 1

- [x] red-team and plan v2
- [x] 48 kHz DSP, 10 roles, 6 voices, 8 textures, smoke
- [x] byte-level tokenizer + grammar mask
- [x] planner, detail, reward, discriminator models
- [x] SFT, detail, reward, DPO loops with resume
- [x] pair builder + filter, reward manifest, preference builder
- [x] audit, retrieval baseline, mel, stems
- [x] 35 tests + full training smoke on Linux and the Runpod 4090
- [x] 1M synthetic scores generated on the CPU box
- [x] compact Spec v2: timed events, digit tokens, constrained decoding
- [x] Spec planner trained: 96.6% free-running exact match, CE 0.0021
- [x] detail degradation harness + identity baseline
- [x] detail harness go/no-go: **no-go** (16.1% vs 25%)
- [ ] generative detail model (noise input, adversarial/flow matching)
- [ ] 20k CC0 pairs, role audit
- [ ] 23M detail model, 5-listener A/B

The detail model is the only remaining go/no-go. The planner is done.
