# Model card — ORB v1

Published on the Hugging Face Hub as
[marcosmaceo/orb-planner](https://huggingface.co/marcosmaceo/orb-planner)
(Apache-2.0) and [marcosmaceo/orb-detail](https://huggingface.co/marcosmaceo/orb-detail)
(CC BY-NC-4.0). An earlier planner (v5) was trained on
[marcosmaceo/orb-scores-100k](https://huggingface.co/datasets/marcosmaceo/orb-scores-100k);
the released v8 was trained on regenerated synthetic data (see Planner below).

## Artifacts

| file | params | bytes | license | what |
|---|---|---|---|---|
| `weights/planner.pt` | 5,096,192 | 20.4 MB | Apache-2.0 | caption or plain text → compact Spec (v8) |
| `weights/detail.pt` | 9,064,387 | 36.3 MB | CC BY-NC-4.0 | high-band detail residual, detail v6 (separate download) |
| total, both models | 14,160,579 | 56.7 MB | mixed | the Apache-2.0 release is the planner alone (5.1M, 20.4 MB) |

Both are plain `torch.save` dicts (`model`/`generator` + `config`), no optimizer state, no pretrained weights. `weights/manifest.json` records the SHA-256 of each file. Safetensors copies for the Hub are produced by `scripts/to_safetensors.py`.

The detail weights are **not** part of the open-source release: they were trained on ESC-50 (CC BY-NC 3.0) and are published separately under non-commercial terms (CC BY-NC-4.0). See `weights/README.md`.

## Planner

- 6-layer decoder-only transformer, d=256, 4 heads, ctx 1024, byte-level tokenizer (395 tokens: 342 shared, one per role and per texture (47 roles, 13 textures), and `NOBED` for texture `none`).
- Input: a caption in the grammar format, e.g. `4.00-second playful cue, vinyl bed, with ui pop @0.60 0.16s -12dB R1 #73`, or, since v8, plain words, e.g. `rain on a tin roof with distant thunder`.
- Output: `Spec(duration, texture, mood, events=[SpecEvent(role, t, dur, gain_db, pan, seed)])`.
- Constrained decoding: structural tokens, categorical ranges, `t + dur <= duration`, and `END_EVENTS` at the event limit are enforced by a mask, so a decoded Spec is always valid.
- Planner v8 (`configs/planner_v8.json`, 5,096,192 params, step 39000 of 40000, batch 32, one H100 80GB on RunPod, about 35 minutes). Warm-started from v7; one new token (`NOBED`). Trained on the 150k v7 grammar pairs plus 150k synthetic plain-text pairs (`orb.freetext`).
- Grammar captions with v8: 500/500 on `heldout_v7.jsonl` (every field 100%), 2000/2000 (100%) on the 2,000 set, 500/500 (100%) on the v6 set.
- Plain-text prompts (`scripts/eval_freetext.py`; a prompt passes only if roles, extra roles, texture, mood, intensity and duration are all right). On our 209-prompt set (frozen, sha256 `7950a21c…`), v8 scores **58.4%**, against 45.0% for v7 behind the site's keyword helper. On a 40-prompt blind set written by someone else, v8 scores **70.0%**, against 77.5% for the helper. The gap between the two sets is real: the templates were written next to our set, and v8 generalises less than our set suggests. The main blind-set failure is an invented extra role for a noun ORB has no sound for.
- Planner v7 (`configs/planner_v7.json`, 5,095,936 params, step 28500 of 30000, kept as `weights/planner_v7.pt`). Same architecture as v5/v6; only the vocabulary grew (31 role and 5 texture tokens).
- Held-out (`data/heldout_v7.jsonl`, 500 captions, 390 using an everyday role): **500/500 exact (100%)**, 0 decode failures, token CE < 0.0001. Every role and scene bed is 100%.
- 2,000-caption held-out set: 2000/2000 (100%), 0 decode failures. The v6 held-out set (500, old captions): 500/500 (the v6 planner scored 98.8% on it).
- On a grammar this regular a perfect held-out score means the parser is solved, not that it understands language: anything outside the caption grammar is still out of distribution.
- Previous planners are kept as `weights/planner_v7.pt` (394 tokens), `weights/planner_v6.pt` (358 tokens) and `weights/planner_v5.pt` (352 tokens); v6 and v5 cannot emit the v7 roles.
- Training data (v8): 150k synthetic plain-text (prompt, Spec) pairs from `orb.freetext` (templates, synonyms, typos and word-order variants; no eval prompt included), plus 150k procedurally generated (caption, Spec) pairs from `orb.grammar.sample_spec` (25% are small scenes: storm, arrival, cave, battle, beach, forest morning, night camp, kitchen, office, city street, sword fight, haunted house); no audio, no external corpus.

## Detail model

**Routing (v0.3): the detail model is skipped for the physically modelled roles
and the scene beds.** All 37 natural/foley/everyday roles and the five scene
beds (ocean, stream, forest, night, traffic) are rendered by DSP only, with or
without `--detail`. `render_stems` puts them in a separate `dsp_only` stem;
`apply_detail_routed` runs the detail model on the rest of the mix and adds the
DSP-only stem back unchanged. A cue that uses only the original ten roles and
eight noise beds takes exactly the old path (bit-identical), and a cue made only
of physical models comes back exactly as the DSP render. Reason: the detail
model learned to restore degraded ESC-50 clips; on a clean physical model it can
only add risk (the one G1 near-miss it ever had was a foley cue; see below).


- Detail v6 (`configs/detail_v6.json`): complex-STFT U-Net, base 64, 4 levels, bilinear resize-conv upsampling, 9.06M params.
- Input: `(2, 513, T)` real/imag STFT of the DSP render; no noise channel.
- Residual is a smoothed mask (kernel 9×3) restricted to bins above 2 kHz, tapered off between 8 and 9.5 kHz, and gated by the input frame energy and a quiet-frame gate (30 dB, 10 dB ramp), so it cannot move the onset or the low band or lift silence.
- Losses: band-energy 2.0, reconstruction 0.05, envelope 0.2, overshoot 1.0, idempotence 5.0; no adversarial loss. Released checkpoint: step 1250 of 1500, batch 8, one H100 80GB, about 2 minutes.
- Training data: ESC-50 folds 1–4 (1599 pairs) degraded to a 40-band mel envelope + lowpass 2–6 kHz. **ESC-50 is CC BY-NC.** The harness weights are for research; retrain on CC0 audio before commercial release.
- No-noise gate (`scripts/eval_detail_artifacts.py`, fold 5, 400 clips): **PASS**. Tonal prominence 2.20 dB (< 6.0), periodicity 0.006 (≤ 0.2), residual periodicity 0.252 (≤ 0.3) on the original 24 cues, energy ratio above 10 kHz and 12 kHz 1.000× (≤ 1.25), quiet-frame rise +0.063 dB (≤ 0.5), onset F1 **0.749** (identity 0.749), MR-STFT improvement **+0.84%**.
- With the 8 natural/foley cues in the gate (32 cues): v6-era routing failed G1 by a hair (residual periodicity 0.304 vs 0.30 on footsteps over gravel, seed 2, with a wind bed). With the v0.3 routing those cues bypass the detail model and the 32-cue gate **passes**: prominence 2.69 dB, output periodicity 0.026, residual 0.268, HF 1.000×, quiet rise +0.063 dB, MR-STFT +0.84%. Same weights.
- The 10% MR-STFT restoration gate is **not met**. v5 (step 4000) reached 1.4% but failed the no-noise gate with tonal stripes from its `ConvTranspose2d` upsampling; v6 trades that for a cleaner, more conservative residual. It is a subtle texture stage, not a restoration model.

## End-to-end

- Caption → Spec → deterministic compiler → DSP render → optional detail → WAV.
- 50 held-out cues (measured with planner v5): **94% exact Spec** (sampling; 96.6% over 500), **100% audit pass**, max peak 0.983, max |DC| 1.5e-4, no clipping. The event stem has no energy before the first event (−77.6 dB); the full mix carries the texture bed from the first sample unless the Spec sets `"texture": "none"` (`"bed_db"` moves the bed level, default −40 dB relative to the events).
- 13-cue demo in `artifacts/demo/` (12/13 exact Spec; all 13 pass the audit).

## Intended use

- Generating short sound-design cues (transitions, UI sounds, impacts, risers,
  ambience beds), common natural, foley and everyday sounds (thunder, doors, footsteps,
  birds, waves, glass, phones, clocks, sirens, applause, …) and scene beds (ocean, stream,
  forest, night, traffic) from a structured caption or a Spec, including by software
  agents through the `orb-mcp` server.
- Research on small, structured, controllable audio generation.

Out of scope: open-vocabulary text-to-audio (plain words are read since v8, but only
into the 47 roles and 13 textures; anything else is approximated or invented), music,
speech, and imitating specific recorded sounds or voices.

## What is not here

- Music (v2).
- A CC0 detail dataset and a generative objective that passes the 10% MR-STFT gate.
- Voices and vocal animals (speech, laughter, crying, coughing, dogs, cats, cows, roosters, sheep): these need recordings or a learned model and are left out. ORB ships 47 roles (10 core + 37 physically modelled) and 13 textures (8 noise beds + 5 scene beds).
- Recorded realism: the natural, foley and everyday roles are physical models, cleaner and more regular than field recordings.
- A learned critic / best-of-N search.

## License

Code: Apache-2.0. Planner weights: trained from scratch on synthetic data. Detail weights: trained from scratch, but on the ESC-50 research harness — treat as non-commercial until retrained on CC0 audio.
