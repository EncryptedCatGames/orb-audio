# Model card — ORB v1

## Artifacts

| file | params | bytes | license | what |
|---|---|---|---|---|
| `weights/planner.pt` | 5,085,184 | 20.4 MB | Apache-2.0 | caption → compact Spec |
| `weights/detail.pt` | 10,269,187 | 41.1 MB | CC BY-NC-4.0 | high-band detail residual (separate download) |
| **total** | **15,354,371** | **61.5 MB** | | |

Both are plain `torch.save` dicts (`model`/`generator` + `config`), no optimizer state, no pretrained weights. `weights/manifest.json` records the SHA-256 of each file. Safetensors copies for the Hub are produced by `scripts/to_safetensors.py`.

The detail weights are **not** part of the open-source release: they were trained on ESC-50 (CC BY-NC-4.0) and are published separately under the same terms. See `weights/README.md`.

## Planner

- 6-layer decoder-only transformer, d=256, 4 heads, ctx 1024, byte-level tokenizer (352 tokens).
- Input: a caption in the grammar format, e.g. `4.00-second playful cue, vinyl bed, with ui pop @0.60 0.16s -12dB R1 #73`.
- Output: `Spec(duration, texture, mood, events=[SpecEvent(role, t, dur, gain_db, pan, seed)])`.
- Constrained decoding: structural tokens, categorical ranges, `t + dur <= duration`, and `END_EVENTS` at the event limit are enforced by a mask, so a decoded Spec is always valid.
- Held-out: **483/500 exact (96.6%)**, 0 decode failures, token CE 0.0029.
- Training data: 100k procedurally generated (caption, Spec) pairs from `orb.grammar.sample_spec`; no audio, no external corpus.

## Detail model

- Complex-STFT U-Net, base 64, 4 levels, 10.27M params.
- Input: `(2, 513, T)` real/imag STFT of the DSP render, plus one Gaussian noise channel.
- Residual is restricted to bins above 2 kHz and gated by the input frame energy, so it cannot move the onset or the low band.
- A learnable noise floor seeds the high band; without it the band-energy loss has zero gradient at zero.
- Training data: ESC-50 folds 1–4 (1599 pairs) degraded to a 40-band mel envelope + lowpass 2–6 kHz. **ESC-50 is CC BY-NC.** The harness weights are for research; retrain on CC0 audio before commercial release.
- Harness (fold 5, 400 clips): high-band ratio **0.337** (identity 0.058), noise sensitivity **0.036**, onset F1 **0.748**, MR-STFT improvement **1.4%**. The 10% MR-STFT gate is **not met**.
- End to end on full-band DSP renders: +3.3% high-band energy, +48 Hz spectral centroid, 100% audit pass. It is a subtle texture stage, not a restoration model.

## End-to-end

- Caption → Spec → deterministic compiler → DSP render → optional detail → WAV.
- 50 held-out cues: **94% exact Spec** (sampling; 96.6% over 500), **100% audit pass**, max peak 0.983, max |DC| 1.5e-4, no clipping, no energy before the first event (−77.6 dB).
- 13-cue demo in `artifacts/demo/` (12/13 exact Spec; all 13 pass the audit).

## What is not here

- Music (v2).
- A CC0 detail dataset and a generative objective that passes the 10% MR-STFT gate.
- The 34-role SFX set; v1 ships the 10 core roles and 8 textures.
- A learned critic / best-of-N search.

## License

Code: Apache-2.0. Planner weights: trained from scratch on synthetic data. Detail weights: trained from scratch, but on the ESC-50 research harness — treat as non-commercial until retrained on CC0 audio.
