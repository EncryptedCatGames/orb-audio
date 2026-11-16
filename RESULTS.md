# Results — 2026-11-16

## Planner: compact Spec v2

- Model: **5.08M params**, 6 layers, d=256, ctx 1024, byte-level tokenizer.
- Data: 100k deterministic (caption, Spec) pairs; 500 held out.
- Target: duration (2–30 s in 0.5 s steps), texture, mood, and 0–8 timed events (`t`, `dur`, `gain`, `pan`, `seed`). Numbers are decimal digit tokens.
- Held-out token CE **0.0029** (ppl 1.003).
- Free-running greedy exact match **483/500 (96.6%)**, zero decode failures.
- Per-field: duration/texture/mood 100%, roles 100%, times 99.6%, durs 99.6%, gains 99.9%, pans 99.9%, seeds 99.7%.
- Remaining failures are single-digit copy errors (e.g. gain `-6` → `-16`); the structured Spec API is the exact path.
- Edge cases verified: 0 events (pure ambience), 1 event (the common agent call), and 8 events all reproduce exactly.

Two bugs found and fixed while widening the distribution:

1. `sample_spec` only produced 2–5 events, so a single-sound caption was out of distribution and the planner emitted zeros for every number. The distribution is now 0–8 events.
2. At `MAX_EVENTS` the constrained-decoding mask skipped `END_EVENTS`, producing an invalid Spec. The mask now always closes the event list; `tests/test_spec.py` covers it.

Commands:

```bash
/workspace/venv/bin/python scripts/gen_scores.py --n 100000 --out data/scores_100k.jsonl --seed 2 --no-music
/workspace/venv/bin/python scripts/gen_scores.py --n 500 --out data/heldout.jsonl --seed 54321 --no-music
/workspace/venv/bin/python scripts/train_spec.py --config configs/spec_v5.json
/workspace/venv/bin/python scripts/eval_spec.py --checkpoint runs/spec_v5/best.pt --data data/heldout.jsonl --n 500
```

## Detail harness

- Data: ESC-50, folds 1–4 train (1599 pairs), fold 5 eval (400 pairs). Harness only; ESC-50 is CC BY-NC and the shipped detail model must be retrained on CC0 before release.
- Degradation: 40-band Slaney mel envelope → shaped noise + tracked sinusoids, lowpass 2–6 kHz, RMS match.
- Identity baseline: MR-STFT **0.2289**, onset F1 **0.749**, high-band ratio **0.058**.
- The first deterministic regression model reached 16.1% MR-STFT but collapsed the high band to 0.172 (absolute loss) or 0.013 (relative loss). Diagnosis: the degradation destroys the high band, and a regression model converges to the conditional mean.
- The current model is a **10.27M complex-STFT U-Net** with a Gaussian noise channel, a soft energy gate, a high-frequency-only residual (above 2 kHz), a learnable noise floor, and a weighted band-energy loss `[0.05, 0.05, 0.1, 0.3, 1.0, 2.0, 4.0]` over seven octave bands.
- Best checkpoint (`detail_v5` step 4000): MR-STFT improvement **1.4%**, onset F1 **0.748**, high-band ratio **0.337** (identity 0.058), noise sensitivity **0.036**.
- The 4–8 kHz band reaches **0.38** of the real energy; 8–16 kHz stays at **0.05**. The model oscillates: later checkpoints collapse back toward identity, and the band loss alone is not enough to hold the top.
- On in-domain (full-band) DSP renders the model is deliberately conservative: **+3.3% high-band energy**, **+48 Hz centroid**, 100% audit pass, 1–5% relative waveform change at correlation 0.999.

Honest reading: the harness gate (10% MR-STFT, high-band 0.7–1.3) is **not met**. The detail model is an optional texture stage; the DSP path is the shippable v1. A CC0 detail dataset and a generative objective (flow matching, or a high-band discriminator with R1) are the next step.

Commands:

```bash
/workspace/venv/bin/python scripts/build_degrade_pairs.py --esc50 /workspace/data/ESC-50-master
/workspace/venv/bin/python scripts/train_detail.py --config configs/detail_v3.json --resume runs/detail_v2/step3000.pt
/workspace/venv/bin/python scripts/eval_detail.py --data data/degrade/eval.jsonl --checkpoint runs/detail_v3/step6000.pt
/workspace/venv/bin/python scripts/band_diag.py runs/detail_v3/step6000.pt
```

## Noise floor fix (2026-11-16)

Reported symptom: audible background hiss under the generated cues. Three causes, all in the deterministic renderer — **no retraining was needed**.

1. **Normalisation.** `render_texture` leaves the bed at −52 dBFS, then `gain = target / rms(mix)` boosted it by up to +32 dB on sparse cues, so the same −30 dB bed landed at −20 dBFS and sat as loud as the sound itself. The mix now normalises the foreground to a 0.89 peak and places the bed `gain_db` below the foreground's *active* (95th-percentile frame) loudness. Bed on a sparse cue: **−27 → −46 dBFS**.
2. **Spectral tail.** `np.interp` clamps past the last control point, so vinyl stayed flat at −36 dB and rain at −24 dB from 12 kHz to Nyquist — a white-noise tail. `_shape_curve` now rolls off −24 dB/octave above 8 kHz and high-passes at 30 Hz. Rain energy above 8 kHz: **26% → 2.2%**.
3. **Grains.** Grains were raw white bursts (`standard_normal * 0.35`), which read as clicks and hiss. They are now band-passed around a random 0.6–6 kHz centre, normalised, and scaled to the bed RMS.

Also: `soft_limit` was applied per stem, so the stems no longer summed to the mix; there is now a single safety gain applied to every stem. The bed's LFO is three incommensurate sines instead of one, which removes the periodic tremolo that made a noise bed read as "filtered white noise".

The detail model was not the cause: it adds **+0.1 dB** in the silent gaps. It is now slightly out of distribution (the bed it sees is quieter and darker), but it still passes the audit and remains an optional stage.

Audit after the fix: 12/12 web examples and 13/13 demo cues pass; max peak 0.905, max |DC| 2.5e-4, 12/13 exact Spec.

## End to end

- Caption → Spec → Score → DSP → WAV, with the planner v5.
- 500 held-out captions: **96.6% exact Spec**, 0 decode failures.
- 50 held-out cues: **94% exact Spec** (small-sample), **100% audit pass**, max peak 0.983, max |DC| 1.5e-4, no clipping.
- No energy before the first event: **−77.6 dB** relative to the event stem.
- The DSP renders are full-band; the bed is rolled off above 8 kHz, so the mix centroid now sits around 2–5 kHz depending on the events.
- The detail stage adds **+3.3% high-band energy** and **+48 Hz centroid**, keeps the audit at 100%, and preserves timing.
- 13-cue demo: 12/13 exact Spec (one single-digit gain copy error), **13/13 audit pass**, high-band ratio 1.00–1.28.

Commands:

```bash
/workspace/venv/bin/python scripts/generate.py --prompt '4.00-second playful cue, vinyl bed, with ui pop @0.60 0.16s -12dB R1 #73' --out out/ui_pop.wav
/workspace/venv/bin/python scripts/eval_end_to_end.py --detail '' --n 20
/workspace/venv/bin/python scripts/make_demo.py --detail '' --out out/demo
```

## Cost

- Pod: RTX 4090 Secure, **$0.89/hr**, ~6 h = **~$5**.
- Both pods terminated and the 100 GB network volume deleted after the artifacts were verified locally. Nothing is still billing.
- Everything needed to reproduce or retrain is in `artifacts/`: slim checkpoints, full checkpoints, configs, the 100k planner pairs, the 1.8 GB detail pairs, and the demo WAVs.
