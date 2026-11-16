# Architecture

Four shipped components, one training-only discriminator, one test-time loop. The constraint is parameter count, not latency.

## 1. Text encoder

| | |
|---|---|
| type | decoder-only transformer, 6 layers, d=448, 7 heads |
| vocab | 8k BPE, trained on captions + score tokens |
| params | ~8M |
| input | caption + constraint tokens (duration, role, tempo, key, energy) |
| output | 448-d per token |

Trained jointly with the planner on 40M caption tokens. No pretrained encoder. The tokenizer is trained once on the caption corpus and frozen before planner training.

A frozen MiniLM or CLAP text tower may be used during development as a teacher. Never in the artifact.

## 2. Planner

| | |
|---|---|
| type | decoder-only transformer, 16 layers, d=512, 8 heads, SwiGLU, RoPE |
| params | ~32M |
| context | 2048 |
| output | score tokens, grammar-masked |
| decode | greedy for eval, temperature 0.8 / top-p 0.95 for sampling |

Training:

1. **SFT**: next-token cross-entropy on 10M synthetic scores, with loss masking so the model is not rewarded for copying the meta block.
2. **DPO**: 500k preference pairs from the critic, β=0.1, one epoch.
3. **GRPO** (optional): critic reward with a KL anchor to the SFT model, KL < 0.05.

**Grammar-constrained decoding** is the most important detail. At each step, a state machine allows only tokens that keep the score valid. The model can be wrong about *which* whoosh, never about whether the JSON parses. This is what makes a 32M planner usable by an agent.

## 3. DSP render

No parameters. 48 kHz, deterministic, sample-accurate.

### Events

Ten roles now, 34 at S1:

- modal resonator bank (impact, bell, pop, click)
- time-varying bandpass noise (whoosh, riser, swell, noise sweep)
- pitch sweeps (sub drop, riser)
- transient click layer
- saturation, room, stereo width

Each role exposes 8–12 physical knobs. The planner emits the role and a sparse override; defaults live in code. Continuous roles are peak-aligned to the score's `apex` by measuring the rendered envelope and shifting the clip (1–2 ms residual).

### Instruments

Six voices, upgraded to physical modeling because latency is free:

| voice | method |
|---|---|
| pad | detuned additive saw stack, slow attack, lowpass |
| pluck | Karplus-Strong / additive with per-harmonic decay |
| bass | sine + soft saw, short decay |
| bell | FM with decaying index |
| perc | noise transient + pitched body |
| strings | bowed model: stick-slip excitation + resonant body |

At 48 kHz with 32–64 harmonics and proper anti-aliasing. Quality comes from voicing, humanization, and mixing, not from a neural vocoder.

### Textures

Eight characters. The preset is a target spectral tilt, a slow LFO, and a grain density. The renderer shapes band-limited noise, adds sparse grains, mid/side width, and a seamless loop crossfade. A 4M head will later predict the 32-band trajectory from a character code; the renderer stays DSP.

## 4. Detail model

The timbre fix, and the highest-risk component.

### Shape

```
X = STFT(DSP render, n_fft=1024, hop=256)
c = score conditioning (role, params, section energy, instrument)
R = U-Net(X, c)                  # complex residual, 2 channels
audio = ISTFT(X + R)
```

| | |
|---|---|
| type | complex-STFT U-Net, 4 down / 4 up blocks, base 64 channels |
| params | ~20M |
| input | 2×T×F complex STFT + conditioning vector |
| output | 2×T×F complex residual |
| sample rate | 48 kHz |
| inference | 1 pass (deterministic) or 4–8 flow steps |

Why a residual instead of waveform generation:

- **Timing is inherited.** The DSP render carries the onset, apex, and duration. The model only adds texture; it cannot move the hit.
- **No vocoder.** ISTFT is the decoder. Phase is anchored to the DSP render.
- **Small.** It learns a correction, not a distribution from scratch.
- **Trainable with the losses that matter.** Multi-resolution STFT + adversarial + phase-anchor.

### Training

Paired data: each real clip is paired with a DSP render of a fitted score.

1. Fit a score to the real clip: role, duration, onset, apex, spectral tilt, pitch. Use earshot `fit` where possible; otherwise a short parameter search.
2. Render the score with the DSP engine.
3. Align by onset; crop to the same length.
4. Train the residual.

Losses:

| loss | weight | purpose |
|---|---|---|
| multi-resolution STFT (3 scales) | 1.0 | spectral realism |
| complex L1 | 0.5 | phase and transient accuracy |
| adversarial (multi-scale discriminator, 2M) | 0.1 | recorded texture |
| phase-anchor (onset/apex) | 0.2 | never move the hit |
| loudness | 0.05 | no level drift |

The discriminator is training-only and is not shipped.

**Phase caveat.** The real clip and the DSP render are aligned by onset, not by absolute phase. The complex L1 is therefore an auxiliary term; the main perceptual loss is the multi-resolution STFT on the *output*, which is phase-insensitive. The model may output a complex mask, but it is not punished for a constant phase offset. If phase prediction proves unstable, switch to a magnitude mask with the DSP phase and keep the adversarial term.

### Fallbacks

If the complex-STFT residual underperforms:

1. **Small vocoder**: 8M HiFi-GAN-style, conditioned on a detailed mel. Adds 8M params.
2. **Sample-grain layer**: a 10M model that retrieves and morphs grains from a CC0 bank. Less general, very realistic.
3. **Accept the ceiling**: ship the DSP-only "designed" sound and narrow the product to timing-critical SFX.

The go/no-go at week 7 decides which path.

## 5. Reward model and critic

### Reward model

| | |
|---|---|
| audio encoder | 4 conv blocks on 64×T mel → 4 transformer layers d=256 |
| text encoder | shared with the planner |
| params | ~6M |
| heads | match (contrastive), realness (binary), sync (regression on apex error) |

Trained with InfoNCE on 2M (caption, render) pairs plus regression on measured properties. Trained on *our renderer's* outputs, including its failure modes.

### Critic

Measurement first, model second:

- loudness (LUFS), true peak, LRA
- spectral balance in 1/3 octaves against the role's reference profile
- onset, apex, tail length
- clicks, DC, clipping, comb ring
- sync to cue times
- optional earshot `critique` for film-level checks

Returns words: "apex 14 ms late", "tail 2.1× too short", "2.4 dB too bright".

## 6. Test-time loop

Quality scales with compute, so spend it.

```
prompt
  → sample 8 scores
  → render + detail (batched)
  → rank with reward model
  → keep top 2
  → 60 s local search over event params and mix gains (CMA-ES)
  → 3 critique-revise rounds on the score
  → return best
```

Budget: ≤5 minutes per 30 s cue on one GPU. On a 4-card box, batch the 8 samples across cards.

Local search space: event `t`, `dur`, `apex`, gain, pan, 8–12 role params; track gain and pan; texture gain and width. 20–40 dimensions. Objective: reward model + model-free measurements + sync error.

## 7. Inference path

```
caption + constraints
  → tokenize (8k BPE)
  → planner (grammar-masked, 8 samples)
  → score JSON
  → DSP render (48 kHz)
  → detail model
  → critic
  → local search / revise
  → best cue
```

Quantization: int8 ONNX for the planner and reward model; fp16 for the detail model. Target <400 MB total.

## 8. What is deliberately not learned

- the score grammar
- the event synth
- the instrument voices
- the texture renderer
- the measurements
- the search loop

Every one of those is deterministic code. The learned parts decide *what* to render and *how to correct the timbre*, never *when* a hit lands. That is the difference between a 66M system that works and a 300M system that almost works.
