# Training on Runpod

Everything below runs on one 24 GB GPU. The current box is a **Runpod RTX 4090 Secure** at **$0.89/hr** with a 100 GB standard network volume at `/workspace` (~$7/month). Stop the pod when no run is active; the volume keeps the data.

## 1. Box

The shipped planner is 5.08M and the detail model is 10.27M, so a 24 GB card is comfortable. Everything here was trained on a single RTX 4090.

| stage | VRAM at batch 4–8 | fits 20 GB |
|---|---|---|
| planner 5.08M, seq 1024 | 4–8 GB | yes |
| detail 10.27M, 2×513×256 | 8–12 GB | yes |
| reward 5M (not used in v1) | <4 GB | yes |
| DPO 41M (not used in v1) | 12–16 GB | yes |

If you hit OOM: halve the batch, or add gradient accumulation. The scripts do not accumulate yet.

## 2. Setup

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

git clone <repo> cool-sounds   # or rsync from the dev box
cd cool-sounds
uv venv --python 3.12
uv pip install -e ".[train,dev]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
.venv/bin/python -m pytest -q
```

All 35 tests should pass. If CUDA is false, the driver or the wheel index is wrong.

On the current pod:

```bash
rsync -a --exclude '/.venv/' --exclude '/data/' --exclude '/out/' \
  -e "ssh -p 10165" ./ root@203.57.40.89:/workspace/cool-sounds/
ssh -p 10165 root@203.57.40.89
cd /workspace/cool-sounds
python3 -m venv --system-site-packages /workspace/venv
/workspace/venv/bin/pip install -q -e . --no-deps
/workspace/venv/bin/pip install -q numpy pytest safetensors scipy
/workspace/venv/bin/python -m pytest -q
```

## 3. Data

### Scores (planner)

```bash
.venv/bin/python scripts/gen_scores.py --n 1000000 --out data/scores.jsonl --no-music
```

- ~9 GB JSONL, ~22 min on 16 cores, single-threaded.
- 1M pairs is enough for the first planner. Scale to 5M only if the loss plateaus early.
- `--no-music` is the v1 scope: SFX + texture. Drop it only when music comes back in v2.
- The planner does **not** need audio. Do not render in its dataloader.

### Detail pairs

```bash
.venv/bin/python scripts/build_pairs.py \
  --input /data/cc0 \
  --out data/pairs \
  --manifest data/pairs/manifest.jsonl \
  --workers 16
```

- Input: any folder of WAV/FLAC/OGG/MP3. ffmpeg converts to 48 kHz mono.
- 20k clips at 16 workers: roughly 1–2 hours.
- Output: paired `*_real.wav` / `*_dsp.wav` plus a manifest with a 32-d conditioning vector and a `fit_score`.
- **Filter before training.** The fit is heuristic. Keep the top 30% by spectral fit. A clean 6k beats a noisy 20k.

```bash
.venv/bin/python scripts/filter_pairs.py \
  --manifest data/pairs/manifest.jsonl \
  --out data/pairs/manifest_filtered.jsonl \
  --keep 0.3
```

### Reward manifest

```bash
.venv/bin/python scripts/build_reward.py --pairs data/pairs/manifest.jsonl --out data/reward/manifest.jsonl
```

Two rows per pair: DSP (real=0) and recording (real=1).

### Preferences

Generated after the planner and reward model exist. Sample N scores per prompt, render, rank with the reward model, keep the best/worst pair.

```bash
.venv/bin/python scripts/build_preferences.py \
  --planner runs/planner_s/last.pt \
  --reward runs/reward_s/last.pt \
  --prompts data/scores.jsonl \
  --out data/preferences.jsonl \
  --n 1000 --samples 4
```

100k pairs is the target. This step is slow; run it on the CPU box while the GPU trains the next stage.

## 4. Training

Run stages in order. Each writes `runs/<name>/log.csv` and `last.pt`.

### Detail sprint (week 2–4, go/no-go)

```bash
.venv/bin/python scripts/train_detail.py --config configs/detail_s.json
```

- 5.8M params, base 48.
- 20k pairs, batch 4, ~20 epochs.
- ~20 min/epoch on one RTX 4000 Ada.
- **Gate:** 5-listener A/B, DSP+detail vs DSP-only. Kill if the delta is under 15 points.

### Detail v2

```bash
.venv/bin/python scripts/train_detail.py --config configs/detail_m.json
```

- 23M params, base 96.
- 100k pairs, batch 4, 10 epochs.
- ~3 hours/epoch on one card.
- Resume: `--resume runs/detail_m/last.pt`.

### Spec planner (v1)

```bash
/workspace/venv/bin/python scripts/gen_scores.py --n 100000 --out data/scores_100k.jsonl --seed 100 --no-music
/workspace/venv/bin/python scripts/gen_scores.py --n 5000 --out data/heldout.jsonl --seed 999 --no-music
/workspace/venv/bin/python scripts/train_spec.py --config configs/spec_v2.json
```

- 4M params, 6 layers, d=256, ctx 1024.
- The target is a `Spec`, not a `Score`: duration, texture, mood, and up to 8 timed events (`t`, `dur`, `gain`, `pan`, `seed`).
- Numbers are decimal digit tokens, not 1024-bin values. A bin makes the model learn arithmetic; digits make it a copy.
- 100k pairs, batch 32, 4k steps. ~8 minutes on the 4090.
- **Metric:** held-out token CE and free-running exact match. Teacher-forced CE alone is not enough.
- Resume: `--resume runs/spec_v2/last.pt`.

Evaluate:

```bash
/workspace/venv/bin/python scripts/eval_spec.py --checkpoint runs/spec_v2/best.pt --data data/heldout.jsonl --n 500
```

Expected: CE ~0.002, 96–97% exact, zero decode failures. The remaining failures are single-digit copy errors.

Smoke:

```bash
/workspace/venv/bin/python scripts/smoke_spec.py --checkpoint runs/spec_v2/best.pt \
  --prompt '12.00-second tense cue, sci fi bed, with whoosh @2.00 0.80s -8dB C #a3, impact @3.20 0.70s -6dB C #1f'
```

The planner is a parser over a deterministic grammar. It is not a language model, and free-form captions outside the grammar are out of distribution. The structured Spec API is the reliable path for agents.

### Detail degradation harness (go/no-go)

```bash
/workspace/venv/bin/python scripts/build_degrade_pairs.py --esc50 /workspace/data/ESC-50-master
/workspace/venv/bin/python scripts/eval_detail.py --data data/degrade/eval.jsonl
/workspace/venv/bin/python scripts/train_detail.py --config configs/detail_harness.json
/workspace/venv/bin/python scripts/eval_detail.py --data data/degrade/eval.jsonl --checkpoint runs/detail_harness/last.pt
```

- ESC-50 is CC BY-NC. Use it for the harness only; discard the weights and never warm-start the CC0 model from them.
- The degradation is time-domain resynthesis (smoothed mel envelope as shaped noise, tracked sinusoids, lowpass, RMS match), then STFT.
- Reuse `PairDataset` with `normalize_stft: true`. The shared per-clip scale is required; unnormalized STFTs have gradients in the hundreds.
- `adv_weight: 0` for the first run. Adversarial training hides whether reconstruction works.
- Identity baseline on fold 5: MR-STFT 0.229, onset F1 0.749, high-band ratio 0.376.
- **Gate:** held-out MR-STFT at least 25% below identity, onset F1 ≥ 0.95, high-band ratio > 1.0.
- **Transfer gate:** repeat with 10 classes held out entirely. If the gain only survives in-class, the model is memorizing timbre.
- **Result (2026-11-16): no-go.** Best regression run: **16.1%** MR-STFT improvement, onset F1 **0.740**, high-band ratio **0.172**. The relative-loss variant: **2.2%**, **0.737**, **0.013**. The degradation destroys the high-frequency information; a regression model either hallucinates energy or collapses to the conditional mean. Do not collect the 20k CC0 set until the objective changes to a generative one (noise input, adversarial or flow matching).

### Reward model

```bash
.venv/bin/python scripts/train_reward.py --config configs/reward_s.json
```

- 5M params.
- 200k rows, batch 8, 3 epochs.
- ~1 hour/epoch.

### DPO

```bash
.venv/bin/python scripts/train_dpo.py --config configs/dpo_s.json
```

- Loads `runs/planner_s/last.pt`.
- 100k preference pairs, batch 4, 1 epoch.
- ~2–3 hours.
- Resume: `--resume runs/dpo_s/last.pt`.

## 5. Monitoring

```bash
tail -f runs/planner_s/log.csv
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 5
```

Healthy signs:

- Spec planner loss falls from ~8.5 to <0.1; validation tracks training instead of rising.
- detail `rec` falls from the identity baseline; with `adv_weight=0`, `d` stays 0.
- reward `match` falls below 1.0; realness accuracy moves off 0.5.

If the planner loss stays at 8.5, the labels are wrong or the learning rate is zero. If validation rises while training falls, check the causal shift in `evaluate()` and `eval_planner.py`; an unshifted metric measures a different task. If detail `rec` does not move after 200 steps, check `normalize_stft`, the residual scale, and whether the degraded/real pair is aligned.

## 6. Resume and checkpoints

- Planner, reward, DPO: `--resume runs/<name>/last.pt`.
- Detail: `--resume runs/<name>/last.pt` restores generator, discriminator, and both optimizers.
- Checkpoints every 500 steps by default. Set `save_every` in the config.

## 7. Multi-GPU

The scripts are single-GPU. To use four cards:

- run four independent stages (detail, planner, reward, eval) on four cards, or
- wrap the planner with `torchrun --nproc_per_node=4` and add DDP. The planner is the only stage that benefits; detail and reward are small.

For the first sprint, one card per stage is simpler and nearly as fast.

## 8. What to run first

1. `gen_scores.py --n 100000 --no-music`
2. `train_spec.py --config configs/spec_v2.json`
3. `eval_spec.py --checkpoint runs/spec_v2/best.pt`
4. `build_degrade_pairs.py`
5. `eval_detail.py` (identity baseline)
6. `train_detail.py --config configs/detail_harness.json`
7. `eval_detail.py --checkpoint runs/detail_harness/last.pt`

The planner is done; the detail model is the bet.
