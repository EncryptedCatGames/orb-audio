# ORB

**A tiny, from-scratch audio engine for agents.** Caption in, 48 kHz stereo out — exact to the frame, editable as JSON, no samples, no pretrained weights.

```
caption ──► planner (5.1M) ──► Spec ──► DSP render ──► [detail (10.3M)] ──► WAV + audit
```

- **15.35M parameters, 61.5 MB**, 48 kHz stereo
- planner: **96.6% exact** on 500 held-out captions, 0 decode failures
- the DSP path needs **only NumPy** — no torch, no weights, no GPU
- timing is a property of the score, so it is exact
- every output is an editable `Spec`, not a blob
- Apache-2.0

## The idea

A sample is a **file**. This is a **function**.

That one change is the whole pitch. Because the sound is generated from a typed
score, you can move an impact from 5.00 s to 5.40 s by editing a number, ask
for 500 non-repeating variants from one spec, diff a sound design in git, and
ship the whole engine in 61 MB with no sample clearance.

It is not trying to beat a recorded sound library on timbre. It is trying to be
the programmable layer underneath one: the transitions, UI ticks, ambience beds
and variations that make up most of a project's audio.

## Quickstart

```bash
git clone https://github.com/stdevMac/orb-audio
cd orb-audio
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# numpy only, no weights, no torch
python examples/04_no_torch.py

# the full pipeline (needs torch)
pip install -e ".[train]"
python examples/01_generate.py
```

```python
from orb.api import generate, save_wav

out = generate("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a")
out["audio"]   # (n, 2) float32 at 48 kHz
out["spec"]    # {'duration': 6.0, 'texture': 'wind', 'mood': 'bright', 'events': [...]}
out["audit"]   # peak, rms_dbfs, dc, click_score, passes, ...
save_wav(out["audio"], "whoosh.wav")
```

Or from the shell:

```bash
orb-generate "8.00-second tense cue, drone bed, with riser @1.00 2.40s -16dB C #51" --out riser.wav
```

Local web demo (pre-rendered examples plus live generation):

```bash
python scripts/build_web_examples.py   # pre-render the examples, once (~2 min)
python web/server.py --port 8765       # http://127.0.0.1:8765
```

## Examples

| File | Needs torch? | Shows |
|---|---|---|
| [`examples/01_generate.py`](examples/01_generate.py) | yes | caption → WAV, full pipeline |
| [`examples/02_edit_spec.py`](examples/02_edit_spec.py) | no | change one number, re-render |
| [`examples/03_variants.py`](examples/03_variants.py) | no | N non-repeating takes from one Spec |
| [`examples/04_no_torch.py`](examples/04_no_torch.py) | no | the shippable DSP engine |

A 12-second trailer cue:

```
12.00-second intense cue, drone bed, with riser @1.00 2.50s -18dB C #55, whoosh @4.20 1.20s -10dB L3 #77, impact @5.00 0.90s -4dB C #66, sub drop @5.00 1.60s -8dB C #88, noise sweep @6.50 2.00s -16dB R2 #97, impact @9.50 0.90s -4dB C #99, bell @9.50 2.00s -12dB R1 #3d
```

## The Spec

The planner emits a small typed object, and the renderer is a pure function of it.

```json
{
  "duration": 12.0,
  "texture": "drone",
  "mood": "intense",
  "events": [
    {"role": "riser",    "t": 1.0, "dur": 2.5, "gain_db": -18.0, "pan": 0.0,   "seed": 85},
    {"role": "impact",   "t": 5.0, "dur": 0.9, "gain_db": -4.0,  "pan": 0.0,   "seed": 102},
    {"role": "sub_drop", "t": 5.0, "dur": 1.6, "gain_db": -8.0,  "pan": 0.0,   "seed": 136}
  ]
}
```

- **10 roles**: whoosh, impact, ui_pop, riser, swell, click, bell, sub_drop, glitch, noise_sweep
- **8 textures**: room_cold, wind, city_hum, vinyl, drone, rain, fire, sci_fi
- **10 moods**: calm, warm, playful, hopeful, bright, tense, dark, eerie, neutral, intense
- up to **8 events**, duration 2–30 s

The caption grammar is documented in [`examples/README.md`](examples/README.md).

## Results

**Planner** — 5.08M parameters, 6 layers, d=256, byte-level tokenizer, constrained decoding.

| Metric | Value |
|---|---|
| Held-out exact match | **483/500 (96.6%)** |
| Decode failures | 0 |
| Token CE | 0.0029 |
| Duration / texture / mood | 100% |
| Times / durs / gains / pans / seeds | 99.6–99.9% |

The remaining 3.4% are single-digit copy errors on dense captions (`-6dB` → `-16dB`). When exactness matters, pass a Spec dict instead of a caption.

**DSP renderer** — 50 held-out cues: 100% audit pass, max peak 0.983, max |DC| 1.5e-4, no clipping, −77.6 dB before the first event.

**Detail model** (optional, separate download) — 10.27M complex-STFT U-Net. It does **not** meet its restoration gate (10% MR-STFT improvement); it is a subtle texture stage. The DSP path is the shippable engine. Full numbers in [`RESULTS.md`](RESULTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md).

## Licensing

The project is split so the open-source part is unambiguous:

| Component | License | Where |
|---|---|---|
| Code | **Apache-2.0** | this repo |
| `weights/planner.pt` | **Apache-2.0** | this repo |
| Detail model | **CC BY-NC-4.0** | separate download, research only |
| 100k planner pairs | **Apache-2.0** | separate dataset |

The detail model was trained on [ESC-50](https://github.com/karolpiczak/ESC-50), which is CC BY-NC-4.0, so those weights cannot be Apache-2.0. They are not included here. See [`weights/README.md`](weights/README.md).

## What this is not

- **Not a sample library replacement.** A recorded door slam beats a synthesized one. The detail model was meant to close that gap and does not yet.
- **Not a music generator.** Music is deferred to v2.
- **Not real-time.** A 12 s cue renders in about a second on CPU with the DSP path; the detail stage takes tens of seconds.
- **Not 34 roles.** v1 ships 10 roles and 8 textures.

## Repo layout

```
src/orb/
  api.py            public API: generate, render_spec, plan
  score.py          the typed Score
  grammar.py        Spec, captions, deterministic compiler
  render/           DSP: events, textures, music, mix, detail
  model/            planner, detail U-Net, tokenizer, grammar mask
  data/             datasets
  train/            training loops and losses
  eval/             audit and metrics
examples/           four runnable examples
scripts/            data generation, training, evaluation, packaging
web/                local demo server and UI
docs/               architecture, score format, data, training, eval
weights/            planner weights (Apache-2.0)
hf/                 ready-to-publish Hugging Face model, dataset and Space
```

Further reading: [`docs/architecture.md`](docs/architecture.md) (how it works),
[`docs/score.md`](docs/score.md) (the Score schema),
[`docs/training.md`](docs/training.md) (GPU runbook),
[`docs/eval.md`](docs/eval.md) (gates), [`PLAN.md`](PLAN.md) (original plan),
[`REVIEW.md`](REVIEW.md) (pre-build red-team).

## Training

The planner trains from scratch on procedurally generated data in minutes on one GPU:

```bash
python scripts/gen_scores.py --n 100000 --out data/scores_100k.jsonl --no-music
python scripts/train_spec.py --config configs/spec_v5.json
python scripts/eval_spec.py --checkpoint runs/spec_v5/best.pt --data data/heldout.jsonl --n 500
```

See [`docs/training.md`](docs/training.md) for the GPU runbook and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a role or texture.

## Citation

```bibtex
@software{orb2026,
  title  = {ORB: a tiny, from-scratch audio engine for agents},
  author = {Marcos Maceo},
  year   = {2026},
  url    = {https://github.com/stdevMac/orb-audio},
  note   = {Version 0.1.0}
}
```

## Acknowledgements

Development was AI-assisted (Anthropic Claude), with the architecture, training
runs, evaluation and release prepared in a single assisted session. The
implementation draws on the following work:

- Ronneberger, Fischer & Brox. *U-Net.* MICCAI 2015.
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
- Wu & He. *Group Normalization.* ECCV 2018.
- Yamamoto, Song & Kim. *Parallel WaveGAN* (multi-resolution STFT loss). ICASSP 2020.
- Miyato et al. *Spectral Normalization for GANs.* ICLR 2018.
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.
- Piczak. *ESC: Dataset for Environmental Sound Classification.* ACM MM 2015.

## License

Code and planner weights: [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
third-party components.
