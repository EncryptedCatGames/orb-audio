# Results — 2026-11-16

## Realism pass: 30 natural and foley roles rebuilt (2026-11-17)

Every natural, foley and everyday role that read wrong was rebuilt from the physics of the real sound (what vibrates, what excites it, how events are spread in time, what the space adds) and checked against real ESC-50 recordings. Thunder is now a near-bolt crack, rumble from the rest of the channel arriving over 1–2 s, and a diffuse tail where lows ring longer than highs (t60 1.8 s below 350 Hz, 1.1 s above), instead of a stylised whoosh. The seed variant contract (`VARIANTS`, `seed % N`) is unchanged, so the planner needs no retraining; every role still renders a 4 s event well under 0.5 s.

Judges, seeds 0–5 per role. AST: mean probability of the role's AudioSet label(s). CLAP rank: rank of the correct role among the 37 role descriptions, zero-shot (1 is best). ESC similarity: CLAP cosine to the centroid of the matching ESC-50 class (real clips score 0.74–0.93 against their own centroid).

| role | AST before → after | CLAP rank before → after | ESC similarity before → after (ceiling) | AST's usual label now |
|---|---|---|---|---|
| thunder | 0.019 → 0.105 | 4.8 → 2.0 | 0.45 → 0.62 | Explosion |
| footsteps | 0.001 → 0.001 | 14.5 → 11.2 | 0.12 → 0.29 | Sound effect |
| water_drop | 0.003 → 0.007 | 2.0 → 2.0 | 0.52 → 0.52 | Plop |
| bird_chirp | 0.218 → 0.492 | 3.3 → 2.8 | 0.50 → 0.55 | Chirp, tweet |
| crickets | 0.501 → 0.662 | 4.3 → 2.5 | 0.49 → 0.47 | Insect |
| wave_crash | 0.008 → 0.013 | 1.5 → 1.7 | 0.51 → 0.52 | Sound effect |
| wind_gust | 0.021 → 0.063 | 2.0 → 2.0 | 0.51 → 0.57 | Waves, surf |
| splash | 0.001 → 0.002 | 1.7 → 1.5 | 0.47 → 0.51 | Slosh |
| frog | 0.041 → 0.166 | 5.5 → 1.3 | 0.15 → 0.23 | Sound effect |
| owl | 0.001 → 0.260 | 14.8 → 3.2 | – | Owl |
| glass_break | 0.117 → 0.140 | 1.0 → 1.0 | 0.72 → 0.69 | Breaking |
| clock_tick | 0.007 → 0.023 | 15.5 → 7.2 | 0.13 → 0.29 | Wood block |
| phone_ring | 0.562 → 0.647 | 5.7 → 3.8 | – | Telephone bell ringing |
| keyboard_typing | 0.022 → 0.017 | 13.3 → 7.5 | 0.10 → 0.22 | Sound effect |
| mouse_click | 0.001 → 0.001 | 8.8 → 4.3 | 0.27 → 0.43 | Finger snapping |
| camera_shutter | 0.002 → 0.068 | 22.0 → 15.8 | – | Sound effect |
| paper_rustle | 0.048 → 0.038 | 6.5 → 4.0 | – | Sound effect |
| cork_pop | 0.061 → 0.082 | 1.0 → 1.0 | – | Knock |
| coin_drop | 0.004 → 0.040 | 2.0 → 2.0 | – | Chink, clink |
| heartbeat | 0.057 → 0.649 | 1.0 → 1.0 | – | Heart sounds, heartbeat |
| applause | 0.004 → 0.279 | 8.3 → 7.2 | 0.19 → 0.34 | Firecracker |
| pouring_water | 0.054 → 0.079 | 2.5 → 2.7 | 0.29 → 0.31 | Plop |
| can_opening | 0.000 → 0.000 | 2.0 → 4.2 | 0.44 → 0.41 | Sound effect |
| clock_alarm | 0.143 → 0.166 | 1.2 → 1.0 | 0.44 → 0.56 | Telephone bell ringing |
| punch | 0.040 → 0.119 | 1.3 → 1.0 | – | Whoosh, swoosh, swish |
| gunshot | 0.007 → 0.027 | 3.2 → 1.5 | – | Explosion |
| siren | 0.388 → 0.732 | 1.5 → 1.2 | 0.65 → 0.69 | Siren |
| engine | 0.009 → 0.038 | 13.0 → 6.5 | 0.15 → 0.19 | Sound effect |
| church_bell | 0.019 → 0.377 | 5.3 → 1.5 | 0.15 → 0.32 | Church bell |
| fireworks | 0.139 → 0.339 | 6.5 → 2.7 | 0.41 → 0.64 | Fireworks |

Unchanged (not reworked): door_slam, knock, explosion, metal_clang, wood_creak, sword_clash, car_horn

Still weak: wind (AST hears waves), engine, clock tick (heard as a wood block), can opening (CLAP rank worse, 2.0 → 4.2), and the frog's bullfrog and peeper variants. Tests: 406 passed, 1 skipped (`test_natural`, `test_foley`, `test_render_baseline`).

## Planner v8: plain-text prompts (2026-11-17)

Planner v8 (`configs/planner_v8.json`) reads plain words as well as the caption grammar. Same architecture as v5–v7 (d256, 6 layers, 4 heads, ctx 1024). The vocabulary grows by one token, 394 → 395: `NOBED` means "no bed" (texture `none`), so a prompt like "a notification ding" can plan a dry cue. The planner now has 5,096,192 params, +256 over v7, and the total for both models is 14,160,579.

Training warm-started from `weights/planner_v7.pt` on 300k rows: the 150k v7 grammar pairs (`data/scores_150k_v7.jsonl`) and 150k synthetic plain-text pairs (`data/freetext_150k.jsonl`, written by `scripts/gen_freetext.py` from `orb.freetext`). The run was 40k steps at batch 32 on one H100 80GB (RunPod pod `xpkopdcapuw7sx`, about 35 minutes, ≈ $2.6; the pod was deleted afterwards). The best checkpoint is step 39000 (val CE 0.036 on `data/heldout_v8.jsonl`, which mixes grammar and free-text rows). The previous planner is kept as `weights/planner_v7.pt`.

**Grammar captions (no regression)**

| held-out set | v8 exact | decode failures |
|---|---|---|
| `heldout_v7.jsonl` (500) | **500/500 (100%)**, every field 100% | 0 |
| `heldout_v7_2k.jsonl` (2,000) | **2000/2000 (100%)**, every field 100% | 0 |
| v6 held-out set (500) | **500/500 (100%)**, every field 100% | 0 |

**Plain-text prompts.** `scripts/eval_freetext.py` decodes each prompt to a Spec. A prompt passes when every check its row states holds: required roles, no extra roles, texture, mood, intensity and duration. No eval prompt appears in the training data, including after normalisation. There are two hand-written sets:

- `data/freetext_eval.jsonl`: 209 prompts, written by us alongside the templates, sha256 `7950a21c4316dbb0370535a93192558fd247e8e34b3c887c9396b6d5f48151b8`. Frozen before training; the templates were not tuned against its failures. The set is split dev/test by md5 of the prompt. The dev half was looked at while writing `orb.freetext`, so some phrasing may have leaked into the templates.
- `data/freetext_blind.jsonl`: 40 prompts written by someone else, never seen while writing templates, sha256 `3cd69f39050b083469b7b833a8c0ac2ddea187c6019938f8952cafc463b04c83`.

The baseline is v7 behind the site's keyword helper (`scripts/helper_translate.mjs`, rules, not a model), which turns plain words into a grammar caption.

| set | v7 + keyword helper | **v8, raw text** | v8 + keyword helper |
|---|---|---|---|
| our set, all 209 | 45.0% | **58.4%** | 45.0% |
| our set, test half (107) | 46.7% | **58.9%** | 46.7% |
| blind set (40) | **77.5%** | 70.0% | **77.5%** |

v8 gains a lot more on our own set (+13.4 points) than on the blind set (−7.5 points against the helper), so the free-text skill generalises less than our set suggests. On the blind set, the raw planner gets 90% of roles, 100% of moods and intensities and 97.5% of textures. Most of its failures are extra roles: for a noun it has no role for, it invents the nearest one instead of leaving it out (75% pass the "no extras" check, against 95% for the helper). On our set the helper loses on intensity ("soft", "huge"), modifiers, typos and synonyms, where the planner is clearly better. By tag, raw v8: plain 74%, word order 77%, typos 72%, synonyms 70%, scenes 59%, UI/game 58%, sequences 30%, counts 20%, long mixed prompts 15%.

Pooled over all 249 prompts, raw v8 is best (60.2%). On the blind set alone, "helper if it can read the prompt, else raw" is best (80%). The site therefore runs plain words through the planner first and shows the helper's caption beside it as a one-click second opinion.

**Web export.** The site planner is int8 per row (`site/assets/planner/planner.q8.bin`, 5,141,014 bytes, same size as v7). On the int8 export, `scripts/check_web_planner.py --n 40 --free 20` gives numpy vs torch token parity on 78/78 prompts and caption → Spec exact on 40/40 for numpy and torch.

**Everything else unchanged**: `tests/test_render_baseline.py` is bit-identical (151 checks). The noise suite (all targets OK), the detail gate (PASS on all four checks, same numbers as v7: G1 2.69 dB, G2 1.000×, G3 +0.063 dB, G4 +0.84%) and the HF/buzz/hiss tests in `tests/test_natural.py` pass. The renderer and the detail model are untouched.

Known limitations: the planner invents roles for nouns it has no sound for; counts ("three knocks") and long multi-clause prompts are weak; and it still has only 47 roles and 13 textures to plan with.

## Everyday library: 31 roles, 5 scene beds, planner v7 (2026-11-16)

31 physically modelled roles in `src/orb/render/natural.py` and 5 scene beds in `src/orb/render/ambience.py`, appended after the foley six and the eight noise textures so no existing seed or index moves. All of them render through DSP only; the detail model is never applied to them.

- **Existing sounds unchanged**: `tests/test_render_baseline.py` (151 checks, 150 pinned renders) is bit-identical.
- **Noise suite** (`scripts/eval_noise.py`): all targets OK, unchanged from v6.
- **Detail gate** (`scripts/eval_detail_artifacts.py --checkpoint weights/detail.pt`): **PASS** on all four checks (G1 prominence 2.69 dB, output periodicity 0.026, residual 0.268; G2 1.000×; G3 +0.063 dB; G4 +0.84%). The v6 G1 near-miss (`footsteps_gravel_wind`, residual 0.304) is gone because foley roles now bypass the detail model; the same weights, `weights/detail.pt`, were not retrained.
- **HF, buzz and hiss tests** (`tests/test_natural.py`, seeds 0/1/2/5 for every role): energy within 2 kHz of Nyquist < 0.1% for all 31 roles; the 17 non-tonal roles keep residual periodicity above 6 kHz ≤ 0.5 and no line more than 15 dB above its neighbours. Tonal roles (birds, crickets, frogs, owl, bells, horns, siren, alarm, phone, coins, glass, ringing metal, swords) are periodic by nature and are exempt from the buzz check only. Each scene bed keeps < 0.1% of its energy above 14 kHz and < 10% above 6 kHz, and ends on an exact zero.
- **Routing**: `render_stems` splits a `dsp_only` stem (the 37 physical-model roles and the 5 scene beds); with the detail model on, a pure physical-model cue comes back exactly as the DSP render, and a cue with only the original roles takes exactly the old path.

### ESC-50 comparison

40 clips per class, median over clips; ORB median over seeds 0–5. Columns: spectral centroid (Hz), energy above 6 kHz (%), energy below 250 Hz (%), crest factor (dB), fraction of frames within 30 dB of the peak. ESC-50 clips run to 22 kHz; ORB is band-limited to 12 kHz on purpose, so its >6 kHz share is lower across the board. ESC clips are 5 s field recordings, usually with a continuous background, which is why their active fraction is near 1.

| role | ESC class | ORB centroid / >6k / <250 / crest / active | ESC median | verdict |
|---|---|---|---|---|
| bird_chirp | chirping_birds | 3027 / 0.2 / 0.0 / 17.8 / 0.29 | 3162 / 2.7 / 1.5 / 17.9 / 1.00 | close |
| crickets | crickets | 4565 / 0.3 / 0.0 / 17.6 / 0.50 | 3772 / 6.8 / 1.7 / 15.0 / 1.00 | brighter carrier, close |
| wave_crash | sea_waves | 1612 / 5.5 / 21.2 / 23.8 / 0.60 | 1357 / 2.1 / 12.0 / 18.0 / 1.00 | close, more dynamic |
| wind_gust | wind | 640 / 0.1 / 60.6 / 17.1 / 0.71 | 262 / 0.0 / 60.3 / 13.1 / 1.00 | brighter whistle, same low share |
| splash | water_drops | 1546 / 1.0 / 2.6 / 29.0 / 0.09 | 1249 / 2.7 / 5.0 / 28.4 / 0.19 | close |
| frog | frog | 940 / 0.0 / 0.0 / 22.2 / 0.24 | 1924 / 0.0 / 0.6 / 19.3 / 0.88 | darker (a bullfrog-ish croak; ESC is mostly tree frogs) |
| glass_break | glass_breaking | 4140 / 5.8 / 0.4 / 26.0 / 0.15 | 5233 / 34.2 / 1.4 / 24.7 / 0.16 | right shape, less top end (12 kHz cap) |
| clock_tick | clock_tick | 1571 / 0.3 / 3.3 / 30.0 / 0.11 | 1828 / 3.3 / 5.0 / 25.5 / 0.46 | close |
| keyboard_typing | keyboard_typing | 2053 / 0.1 / 8.7 / 29.2 / 0.25 | 2251 / 5.7 / 17.2 / 30.1 / 0.61 | close |
| mouse_click | mouse_click | 2315 / 0.4 / 3.5 / 39.9 / 0.02 | 2816 / 21.1 / 19.7 / 31.2 / 0.31 | close centroid, cleaner (no handling noise) |
| wood_creak | door_wood_creaks | 647 / 0.0 / 7.1 / 21.8 / 0.40 | 1884 / 2.1 / 5.9 / 20.2 / 0.67 | darker (a floorboard more than a door hinge) |
| applause | clapping | 2735 / 5.1 / 0.0 / 21.9 / 0.67 | 1699 / 1.9 / 1.3 / 20.4 / 0.97 | brighter (a crowd of small claps vs ESC's single person) |
| pouring_water | pouring_water | 1261 / 1.5 / 0.5 / 16.5 / 0.59 | 2358 / 9.2 / 0.8 / 26.3 / 0.64 | darker, smoother |
| can_opening | can_opening | 4474 / 28.7 / 0.5 / 32.1 / 0.11 | 5940 / 43.1 / 1.0 / 25.7 / 0.10 | close |
| clock_alarm | clock_alarm | 2917 / 3.1 / 0.0 / 11.7 / 0.34 | 4242 / 13.2 / 0.0 / 12.9 / 0.64 | lower beep pitch, same envelope |
| car_horn | car_horn | 1187 / 0.5 / 0.0 / 19.3 / 0.15 | 1205 / 0.3 / 14.0 / 16.1 / 0.90 | close |
| siren | siren | 1339 / 0.5 / 0.0 / 10.6 / 0.92 | 1083 / 0.0 / 0.3 / 11.8 / 1.00 | close |
| engine | engine | 266 / 0.1 / 74.5 / 16.9 / 0.78 | 359 / 0.4 / 76.2 / 13.2 / 1.00 | close |
| church_bell | church_bells | 466 / 0.0 / 13.8 / 17.9 / 0.78 | 919 / 0.0 / 3.5 / 14.2 / 1.00 | darker (one big tower bell vs ESC's peals) |
| fireworks | fireworks | 1211 / 0.1 / 12.5 / 21.7 / 0.47 | 628 / 0.3 / 23.7 / 29.1 / 0.33 | brighter (more crackle, less distant boom) |
| bed: ocean | sea_waves | 1027 / 2.0 / 22.9 / 17.3 / 1.00 | 1357 / 2.1 / 12.0 / 18.0 / 1.00 | close |
| bed: stream | pouring_water | 1413 / 1.2 / 0.3 / 15.7 / 0.99 | 2358 / 9.2 / 0.8 / 26.3 / 0.64 | darker (no stream class in ESC) |
| bed: forest | chirping_birds | 3187 / 0.4 / 0.4 / 17.2 / 0.76 | 3162 / 2.7 / 1.5 / 17.9 / 1.00 | close |
| bed: night | crickets | 4421 / 0.3 / 0.5 / 18.2 / 0.99 | 3772 / 6.8 / 1.7 / 15.0 / 1.00 | close |
| bed: traffic | engine | 251 / 0.1 / 75.5 / 14.7 / 1.00 | 359 / 0.4 / 76.2 / 13.2 / 1.00 | close |

No ESC-50 class (centroid in Hz): owl 397, phone_ring 641, camera_shutter 990, paper_rustle 3271, cork_pop 899, coin_drop 4035, metal_clang 1014, heartbeat 86, punch 187, sword_clash 1709, gunshot 267.

ESC-50 coverage: 20 classes have a dedicated new role (table above; wind and water drops were also covered before, by the `wind` bed and the v6 `water_drop` role). Rain and crackling fire were already beds, and thunderstorm, footsteps and door knock are v6 foley roles, so 25 of the 50 classes are covered.

Out of scope, by design: voices and vocal animals (dog, rooster, pig, cow, cat, hen, sheep, crow, crying baby, sneezing, laughing, coughing, breathing, snoring). A listener names these by vocal timbre, which a small physical model does not reach; they need recordings or a learned model.

Not attempted in this pass: drinking/sipping, brushing teeth, toilet flush, vacuum cleaner, washing machine, hand saw, chainsaw, helicopter, airplane, train. These are long broadband machine or body noises; without picture they mostly read as filtered noise, so a DSP model would add little over the existing `city_hum`/`wind` beds. They are the next candidates if recordings or a learned model are added.

### Planner v7

Planner v7 (`configs/planner_v7.json`): same architecture as v5/v6 (d256, 6 layers, 4 heads, ctx 1024), vocabulary 358 → 394 (31 role + 5 texture tokens), 5,095,936 params. 150k pairs (`data/scores_150k_v7.jsonl`, 25% scenes across the 12 scene kinds, no music), 30k steps at batch 32 on one H100 80GB (RunPod pod `s461plpdtwn0bo`, about 26 minutes of pod time, ≈ $1.9); best checkpoint step 28500, val CE < 0.0001.

| held-out set | exact | decode failures |
|---|---|---|
| `heldout_v7.jsonl` (500) | **500/500 (100%)** | 0 |
| …captions with a v6/v7 role | 448/448 | 0 |
| …captions with an everyday (v7) role | 390/390 | 0 |
| …old roles only | 52/52 | 0 |
| …captions not in the training set | 465/465 | 0 |
| `heldout_v7_2k.jsonl` (2,000) | **2000/2000 (100%)** | 0 |
| v6 held-out set (500, old captions) | 500/500 (100%; v6 planner 98.8%) | 0 |

Per role and per scene bed, every one of the 47 roles and 5 new textures is 100% on both v7 sets (23–71 captions per role on the 500 set, 104–289 on the 2,000 set). Full breakdown: `eval_subsets.py` output. All fields (duration, texture, mood, event count, roles, times, durs, gains, pans, seeds) are 100%. The previous planner is kept as `weights/planner_v6.pt`.

## Natural and foley roles + planner v6 (2026-11-16)

Six physically modelled roles in `src/orb/render/foley.py`: thunder, door_slam, knock, footsteps, water_drop, explosion, appended after the original ten so no existing seed moves.

- **Existing sounds unchanged**: `tests/test_render_baseline.py` pins 150 renders recorded before the change (17 site library cues, 6 playground presets, the noise suite, every original role and texture, single events). All 150 are bit-identical (SHA-256 of the float32 samples).
- **Noise suite** (`scripts/eval_noise.py`, 38 cues): all pass, targets OK. New roles in isolation: nothing before the event (−inf), floor −inf (explosion −142 dBFS), energy above 8 kHz 0.00–3.44% (door latch), above 12 kHz ≤ 0.38%. Scenes without a bed: floor −inf / −106 / −125 dBFS.
- **Detail gate** (`scripts/eval_detail_artifacts.py`, now 32 cues, `weights/detail.pt` not retrained): G2, G3, G4 pass; G1 fails on one cue, `footsteps_gravel_wind`, residual periodicity **0.304** (limit 0.30; the old worst was 0.252). The output periodicity on that cue is 0.003 and its prominence 0.15 dB; other gravel seeds give −0.07 to 0.19. Use the DSP path for foley cues until the detail model is retrained.
- **ESC-50 comparison** (40 clips per class; spectral centroid): thunder far/mid seeds 123–156 Hz and near 352 Hz vs ESC 225 Hz, crest 18–21 dB vs 18.1; explosion 576–671 Hz vs 628 Hz; footsteps 373–449 Hz (hard/stone) and 2.5 kHz (gravel) vs 714 Hz; knock 490–572 Hz vs 341 Hz; water drop 607–2568 Hz vs 1249 Hz. ESC-50 has no door-slam class.

Planner v6 (`configs/planner_v6.json`): same architecture as v5 (d256, 6 layers, 4 heads, ctx 1024), vocabulary 352 → 358, 5,086,720 params. 100k pairs (`data/scores_100k_v6.jsonl`, 20% scenes), 20k steps at batch 32 on one H100 80GB (RunPod, about 12 minutes); best checkpoint step 19500, val CE 0.0010.

| held-out set | exact | decode failures |
|---|---|---|
| `heldout_v6.jsonl` (500; 395 use a new role) | **494/500 (98.8%)** | 0 |
| …captions with a new role | 389/395 (98.5%) | 0 |
| …old roles only | 105/105 (100%) | 0 |
| …captions not in the training set | 456/462 (98.7%) | 0 |
| `heldout_v6_2k.jsonl` (2,000) | 1984/2000 (99.2%) | 0 |
| v5 held-out set (500, old roles only) | 494/500 (98.8%; v5 planner 96.6%) | 0 |

Per new role (500 set): thunder 97.5%, door_slam 96.9%, knock 97.6%, footsteps 97.7%, water_drop 96.1%, explosion 97.2% (2,000 set: 98.7–99.4%).

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
- Detail v5 (superseded by v6, below) was a **10.27M complex-STFT U-Net** with a Gaussian noise channel, a soft energy gate, a high-frequency-only residual (above 2 kHz), a learnable noise floor, and a weighted band-energy loss `[0.05, 0.05, 0.1, 0.3, 1.0, 2.0, 4.0]` over seven octave bands.
- Best checkpoint (`detail_v5` step 4000): MR-STFT improvement **1.4%**, onset F1 **0.748**, high-band ratio **0.337** (identity 0.058), noise sensitivity **0.036**.
- The 4–8 kHz band reaches **0.38** of the real energy; 8–16 kHz stays at **0.05**. The model oscillates: later checkpoints collapse back toward identity, and the band loss alone is not enough to hold the top.
- On in-domain (full-band) DSP renders v5 was deliberately conservative: **+3.3% high-band energy**, **+48 Hz centroid**, 100% audit pass, 1–5% relative waveform change at correlation 0.999.

- v5 **failed the no-noise gate** (`scripts/eval_detail_artifacts.py`): its `ConvTranspose2d` upsampling left checkerboard tonal stripes around 10–13 kHz plus a ~4 kHz line. It is kept as `weights/detail_v5.pt` for history only.

### Detail v6 (released)

- `configs/detail_v6.json`, **9,064,387 params (9.06M)**. Complex-STFT U-Net, base 64, 4 levels, bilinear resize-conv upsampling (replaces `ConvTranspose2d`), mask residual with a smoothed mask (kernel 9×3), residual tapered off between 8 and 9.5 kHz, quiet-frame gate (30 dB, 10 dB ramp), energy gate, no noise channel.
- No adversarial loss (`adv_weight` 0). Losses: band-energy 2.0, reconstruction 0.05, envelope 0.2, overshoot 1.0, idempotence 5.0.
- Same ESC-50 pairs (folds 1–4 train = 1599 pairs, fold 5 eval = 400 pairs), batch 8, 1500 steps on one NVIDIA H100 80GB (RunPod), about 2 minutes. The gate was run on checkpoints 250..1500 and all passed; the released checkpoint is **step 1250**, which had the best MR-STFT gain and the lowest periodicity and quiet-rise.

No-noise gate, step 1250, 400 eval clips:

| gate | metric | v6 | limit |
|---|---|---|---|
| G1 tonal | prominence (DSP cues) | 2.20 dB | < 6.0 |
| G1 tonal | periodicity, output | 0.006 | ≤ 0.2 |
| G1 tonal | periodicity, residual | 0.252 | ≤ 0.3 |
| G1 tonal | prominence, eval pairs (max / median) | 3.71 / 0.86 dB | < 6.0 |
| G2 HF noise | energy ratio > 10 kHz | 1.000× | ≤ 1.25 |
| G2 HF noise | energy ratio > 12 kHz | 1.000× | ≤ 1.25 |
| G3 quiet | quiet-frame rise | +0.063 dB | ≤ 0.5 |
| G4 useful | MR-STFT improvement | +0.84% | ≥ 0 |
| G4 useful | onset F1 (identity 0.749) | 0.749 | ≥ identity − 0.02 |
| | **GATE** | **PASS** | |

v6 is a more conservative, cleaner and subtler texture stage than v5 (+0.84% MR-STFT against v5's 1.4%, but no tonal artifacts).

```bash
/workspace/venv/bin/python scripts/train_detail.py --config configs/detail_v6.json
/workspace/venv/bin/python scripts/eval_detail_artifacts.py --checkpoint runs/detail_v6/step1250.pt
```

Honest reading: the harness gate (10% MR-STFT, high-band 0.7–1.3) is **not met** by v5 or v6. The detail model is an optional texture stage; the DSP path is the shippable v1. A CC0 detail dataset and a generative objective (flow matching, or a high-band discriminator with R1) are the next step.

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

The detail model was not the cause: it adds **+0.1 dB** in the silent gaps (v5; it has since been retrained as detail v6, see above, which adds +0.063 dB in quiet frames). It is now slightly out of distribution (the bed it sees is quieter and darker), but it still passes the audit and remains an optional stage.

Audit after the fix: 12/12 web examples and 13/13 demo cues pass; max peak 0.905, max |DC| 2.5e-4, 12/13 exact Spec.

## Noise pass (2026-11-16)

Goal: no audible noise in released samples. All changes are in the deterministic renderer. The planner, its tokenizer and the caption grammar are unchanged (96.6% exact still applies), and so is the detail model (it was since retrained as detail v6; see "Detail v6" above).

What changed:

1. **Optional bed.** A Spec dict can set `texture: "none"` (no bed; at least one event required) and `bed_db` (−60 to −6 dB, default −40 since the second round below, was −30; not allowed with `"none"`). Both are validated in `orb.grammar.Spec`, round-trip through `orb.api`, and are exposed in the MCP / tool schema. A grammar caption can say `no bed` (second round); `bed_db` is Spec only.
2. **Bed edges.** Each texture has its DC removed per texture, then a 30 ms raised-cosine fade-in and an 80 ms fade-out. Grains have a 1.5 ms attack, so they do not click. The `sci_fi` curve is darker above 2 kHz (−38 dB at 8 kHz, was flat to 12 kHz).
3. **Band-limited attacks.** The noise bursts in impact, ui_pop, click and the music perc are `shaped_burst`: zero-phase Butterworth band-pass, energy-matched to the raw burst, and they draw the same rng samples, so determinism is kept. Impact and glitch are low-passed at 12 kHz after saturation.
4. **Darker reverb.** The IR was raw white noise, with 67% of its energy above 8 kHz. It is now band-limited to 80 Hz–7 kHz, and the band above 3 kHz decays 2.5× faster (air absorption). That leaves 1.3% of the energy above 8 kHz and 0.07% above 12 kHz. In the mixer the tail now rings out past the event end instead of being cut (the cut was itself a click), and every cue gets a 10 ms end fade on the event and music stems.

`scripts/eval_noise.py --baseline-src <old src>` renders a fixed suite with both renderers and prints the table below. In BEFORE, the `none` cues are rendered with `room_cold` at −30 dB, which is what a user got before. Abbreviations:

- `pre`: level before the first event, in dBFS
- `floor`: 10th-percentile level of 20 ms frames, in dBFS
- `>8k` and `>12k`: share of the cue's energy above that frequency
- `flat`: spectral flatness of the quietest 10% of frames

| cue | BEFORE pre / floor | AFTER pre / floor | >12k before → after | flat before → after | timing |
|---|---|---|---|---|---|
| role/click (none) | −54 / −54 | −inf / −inf | 25.4% → 5.0% | 0.007 → – | same |
| role/impact (none) | −39 / −39 | −inf / −inf | 0.001% → 0.000% | 0.007 → – | same |
| ui/pop_none | −49 / −50 | −inf / −inf | 0.002% → 0.000% | 0.007 → – | same |
| ui/pack_none | −45 / −45 | −inf / −inf | 0.097% → 0.016% | 0.007 → – | same |
| ui/click_glitch_none | −45 / −46 | −inf / −inf | 1.10% → 0.26% | 0.008 → – | same |
| readme/trailer_none | −42 / −44 | −inf / −164 | 0.001% → 0.000% | 0.007 → – | same |
| readme/trailer_drone (bed) | −44 / −44 | −44 / −45 | 0.001% → 0.001% | 0.005 → 0.005 | same |
| ui/pop_vinyl_bed (bed) | −49 / −50 | −49 / −50 | 0.004% → 0.002% | 0.016 → 0.011 | same |
| texture/vinyl alone | – / −23 | – / −23 | 0.055% → 0.039% | 0.015 → 0.011 | – |
| texture/rain alone | – / −24 | – / −23 | 0.127% → 0.078% | 0.034 → 0.018 | – |
| texture/sci_fi alone | – / −23 | – / −24 | 0.115% → 0.023% | 0.017 → 0.005 | – |

All 24 cues pass the audit both before and after. Each event is also rendered alone, and two timing points are compared: its start (first sample above −100 dBFS) and its −20 dB onset.

- **Start:** identical for every transient role. The slow-ramp riser and swell start 6–8 samples (≤0.17 ms) later, because their first samples sit just under −100 dBFS.
- **Onset:** within 0.25 ms for every transient role.

The onset is not gated for riser, swell and noise_sweep. Their −20 dB point sits on a ramp that lasts seconds, so a 1 dB level change moves it by tens of ms. Their envelopes match the old ones to within about 1 dB along the ramp.

Bed spectra (each bed placed at −50 dBFS):

- Content above 10 kHz sits at about −74 to −81 dBFS.
- Spectral flatness above 10 kHz is 0.15–0.18. Before, vinyl was 0.34 and rain was 0.41, because grain clicks sprayed energy up to Nyquist.
- The spectrograms show the old full-band vertical streaks (vinyl grains, white attack bursts at impacts and clicks) are gone.

Targets met:

- `texture: "none"`: gaps and lead-in at or below −90 dBFS (measured −inf).
- No flat hiss above 10 kHz (`tests/test_render.py::test_beds_are_ambience_not_hiss`).
- All audits pass.
- Event timing preserved.

What this pass does not do (the second round below addresses the first point):

- The default bed level was still −30 dB under the foreground. Planner-made cues therefore always carried a quiet, non-white bed (about −45 to −50 dBFS on sparse cues).
- The renderer output changed. The detail model's training pairs (`build_pairs`, `smoke_train`) are rendered by this code, so any retrain should regenerate them. (The released detail v6 is trained on the ESC-50 degradation pairs from `build_degrade_pairs.py`, which do not go through the renderer and are unchanged.)
- The trailer cue renders about 15% slower, because of the FFT band-limits and the longer reverb tails.

```bash
.venv/bin/python scripts/eval_noise.py                                   # AFTER only, checks targets
.venv/bin/python scripts/eval_noise.py --baseline-src /path/to/old/src   # BEFORE vs AFTER
```

### Second round (review fixes)

A listening review of the first round found four more problems. Each is fixed and checked by `scripts/eval_noise.py` and the test suite.

1. **Whoosh started with a click.** The fade-in ran before `align_peak`, which can shift a continuous clip up to 50 ms left. That discarded the fade and started the clip mid-waveform: a step from digital silence to −8..−23 dBFS on every whoosh. Continuous roles (whoosh, riser, swell, noise_sweep) are now aligned first and faded after, with a 5 ms raised-cosine fade-in and a 20–80 ms raised-cosine fade-out (a noise_sweep ends at ~13% of its peak, and the old 10 ms linear ramp from there was a small tick). Worst first 0.5 ms over 16 seeds, alone on silence: whoosh −10.3 → −44.1 dBFS, noise_sweep −58.2 → −89.8, swell −80.0 → −98.9, riser −89.9 → −107.5. Transient roles are unchanged.
2. **Every caption still carried the bed at −30 dB.** The default `bed_db` is now **−40 dB** (`orb.grammar.DEFAULT_BED_DB`). The planner has no slot for the level, so every planner cue gets the new default: before the first event, a single pop on `room_cold` goes −48 → −58 dBFS, a pop on vinyl −49 → −59, the README trailer −44 → −54. Ambience-only cues (no events) are unchanged: the bed is the sound and still takes the loudness target. `spec_from_score` reads the training scores' −30 dB (`TRAIN_BED_DB`) as the default, so held-out Specs still compare equal to planner output.
3. **Bed-less captions went to the planner.** The grammar now writes `no bed` for a Spec with `texture: "none"`, and `orb.grammar.parse_caption` reads any caption in the exact grammar. `generate()`, `orb-generate` and the `orb_generate` tool render a `no bed` caption (or one with no bed clause) straight from that Spec, without loading the planner or torch; `out["planned"]` / `source: "caption"` say so. Captions with a bed still go through the planner, unchanged.
4. **Smaller items.** Impact and glitch are low-passed at 12 kHz with an 8th-order slope (was 4th; ~48 dB down at 24 kHz instead of ~24): glitch energy above 16 kHz −64.0 → −76.6 dBFS. `orb.api.save_wav` (used by the CLI and the tools) rounds to the nearest 16-bit step instead of truncating toward zero; no dither is added, so digital silence stays exactly zero.

Timing: transient roles are unchanged to the sample. The continuous roles' first sample above −100 dBFS moves up to 2 ms later (the new 5 ms fade-in; riser +94 samples, swell +52, noise_sweep +29), and the whoosh's −20 dB onset by 2.5 ms (the old onset was the click). `eval_noise.py` gates continuous roles at 5 ms for both.

## End to end

- Caption → Spec → Score → DSP → WAV, with the planner v5.
- 500 held-out captions: **96.6% exact Spec**, 0 decode failures.
- 50 held-out cues: **94% exact Spec** (small-sample), **100% audit pass**, max peak 0.983, max |DC| 1.5e-4, no clipping.
- Event stem before the first event: **−77.6 dB** relative to the event stem's peak. This was measured on the event stem only. The mix always carried the ambience bed (about −40 to −50 dBFS on sparse cues), so the mix itself was never silent before the first event. Since the noise pass (above), a Spec with `texture: "none"` is digitally silent (−inf dBFS) before and between events.
- The DSP renders are full-band; the bed is rolled off above 8 kHz, so the mix centroid now sits around 2–5 kHz depending on the events.
- The detail stage (v5, measured before the v6 retrain) adds **+3.3% high-band energy** and **+48 Hz centroid**, keeps the audit at 100%, and preserves timing. Detail v6 leaves energy above 10 kHz at 1.000× on the no-noise gate cues.
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
