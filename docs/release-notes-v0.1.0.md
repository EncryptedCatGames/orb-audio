# ORB v0.1.0: sound as code

ORB is a tiny audio engine, trained from scratch, that turns a one-line caption into 48 kHz
stereo sound. In between, the sound exists as a typed, editable JSON **Spec**. The open
(Apache-2.0) engine is a 5.1M-parameter planner plus a NumPy DSP renderer: no samples, no
pretrained weights, and the renderer runs on NumPy alone. An optional 9.1M detail model is
published separately under CC BY-NC-4.0 and is not part of the Apache-2.0 release.

```
caption ──► planner (5.1M) ──► Spec (JSON) ──► DSP render ──► [detail (9.1M)] ──► WAV + audit
```

**[Watch the demo](https://stdevmac.github.io/orb-audio/)** ·
**[Try it live](https://huggingface.co/spaces/marcosmaceo/orb)** ·
**[Planner on the Hub](https://huggingface.co/marcosmaceo/orb-planner)** ·
**[Dataset](https://huggingface.co/datasets/marcosmaceo/orb-planner-data)**

## Why it is different

A sample is a file. ORB treats sound as a function. Every cue is a small document with
roles, start times, durations, gains, pans and seeds, and the renderer is a pure function of
that document. That gives you:

- **Frame-exact timing.** An impact at 5.00 s lands at 5.00 s. Sound can be generated *from*
  an animation's keyframes instead of being nudged to fit afterwards.
- **Edits, not re-rolls.** "Later and quieter" is a two-number diff that you can review in git.
- **Endless variants.** Change the seeds and you get non-repeating takes of the same design.
- **A checkable output.** Every render returns an audit: peak, DC, click score, and silence
  before the first event.
- **Agent-native.** An agent can read, write and reason about JSON far better than a waveform.

## Highlights

- **Planner v8 (5.10M params)**: 500/500 (100%) exact Spec match on held-out grammar
  captions (2000/2000 on a larger set), with 0 decode failures thanks to grammar-constrained
  decoding. It also reads plain words ("rain on a tin roof with distant thunder"): 58.4% of
  our 209 free-text prompts and 70.0% of a 40-prompt blind set are planned fully right.
- **DSP renderer (NumPy only)**: 47 roles (10 core + 37 physically modelled natural, foley
  and everyday sounds), 13 textures (8 noise beds + 5 scene beds) and 10 moods at 48 kHz
  stereo. On 50 held-out cues: 100% audit pass, max peak 0.983, no clipping.
- **`orb-mcp`**: a Model Context Protocol server, so Claude Code, Claude Desktop, Cursor and
  other MCP clients can plan, render and edit cues as tools. See
  [`docs/agents.md`](https://github.com/stdevMac/orb-audio/blob/main/docs/agents.md).
- **`orb-generate`**: caption to WAV from the shell.
- **A Hugging Face Space** that runs the planner + DSP path on a free CPU.
- **A project page** on GitHub Pages with a demo video of ORB cues cut to motion design.
- **The training set**: the exact 300k rows v8 was trained on (150k grammar pairs, 150k
  plain-text pairs) plus the held-out sets, Apache-2.0.

## Install

ORB is not on PyPI yet; install the package `orb-audio` (import name `orb`) from GitHub:

```bash
# DSP engine only (numpy)
pip install "orb-audio @ git+https://github.com/stdevMac/orb-audio@v0.1.0"

# caption -> audio: the planner, downloaded from the Hub on first use
pip install "orb-audio[train,hub] @ git+https://github.com/stdevMac/orb-audio@v0.1.0"

# MCP server for agents (add train for caption mode)
pip install "orb-audio[mcp] @ git+https://github.com/stdevMac/orb-audio@v0.1.0"
```

```python
from orb.api import generate, save_wav

out = generate("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a")
save_wav(out["audio"], "whoosh.wav")
print(out["spec"], out["audit"]["passes"])
```

`generate()` looks for the planner in this order: `planner_path=`, `$ORB_PLANNER`,
`./weights/planner.pt`, the clone's `weights/` folder (`.pt`, or `.safetensors` +
`config.json`), then a cached download of
[marcosmaceo/orb-planner](https://huggingface.co/marcosmaceo/orb-planner) if
`huggingface_hub` is installed. Otherwise it raises an error that says what to install or
pass. The CC BY-NC detail model is never downloaded automatically.

For sounds with no background bed (UI pops and clicks), render a Spec dict with
`"texture": "none"`, or write `no bed` in a grammar caption (read exactly, without the
planner). `"bed_db"` sets the bed level relative to the events (default −40 dB).

## Artifacts

| Artifact | Where | License |
|---|---|---|
| Code, DSP engine, MCP server | this repository | Apache-2.0 |
| Planner weights | `planner.pt` in the repo (Git LFS); `planner.safetensors` + `config.json` as release assets and on the [Hub](https://huggingface.co/marcosmaceo/orb-planner) | Apache-2.0 |
| Planner training data (300k rows) | [Hub dataset](https://huggingface.co/datasets/marcosmaceo/orb-planner-data) | Apache-2.0 |
| Demo video (`orb-demo.mp4`) | release asset, [project page](https://stdevmac.github.io/orb-audio/) | Apache-2.0, except the third-party photos and generated images credited in [NOTICE](https://github.com/stdevMac/orb-audio/blob/main/NOTICE) |
| Detail model (9.06M) | [Hub](https://huggingface.co/marcosmaceo/orb-detail) only | **CC BY-NC-4.0**, research only |

SHA-256 of `weights/planner.pt`: `4327864eff1a26f5b79497e61a9dbc76bd97527189ed81463f6c90a3f35b235c`

## Known limitations

These are stated plainly because they matter:

- **The detail model does not meet its own gate.** Detail v6 passes the no-noise artifact gate,
  but it improves MR-STFT by only 0.84% against a 10% target, so treat it as a subtle texture
  stage rather than a restoration model. It was
  trained on ESC-50 (CC BY-NC), so it ships separately under CC BY-NC-4.0. The DSP path is
  the shippable engine.
- **Plain words are best-effort.** Grammar captions are read exactly; plain words only map
  onto the 47 roles and 13 textures. For a noun ORB has no sound for, the planner may invent
  the nearest role, and counts ("three knocks") and long multi-clause prompts are weak. Pass
  a grammar caption or a Spec dict when exactness matters.
- **Not a sample library.** A recorded door slam still beats a synthesized one.
- **No music yet** (v2), and no voices or vocal animals.

## Thanks

Development was AI-assisted (Anthropic Claude). See the README for the papers this work
builds on, and [`CHANGELOG.md`](https://github.com/stdevMac/orb-audio/blob/main/CHANGELOG.md)
for the full list of changes.

---

## Announcement post (short form)

> **ORB v0.1.0 is out: sound as code.**
>
> ORB is a small audio engine trained from scratch: a 5.1M-parameter planner and a NumPy
> DSP renderer. You give it a caption and it
> gives you 48 kHz stereo, but in between the sound is a JSON Spec you can edit. Move an
> impact by 0.4 s by changing one number. Get 500 non-repeating variants from one design.
> Diff your sound design in git.
>
> - 100% exact caption → Spec on held-out grammar captions, 0 invalid outputs, and plain words too
> - the renderer is pure NumPy: no GPU, no weights, no samples
> - an MCP server, so your agent can design sound alongside your motion graphics
> - Apache-2.0 code and planner weights, plus a 100k training set (the optional
>   detail model is separate and non-commercial)
>
> Demo video: https://stdevmac.github.io/orb-audio/
> Try it: https://huggingface.co/spaces/marcosmaceo/orb
> Code: https://github.com/stdevMac/orb-audio
>
> It does not replace recorded samples, and the optional neural "detail" stage misses its
> own quality gate. The release notes say so plainly. What it does well is timing,
> structure and control, and that turns out to be most of what an agent needs.
