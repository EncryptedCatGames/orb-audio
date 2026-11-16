# Weights

## What is here

| File | License | Size | Notes |
|---|---|---|---|
| `planner.pt` | **Apache-2.0** | 20 MB | caption → Spec, 5.08M params |
| `detail.pt` | **CC BY-NC-4.0** | 41 MB | optional texture stage, **not distributed** |
| `manifest.json` | — | — | SHA-256 and parameter counts |

`planner.pt` is trained from scratch on procedurally generated data. No
pretrained weights, no external audio, no human labels.

## Why the detail model is not here

The detail model was trained on [ESC-50](https://github.com/karolpiczak/ESC-50),
which is **CC BY-NC-4.0**. A model trained on that data cannot be redistributed
under Apache-2.0, so it is published separately under the same non-commercial
terms:

- Hugging Face: `YOUR_HF_USER/orb-detail` (CC BY-NC-4.0, research only)

The DSP path is a complete cue without it. If you have the file, drop it at
`weights/detail.pt` and the web demo and `--detail` flag will pick it up.

## Formats

The `.pt` files are plain `torch.save` dicts:

- planner: `{"model": state_dict, "config": {...}, "step": int}`
- detail: `{"generator": state_dict, "config": {...}, "step": int}`

Safetensors copies for the Hub are generated with:

```bash
python scripts/to_safetensors.py
```

`orb.api.load_planner` and `load_detail` accept either format.

## Regenerating

```bash
python scripts/package_artifacts.py --out weights   # from artifacts/final
python scripts/to_safetensors.py                    # Hub copies
```

Verify the hashes against `manifest.json`:

```bash
shasum -a 256 weights/planner.pt
```
