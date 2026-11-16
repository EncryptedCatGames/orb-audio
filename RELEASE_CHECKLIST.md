# Release checklist

Everything is prepared locally. **Nothing has been pushed.** Work top to bottom;
each step depends on the one above it.

> `hf/` is gitignored in the main repo: it is a staging area where each
> subfolder becomes its own repository. Do not `git clean -fdx` before pushing
> it, and do not `git add hf/` from the main repo.

## 0. Fill in the placeholders

Author fields are already set to Marcos Maceo. Two placeholders remain:

```bash
grep -rn "YOUR_ORG\|YOUR_HF_USER" \
  --include="*.md" --include="*.toml" --include="*.cff" --include="*.json" --include="Dockerfile" .
```

| Placeholder | Replace with |
|---|---|
| `YOUR_ORG` | GitHub org or user |
| `YOUR_HF_USER` | Hugging Face user or org |

Also decide the PyPI distribution name. `pyproject.toml` currently uses `orb`;
`orb-audio` is safer if `orb` is taken.

## 1. Verify locally

```bash
python -m pytest -q                       # 36 passed
python examples/04_no_torch.py            # must not import torch
python examples/01_generate.py
python examples/02_edit_spec.py
python examples/03_variants.py --n 3
python web/server.py --port 8765          # open http://127.0.0.1:8765
```

Check `weights/planner.pt` matches `weights/manifest.json`:

```bash
shasum -a 256 weights/planner.pt
```

## 2. Push the GitHub repo

```bash
git init
git lfs install
git add .
git commit -m "ORB 0.1.0: from-scratch caption-to-audio engine"
git branch -M main
git remote add origin https://github.com/YOUR_ORG/orb-audio.git
git push -u origin main
git tag -a v0.1.0 -m "ORB 0.1.0" && git push origin v0.1.0
```

Confirm LFS is active (`git lfs ls-files` should list `weights/planner.pt`).
Turn on branch protection and enable the `tests` workflow.

## 3. Publish the planner model

```bash
pip install -U huggingface_hub
huggingface-cli login

cd hf/model-planner
huggingface-cli repo create orb-planner --type model
git init && git lfs install
git remote add origin https://huggingface.co/YOUR_HF_USER/orb-planner
git add . && git commit -m "ORB planner 0.1.0"
git push origin main
```

Check the card renders and the license badge reads Apache-2.0.

## 4. Publish the dataset

```bash
cd hf/dataset-scores
huggingface-cli repo create orb-scores-100k --type dataset
git init && git lfs install
git remote add origin https://huggingface.co/datasets/YOUR_HF_USER/orb-scores-100k
git add . && git commit -m "ORB scores 100k"
git push origin main
```

## 5. Publish the detail model (non-commercial)

```bash
cd hf/model-detail
huggingface-cli repo create orb-detail --type model
git init && git lfs install
git remote add origin https://huggingface.co/YOUR_HF_USER/orb-detail
git add . && git commit -m "ORB detail 0.1.0 (CC BY-NC-4.0)"
git push origin main
```

The card must keep the non-commercial warning at the top.

## 6. Publish the Space

```bash
cd hf/space
huggingface-cli repo create orb --type space --space_sdk docker
git init && git lfs install
git remote add origin https://huggingface.co/spaces/YOUR_HF_USER/orb
git add . && git commit -m "ORB demo"
git push origin main
```

The Dockerfile clones the GitHub repo at `v0.1.0` and downloads the planner
from the Hub, so steps 2 and 3 must be done first. Build takes a few minutes.

## 7. After publishing

- [ ] README links resolve (GitHub, HF model, HF dataset, Space)
- [ ] `pip install "orb[train] @ git+https://github.com/YOUR_ORG/orb-audio"` works in a clean venv
- [ ] The Space loads and generates a cue
- [ ] The detail model card shows CC BY-NC-4.0
- [ ] `CITATION.cff` validates (`cffconvert --validate`)
- [ ] Add the repo to the HF model card's `repository` field if desired

## Not included on purpose

- `artifacts/` (2.2 GB of training output) — reproducible from `scripts/`
- `artifacts/degrade_audio/` (1.8 GB, ESC-50 derived, CC BY-NC)
- `weights/detail.pt` (CC BY-NC, separate repo)
- `web/out/` (generated cache)
