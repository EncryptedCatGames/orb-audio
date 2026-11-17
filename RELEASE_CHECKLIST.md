# Release checklist — ORB v0.1.0

Everything is prepared locally, and **nothing has been pushed**. Two scripts do the
mechanical work:

| Script | What it does | Changes anything? |
|---|---|---|
| `scripts/release_check.sh` | every local preflight check; exits non-zero on failure | no |
| `scripts/release_publish.sh` | GitHub repo + Pages + tag + Release, then the four Hugging Face repos | **only with `--yes`**; dry run by default |

Identities: GitHub **`stdevMac/orb-audio`**, Hugging Face **`marcosmaceo`**, PyPI
distribution **`orb-audio`** (`orb` is taken on PyPI; the import name is still `orb`).
Pages URL: <https://stdevmac.github.io/orb-audio/>.

> `hf/` is gitignored in the main repo. It is a staging area where each subfolder becomes
> its own Hugging Face repository. Do not `git clean -fdx` before publishing, and do not
> `git add hf/` from the main repo.

## 0. One-time tooling

```bash
brew install gh git-lfs            # gh is already logged in as stdevMac
git lfs install
pip install -U "huggingface_hub[cli]"   # into the python you pass as PY (default .venv-torch)
hf auth login                      # or: export HF_TOKEN=hf_...
```

## 1. Finish the pieces built in parallel

- [ ] **Demo video**: `site/assets/orb-demo.mp4` (< 25 MB) and the poster
      `site/assets/orb-demo-poster.jpg`. The README embeds the poster and links the video.
- [ ] **Project page**: `site/` plus `.github/workflows/pages.yml` (Actions deploy of `site/`).
      Open `site/index.html` locally and check that audio plays.
- [ ] **MCP server**: `src/orb/mcp_server.py` defines `main()`, documented in `docs/agents.md`.
      Smoke test: `pip install -e ".[mcp,train]" && orb-mcp`, then connect a client
      (for example `claude mcp add orb -- orb-mcp`) and render one cue.

## 2. Preflight (must pass)

```bash
scripts/release_check.sh          # or: make release-check
```

It checks:

- `pyproject.toml`: name `orb-audio`, the `train`/`mcp`/`dev` extras, and that the `orb-generate` and `orb-mcp` entry points resolve to real functions
- version `0.1.0` agrees across pyproject, CITATION.cff, CHANGELOG, the Space `ORB_REF` and `docs/release-notes-v0.1.0.md`
- Git LFS: rules for `*.pt` / `*.safetensors`, `weights/planner.pt` stored as an LFS object, no non-LFS file over 25 MB, `hf/` and the CC BY-NC detail weights untracked
- `weights/planner.pt` SHA-256 and size match `weights/manifest.json`
- `hf/model-planner/planner.safetensors` is tensor-identical to `planner.pt`, the config matches, and it decodes a caption exactly
- the detail safetensors loads, and the dataset has 100,000 valid rows
- the `hf/*/README.md` front-matter is valid Hub YAML (license, tags, pipeline_tag, library_name, datasets, language, configs, Space sdk/app_port)
- the Space Dockerfile clones `v0.1.0`, downloads the planner and config from the Hub, exposes 7860 and runs as uid 1000
- no placeholder markers, stale `orb[...]` install names or broken relative links in the README and docs
- `pytest -q`, all examples, the `orb-generate` CLI, and the DSP path in a venv **without torch**

`--fast` skips the tests and examples.

## 3. Commit

```bash
git add -A && git status        # check that hf/, out/, data/ and artifacts/ are NOT listed
git commit -m "Release 0.1.0"
```

## 4. Rehearse, then publish

```bash
scripts/release_publish.sh                          # dry run: prints every command, checks logins
scripts/release_publish.sh --yes                    # publishes everything, in order
```

What `--yes` does, in order. Each step can be re-run alone with `--only <step>`:

| Step | Action |
|---|---|
| `github` | `gh repo create stdevMac/orb-audio --public`, add `origin`, push `main` (LFS objects included), set topics |
| `pages` | `gh api -X POST repos/stdevMac/orb-audio/pages -f build_type=workflow`, then trigger `pages.yml` |
| `release` | tag `v0.1.0`, push it, then `gh release create` with the CHANGELOG section as notes; attaches `planner.safetensors`, its config and the demo MP4 |
| `hf-planner` | create and upload `marcosmaceo/orb-planner` (Apache-2.0) |
| `hf-dataset` | create and upload `datasets/marcosmaceo/orb-planner-data` (Apache-2.0) |
| `hf-detail` | create and upload `marcosmaceo/orb-detail` (**CC BY-NC-4.0**; skip it with `--skip-detail`) |
| `hf-space` | create and upload `spaces/marcosmaceo/orb` (Docker). It refuses to run until the GitHub tag exists |

For the long-form GitHub Release body, use
`--only release --notes-file docs/release-notes-v0.1.0.md`, or edit the release afterwards.
`--private` publishes everything as private, which is useful for a full rehearsal.

The Space build clones `stdevMac/orb-audio@v0.1.0` and downloads the planner from the Hub,
so `github`, `release` and `hf-planner` must be done before `hf-space` (the default order
handles this). The first build takes about 5 minutes.

## 5. After publishing

- [ ] CI is green on `main` and on the tag (the README badge)
- [ ] <https://stdevmac.github.io/orb-audio/> loads, the video plays, and the audio examples play
- [ ] README badges and links resolve (GitHub, Pages, HF model, dataset and Space)
- [ ] The poster image in the README renders and links to the Pages site
- [ ] `pip install "orb-audio[train] @ git+https://github.com/stdevMac/orb-audio@v0.1.0"` works in a clean venv
- [ ] `pip install "orb-audio[mcp,train] @ git+https://github.com/stdevMac/orb-audio@v0.1.0" && orb-mcp` starts
- [ ] The Space builds, loads and generates a cue
- [ ] The HF planner card shows Apache-2.0, `text-to-audio`, the dataset link and the eval result
- [ ] The detail model card shows CC BY-NC-4.0 and opens with the non-commercial warning
- [ ] The dataset viewer renders both configs (`grammar`, `freetext`)
- [ ] `CITATION.cff` validates (`pipx run cffconvert --validate`); GitHub shows "Cite this repository"
- [ ] Turn on branch protection for `main` (require the `tests` checks)
- [ ] Post the announcement (the short form is at the end of `docs/release-notes-v0.1.0.md`)
- [ ] Optional: publish to PyPI as `orb-audio` (`python -m build && twine upload dist/*`). Weights are not in the sdist or wheel; the planner comes from LFS or the Hub.

## Not included on purpose

- `artifacts/` (2.2 GB of training output), which is reproducible from `scripts/`
- `artifacts/degrade_audio/` (1.8 GB, derived from ESC-50, CC BY-NC)
- `weights/detail.pt` (CC BY-NC, separate Hub repo)
- `web/out/`, `out/` (generated cache)
