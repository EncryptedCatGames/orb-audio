#!/usr/bin/env bash
# Publish ORB to GitHub and Hugging Face.
#
# DRY RUN BY DEFAULT: without --yes this only prints what it would do and
# checks that the tools and logins are in place. Nothing is created, pushed,
# tagged or uploaded unless you pass --yes.
#
#   scripts/release_publish.sh                     # dry run, all steps
#   scripts/release_publish.sh --yes               # really publish, all steps
#   scripts/release_publish.sh --yes --only hf     # just the Hugging Face uploads
#
# Options:
#   --yes                 actually run the commands
#   --only STEPS          comma list of: github,pages,release,hf-planner,hf-dataset,hf-detail,hf-space
#                         (also: hf = all four hf-* steps). Default: all, in that order.
#   --skip-check          do not run scripts/release_check.sh first
#   --skip-detail         never publish the CC BY-NC detail model
#   --private             create the GitHub repo and HF repos as private (useful for a rehearsal)
#   --notes-file PATH     release notes for `gh release create`
#                         (default: the CHANGELOG section for this version)
#
# Environment overrides:
#   GH_REPO   (stdevMac/orb-audio)      HF_USER (marcosmaceo)
#   HF_PLANNER_REPO (orb-planner)  HF_DETAIL_REPO (orb-detail)
#   HF_DATASET_REPO (orb-planner-data)  HF_SPACE_REPO (orb)
#   PY        python with huggingface_hub installed (default: .venv-torch/bin/python, then python3)
#
# Order matters: the Space Dockerfile clones the GitHub tag and downloads the
# planner from the Hub, so github -> release -> hf-planner must precede hf-space.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

YES=0; SKIP_CHECK=0; SKIP_DETAIL=0; PRIVATE=0; ONLY=""; NOTES_FILE=""
while (( $# )); do
  case "$1" in
    --yes) YES=1 ;;
    --skip-check) SKIP_CHECK=1 ;;
    --skip-detail) SKIP_DETAIL=1 ;;
    --private) PRIVATE=1 ;;
    --only) ONLY="${2:?--only needs a value}"; shift ;;
    --notes-file) NOTES_FILE="${2:?--notes-file needs a path}"; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

GH_REPO="${GH_REPO:-stdevMac/orb-audio}"
HF_USER="${HF_USER:-marcosmaceo}"
HF_PLANNER="$HF_USER/${HF_PLANNER_REPO:-orb-planner}"
HF_DETAIL="$HF_USER/${HF_DETAIL_REPO:-orb-detail}"
HF_DATASET="$HF_USER/${HF_DATASET_REPO:-orb-planner-data}"
HF_SPACE="$HF_USER/${HF_SPACE_REPO:-orb}"
GH_OWNER="${GH_REPO%%/*}"; GH_NAME="${GH_REPO##*/}"
PAGES_URL="https://$(echo "$GH_OWNER" | tr '[:upper:]' '[:lower:]').github.io/$GH_NAME/"
if [[ -z "${PY:-}" ]]; then
  if [[ -x .venv-torch/bin/python ]]; then PY=.venv-torch/bin/python; else PY=python3; fi
fi

VERSION="$("$PY" -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')"
TAG="v$VERSION"
VISIBILITY=public; [[ $PRIVATE -eq 1 ]] && VISIBILITY=private

if [[ -t 1 ]]; then B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; N=$'\e[0m'; else B='' D='' G='' R='' Y='' N=''; fi
say()  { echo "${B}$*${N}"; }
note() { echo "  ${D}$*${N}"; }
die()  { echo "${R}error:${N} $*" >&2; exit 1; }

# run CMD...: print it; execute only with --yes.
run() {
  printf '  %s$%s' "$G" "$N"; printf ' %q' "$@"; echo
  if [[ $YES -eq 1 ]]; then "$@"; fi
}

ALL_STEPS="github pages release hf-planner hf-dataset hf-detail hf-space"
if [[ -z "$ONLY" ]]; then
  STEPS="$ALL_STEPS"
else
  STEPS=""
  for s in $(echo "$ONLY" | tr ',' ' '); do
    case "$s" in
      hf) STEPS="$STEPS hf-planner hf-dataset hf-detail hf-space" ;;
      github|pages|release|hf-planner|hf-dataset|hf-detail|hf-space) STEPS="$STEPS $s" ;;
      *) die "unknown step '$s' (valid: $ALL_STEPS, hf)" ;;
    esac
  done
fi
want() { [[ " $STEPS " == *" $1 "* ]]; }

if [[ $YES -eq 1 ]]; then
  say "ORB $TAG: PUBLISHING for real (steps: $STEPS)"
else
  say "ORB $TAG: DRY RUN (nothing will be changed). Re-run with --yes to publish."
  note "steps: $STEPS"
fi
note "GitHub: $GH_REPO ($VISIBILITY)   Pages: $PAGES_URL"
note "HF: $HF_PLANNER, $HF_DATASET, $HF_DETAIL, spaces/$HF_SPACE"
echo

# ---------------------------------------------------------------------------
say "0. Preflight"
if [[ $SKIP_CHECK -eq 0 ]]; then
  if [[ $YES -eq 1 ]]; then
    scripts/release_check.sh || die "release_check.sh failed; fix it before publishing"
  else
    note "would run scripts/release_check.sh (run it yourself, or pass --skip-check)"
  fi
fi

missing=()
command -v git >/dev/null || missing+=(git)
git lfs version >/dev/null 2>&1 || missing+=("git-lfs (brew install git-lfs)")
if want github || want pages || want release; then
  command -v gh >/dev/null || missing+=("gh (brew install gh)")
fi
if want hf-planner || want hf-dataset || want hf-detail || want hf-space; then
  "$PY" -c "import huggingface_hub" 2>/dev/null || missing+=("huggingface_hub for $PY (pip install -U 'huggingface_hub[cli]')")
fi
if (( ${#missing[@]} )); then
  for m in "${missing[@]}"; do echo "  ${Y}missing:${N} $m"; done
  [[ $YES -eq 1 ]] && die "install the missing tools first"
fi

if want github || want pages || want release; then
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    note "gh logged in as $(gh api user -q .login 2>/dev/null || echo '?')"
  else
    echo "  ${Y}gh is not logged in${N} (gh auth login)"; [[ $YES -eq 1 ]] && die "gh auth login first"
  fi
fi
if want hf-planner || want hf-dataset || want hf-detail || want hf-space; then
  if who="$("$PY" -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null)"; then
    note "Hugging Face logged in as $who"
    [[ "$who" == "$HF_USER" ]] || echo "  ${Y}note:${N} logged in as '$who' but publishing under '$HF_USER' (fine if you can write to that org)"
  else
    echo "  ${Y}not logged in to Hugging Face${N} (hf auth login, or export HF_TOKEN)"; [[ $YES -eq 1 ]] && die "log in to Hugging Face first"
  fi
fi

if want github || want release; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "  ${Y}working tree is not clean${N}; commit the release changes first:"
    git status --short | head -20 | sed 's/^/    /'
    [[ $YES -eq 1 ]] && die "commit or stash changes before publishing"
  fi
  [[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || { echo "  ${Y}not on main${N}"; [[ $YES -eq 1 ]] && die "check out main"; }
fi
echo

# ---------------------------------------------------------------------------
if want github; then
  say "1. GitHub repository: $GH_REPO"
  run git lfs install --local
  if gh repo view "$GH_REPO" >/dev/null 2>&1; then
    note "repo exists"
  else
    run gh repo create "$GH_REPO" "--$VISIBILITY" \
      --description "Sound as code: a 15M-parameter, from-scratch caption-to-audio engine for agents. Caption -> JSON Spec -> 48 kHz DSP." \
      --homepage "$PAGES_URL"
  fi
  if git remote get-url origin >/dev/null 2>&1; then
    note "remote origin = $(git remote get-url origin)"
  else
    run git remote add origin "https://github.com/$GH_REPO.git"
  fi
  run git push -u origin main
  run git lfs push origin main
  run gh repo edit "$GH_REPO" --enable-issues --enable-wiki=false \
    --add-topic audio --add-topic sound-effects --add-topic text-to-audio --add-topic procedural-audio \
    --add-topic dsp --add-topic agents --add-topic mcp --add-topic sound-design --add-topic motion-design \
    --add-topic pytorch --add-topic numpy
  echo
fi

# ---------------------------------------------------------------------------
if want pages; then
  say "2. GitHub Pages (Actions build) -> $PAGES_URL"
  if gh api "repos/$GH_REPO/pages" >/dev/null 2>&1; then
    run gh api -X PUT "repos/$GH_REPO/pages" -f build_type=workflow
  else
    run gh api -X POST "repos/$GH_REPO/pages" -f build_type=workflow
  fi
  if [[ -f .github/workflows/pages.yml ]]; then
    run gh workflow run pages.yml --repo "$GH_REPO" --ref main
    note "watch it: gh run watch --repo $GH_REPO \$(gh run list --repo $GH_REPO --workflow pages.yml -L1 --json databaseId -q '.[0].databaseId')"
  else
    echo "  ${Y}.github/workflows/pages.yml not found${N}: Pages is enabled but nothing will deploy it"
  fi
  echo
fi

# ---------------------------------------------------------------------------
if want release; then
  say "3. Tag $TAG and GitHub Release"
  if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    note "tag $TAG exists locally"
  else
    run git tag -a "$TAG" -m "ORB $VERSION"
  fi
  run git push origin "$TAG"

  notes="$NOTES_FILE"
  if [[ -z "$notes" ]]; then
    notes="$(mktemp "${TMPDIR:-/tmp}/orb-notes.XXXXXX")"
    # The CHANGELOG section for this version, without its heading.
    awk -v v="$VERSION" '
      $0 ~ "^## \\[" v "\\]" {on=1; next}
      on && /^## \[/ {exit}
      on && /^\[[^]]+\]: http/ {exit}
      on {print}' CHANGELOG.md > "$notes"
    {
      echo
      echo "**Demo:** $PAGES_URL · **Space:** https://huggingface.co/spaces/$HF_SPACE · **Model:** https://huggingface.co/$HF_PLANNER · **Dataset:** https://huggingface.co/datasets/$HF_DATASET"
    } >> "$notes"
    [[ -s "$notes" ]] || die "could not extract [$VERSION] from CHANGELOG.md"
    note "release notes: CHANGELOG [$VERSION] section ($(wc -l < "$notes" | tr -d ' ') lines) -> $notes"
    note "(for the long-form notes use --notes-file docs/release-notes-$TAG.md)"
  fi

  for a in hf/model-planner/planner.safetensors hf/model-planner/config.json; do
    [[ -f "$a" ]] || die "release asset missing: $a"
  done
  # "path#label" sets the display label shown on the release page.
  assets=(hf/model-planner/planner.safetensors "hf/model-planner/config.json#planner config.json")
  [[ -f site/assets/orb-demo.mp4 ]] && assets+=("site/assets/orb-demo.mp4#ORB demo video (mp4)")
  if gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
    note "release $TAG exists; uploading assets with --clobber"
    run gh release upload "$TAG" --repo "$GH_REPO" --clobber "${assets[@]}"
  else
    run gh release create "$TAG" --repo "$GH_REPO" --verify-tag \
      --title "ORB $VERSION: sound as code" --notes-file "$notes" --latest \
      "${assets[@]}"
  fi
  echo
fi

# ---------------------------------------------------------------------------
# hf_upload REPO_ID TYPE FOLDER MESSAGE [space_sdk]
hf_upload() {
  local repo="$1" type="$2" folder="$3" msg="$4" sdk="${5:-}"
  [[ -d "$folder" ]] || die "$folder missing"
  local private_py=False; [[ $PRIVATE -eq 1 ]] && private_py=True
  printf '  %s$%s %s\n' "$G" "$N" "create_repo($repo, repo_type=$type${sdk:+, space_sdk=$sdk}, exist_ok=True) + upload_folder($folder)"
  note "CLI equivalent: hf upload $repo $folder . --repo-type $type --commit-message \"$msg\""
  [[ $YES -eq 1 ]] || return 0
  "$PY" - "$repo" "$type" "$folder" "$msg" "$sdk" "$private_py" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, rtype, folder, msg, sdk, private = sys.argv[1:7]
api = HfApi()
kw = {"space_sdk": sdk} if rtype == "space" and sdk else {}
url = api.create_repo(repo, repo_type=rtype, exist_ok=True, private=(private == "True"), **kw)
print("   repo:", url)
info = api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=folder, commit_message=msg,
                         ignore_patterns=[".git/*", ".DS_Store", "**/.DS_Store", "__pycache__/*"])
print("   commit:", getattr(info, "commit_url", info))
EOF
}

if want hf-planner; then
  say "4. HF model: $HF_PLANNER (Apache-2.0)"
  hf_upload "$HF_PLANNER" model hf/model-planner "ORB planner $VERSION"
  echo
fi
if want hf-dataset; then
  say "5. HF dataset: $HF_DATASET (Apache-2.0)"
  hf_upload "$HF_DATASET" dataset hf/dataset-scores "ORB planner data ($VERSION)"
  echo
fi
if want hf-detail; then
  if [[ $SKIP_DETAIL -eq 1 ]]; then
    say "6. HF model: $HF_DETAIL (skipped: --skip-detail)"
  else
    say "6. HF model: $HF_DETAIL (CC BY-NC-4.0, research only)"
    grep -q "^license: cc-by-nc-4.0" hf/model-detail/README.md || die "detail card must declare cc-by-nc-4.0"
    hf_upload "$HF_DETAIL" model hf/model-detail "ORB detail $VERSION (CC BY-NC-4.0)"
  fi
  echo
fi
if want hf-space; then
  say "7. HF Space: spaces/$HF_SPACE (docker)"
  if [[ $YES -eq 1 ]] && command -v gh >/dev/null; then
    gh api "repos/$GH_REPO/git/ref/tags/$TAG" >/dev/null 2>&1 || die "the Space clones $GH_REPO@$TAG, which is not on GitHub yet; run the github and release steps first"
  else
    note "requires $GH_REPO@$TAG on GitHub and $HF_PLANNER on the Hub (the Dockerfile clones / downloads them)"
  fi
  hf_upload "$HF_SPACE" space hf/space "ORB demo $VERSION" docker
  note "build log: https://huggingface.co/spaces/$HF_SPACE?logs=build"
  echo
fi

# ---------------------------------------------------------------------------
say "Links"
echo "  GitHub   https://github.com/$GH_REPO"
echo "  Release  https://github.com/$GH_REPO/releases/tag/$TAG"
echo "  Pages    $PAGES_URL"
echo "  Model    https://huggingface.co/$HF_PLANNER"
echo "  Dataset  https://huggingface.co/datasets/$HF_DATASET"
echo "  Detail   https://huggingface.co/$HF_DETAIL"
echo "  Space    https://huggingface.co/spaces/$HF_SPACE"
if [[ $YES -eq 0 ]]; then echo; say "Dry run complete. Nothing was changed."; fi
