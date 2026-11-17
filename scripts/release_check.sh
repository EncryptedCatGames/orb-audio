#!/usr/bin/env bash
# Local release preflight for ORB. Read-only: it never commits, pushes or
# publishes. Exits non-zero if any check FAILs.
#
#   scripts/release_check.sh            # everything
#   scripts/release_check.sh --fast     # skip the test suite and examples
#
# Interpreters (override with env vars):
#   PY   python with torch + safetensors   (default: first of .venv-torch, .venv, python3 that has them)
#   NPY  python WITHOUT torch, for the DSP-only check (default: .venv/bin/python if torch-free; else skipped)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

FAST=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

has_torch() { "$1" -c 'import torch, safetensors' >/dev/null 2>&1; }
if [[ -z "${PY:-}" ]]; then
  for cand in .venv-torch/bin/python .venv/bin/python python3; do
    if command -v "$cand" >/dev/null 2>&1 && has_torch "$cand"; then PY="$cand"; break; fi
  done
  [[ -n "${PY:-}" ]] || { echo "no python with torch + safetensors found; set PY=..." >&2; exit 1; }
fi
if [[ -z "${NPY:-}" && -x .venv/bin/python ]] && ! has_torch .venv/bin/python; then NPY=.venv/bin/python; fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -t 1 ]]; then G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'; else G='' R='' Y='' B='' N=''; fi
PASS=0; FAILS=(); WARNS=()
ok()   { PASS=$((PASS+1)); echo "  ${G}PASS${N} $*"; }
bad()  { FAILS+=("$*");    echo "  ${R}FAIL${N} $*"; }
warn() { WARNS+=("$*");    echo "  ${Y}WARN${N} $*"; }
section() { echo; echo "${B}== $*${N}"; }

# check "<label>" <command...>: PASS if the command exits 0, else FAIL with its output.
check() {
  local label="$1"; shift
  local out
  if out="$("$@" 2>&1)"; then ok "$label"; else bad "$label"; echo "$out" | tail -20 | sed 's/^/       /'; fi
}

VERSION="$("$PY" -c 'import tomllib;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])' 2>/dev/null || true)"
[[ -n "$VERSION" ]] || { echo "cannot read version from pyproject.toml with $PY" >&2; exit 1; }
echo "${B}ORB release preflight${N}  version=$VERSION  PY=$PY  NPY=${NPY:-<none>}"

# ---------------------------------------------------------------------------
section "Metadata and version consistency"

check "pyproject: name, extras, entry points" "$PY" - <<'EOF'
import tomllib, pathlib, re
d = tomllib.load(open("pyproject.toml", "rb"))["project"]
errs = []
if d["name"] != "orb-audio": errs.append(f"name is {d['name']!r}, expected 'orb-audio' ('orb' is taken on PyPI)")
for k in ("description", "readme", "license", "authors", "urls", "classifiers", "keywords", "requires-python"):
    if not d.get(k): errs.append(f"missing {k}")
for extra in ("train", "mcp", "dev"):
    if extra not in d.get("optional-dependencies", {}): errs.append(f"missing extra [{extra}]")
scripts = d.get("scripts", {})
for name, target in (("orb-generate", "orb.cli:main"), ("orb-mcp", "orb.mcp_server:main")):
    if scripts.get(name) != target: errs.append(f"entry point {name} != {target}")
for name, target in scripts.items():
    mod, func = target.split(":")
    path = pathlib.Path("src", *mod.split(".")).with_suffix(".py")
    if not path.exists():
        errs.append(f"entry point {name}: {path} does not exist"); continue
    if not re.search(rf"^def {func}\(", path.read_text(), re.M):
        errs.append(f"entry point {name}: no top-level def {func}() in {path}")
for k, v in d.get("urls", {}).items():
    if not v.startswith("https://"): errs.append(f"url {k} is not https: {v}")
if errs: raise SystemExit("\n".join(errs))
EOF

check "version $VERSION consistent (CITATION, CHANGELOG, Space ORB_REF, release notes)" "$PY" - "$VERSION" <<'EOF'
import sys, re, pathlib
v = sys.argv[1]; errs = []
cff = pathlib.Path("CITATION.cff").read_text()
if not re.search(rf"^version:\s*\"?{re.escape(v)}\"?\s*$", cff, re.M): errs.append("CITATION.cff version")
if f"## [{v}]" not in pathlib.Path("CHANGELOG.md").read_text(): errs.append(f"CHANGELOG.md has no '## [{v}]' section")
if f"ARG ORB_REF=v{v}" not in pathlib.Path("hf/space/Dockerfile").read_text(): errs.append(f"hf/space/Dockerfile ORB_REF != v{v}")
if not pathlib.Path(f"docs/release-notes-v{v}.md").exists(): errs.append(f"docs/release-notes-v{v}.md missing")
if errs: raise SystemExit("\n".join(errs))
EOF

check "CITATION.cff is valid YAML with required fields" "$PY" - <<'EOF'
import yaml
c = yaml.safe_load(open("CITATION.cff"))
missing = [k for k in ("cff-version", "message", "title", "authors", "version", "license") if not c.get(k)]
if missing: raise SystemExit(f"missing: {missing}")
EOF

# ---------------------------------------------------------------------------
section "Git LFS and repository hygiene"

if git lfs version >/dev/null 2>&1; then ok "git-lfs installed ($(git lfs version | cut -d' ' -f1))"; else bad "git-lfs not installed"; fi

for pat in '*.pt' '*.safetensors' '*.wav' '*.flac'; do
  if grep -qE "^\\${pat} .*filter=lfs" .gitattributes; then ok ".gitattributes routes $pat through LFS"; else bad ".gitattributes has no LFS rule for $pat"; fi
done
if [[ "$(git check-attr filter -- weights/planner.pt | awk '{print $NF}')" == "lfs" ]]; then
  ok "weights/planner.pt has filter=lfs"
else
  bad "weights/planner.pt is not LFS-tracked"
fi
if grep -qx 'weights/planner.pt' <<<"$(git lfs ls-files -n 2>/dev/null)"; then ok "weights/planner.pt is stored as an LFS object"; else bad "weights/planner.pt is not in 'git lfs ls-files'"; fi

if [[ -z "$(git ls-files hf)" ]]; then ok "hf/ staging area is not tracked by git"; else bad "files under hf/ are tracked in the main repo"; fi
for f in weights/detail.pt weights/detail.safetensors; do
  if [[ -n "$(git ls-files "$f")" ]]; then bad "$f (CC BY-NC) is tracked"; fi
done
ok "CC BY-NC detail weights are not tracked"

# Every file that would be pushed (tracked + untracked-not-ignored) must be
# small unless LFS handles it. GitHub hard-limits at 100 MB, warns at 50 MB.
check "no non-LFS file over 25 MB would be pushed (video < 25 MB)" "$PY" - <<'EOF'
import subprocess, os
files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, text=True, check=True).stdout.split("\0")
files = [f for f in files if f and os.path.isfile(f)]
attrs = subprocess.run(["git", "check-attr", "-z", "filter", "--stdin"], input="\0".join(files),
                       capture_output=True, text=True, check=True).stdout.split("\0")
lfs = {attrs[i] for i in range(0, len(attrs) - 2, 3) if attrs[i + 2] == "lfs"}
errs, warns = [], []
for f in files:
    size = os.path.getsize(f)
    if f in lfs: continue
    if size > 25e6: errs.append(f"{f}: {size/1e6:.1f} MB and not in LFS")
    elif size > 10e6: warns.append(f"{f}: {size/1e6:.1f} MB")
for w in warns: print("note (10-25 MB, not LFS):", w)
if errs: raise SystemExit("\n".join(errs))
EOF

if command -v uv >/dev/null 2>&1; then
  check "sdist + wheel build (orb-audio, entry points, no weights or node_modules)" bash -c '
    set -e
    d="$(mktemp -d)"; trap "rm -rf \"$d\"" EXIT
    uv build --quiet --out-dir "$d" >/dev/null 2>&1 || { echo "uv build failed"; uv build --out-dir "$d" 2>&1 | tail -20; exit 1; }
    whl=$(ls "$d"/orb_audio-*.whl); sdist=$(ls "$d"/orb_audio-*.tar.gz)
    unzip -p "$whl" "*/entry_points.txt" | grep -q "orb-mcp = orb.mcp_server:main"
    unzip -p "$whl" "*/entry_points.txt" | grep -q "orb-generate = orb.cli:main"
    if tar -tzf "$sdist" | grep -E "node_modules|\.pt$|\.safetensors$|/weights/|/hf/"; then echo "sdist contains weights or node_modules"; exit 1; fi
    size=$(wc -c < "$sdist"); [ "$size" -lt 5000000 ] || { echo "sdist is $size bytes"; exit 1; }
  '
else
  warn "uv not found; skipped the sdist/wheel build check"
fi

# ---------------------------------------------------------------------------
section "Weights"

check "weights/*.pt sha256 and size match weights/manifest.json" "$PY" - <<'EOF'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("weights/manifest.json").read_text())["files"]
errs = []
for name, meta in m.items():
    p = pathlib.Path("weights", name)
    if not p.exists():
        if name == "planner.pt": errs.append("weights/planner.pt missing")
        else: print(f"skip {name} (not present locally; distributed separately)")
        continue
    data = p.read_bytes()
    if data.startswith(b"version https://git-lfs"): errs.append(f"{p} is an LFS pointer; run git lfs pull"); continue
    h = hashlib.sha256(data).hexdigest()
    if h != meta["sha256"]: errs.append(f"{p}: sha256 {h} != {meta['sha256']}")
    if len(data) != meta["bytes"]: errs.append(f"{p}: {len(data)} bytes != {meta['bytes']}")
    if not errs: print(f"{name} ok {h[:16]}…")
if errs: raise SystemExit("\n".join(errs))
EOF

check "hf/model-planner: safetensors == planner.pt, config matches, decodes a caption exactly" "$PY" - <<'EOF'
import json, torch
from safetensors.torch import load_file
from orb.api import load_planner, plan, spec_to_dict
d = "hf/model-planner"
cfg = json.load(open(f"{d}/config.json"))
ck = torch.load("weights/planner.pt", map_location="cpu", weights_only=True)
sd = load_file(f"{d}/planner.safetensors")
assert set(sd) == set(ck["model"]), "tensor names differ"
assert all(torch.equal(sd[k], ck["model"][k]) for k in sd), "tensor values differ"
assert cfg["model"] == {k: ck["config"]["model"][k] for k in cfg["model"]}, "config.model differs from checkpoint"
tok, model = load_planner(f"{d}/planner.safetensors")
n = sum(p.numel() for p in model.parameters())
assert n == cfg["parameters"], f"params {n} != config {cfg['parameters']}"
assert tok.vocab_size == cfg["vocab_size"], "vocab size mismatch"
spec = spec_to_dict(plan("6.00-second bright cue, wind bed, with whoosh @1.00 0.90s -6dB L2 #2a", tok, model))
want = {"duration": 6.0, "texture": "wind", "mood": "bright",
        "events": [{"role": "whoosh", "t": 1.0, "dur": 0.9, "gain_db": -6.0, "pan": -0.5, "seed": 42}]}
assert spec == want, f"decoded {spec}"
print(f"planner ok: {n:,} params")
EOF

if [[ -f hf/model-detail/detail.safetensors ]]; then
  check "hf/model-detail: safetensors loads and matches config" "$PY" - <<'EOF'
import json
from orb.api import load_detail
d = "hf/model-detail"
cfg = json.load(open(f"{d}/config.json"))
m = load_detail(f"{d}/detail.safetensors")
n = sum(p.numel() for p in m.parameters())
assert n == cfg["parameters"], f"params {n} != {cfg['parameters']}"
print(f"detail ok: {n:,} params")
EOF
else
  warn "hf/model-detail/detail.safetensors not present; detail repo will not be publishable"
fi

if [[ -f hf/dataset-scores/data/grammar/train.jsonl.gz && -f hf/dataset-scores/data/freetext/train.jsonl.gz ]]; then
  check "hf/dataset-scores: grammar + freetext splits are valid JSON with the expected keys and counts" "$PY" - <<'EOF'
import gzip, json
want = {"grammar/train": (150_000, "score"), "grammar/test": (500, "score"),
        "freetext/train": (150_000, "spec"), "freetext/validation": (500, "spec")}
for name, (count, key) in want.items():
    n = 0
    with gzip.open(f"hf/dataset-scores/data/{name}.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line); n += 1
            assert set(row) == {"caption", key}, f"{name} row {n}: {sorted(row)}"
    assert n == count, f"{name}: {n} rows"
EOF
else
  bad "hf/dataset-scores/data/{grammar,freetext}/*.jsonl.gz missing"
fi

# ---------------------------------------------------------------------------
section "Hugging Face cards and Space"

check "hf/*/README.md front-matter is valid Hub metadata" "$PY" - <<'EOF'
import re, yaml, pathlib
want = {
    "hf/model-planner": dict(license="apache-2.0", keys=["library_name", "pipeline_tag", "tags", "datasets", "language"]),
    "hf/model-detail":  dict(license="cc-by-nc-4.0", keys=["library_name", "pipeline_tag", "tags"]),
    "hf/dataset-scores": dict(license="apache-2.0", keys=["task_categories", "language", "size_categories", "configs", "pretty_name"]),
    "hf/space":         dict(license="apache-2.0", keys=["title", "sdk", "app_port", "short_description"]),
}
errs = []
for d, spec in want.items():
    text = pathlib.Path(d, "README.md").read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m: errs.append(f"{d}: no YAML front-matter"); continue
    meta = yaml.safe_load(m.group(1))
    if meta.get("license") != spec["license"]: errs.append(f"{d}: license {meta.get('license')!r} != {spec['license']!r}")
    for k in spec["keys"]:
        if k not in meta: errs.append(f"{d}: missing {k}")
    if d == "hf/space":
        if meta.get("sdk") != "docker" or meta.get("app_port") != 7860: errs.append("space: sdk must be docker, app_port 7860")
        if len(meta.get("short_description", "")) > 60: errs.append("space: short_description > 60 chars")
    if d == "hf/dataset-scores":
        for c in meta["configs"]:
            for df in c["data_files"]:
                if not pathlib.Path(d, df["path"]).exists(): errs.append(f"{d}: data_files {df['path']} missing")
    if d == "hf/model-detail" and "Non-commercial" not in text[:1500]: errs.append("detail card must lead with the non-commercial warning")
for d in ("hf/model-planner", "hf/model-detail", "hf/dataset-scores"):
    ga = pathlib.Path(d, ".gitattributes")
    if not ga.exists() or "filter=lfs" not in ga.read_text(): errs.append(f"{d}: .gitattributes with LFS rules missing")
if errs: raise SystemExit("\n".join(errs))
EOF

check "Space Dockerfile: v$VERSION, planner from Hub, port 7860, uid 1000" "$PY" - "$VERSION" <<'EOF'
import sys, re, pathlib
v = sys.argv[1]
t = pathlib.Path("hf/space/Dockerfile").read_text()
need = {
    f"clones the v{v} tag": f"ARG ORB_REF=v{v}" in t and "git clone" in t and '--branch "${ORB_REF}"' in t,
    "downloads planner.safetensors": "planner.safetensors" in t and "hf_hub_download" in t,
    "downloads config.json next to it": "config.json" in t,
    "does not download the detail model": "detail.safetensors" not in t,
    "exposes 7860": re.search(r"^EXPOSE 7860$", t, re.M) is not None and '"--port", "7860"' in t,
    "binds 0.0.0.0": '"0.0.0.0"' in t,
    "creates uid 1000": re.search(r"useradd .*-u 1000", t) is not None,
    "runs as non-root": re.search(r"^USER (user|1000)$", t, re.M) is not None,
}
bad = [k for k, ok in need.items() if not ok]
if bad: raise SystemExit("Dockerfile: " + ", ".join(bad))
EOF

# ---------------------------------------------------------------------------
section "Docs: placeholders and links"

check "no placeholders or stale names in release-facing files" "$PY" - <<'EOF'
import pathlib, re
files = ["README.md", "CHANGELOG.md", "MODEL_CARD.md", "CITATION.cff", "pyproject.toml", "NOTICE",
         "CONTRIBUTING.md", "RELEASE_CHECKLIST.md", "examples/README.md", "weights/README.md",
         "hf/space/Dockerfile"]
files += [str(p) for p in pathlib.Path("hf").glob("*/README.md")]
files += [str(p) for p in pathlib.Path("docs").glob("*.md")]
files += [str(p) for p in pathlib.Path("site").glob("*.html")]
pats = [
    (r"\b(TODO|FIXME|TBD|XXX)\b", "marker"),
    (r"your[-_ ]?(org|user|name|username|repo|handle)\b|YOUR_[A-Z]+|<(org|user|owner)>|\bREPLACE[_ ]?ME\b|lorem ipsum", "placeholder"),
    (r"\bexample\.com\b", "placeholder domain"),
    (r"pip install [\"']?orb\[|\"orb\[", "old distribution name 'orb[...]' (now orb-audio)"),
    (r"\]\((\./)?(PLAN|REVIEW)\.md\)", "link to moved PLAN.md/REVIEW.md (now docs/internal/)"),
]
hits = []
for f in files:
    p = pathlib.Path(f)
    if not p.exists(): continue
    for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
        for pat, why in pats:
            if re.search(pat, line, re.I if why != "marker" else 0):
                hits.append(f"{f}:{i}: {why}: {line.strip()[:110]}")
if hits: raise SystemExit("\n".join(hits))
EOF

check "relative links and images in README / docs resolve" "$PY" - <<'EOF'
import pathlib, re
files = ["README.md", "CONTRIBUTING.md", "MODEL_CARD.md", "examples/README.md"] + [str(p) for p in pathlib.Path("docs").glob("*.md")]
link = re.compile(r"\]\(([^)\s]+)\)|(?:src|href)=\"([^\"]+)\"")
missing = []
for f in files:
    p = pathlib.Path(f)
    if not p.exists(): continue
    for m in link.finditer(p.read_text()):
        target = (m.group(1) or m.group(2)).split("#")[0]
        if not target or re.match(r"^[a-z]+:", target): continue
        if not (p.parent / target).exists(): missing.append(f"{f}: {target}")
if missing: raise SystemExit("missing:\n" + "\n".join(sorted(set(missing))))
EOF

# ---------------------------------------------------------------------------
if [[ $FAST -eq 0 ]]; then
  section "Tests and examples"
  check "pytest -q" "$PY" -m pytest -q -p no:cacheprovider
  check "examples/01_generate.py (planner + DSP)" "$PY" examples/01_generate.py
  check "examples/02_edit_spec.py" "$PY" examples/02_edit_spec.py
  check "examples/03_variants.py --n 3" "$PY" examples/03_variants.py --n 3
  check "examples/04_no_torch.py" "$PY" examples/04_no_torch.py
  check "orb-generate CLI --json" "$PY" -m orb.cli "4.00-second playful cue, vinyl bed, with ui pop @0.60 0.16s -12dB R1 #73" --out out/release_check_cli.wav --json
  if [[ -n "${NPY:-}" ]]; then
    if "$NPY" -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("torch") is None else 1)' 2>/dev/null; then
      check "DSP path runs in a venv WITHOUT torch ($NPY)" bash -c "'$NPY' examples/04_no_torch.py && '$NPY' examples/02_edit_spec.py"
    else
      warn "NPY=$NPY has torch installed; cannot prove the DSP path is torch-free"
    fi
  else
    warn "no torch-free python (NPY) found; DSP-only check skipped"
  fi
else
  section "Tests and examples"
  warn "skipped (--fast)"
fi

# ---------------------------------------------------------------------------
section "Summary"
echo "  ${G}${PASS} passed${N}, ${Y}${#WARNS[@]} warnings${N}, ${R}${#FAILS[@]} failed${N}"
if (( ${#FAILS[@]} )); then
  for f in "${FAILS[@]}"; do echo "  ${R}-${N} $f"; done
  exit 1
fi
echo "  Ready for scripts/release_publish.sh (dry run first)."
