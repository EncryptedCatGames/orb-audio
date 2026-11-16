#!/usr/bin/env bash
# One-shot setup for a Linode GPU box (Ubuntu 22.04/24.04).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y ffmpeg git curl

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.12
uv pip install -e ".[train,dev]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available; check the driver and the wheel index"
PY

.venv/bin/python -m pytest -q
echo "setup ok"
