PY ?= python
VENV ?= .venv

.PHONY: help install install-all test examples demo serve mcp package release-check release-dry-run clean

help:
	@echo "install       create a venv and install the package with dev deps (numpy only)"
	@echo "install-all   same, plus torch, the MCP server and huggingface_hub"
	@echo "test      run the test suite"
	@echo "examples  run the four examples into out/"
	@echo "demo      regenerate the curated demo set into out/demo"
	@echo "serve     run the local web demo on :8765"
	@echo "mcp       run the MCP server for agents (stdio)"
	@echo "package   rebuild weights/ from artifacts/final"
	@echo "release-check     run every local release preflight check"
	@echo "release-dry-run   print what scripts/release_publish.sh would do (publishes nothing)"
	@echo "clean     remove generated output"

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

install-all:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[all]"

test:
	$(VENV)/bin/python -m pytest -q

examples:
	$(VENV)/bin/python examples/01_generate.py
	$(VENV)/bin/python examples/02_edit_spec.py
	$(VENV)/bin/python examples/03_variants.py --n 4
	$(VENV)/bin/python examples/04_no_torch.py

demo:
	$(VENV)/bin/python scripts/make_demo.py --out out/demo

serve:
	$(VENV)/bin/python web/server.py --port 8765

mcp:
	$(VENV)/bin/orb-mcp

package:
	$(VENV)/bin/python scripts/package_artifacts.py --out weights

release-check:
	scripts/release_check.sh

release-dry-run:
	PY=$(VENV)/bin/python scripts/release_publish.sh

clean:
	rm -rf out runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
