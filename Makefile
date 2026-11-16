PY ?= python
VENV ?= .venv

.PHONY: help install test examples demo serve package clean

help:
	@echo "install   create a venv and install the package with dev deps"
	@echo "test      run the test suite"
	@echo "examples  run the four examples into out/"
	@echo "demo      regenerate the curated demo set into out/demo"
	@echo "serve     run the local web demo on :8765"
	@echo "package   rebuild weights/ from artifacts/final"
	@echo "clean     remove generated output"

install:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

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

package:
	$(VENV)/bin/python scripts/package_artifacts.py --out weights

clean:
	rm -rf out runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
