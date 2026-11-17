## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `pytest -q` passes
- [ ] `python examples/04_no_torch.py` runs (the DSP path must never import torch)
- [ ] If the renderer changed: `python examples/01_generate.py` still passes the audit
- [ ] If a role or texture was added: planner data regenerated and retrained (see CONTRIBUTING.md)
- [ ] CHANGELOG.md updated under `[Unreleased]`
