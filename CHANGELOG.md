# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-16

First public release.

### Added

- Typed `Score` / `Spec` schema with validation and JSON serialisation.
- Deterministic DSP renderers at 48 kHz: 10 SFX roles, 8 texture beds, and the
  music renderer (music is deferred to v2).
- Compact Spec planner: 5.08M parameters, byte-level tokenizer, constrained
  decoding, **96.6% exact** on 500 held-out captions with 0 decode failures.
- Optional detail model: 10.27M complex-STFT U-Net with a noise channel, an
  energy gate, and a high-frequency-only residual.
- `orb.api` public surface: `generate`, `render_spec`, `plan`, `load_planner`,
  `load_detail`, `spec_to_dict`, `dict_to_spec`, `save_wav`.
- Local web demo (`web/`) with pre-rendered examples and live generation.
- Training, evaluation, and packaging scripts.
- 36 tests.

### Fixed

- Background hiss: the bed absorbed the mix normalisation gain on sparse cues,
  `np.interp` left a flat white-noise tail above the last spectral control
  point, and grains were unfiltered white bursts. See `RESULTS.md`.
- DC offset from one-sided transients and slow texture LFOs.
- Constrained decoding skipped `END_EVENTS` at the event limit.
- Planner event distribution excluded 0 and 1 event, so single-sound captions
  emitted zeros.
- `torch.istft` window device mismatch during CUDA inference.
- Detail inference fed whole clips to a model trained on 256-frame crops.

### Known limitations

- The detail model does not meet its restoration gate (10% MR-STFT); it is an
  optional texture stage, and the DSP path is the shippable engine.
- The detail weights are CC BY-NC-4.0 and are distributed separately.
- Music is not implemented (v2).
- The planner has a ~3.4% single-digit copy error on dense captions; pass a
  Spec dict when exactness matters.
