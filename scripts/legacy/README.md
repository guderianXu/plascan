# Legacy Python backends

This directory retains manual compatibility paths that are not part of the
default GUI or CLI workflow.

- `run_full_pipeline_test.py` is available only through
  `scripts/workflows/run_full_pipeline.py --legacy-stereo-test`.
- `run_lightglue.py` is a Python fallback for feature-file matching; the normal
  workflow uses the C++ TorchScript LightGlue matcher.

New workflow features must target the C++ pipeline rather than these scripts.
