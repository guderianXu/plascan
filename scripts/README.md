# PlaScan scripts

Scripts are grouped by responsibility. Run commands from the repository root so
relative model, build, and test-data paths resolve consistently.

| Directory | Purpose |
| --- | --- |
| `models/` | Download, export, and maintain model resources. |
| `workflows/` | User-invoked reconstruction, feature, and matching workflows. |
| `bench/` | Deterministic performance and prepared-dataset benchmark runners. |
| `env/` | Local Python, LibTorch, vcpkg, and CMake environment setup. |
| `build_win/` | Windows CUDA build and developer-shell entrypoints. |
| `marker_targets/` | Marker corpus import and printable-target verification. |
| `validation/` | Regression, third-party comparison, and reproducibility tools. |
| `legacy/` | Compatibility-only paths excluded from normal GUI and CLI workflows. |

`validation/experiments/` stores historical profiles. It is not part of the
default validation baseline.
