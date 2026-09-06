# PlaScan scripts

Scripts are grouped by responsibility. Run commands from the repository root so
relative model, build, and test-data paths resolve consistently.

| Directory | Purpose |
| --- | --- |
| `models/` | Download, export, and maintain model resources. |
| `workflows/` | User-invoked reconstruction, feature, and matching workflows. |
| `bench/` | Deterministic performance and prepared-dataset benchmark runners. |
| `env/` | Local Python, vcpkg, and CMake environment setup. |
| `build_win/` | Windows CUDA build and developer-shell entrypoints. |
| `marker_targets/` | Marker corpus import and printable-target verification. |
| `validation/` | Regression, third-party comparison, and reproducibility tools. |
| `dev/` | Developer-only local runtimes, including isolated browser GUI testing. |
| `legacy/` | Compatibility-only paths excluded from normal GUI and CLI workflows. |

To run the real Qt GUI in an isolated browser-accessible display, see
[`docs/GUI_BROWSER_TESTING.md`](../docs/GUI_BROWSER_TESTING.md).
The `dev/browser_gui.py` launcher owns the isolated Linux runtime. Agents should use
`dev/browser_agent.py` for compact inspection, semantic actions, nested waits/assertions, JSONL
watching, guarded form updates, task cancellation, and diagnostic bundles. The lower-level
`dev/browser_gui_scenario.py` executes reusable semantic JSON scenarios.

`validation/experiments/` stores historical profiles. It is not part of the
default validation baseline.
