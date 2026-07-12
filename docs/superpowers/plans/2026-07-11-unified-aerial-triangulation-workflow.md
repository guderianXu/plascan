# Unified Aerial Triangulation Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GUI and CLI execute the same tie-point preparation, SfM, BA, and result-reporting workflow.

**Architecture:** Extend `AerialTriangulationWorkflow` with a resolved `MatchPhotosTask` configuration and an injectable tie-point runner. The workflow owns execution order and cache semantics; GUI and CLI only supply project inputs and consume the merged result. SfM always consumes the files produced or explicitly reused by the workflow and never launches a second matcher in the same run.

**Tech Stack:** C++17, Qt6 Core, MatchPhotosTask, AerialTriangulationService, GTest, CMake/Ninja, CUDA SIFT and LightGlue.

---

### Task 1: Define and test the unified workflow contract

**Files:**
- Modify: `src/core/aerial_triangulation/AerialTriangulationWorkflow.h`
- Modify: `tests/test_aerial_triangulation_workflow.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] Add failing tests asserting that reset alignment forces tie-point preparation and feature regeneration.
- [ ] Add a failing test asserting that the service receives `autoGenerateMissingMatches=false` after preparation.
- [ ] Add fake-runner tests for preparation-before-SfM ordering, preparation failure short-circuiting, and merged feature/match results.
- [ ] Build `test_aerial_triangulation_workflow` and confirm failures are caused by missing workflow fields and runner behavior.

### Task 2: Implement shared tie-point configuration and execution

**Files:**
- Modify: `src/core/aerial_triangulation/AerialTriangulationWorkflow.h`
- Modify: `src/core/aerial_triangulation/AerialTriangulationWorkflow.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] Add project asset, feature, match, mask, preparation-policy, and manual-pair fields to workflow options.
- [ ] Resolve `MatchPhotosOptions` once from quality, device, sequence mode, keypoint/tie-point limits, guided matching, masks, and fixed-point policy.
- [ ] Implement reset cleanup scoped to the current project's match/no-match/tie-point files and force `reuseExistingFeatures=false`.
- [ ] Execute the injected or default `MatchPhotosTask` runner before SfM and map progress to `0-35%`.
- [ ] Merge generated feature/match/tie-point records into the workflow result and set explicit preparation diagnostics.
- [ ] Run the workflow tests until green.

### Task 3: Migrate the GUI to one background workflow

**Files:**
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.h`
- Modify: `tests/test_source_contracts.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] Add failing source-contract tests requiring one workflow call and forbidding the old recursive `prepareAerialTriangulationTiePoints -> runAerialTriangulationSfm` path.
- [ ] Populate workflow asset directories and `ProjectIO::maskPathsForImages()` in the GUI adapter.
- [ ] Remove the standalone preparation runner and callback recursion.
- [ ] Write merged feature, match, tie-point, camera, sparse-cloud, and report records from one workflow result.
- [ ] Preserve cancellation, project-switch guards, atomic `replaceImageCameras()`, and user-visible progress.
- [ ] Build `plascan_gui` and run the GUI contract tests.

### Task 4: Migrate the aerial-triangulation CLI

**Files:**
- Modify: `src/cli/cli_aerial_triangulation.cpp`
- Modify: `tests/test_cli_contracts.cpp`

- [ ] Add failing tests for resolved preparation mode and preparation diagnostics in dry-run/run reports.
- [ ] Map mask directory, sequence mode, keypoint/tie-point limits, reset semantics, and reuse-only mode into the shared workflow options.
- [ ] Make reset alignment override `--no-auto-generate-missing-matches` and record that decision.
- [ ] Include preparation counts, candidate-pair counts, and stage status in `aerial_triangulation_cli_report.json`.
- [ ] Build the CLI and run CLI contract tests.

### Task 5: Verify clean-cache GUI-equivalent reconstruction

**Files:**
- Update if behavior changed: `src/core/sfm/README.md`
- Update if module boundaries changed: `docs/PROJECT_ARCHITECTURE.md`

- [ ] Build `plascan_gui`, `aerial_triangulation_cli`, and focused test targets with `-j 32`.
- [ ] Run project-data, match-file, workflow, CLI-contract, source-contract, SfM-parameter, and SfM-pipeline tests.
- [ ] Run `dino` in a new project/output directory with highest quality, SIFT+LightGlue, sequence window 8, keypoint mask mode, 40000 keypoints, 4000 tie points, reset alignment, and no reusable feature/match cache.
- [ ] Assert the report records tie-point preparation as executed and registers 16/16 real cameras.
- [ ] Run the same clean configuration a second time in another new directory and compare registered count, actual match-graph edges, selected focal candidate, and reprojection error.
- [ ] Run `git diff --check` on all touched files and report any unrelated existing workspace changes separately.
