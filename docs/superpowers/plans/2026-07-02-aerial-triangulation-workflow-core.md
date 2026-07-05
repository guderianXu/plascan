# Aerial Triangulation Workflow Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a core workflow class that maps Metashape-style "Align Photos" dialog settings into PlaScan SfM/BA execution options and lets the workflow-menu aerial triangulation action run the existing sparse reconstruction pipeline.

**Architecture:** Keep GUI responsibilities limited to dialog, preflight, progress, cancellation, and project write-back. Add `AerialTriangulationWorkflow` under `src/core/pipeline` to receive user-level options, resolve algorithm-level settings, and call the existing `SFMService`. The first version reuses existing SfM/BA modules instead of rewriting reconstruction internals.

**Tech Stack:** C++17, Qt6 Core/Widgets, CMake, GTest, existing `SFMService`, `IncrementalSfmOptions`, and GUI task runner.

---

### Task 1: Core Workflow Options And Config Resolution

**Files:**
- Create: `src/core/pipeline/AerialTriangulationWorkflow.h`
- Create: `src/core/pipeline/AerialTriangulationWorkflow.cpp`
- Test: `tests/test_aerial_triangulation_workflow.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing config tests**

Add tests that include `AerialTriangulationWorkflow.h` and assert:
- `quality="highest"` maps to `SFMServiceOptions::quality == 3`, no image downscale, high keypoint limit.
- `quality="low"` maps to a lower SfM quality and reduced feature max dimension.
- `genericPreselection=false`, `referencePreselection=true`, `referenceMode="sequence"` maps to sequence-window pair planning.
- `keypointLimit` and `tiepointLimit` drive `featureMaxKeypoints`, `maxTiePointsPerImage`, and `maxTiePointsPerGridCell`.

- [ ] **Step 2: Verify tests fail**

Run:
```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow --config Release -j 8
```
Expected: compilation fails because `AerialTriangulationWorkflow.h` does not exist.

- [ ] **Step 3: Implement options and resolver**

Create:
- `AerialTriangulationWorkflowOptions`
- `AerialTriangulationResolvedConfig`
- `AerialTriangulationWorkflowResult`
- `AerialTriangulationWorkflow::resolveConfig()`
- `AerialTriangulationWorkflow::run()`

`run()` should call `SFMService::run(resolveConfig(options).sfmOptions)` and return a wrapper result containing both resolved config and `SFMServiceResult`.

- [ ] **Step 4: Verify config tests pass**

Run:
```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow --config Release -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R AerialTriangulationWorkflowCoreTest
```
Expected: all new core workflow tests pass.

### Task 2: GUI Dialog Mapping And Launch Integration

**Files:**
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.h`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Update failing GUI tests**

Change the existing test that says workflow dialog does not start workflow. New expected behavior:
- `openWorkflowAerialTriangulationDialog()` still uses `AerialTriangulationDialog`.
- On accepted dialog it saves settings and calls `startAerialTriangulationWorkflow(...)`.
- Sparse reconstruction menu's formal aerial triangulation dialog still uses `ThreeDReconstructionDialog::Mode::AerialTriangulation`.

- [ ] **Step 2: Verify GUI test fails**

Run:
```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=AerialTriangulationWorkflowTest.WorkflowDialogStartsAerialTriangulationWorkflow
```
Expected: fails because the workflow dialog currently only saves settings.

- [ ] **Step 3: Implement accepted-dialog launch**

In `openWorkflowAerialTriangulationDialog()`, after `dlg.exec() == QDialog::Accepted`, save `dlg.collectSettings()` and call `startAerialTriangulationWorkflow(settings)`.

- [ ] **Step 4: Use core workflow resolver in SFM launch**

In `launchAerialTriangulationSfm()`, replace local manual mapping of quality, device, feature algorithm, match algorithm, guided matching, and tie point limits with `AerialTriangulationWorkflow::resolveConfig(...)`. Keep preflight and project write-back in the GUI controller.

- [ ] **Step 5: Verify GUI tests pass**

Run:
```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=AerialTriangulationWorkflowTest.*:AerialTriangulationDialogTest.*:MainMenuTest.WorkflowMenuExposesAerialTriangulationDialogBeforeThreeDReconstruction
```
Expected: all matching GUI tests pass.

### Task 3: Build Verification

**Files:**
- No new files.

- [ ] **Step 1: Build GUI and tests**

Run:
```powershell
cmd /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow test_gui_project_utils plascan_gui --config Release -j 8"
```
Expected: build exits 0.

- [ ] **Step 2: Run targeted tests**

Run:
```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R "AerialTriangulationWorkflow|AerialTriangulationDialog|MainMenuTest.WorkflowMenu"
```
Expected: targeted tests pass.

### Self-Review

- The plan keeps the first implementation scoped to workflow orchestration and parameter resolution.
- It does not add GCP, sensor groups, or full self-calibration staging yet; those are separate product-level phases.
- It requires tests to fail before production code is added.
