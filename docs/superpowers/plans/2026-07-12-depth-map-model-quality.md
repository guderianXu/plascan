# Depth Map Model Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Dino depth-map and mesh quality while retaining CUDA PatchMatch and multithreaded streaming fusion.

**Architecture:** Fix confidence at the CUDA photometric-cost source, reuse one depth postprocessor for memory and disk paths, keep strict multi-view fusion in production model generation, then stabilize closed-object normals before Poisson. GUI and CLI continue to call the shared core workflow.

**Tech Stack:** C++17, CUDA 13.1, OpenCV, Qt6, PlaPoint/PlaMatrix, GTest, CMake/CTest.

---

### Task 1: Robust multi-source PatchMatch confidence

**Files:**
- Create: `src/core/mvs/PatchMatchPhotometricCost.h`
- Modify: `src/core/mvs/PatchMatchCUDA.cu`
- Test: `src/core/mvs/tests/test_patchmatch_photometric_cost.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`

- [ ] Add a failing test showing that one high NCC and one failed source cannot produce high confidence, while two agreeing sources can.
- [ ] Run the focused GTest and verify the old aggregation fails the new expectation.
- [ ] Add a fixed-size host/device robust NCC aggregator requiring at least two and a majority of source views.
- [ ] Use the aggregator in `evalHypCost()` without moving pixel work off CUDA.
- [ ] Run the focused GTest and existing MVS tests.

### Task 2: Preserve full fine-level search

**Files:**
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Test: `src/core/mvs/tests/test_mvs_quality_policy.cpp`

- [ ] Add a failing policy test proving high hint coverage does not reduce requested fine iterations.
- [ ] Extract the coarse/fine iteration policy into a testable helper.
- [ ] Keep coarse initialization but always retain the requested fine iteration count.
- [ ] Build and run the policy and source-planner tests.

### Task 3: Reuse depth postprocessing for stored frames

**Files:**
- Create: `src/core/mvs/DepthMapPostprocessor.h`
- Create: `src/core/mvs/DepthMapPostprocessor.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/DepthFrameUtils.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`
- Test: `src/core/mvs/tests/test_depth_map_postprocessor.cpp`

- [ ] Add failing tests for confidence rejection, an isolated relative-depth spike, and a small valid component.
- [ ] Move the existing filters behind `postprocessDepthMap()` and keep `DepthPostProcessStats` unchanged.
- [ ] Call the shared postprocessor from both in-memory generation and `buildStoredFusionFrame()`.
- [ ] Preserve confidence until postprocessing completes, then release it before fusion to cap memory.
- [ ] Run postprocessor, manifest, and fusion tests.

### Task 4: Restore production multi-view fusion

**Files:**
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `src/gui/project/support/ProjectDenseWorkflowConfig.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] Add a failing source-contract test that a 16-frame transient model request does not cap `minNumPixels` at 2.
- [ ] Remove the frame-count relaxation; retain it only for an explicit `fast_preview` profile.
- [ ] Keep existing threaded window fusion and cancellation behavior.
- [ ] Run GUI project and model-workflow tests.

### Task 5: Stabilize closed-object normals

**Files:**
- Modify: `src/core/mesh/PointCloudPreprocess.h`
- Modify: `src/core/mesh/PointCloudPreprocess.cpp`
- Modify: `src/core/mesh/SurfaceReconstructor.cpp`
- Test: `tests/test_mesh_reconstructor.cpp`

- [ ] Add a failing test with alternating local normals on a concave synthetic cloud.
- [ ] Implement KDTree neighborhood orientation propagation and one global outward decision.
- [ ] Run CPU neighbor queries in parallel; do not use brute-force GPU KNN for large clouds.
- [ ] Replace per-point centroid flipping with the neighborhood-consistent path.
- [ ] Run all mesh tests.

### Task 6: Dino end-to-end verification

**Files:**
- No production file required.

- [ ] Build `plascan_gui`, `three_d_reconstruction_cli`, `mesh_reconstruct_cli`, and related tests in `build/windows-vcpkg-cuda-release`.
- [ ] Run Dino 16 images into a new output directory without overwriting existing user results.
- [ ] Record depth jump percentiles, fusion point distribution, normal statistics, mesh counts, and algorithms.
- [ ] Run `ctest --output-on-failure -R "Mvs|Depth|Fusion|Mesh|ModelGeneration|Gui"`.
- [ ] Run `git diff --check` for changed files and report any remaining Poisson/TSDF limitation.
