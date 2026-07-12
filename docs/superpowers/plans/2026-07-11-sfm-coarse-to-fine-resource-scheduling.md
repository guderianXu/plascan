# SfM Coarse-to-Fine Resource Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated full-strength focal/initial-pair SfM attempts with bounded-parallel coarse evaluation followed by one full refinement, while preserving dino 16/16 registration and selecting CPU/CUDA BA by measured problem size.

**Architecture:** Add a small pure `SfmSearchPolicy` module for candidate scoring and worker allocation. Extend incremental SfM results with the selected initial pair and add an internal coarse execution profile. The aerial-triangulation service evaluates focal candidates without writing outputs, ranks them, then replays the best focal/pair once with the existing full-quality path and emits resource diagnostics.

**Tech Stack:** C++17, Qt6 Core/Concurrent, OpenMP, existing SfM/BA modules, GoogleTest, CMake, `aerial_triangulation_cli`.

---

### Task 1: Add pure search policy and scheduler tests

**Files:**
- Create: `src/core/aerial_triangulation/SfmSearchPolicy.h`
- Create: `src/core/aerial_triangulation/SfmSearchPolicy.cpp`
- Create: `tests/test_sfm_search_policy.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `src/cli/CMakeLists.txt`

- [ ] **Step 1: Write failing policy tests**

Cover worker allocation, lexicographic quality ranking, deterministic tie-breaking and replay budget:

```cpp
TEST(SfmSearchPolicyTest, AllocatesFourWorkersAcrossThirtyTwoThreads)
{
    const auto budget = xjw::sfm_search::allocateWorkers(6, 32);
    EXPECT_EQ(budget.workerCount, 4);
    EXPECT_EQ(budget.threadsPerWorker, 8);
}

TEST(SfmSearchPolicyTest, RegistrationCoverageDominatesRms)
{
    SfmCandidateSummary full{0, 3.2, 1, 2, 16, 4021, 0.6501, true};
    SfmCandidateSummary partial{1, 1.2, 3, 4, 15, 8000, 0.20, true};
    EXPECT_TRUE(isBetterCandidate(full, partial));
}
```

- [ ] **Step 2: Run RED test**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_sfm_search_policy -j 20
```

Expected: FAIL because `SfmSearchPolicy` and the test target do not exist.

- [ ] **Step 3: Implement the minimal pure policy**

Define `SfmCandidateSummary`, `SfmWorkerBudget`, `allocateWorkers`, `isBetterCandidate`, `rankCandidates`, and `replayCandidateIndices`. Worker allocation must implement:

```cpp
workerCount = std::min({candidateCount, std::max(1, totalThreads / 8), 4});
threadsPerWorker = std::max(1, totalThreads / workerCount);
```

Ranking order is registered images, success, point count, finite lower RMS, then stable candidate index. Replay returns at most three ranked indices.

- [ ] **Step 4: Register the source and test target in CMake**

Add `SfmSearchPolicy.cpp` to GUI and all CLI targets that compile `AerialTriangulationService.cpp`. Link `test_sfm_search_policy` with `GTest::gtest_main` and include `${CMAKE_SOURCE_DIR}/src`.

- [ ] **Step 5: Run GREEN test**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_sfm_search_policy -j 20
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_sfm_search_policy.exe --gtest_brief=1
```

Expected: all policy tests pass.

### Task 2: Add an explicit coarse SfM execution profile

**Files:**
- Modify: `src/core/sfm/pipeline/IncrementalSfm.h`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
- Modify: `tests/test_sfm_pipeline.cpp`
- Modify: `tests/test_source_contracts.cpp`

- [ ] **Step 1: Write failing profile and selected-pair tests**

Add tests requiring:

```cpp
EXPECT_EQ(options.executionProfile, SfmExecutionProfile::FullRefinement);
EXPECT_EQ(result.selectedInitialImageId1, expectedFirst);
EXPECT_EQ(result.selectedInitialImageId2, expectedSecond);
```

Add a source contract requiring coarse mode to cap `maxInitPairCandidates` at 6, `baOptions.maxIterations` at 5, `iterativeBARounds` at 1, widen local BA to every 6 registrations, and disable iteration-level INFO logs.

- [ ] **Step 2: Run RED tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_sfm_pipeline test_source_contracts -j 20
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_sfm_pipeline.exe --gtest_filter=*ExecutionProfile*:*SelectedInitialPair*
```

Expected: FAIL because the execution profile and selected-pair fields do not exist.

- [ ] **Step 3: Implement the coarse profile**

Add:

```cpp
enum class SfmExecutionProfile
{
    FullRefinement,
    CoarseEvaluation
};
```

`IncrementalSfmOptions` receives `executionProfile`. `IncrementalSfmResult` receives selected initial image IDs and BA call/iteration counters. In coarse mode, normalize options once in the constructor or a focused helper:

```cpp
maxInitPairCandidates = std::min(maxInitPairCandidates, 6);
baOptions.maxIterations = std::min(baOptions.maxIterations, 5);
iterativeBARounds = 1;
localBAInterval = std::max(localBAInterval, 6);
globalBAInterval = std::numeric_limits<int>::max();
baOptions.refineSharedFocalLength = false;
```

Keep local BA available at its configured interval so PnP still receives a usable model. Record the candidate pair selected at the end of multi-seed evaluation and for explicit-pair execution.

- [ ] **Step 4: Throttle coarse BA logging**

Add a BA option that controls per-iteration INFO output. Full refinement keeps current logging; coarse evaluation emits one summary per BA and stores counters in the result.

- [ ] **Step 5: Run GREEN tests and SfM regression tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_sfm_pipeline test_source_contracts -j 20
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_sfm_pipeline.exe --gtest_brief=1
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=SfmSourceContractTest.* --gtest_brief=1
```

Expected: execution-profile tests pass; report unrelated pre-existing source-contract failures separately.

### Task 3: Separate candidate evaluation from output-producing refinement

**Files:**
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.h`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `tests/test_aerial_triangulation_workflow.cpp`
- Modify: `tests/test_source_contracts.cpp`

- [ ] **Step 1: Write failing service contract tests**

Require an internal execution mode with these behaviors:

```cpp
EXPECT_FALSE(coarse.writeOutputs);
EXPECT_FALSE(coarse.enableGuidedRematching);
EXPECT_EQ(coarse.executionProfile, SfmExecutionProfile::CoarseEvaluation);
EXPECT_TRUE(full.writeOutputs);
```

Require full replay to accept the focal and initial-pair hint selected by coarse evaluation.

- [ ] **Step 2: Run RED tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_aerial_triangulation_workflow test_source_contracts -j 20
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_aerial_triangulation_workflow.exe --gtest_filter=*Coarse*:*Replay*
```

Expected: FAIL because the service has only the output-producing attempt path.

- [ ] **Step 3: Add internal attempt controls**

Extend service options with internal-only fields defaulting to current behavior:

```cpp
bool writeSfmOutputs = true;
bool useInitialPairHint = false;
ImageId initialImageId1 = kInvalidImageId;
ImageId initialImageId2 = kInvalidImageId;
SfmExecutionProfile sfmExecutionProfile = SfmExecutionProfile::FullRefinement;
QString searchCandidateId;
```

When `writeSfmOutputs` is false, skip PLY/report/project-result writes and all progress callbacks that mutate GUI state. Matching cache reads remain enabled, but coarse attempts must set `autoGenerateMissingMatches=false` so parallel candidates cannot write shared match files.

- [ ] **Step 4: Wire selected-pair hints into IncrementalSfM**

For full replay, set `autoSelectInitPair=false` and pass the chosen IDs. Copy selected pair and BA counters into `AerialTriangulationServiceResult.sfmDiagnostics`.

- [ ] **Step 5: Run GREEN service tests**

Run the two test binaries from Step 2 and confirm the new tests pass.

### Task 4: Implement bounded-parallel focal coarse search and single full replay

**Files:**
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/core/aerial_triangulation/SfmSearchPolicy.h`
- Modify: `src/core/aerial_triangulation/SfmSearchPolicy.cpp`
- Modify: `tests/test_sfm_search_policy.cpp`
- Modify: `tests/test_source_contracts.cpp`

- [ ] **Step 1: Write failing orchestration tests**

Test that six focal scales on 32 threads produce four workers, each coarse candidate disables output and uses eight threads, and only the ranked winner is invoked in full mode. Test serial fallback when worker creation reports failure and cancellation before scheduling the next batch.

- [ ] **Step 2: Run RED tests**

Run the policy and source-contract binaries; confirm failures identify missing coarse orchestration.

- [ ] **Step 3: Implement candidate orchestration**

Replace `runAdaptiveFocalSweep`'s full attempt loop with:

```text
build coarse options for each focal scale
execute at most workerCount attempts concurrently
collect deterministic summaries
rank summaries
replay up to three candidates sequentially in full mode until production quality passes
```

Use a bounded batch of `std::async(std::launch::async, ...)` tasks. Each task owns its options and service result, receives `threadsPerWorker`, has no GUI progress callback, performs no output writes, and prefixes aggregate logs with its candidate ID. Never run multiple full refinements concurrently.

- [ ] **Step 4: Add early comparison and fallback behavior**

If a coarse candidate reaches full registration, still evaluate the nearest remaining focal scale before final ranking. If all concurrent attempts fail to launch, rerun the same list serially. Full replay tries no more than the top three ranked candidates.

- [ ] **Step 5: Add search diagnostics**

Populate candidate count, worker count, worker threads, coarse/full BA calls, selected focal/pair, backend counts and elapsed times in `sfmDiagnostics`. Log cache reuse explicitly so a no-GPU frontend workload is understandable.

- [ ] **Step 6: Run GREEN orchestration tests**

Run policy, workflow and source-contract tests. Confirm deterministic rankings across repeated test runs.

### Task 5: Improve BA resource diagnostics without forcing small problems to CUDA

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`
- Modify: `tests/test_ba_cuda_contracts.py`

- [ ] **Step 1: Write failing backend-reason tests**

For a 16-camera, low-observation problem, require `LegacyCpu` plus reason `below_cuda_problem_size`. For a sufficiently large CUDA-capable problem, retain current CUDA selection. Test that requested thread count is included in diagnostics.

- [ ] **Step 2: Run RED tests**

Build and run `test_bundle_adjust_backend_selection`; run `python tests/test_ba_cuda_contracts.py` with the repository Python environment.

- [ ] **Step 3: Add structured backend selection diagnostics**

Return or fill a `BABackendDecision` containing selected backend and reason while retaining the existing `selectBackendForProblem` API as a thin wrapper. Log cameras, tracks, observations, threads, backend and reason once per formal BA. Coarse BA accumulates counts without per-iteration INFO logs.

- [ ] **Step 4: Run GREEN backend tests**

Confirm small problems remain CPU, large problems retain CUDA selection, and no threshold is lowered.

### Task 6: Build, regress and benchmark dino

**Files:**
- Modify only if diagnostics reveal a tested defect in the preceding implementation.

- [ ] **Step 1: Build all affected targets with parallel compilation**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target aerial_triangulation_cli plascan_gui test_sfm_search_policy test_sfm_pipeline test_aerial_triangulation_workflow test_bundle_adjust_backend_selection test_source_contracts -j 20
```

- [ ] **Step 2: Run focused regression tests**

Run all newly added tests plus existing SfM/BA tests. Record any unrelated dirty-worktree failures without masking them.

- [ ] **Step 3: Run the dino CLI benchmark**

Use the existing 16-image dino input and GUI-equivalent SIFT/LightGlue, sequence, mask and reset-alignment options. Write output to a new timestamped directory under `E:/code/test` and preserve the log for comparison.

- [ ] **Step 4: Verify quality and performance gates**

Require:

```text
registered_images = 16/16
points3d >= 3216 (80% of 4021)
mean_reproj_error <= 0.7001 px
BA calls reduced by at least 70% from 595
elapsed time reduced by at least 60% from 81 seconds
```

If quality passes but timing misses, profile candidate scheduling before changing mathematical thresholds. If quality fails, use the ranked replay diagnostics to identify whether coarse selection or full replay diverged.

- [ ] **Step 5: Run GUI build smoke verification**

Confirm `plascan_gui` links and the aerial-triangulation dialog can start a task without blocking the GUI thread. Do not alter or delete the user's existing dino project outputs.

### Task 7: Synchronize architecture documentation

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/sfm/README.md`

- [ ] **Step 1: Document the search policy boundary**

Describe coarse candidate evaluation, bounded CPU scheduling, single full replay, cache reuse semantics and BA backend selection.

- [ ] **Step 2: Verify documentation against code**

Search for the final option, diagnostic and function names and ensure documentation uses exactly those names.

- [ ] **Step 3: Review the final diff**

Run `git diff --check` and `git status --short`. Do not commit or push until the user explicitly requests it, per repository instructions.
