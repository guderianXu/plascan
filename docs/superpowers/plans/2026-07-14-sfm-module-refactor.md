# SfM Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant SfM code, reuse PlaPoint for generic spatial algorithms, split oversized responsibilities, make algorithm targets Qt-free, and preserve current numerical behavior.

**Architecture:** Implement the approved design in independently verifiable stages. Keep current algorithms and thresholds unchanged while moving code behind pure C++ interfaces; Qt JSON and project file handling move into `sfm_project`, while `sfm_core` and `sfm_postprocess` contain only standard C++, OpenCV, PlaPoint, Camera, Intersection, BundleAdjust, and ControlPoints dependencies.

**Tech Stack:** C++17, CMake, GTest, OpenCV, PlaPoint/PlaMatrix, Ceres/BundleAdjust, Qt 6 only in project adapters.

**Repository rule:** Do not create implementation commits unless the user explicitly requests them. The commit steps normally required by the planning template are replaced with status/diff checkpoints.

---

### Task 1: Move SfM-owned tests into the module

**Files:**
- Create: `src/core/sfm/test/CMakeLists.txt`
- Move: `tests/test_sfm_filter.cpp` → `src/core/sfm/test/test_sfm_filter.cpp`
- Move: `tests/test_sparse_point_cloud_processor.cpp` → `src/core/sfm/test/test_sparse_point_cloud_processor.cpp`
- Move: `tests/test_sparse_point_cloud_workspace.cpp` → `src/core/sfm/test/test_sparse_point_cloud_workspace.cpp`
- Move: `tests/test_sfm_params.cpp` → `src/core/sfm/test/test_sfm_params.cpp`
- Move: `tests/test_sfm_quality_report.cpp` → `src/core/sfm/test/test_sfm_quality_report.cpp`
- Move: `tests/test_sfm_pipeline.cpp` → `src/core/sfm/test/test_sfm_pipeline.cpp`
- Move: `tests/test_sfm_prior_tracks.cpp` → `src/core/sfm/test/test_sfm_prior_tracks.cpp`
- Move: `tests/test_ba_input_builder.cpp` → `src/core/sfm/test/test_ba_input_builder.cpp`
- Move: `tests/test_initial_sparse_triangulator.cpp` → `src/core/sfm/test/test_initial_sparse_triangulator.cpp`
- Move: `tests/test_reference_terrain_prior.cpp` → `src/core/sfm/test/test_reference_terrain_prior.cpp`
- Move: `tests/test_multiview_track_builder.cpp` → `src/core/sfm/test/test_multiview_track_builder.cpp`
- Modify: `src/core/sfm/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Test: `tests/test_source_contracts.cpp`

- [ ] **Step 1: Add a failing source contract for module-owned test placement**

Add a contract that requires `src/core/sfm/test/CMakeLists.txt`, rejects the listed source files under root `tests/`, and permits cross-module tests such as `test_sfm_pair_planner.cpp` to remain at root.

```cpp
TEST(SfmModuleContractTest, ModuleOwnedTestsLiveBesideSfm)
{
    EXPECT_TRUE(QFileInfo(QStringLiteral("src/core/sfm/test/CMakeLists.txt")).exists());
    EXPECT_FALSE(QFileInfo(QStringLiteral("tests/test_sfm_pipeline.cpp")).exists());
    EXPECT_FALSE(QFileInfo(QStringLiteral("tests/test_sparse_point_cloud_processor.cpp")).exists());
    EXPECT_TRUE(QFileInfo(QStringLiteral("tests/test_sfm_pair_planner.cpp")).exists());
}
```

- [ ] **Step 2: Run the contract and observe the expected failure**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_source_contracts -j 4
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R '^SfmModuleContractTest\.ModuleOwnedTestsLiveBesideSfm$'
```

Expected: FAIL because the module test directory is absent and root test files still exist.

- [ ] **Step 3: Move tests and create module-local CMake registration**

Use the existing target names so CTest names stay stable. Add to `src/core/sfm/CMakeLists.txt`:

```cmake
if(BUILD_TESTS)
    add_subdirectory(test)
endif()
```

Remove only the migrated target blocks from `tests/CMakeLists.txt`.

- [ ] **Step 4: Configure, build, and run all moved targets**

Run CMake configure, build the moved targets, then run:

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R '^(Sfm|SparsePointCloud|InitialSparse|ReferenceTerrain|MultiView|BaInput)'
```

Expected: all selected tests pass and the source contract turns green.

- [ ] **Step 5: Checkpoint the diff**

Run `git diff --check` and confirm changes are limited to test relocation and registration.

### Task 2: Delete dead types and compatibility aliases

**Files:**
- Delete: `src/core/sfm/common/SparsePointCloud.h`
- Delete: `src/core/sfm/common/PhotogrammetryPointAttributes.h`
- Delete: `src/core/sfm/filtering/SfmPointCloudFilter.h`
- Delete: `src/core/sfm/filtering/SfmPointCloudFilter.cpp`
- Rename: `src/core/sfm/triangulation/InitialSparsePointCloudTriangulator.h` → `InitialSparsePointFilter.h`
- Rename: `src/core/sfm/triangulation/InitialSparsePointCloudTriangulator.cpp` → `InitialSparsePointFilter.cpp`
- Modify: `src/core/sfm/filtering/SparsePointCloudProcessor.h/.cpp`
- Modify: `src/core/sfm/TriangulationService.cpp`
- Modify: `src/core/sfm/CMakeLists.txt`
- Modify: module tests and root source contracts

- [ ] **Step 1: Add failing contracts for removed APIs**

Require the deleted files and symbols to be absent:

```cpp
expectNotContainsAll(processorHeader, {
    "SparseCloudLocalOptimOptions",
    "SparseCloudLocalOptimResult",
    "localOptim(",
});
EXPECT_FALSE(QFileInfo(QStringLiteral("src/core/sfm/filtering/SfmPointCloudFilter.h")).exists());
EXPECT_TRUE(QFileInfo(QStringLiteral("src/core/sfm/triangulation/InitialSparsePointFilter.h")).exists());
```

- [ ] **Step 2: Run and observe RED**

Expected: contract fails on existing aliases and old files.

- [ ] **Step 3: Rename the real type without an alias**

Expose only:

```cpp
class InitialSparsePointFilter
{
public:
    static InitialSparseTriangulationResult filter(
        const std::vector<Camera> &cameras,
        const std::vector<BATrack> &tracks,
        const InitialSparseTriangulationOptions &options = {});
};
```

Update all callers and tests to `InitialSparsePointFilter::filter`.

- [ ] **Step 4: Remove dead filters and local-optimization aliases**

Replace the only production `localOptim` call with `spatialCleanup`, update tests, and remove obsolete target sources.

- [ ] **Step 5: Build and run affected tests**

Build `sfm`, `test_initial_sparse_triangulator`, `test_sparse_point_cloud_processor`, and `test_source_contracts`; run their discovered tests.

### Task 3: Make PlaPoint the only generic spatial backend

**Files:**
- Modify: `src/core/sfm/filtering/SparsePointCloudWorkspace.h/.cpp`
- Modify: `src/core/sfm/filtering/SparsePointCloudProcessor.cpp`
- Modify: `src/core/sfm/graph/ObservationNetworkBuilder.h/.cpp`
- Create: `src/core/sfm/common/DisjointSet.h`
- Modify: `src/core/sfm/tracks/MultiViewTrackBuilder.cpp`
- Test: module workspace/processor/multiview tests
- Test: create `src/core/sfm/test/test_observation_network_builder.cpp`

- [ ] **Step 1: Add failing tests for PlaPoint-backed behavior**

Cover stable removed indices for statistical and radius filtering, and compare KDTree network edges with the current expected graph.

```cpp
TEST(ObservationNetworkBuilderTest, KdTreeKeepsNearestMatchedNeighbors)
{
    const ObservationNetwork result = ObservationNetworkBuilder::build(names, edges, gps, config);
    EXPECT_EQ(edgeKeys(result), expectedEdgeKeys);
}
```

- [ ] **Step 2: Observe RED for the new structural requirement**

Add a source contract rejecting `ObservationNetworkBuilder::KDNode`, `buildKD`, and `queryKD`; run it before production changes.

- [ ] **Step 3: Replace the custom KDTree**

Use:

```cpp
using SpatialTree = plapoint::search::SpatialKdTree<2, double>;
```

Convert valid GPS coordinates to a local metric/equivalent coordinate representation exactly as the old implementation did before neighbor lookup. Preserve sorting and tie-breaking.

- [ ] **Step 4: Share one internal DisjointSet**

Move path compression and union-by-rank into `common/DisjointSet.h`; use it from MST and multiview track construction without exposing it outside `sfm_core`.

- [ ] **Step 5: Verify numerical equivalence**

Run workspace, processor, observation network, and multiview track tests. Confirm exact removed indices and graph edge sets.

### Task 4: Introduce pure C++ projection and triangulation geometry

**Files:**
- Create: `src/core/sfm/geometry/ProjectionGeometry.h/.cpp`
- Create: `src/core/sfm/geometry/TriangulationQuality.h/.cpp`
- Create: `src/core/sfm/geometry/OpenCvCameraAdapter.h/.cpp`
- Modify: `src/core/sfm/triangulation/Triangulator.cpp`
- Modify: `src/core/sfm/triangulation/InitialSparsePointFilter.cpp`
- Modify: `src/core/sfm/pose/PnpSolver.cpp`
- Modify: `src/core/sfm/BaInputBuilder.cpp`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.cpp`
- Test: create three geometry test files under `src/core/sfm/test`

- [ ] **Step 1: Add characterization tests before extraction**

Record current outputs for normal projection, signed projection fallback, flipped depth axes, U/V sign changes, pair triangulation angle, and RMS reprojection error.

```cpp
TEST(ProjectionGeometryTest, SignedFallbackMatchesCurrentCameraProjection)
{
    const ProjectionResult result = projectForReprojection(camera, worldPoint);
    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.pixel[0], expectedU, 1e-12);
    EXPECT_NEAR(result.pixel[1], expectedV, 1e-12);
}
```

- [ ] **Step 2: Run characterization tests against the wished-for API and observe RED**

Expected: compile failure because the geometry headers do not exist.

- [ ] **Step 3: Implement minimal pure C++ geometry APIs**

Use standard C++ result structures and OpenCV only inside `OpenCvCameraAdapter`; no Qt headers or types.

- [ ] **Step 4: Migrate one caller at a time**

Order: `InitialSparsePointFilter`, `BaInputBuilder`, `Triangulator`, `PnpSolver`, then initial-pair logic in `IncrementalSfm`. After each caller, run its focused tests.

- [ ] **Step 5: Reject duplicated local helpers**

Add source contracts that reject local `reprojectionErrorPx` and repeated depth-sign camera matrix construction outside `geometry/`.

### Task 5: Split BaInputBuilder project responsibilities

**Files:**
- Move: `BaInputBuilder.h/.cpp` → `project/BaInputBuilder.h/.cpp`
- Create: `project/ProjectMatchInputReader.h/.cpp`
- Create: `project/BaTrackBuilder.h/.cpp`
- Create: `project/SurveyControlBaAdapter.h/.cpp`
- Create: `project/MarkerBaAdapter.h/.cpp`
- Modify: GUI/CLI callers and module CMake
- Test: split `test_ba_input_builder.cpp` into focused module tests

- [ ] **Step 1: Add behavior snapshots for BA input**

For fixed metadata fixtures, assert camera count, track observations, initial points, marker flags, control point sigmas, scale bars, and error codes.

- [ ] **Step 2: Add compile-time interface tests for the new components and observe RED**

Each component receives typed data and has one responsibility. JSON is permitted only in `ProjectMatchInputReader`, `SurveyControlBaAdapter`, and `MarkerBaAdapter` because these belong to `sfm_project`.

- [ ] **Step 3: Extract project match reading**

Move feature-index sidecar and project metadata parsing without changing key names or fallback order.

- [ ] **Step 4: Extract survey and marker adapters**

Move GCP/checkpoint/scale-bar parsing and marker observation conversion verbatim, then replace duplicated triangulation with `TriangulationQuality`/`ProjectionGeometry`.

- [ ] **Step 5: Reduce BaInputBuilder to orchestration**

`buildBaInputFromMeta` validates input, invokes adapters, and assembles `BaInputBuildResult`; it must not contain local geometry algorithms.

- [ ] **Step 6: Verify GUI/CLI JSON compatibility**

Run module BA tests, bundle-adjust CLI contract tests, and GUI project utility tests.

### Task 6: Split IncrementalSfm while preserving its public API

**Files:**
- Create: `pipeline/InitialPairInitializer.h/.cpp`
- Create: `pipeline/ImageRegistrationEngine.h/.cpp`
- Create: `pipeline/KnownPoseReconstructor.h/.cpp`
- Create: `pipeline/SfmBundleAdjustCoordinator.h/.cpp`
- Modify: `pipeline/IncrementalSfm.h/.cpp`
- Test: split existing SfM pipeline tests into component tests

- [ ] **Step 1: Add deterministic characterization fixtures**

Record selected initial pair, registered image IDs, point/track associations, BA statistics, permanently failed images, and stable RANSAC behavior for existing synthetic fixtures.

- [ ] **Step 2: Define shared state without copying reconstruction**

Use a non-owning context:

```cpp
struct IncrementalSfmContext
{
    SfmReconstruction &reconstruction;
    CorrespondenceGraph &correspondenceGraph;
    const IncrementalSfmOptions &options;
};
```

Components may hold references only for the duration of `run()`.

- [ ] **Step 3: Extract InitialPairInitializer**

Move candidate selection, E/F/H estimation, relative-pose recovery, trial reset, and scoring without reordering candidate iteration or RANSAC calls.

- [ ] **Step 4: Extract ImageRegistrationEngine**

Move next-image selection, PnP preparation, sequence pose guesses, registration validation, defer/retry bookkeeping, and visibility-cache updates.

- [ ] **Step 5: Extract KnownPoseReconstructor**

Move the fixed-known-pose path, known-pose soft priors, PnP refinement, and alignment to supplied pose priors.

- [ ] **Step 6: Extract SfmBundleAdjustCoordinator**

Move local/global BA, control network integration, BA backend diagnostics, negative-depth filtering, and retriangulation coordination.

- [ ] **Step 7: Reduce IncrementalSfm to orchestration**

Keep its current public methods and result type, but make `run()` coordinate components and assemble the final result.

- [ ] **Step 8: Run deterministic equivalence tests after every extraction**

Any mismatch in registered IDs, selected seed, tracks, or BA configuration blocks progress to the next component.

### Task 7: Remove Qt from quality and algorithm APIs

**Files:**
- Replace: `quality/SfmQualityReport.h/.cpp`
- Create: `quality/SfmQualityMetrics.h/.cpp`
- Create: `quality/SfmError.h`
- Create: `project/SfmQualityJsonSerializer.h/.cpp`
- Modify: `AerialTriangulationService.cpp` and project/CLI callers
- Test: module metrics tests and root JSON contract tests

- [ ] **Step 1: Add a failing no-Qt source contract**

Scan algorithm directories and reject Qt includes/types:

```cpp
expectNotContainsAll(algorithmSources, {
    "#include <Q",
    "QString",
    "QJsonObject",
    "QJsonArray",
    "QObject",
});
```

- [ ] **Step 2: Add metrics-to-JSON golden tests**

Capture the current JSON keys, units, warning names, and numeric values before moving serialization.

- [ ] **Step 3: Introduce pure C++ metrics and errors**

`SfmQualityMetrics` contains standard containers and numeric fields. `SfmErrorCode` and `std::string diagnostic` replace algorithm-layer localized errors.

- [ ] **Step 4: Move Qt JSON serialization to sfm/project**

Serialize exactly the current keys and values; keep translated UI text outside algorithms.

- [ ] **Step 5: Run no-Qt and JSON compatibility tests**

The source contract and all existing report consumers must pass.

### Task 8: Split CMake targets and enforce dependency direction

**Files:**
- Modify: `src/core/sfm/CMakeLists.txt`
- Modify: callers linking `sfm`
- Modify: module test CMake
- Test: root source contracts

- [ ] **Step 1: Add failing CMake source contracts**

Require `sfm_core`, `sfm_postprocess`, `sfm_project`, and an `INTERFACE` target `sfm`; reject Qt links from the first two targets.

- [ ] **Step 2: Define sfm_core**

Include common, geometry, graph, tracks, pose, triangulation, reconstruction, and pipeline sources. Link Camera/Intersection/BundleAdjust/ControlPoints/OpenCV as required, but no Qt.

- [ ] **Step 3: Define sfm_postprocess**

Include filtering and pure metrics. Link `sfm_core` and PlaPoint; no Qt.

- [ ] **Step 4: Define sfm_project**

Include project JSON adapters, preview triangulation service, and JSON serializers. Link Qt Core/Gui, common project libraries, `sfm_core`, and `sfm_postprocess`.

- [ ] **Step 5: Keep sfm as an INTERFACE aggregation target**

```cmake
add_library(sfm INTERFACE)
target_link_libraries(sfm INTERFACE sfm_core sfm_postprocess sfm_project)
```

This target provides build aggregation only; no forwarding headers or C++ aliases are permitted.

- [ ] **Step 6: Build every consumer and verify link boundaries**

Build GUI, all CLI targets, and all module tests. Inspect generated dependencies or source contracts to confirm no Qt link leaks into algorithm targets.

### Task 9: Documentation and full verification

**Files:**
- Modify: `src/core/sfm/README.md`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: any directly affected SfM workflow documentation

- [ ] **Step 1: Update module documentation**

Document final directories, target dependencies, PlaPoint ownership, no-Qt rules, test placement, and the distinction between official SfM and two-view preview filtering.

- [ ] **Step 2: Run static checks**

Run searches for deleted files, old aliases, duplicate geometry helpers, Qt types in algorithm directories, and root-owned SfM tests. Run `git diff --check`.

- [ ] **Step 3: Configure and build the complete project**

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release -j 4
```

Expected: exit code 0.

- [ ] **Step 4: Run focused SfM and project tests**

Run module tests plus aerial triangulation, bundle adjust, GUI project utilities, CLI contracts, and source contracts.

- [ ] **Step 5: Run full CTest**

Set the existing runtime `PATH` and `PROJ_DATA`, then run:

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure
```

Expected: all enabled tests pass; the configured CUDA benchmark may remain disabled.

- [ ] **Step 6: Review the final diff against the design**

Confirm every requirement in `docs/superpowers/specs/2026-07-14-sfm-module-refactor-design.md` is implemented, and report any numerical or dependency deviation instead of hiding it.
