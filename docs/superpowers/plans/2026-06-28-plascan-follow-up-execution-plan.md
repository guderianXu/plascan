# PlaScan Follow-Up Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring PlaScan's reconstruction pipeline closer to production photogrammetry behavior by reusing completed matching work, adding guided rematching as a controlled second pass, improving dense depth/mesh quality, and making GUI workflows cancellable and diagnosable.

**Architecture:** Keep the current C++/Qt/CMake structure. Add small core services for SfM/MVS quality decisions, then let GUI managers consume metadata and reports instead of rescanning output directories or rerunning expensive stages blindly.

**Tech Stack:** C++17, Qt6, OpenCV, CMake, GTest, CUDA optional MVS/feature acceleration, Python comparison scripts.

---

## File Structure

- `src/core/pipeline/GuidedRematchService.h/.cpp`
  - Owns append-only guided rematch candidate generation and merge behavior.
  - It must not overwrite stable existing matches.
- `src/core/pipeline/SFMService.h/.cpp`
  - Orchestrates opt-in guided rematching, match reuse, report generation, and second-pass track rebuild.
- `src/core/pipeline/ReconstructionPrerequisiteReport.h/.cpp`
  - Decides whether upstream data is missing, incomplete, failed, or already completed but unusable.
- `src/core/sfm/quality/*`
  - Stores SfM graph, track, BA, registration, and sparse cloud health metrics.
- `src/core/mvs/MvsQualityReport.h/.cpp`
  - Stores depth/fusion/mesh quality metrics and gates MVS outputs before downstream products.
- `src/core/mvs/DepthMapGenerator.cpp`
  - Handles source view selection, memory-adaptive scheduling, depth post-processing, cancellation, and metadata writes.
- `src/core/mvs/DepthMapFusion.cpp`
  - Handles confidence-aware dense cloud fusion and color/confidence preservation.
- `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
  - Runs dense reconstruction asynchronously and registers depth/dense/mesh outputs.
- `src/gui/project/manager/ProjectTerrainProductsManager.cpp`
  - Runs DEM/DOM asynchronously and consumes explicit dense output paths.
- `src/gui/widgets/DataTreeWidget.cpp`
  - Displays project outputs sorted by filename and refreshed from metadata.
- `testData/compare_point_cloud_to_lidar.py`
  - Compares PlaScan dense products with Metashape/LiDAR references using robust distance statistics.
- `tests/test_guided_rematch_service.cpp`
  - Unit tests for guided rematch eligibility, candidate generation, and append-only merge.
- `tests/test_reconstruction_prerequisites.cpp`
  - Unit tests for matching/SfM prerequisite decisions.
- `tests/test_sfm_quality_report.cpp`
  - Unit tests for SfM quality gates and JSON summaries.
- `tests/test_mvs_*`
  - Unit tests for MVS source selection, memory settings, manifest records, and product identity.

---

## Task 1: Finish Guided Rematching Integration

**Purpose:** Turn the current guided rematch service from an isolated unit into an actual opt-in second pass inside SfM.

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.h`
- Modify: `E:/code/plascan/src/core/pipeline/GuidedRematchService.cpp`
- Test: `E:/code/plascan/tests/test_guided_rematch_service.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add a failing test for opt-in second-pass execution**

Add a test that requires `SFMService.cpp` to call guided rematch merge only when `enableGuidedRematching=true`.

```cpp
TEST(SfmSparseResultMetadataTest, GuidedRematchingIsOptInAndAppendOnly)
{
    const QString source =
        readTextFile(QDir::fromNativeSeparators("E:/code/plascan/src/core/pipeline/SFMService.cpp"));

    EXPECT_TRUE(source.contains("opts.enableGuidedRematching"));
    EXPECT_TRUE(source.contains("generateGuidedRematchCandidates"));
    EXPECT_TRUE(source.contains("mergeGuidedRematchMatches"));
    EXPECT_TRUE(source.contains("replacesExistingMatch = false"));
}
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmSparseResultMetadataTest.GuidedRematchingIsOptInAndAppendOnly' `
  -Jobs 8
```

Expected: FAIL until `SFMService.cpp` contains the real integration calls.

- [ ] **Step 3: Implement second-pass orchestration**

In `SFMService.cpp`, after initial SfM has registered cameras and built weak-pair diagnostics:

```cpp
if (opts.enableGuidedRematching)
{
    // Build GuidedRematchInput for each weak but eligible registered pair.
    // Generate append-only candidates.
    // Merge candidates into the existing pair match set.
    // Rebuild tracks and run BA once more.
    // Record guided_matching_enabled=true and guided_matching_added_matches.
}
else
{
    // Keep current behavior: report guided matching disabled and do not rerun matching.
}
```

- [ ] **Step 4: Run guided rematch and prerequisite tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematchServiceTest|SfmGuidedMatchPlannerTest|ReconstructionPrerequisiteReport|SfmSparseResultMetadataTest' `
  -Jobs 8
```

Expected: PASS.

---

## Task 2: Stop Redundant Matching During Aerial Triangulation

**Purpose:** If the user already completed matching, aerial triangulation must consume existing match results and report unusable matches instead of starting another expensive matching pass.

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.cpp`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `E:/code/plascan/src/gui/dialogs/FeaturePairPlanner.cpp`
- Test: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`
- Test: `E:/code/plascan/tests/test_feature_pair_planner.cpp`

- [ ] **Step 1: Add a failing prerequisite test**

Add a case where a completed matching pass has zero usable SfM edges but many failed/skipped pairs.

```cpp
TEST(ReconstructionPrerequisiteReportTest, CompletedMatchingWithNoSfmEdgesDoesNotRequestFullRematch)
{
    ReconstructionPrerequisiteReport report;
    report.matchingPassCompleted = true;
    report.usableSfmEdgeCount = 0;
    report.failedPairCount = 592;
    report.skippedPairCount = 984;

    const auto decision = inspectReconstructionPrerequisites(report);

    EXPECT_FALSE(decision.shouldRunFullMatching);
    EXPECT_EQ(decision.status, ReconstructionPrerequisiteStatus::CompletedButUnusable);
}
```

- [ ] **Step 2: Run the prerequisite tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|FeaturePairPlanner' `
  -Jobs 8
```

Expected: FAIL until the decision distinguishes missing input from completed-but-unusable input.

- [ ] **Step 3: Implement no-rematch decision and GUI message**

In `MenuWorkflowController.cpp`, show a targeted message:

```cpp
tr("匹配阶段已完成，但没有可用于空三的连接边。请打开匹配诊断查看 failed/skipped pair、几何内点数和连通分量；空三不会自动重新跑完整匹配。")
```

- [ ] **Step 4: Re-run tests**

Run the same command from Step 2.

Expected: PASS.

---

## Task 3: Add PlaScan vs Metashape Dense Cloud Diagnostics

**Purpose:** Explain why PlaScan dense products are rougher by comparing against the Metashape reference point cloud using reproducible metrics.

**Files:**
- Modify: `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
- Create: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_dense_comparison.md`
- Test: `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py`

- [ ] **Step 1: Add a failing Python test for vertical roughness gates**

```python
def test_vertical_gate_marks_rough_cloud_as_failed(tmp_path):
    report = {
        "vertical": {
            "p95_abs": 1.20,
            "median_abs": 0.35,
        },
        "gates": {
            "vertical_p95_max": 0.50,
            "vertical_median_max": 0.15,
        },
    }

    assert classify_vertical_quality(report) == "fail"
```

- [ ] **Step 2: Run the Python test**

Run:

```powershell
python -m pytest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py -q
```

Expected: FAIL until the helper exists.

- [ ] **Step 3: Implement vertical quality classification**

In `compare_point_cloud_to_lidar.py`:

```python
def classify_vertical_quality(report: dict) -> str:
    vertical = report.get("vertical", {})
    gates = report.get("gates", {})
    if vertical.get("p95_abs", 0.0) > gates.get("vertical_p95_max", float("inf")):
        return "fail"
    if vertical.get("median_abs", 0.0) > gates.get("vertical_median_max", float("inf")):
        return "fail"
    return "pass"
```

- [ ] **Step 4: Generate a comparison report**

Run:

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --output-json E:/code/plascan/docs/reports/agisoft_aerial_gcps_dense_comparison.json
```

Expected: JSON contains point count, distance P50/P84/P95, vertical roughness, and pass/fail gates.

---

## Task 4: Improve MVS Depth Quality Before Fusion

**Purpose:** Reduce isolated red speckles, depth spikes, and rough mesh artifacts by filtering unreliable depth before dense fusion.

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsTypes.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.cpp`
- Test: `E:/code/plascan/tests/test_mvs_types.cpp`
- Test: `E:/code/plascan/tests/test_mvs_pipeline.cpp`

- [ ] **Step 1: Add a failing test for depth result metadata**

```cpp
TEST(MvsTypesTest, DepthResultRecordsConfidenceAndMaskPaths)
{
    MvsDepthResult result;
    result.rawDepthPath = "depth_001.raw";
    result.rawConfidencePath = "confidence_001.raw";
    result.validMaskPath = "mask_001.png";
    result.previewPath = "depth_001.png";

    EXPECT_FALSE(result.rawConfidencePath.empty());
    EXPECT_FALSE(result.validMaskPath.empty());
}
```

- [ ] **Step 2: Run MVS type and manifest tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsTypes|MvsWorkspaceManifest|MvsDepthPostprocess|MvsQualityReport' `
  -Jobs 8
```

Expected: FAIL until metadata paths are represented consistently.

- [ ] **Step 3: Add post-filtering gates**

In `DepthMapGenerator.cpp`, apply:

- minimum confidence threshold
- local speckle component removal
- small isolated depth spike removal
- minimum source-view support where available

- [ ] **Step 4: Re-run focused tests**

Run the command from Step 2.

Expected: PASS.

---

## Task 5: Fix Dense Cloud and Mesh Product Identity

**Purpose:** The `3D模型` tree node must show real mesh files as meshes, dense cloud files as point clouds, and never silently show dense cloud as a mesh.

**Files:**
- Modify: `E:/code/plascan/src/gui/project/support/ProjectResultRecords.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectSupportUtils.cpp`
- Modify: `E:/code/plascan/src/gui/widgets/DataTreeWidget.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add a failing product identity test**

```cpp
TEST(ProjectResultRecordsTest, MeshRecordRequiresFaces)
{
    ProjectModelRecord record;
    record.path = "model_from_mesh.ply";
    record.vertexCount = 3286949;
    record.faceCount = 0;

    EXPECT_FALSE(isMeshProduct(record));
    EXPECT_TRUE(isDenseCloudProduct(record));
}
```

- [ ] **Step 2: Run GUI project tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ProjectResultRecords|ProjectSupportUtils|GuiProjectUtils|DataTree' `
  -Jobs 8
```

Expected: FAIL until face count and product role are checked.

- [ ] **Step 3: Implement strict product role logic**

Rules:

- `faceCount > 0` means mesh.
- `vertexCount > 0 && faceCount == 0` means dense point cloud.
- UI labels must show `[V:n F:m]` for meshes and `[点:n]` for point clouds.

- [ ] **Step 4: Re-run GUI tests**

Run the command from Step 2.

Expected: PASS.

---

## Task 6: Make DEM/DOM and Mesh Jobs Truly Async

**Purpose:** Avoid GUI freezes and stale metadata consumption during DEM/DOM and mesh generation.

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add a source-level failing test for QPointer guarded callbacks**

```cpp
TEST(GuiAsyncSafetyTest, TerrainProductsUseGuardedAsyncCallbacks)
{
    const QString source =
        readTextFile(QDir::fromNativeSeparators("E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp"));

    EXPECT_TRUE(source.contains("QPointer<ProjectTerrainProductsManager>"));
    EXPECT_TRUE(source.contains("QtConcurrent::run"));
    EXPECT_FALSE(source.contains("QtConcurrent::run([this"));
}
```

- [ ] **Step 2: Run the async safety test**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuiAsyncSafetyTest' `
  -Jobs 8
```

Expected: FAIL until callbacks are guarded.

- [ ] **Step 3: Refactor async calls**

Use this pattern:

```cpp
QPointer<ProjectTerrainProductsManager> guard(this);
QtConcurrent::run([guard, task = std::move(task)]() mutable
{
    const auto result = task();
    QMetaObject::invokeMethod(qApp, [guard, result]()
    {
        if (!guard)
        {
            return;
        }
        guard->handleTaskFinished(result);
    }, Qt::QueuedConnection);
});
```

- [ ] **Step 4: Re-run GUI async tests**

Run the command from Step 2.

Expected: PASS.

---

## Task 7: Full Windows CUDA Build and Focused Regression

**Purpose:** Validate the first follow-up batch without claiming the whole six-phase roadmap is complete.

**Files:**
- No source files changed in this task.

- [ ] **Step 1: Build with the fixed Windows CUDA script**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -Jobs 8
```

Expected: build completes.

- [ ] **Step 2: Run focused regression tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematch|ReconstructionPrerequisite|SfmQuality|MvsQuality|MvsDepth|FusionDepth|MvsWorkspace|ProjectResult|GuiProject|FeaturePair' `
  -Jobs 8
```

Expected: focused tests pass. If any fail, record exact test name and failure reason before continuing.

- [ ] **Step 3: Optional full ctest**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure
```

Expected: document pass/fail. If historical terrain failures appear, list them separately and do not claim full pass.

---

## Task 8: Documentation and Release Notes

**Purpose:** Keep the project understandable after the optimization batch.

**Files:**
- Modify: `E:/code/plascan/CHANGELOG.md`
- Modify or create: `E:/code/plascan/docs/releases/v1.1.7.md`
- Modify: `E:/code/plascan/docs/PROJECT_ARCHITECTURE.md`
- Modify: `E:/code/plascan/README.md`

- [ ] **Step 1: Update changelog**

Add a section:

```markdown
## v1.1.7

### 新增
- Guided rematching second-pass infrastructure for registered weak pairs.

### 优化
- Aerial triangulation now reuses completed matching results and reports unusable match graphs instead of rerunning full matching.
- MVS depth products record confidence/mask metadata for safer fusion.

### 修复
- Dense cloud and mesh outputs are classified by real geometry content.
- DEM/DOM and mesh jobs avoid GUI-thread blocking and stale metadata.

### 验证
- `GuidedRematch|ReconstructionPrerequisite|SfmQuality|MvsQuality|GuiProject` focused tests.

### 已知问题
- Full production-quality parity with Metashape is not complete; dense reconstruction quality still needs dataset-level tuning.
```

- [ ] **Step 2: Create release detail document**

Create `docs/releases/v1.1.7.md` with sections:

- 新增
- 优化
- 修复
- 验证
- 已知问题

- [ ] **Step 3: Update architecture docs**

Mention:

- guided rematch service
- match prerequisite report
- MVS quality report
- product identity rules

- [ ] **Step 4: Run a docs-only status check**

Run:

```powershell
git status --short
```

Expected: only intended source, tests, and docs files are listed.

---

## Self-Review

- Spec coverage: Covers matching reuse, guided rematching, PlaScan vs Metashape diagnostics, MVS depth quality, dense cloud/mesh identity, async GUI safety, verification, and release docs.
- Placeholder scan: No task uses TBD/TODO/fill-in placeholders. Each task includes exact files, tests, commands, and expected results.
- Type consistency: `GuidedRematch*`, `ReconstructionPrerequisite*`, `Mvs*`, and project product records are named consistently with the current PlaScan modules.

