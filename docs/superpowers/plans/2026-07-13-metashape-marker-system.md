# PlaScan Metashape 式完整标记点系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PlaScan GUI 中实现 Metashape 式完整标记点系统，包括照片右键量测、控制点/检查点/无坐标标记、比例尺、自动标靶检测、预测投影、CRS、空三前先验轨迹、绝对定向、CPU/CUDA BA 和兼容标靶打印。

**Architecture:** 新建无 Qt Widgets 依赖的 `control_points` 核心库，以 `MarkerSet` 为唯一运行时真源并原子保存到 `assets/control_points/marker_set.json`。GUI 通过命令和 repository 修改标记数据，主照片画布负责快速右键量测，双影像聚焦量测器负责批量复核；SfM 通过正式 `PriorTrack` 接口消费人工/标靶轨迹，控制网络在相对 SfM 后执行绝对定向和约束 BA。

**Tech Stack:** C++17、CMake、Qt6 Core/Gui/Widgets/Concurrent、OpenCV、GDAL/OGR、AprilTag 3（vcpkg `apriltag`）、现有 SfM/BundleAdjust/CUDA、GTest。

## Global Constraints

- 实现必须符合 `docs/superpowers/specs/2026-07-13-metashape-marker-system-design.md`。
- `src/core/control_points` 不得依赖 Qt Widgets；GUI 只能通过公开 service/repository 接口调用核心能力。
- 标记投影统一保存原始未旋转影像像素坐标，不能保存视图坐标。
- `MarkerSet` 是唯一运行时数据模型；旧 `survey_control` 成功迁移后删除，不保留双写兼容层。
- 自动检测和打印兼容圆形 12/14/16/20 bit、AprilTag 16h5/25h9/36h10/36h11/Circle21h7/Standard41h12/Standard52h13，以及非编码标靶。
- 所有影像量测、自动检测和预测必须应用项目蒙版。
- 控制点参与绝对定向和 BA；检查点只计算误差；检查比例尺只计算误差。
- GUI 长任务必须后台执行，具有真实进度、取消和关闭窗口安全性。
- 每个任务严格执行 RED -> GREEN -> REFACTOR；提交前运行该任务相关测试。
- C++ 私有成员使用 `_lowerCamel`，Allman 花括号，新增复杂逻辑添加简短中文注释。

---

## File Map

### New core module

```text
src/core/control_points/
├── CMakeLists.txt
├── README.md
├── model/MarkerTypes.h
├── model/MarkerSet.h/.cpp
├── model/MarkerSetValidator.h/.cpp
├── io/MarkerSetJson.h/.cpp
├── io/MarkerSetStore.h/.cpp
├── io/SurveyControlMigration.h/.cpp
├── io/MarkerCsv.h/.cpp
├── commands/MarkerChangeSet.h/.cpp
├── geometry/MarkerGeometry.h/.cpp
├── geometry/MarkerProjectionPredictor.h/.cpp
├── reference/CoordinateReference.h/.cpp
├── registration/PriorTrack.h
├── registration/ControlNetworkSolver.h/.cpp
├── detection/MarkerDetector.h
├── detection/AprilTagDetector.h/.cpp
├── detection/CircularTargetCodebook.h/.cpp
├── detection/CircularTargetDetector.h/.cpp
├── detection/NonCodedTargetDetector.h/.cpp
├── detection/DetectionMerger.h/.cpp
├── print/MarkerPdfWriter.h/.cpp
├── quality/MarkerQualityReport.h/.cpp
└── tests/*.cpp
```

### New GUI marker package

```text
src/gui/markers/
├── ProjectMarkerRepository.h/.cpp
├── MarkerWorkspaceController.h/.cpp
├── MarkerOverlayItems.h/.cpp
├── MarkerReferencePanel.h/.cpp
├── MarkerProjectionPanel.h/.cpp
├── MarkerFocusMeasurementDialog.h/.cpp
├── DetectMarkersDialog.h/.cpp
├── PrintMarkersDialog.h/.cpp
└── MarkerTaskRunner.h/.cpp
```

### Existing integration points

- `src/core/CMakeLists.txt`
- `cmake/PlascanPackages.cmake`
- `vcpkg.json`
- `src/core/sfm/pipeline/IncrementalSfm.h/.cpp`
- `src/core/aerial_triangulation/AerialTriangulationService.h/.cpp`
- `src/core/sfm/BaInputBuilder.h/.cpp`
- `src/core/bundle_adjust/*`
- `src/gui/widgets/CanvasWidget.h/.cpp`
- `src/gui/views/LayerRenderer.h/.cpp`
- `src/gui/widgets/DualImageViewer.h/.cpp`
- `src/gui/project/data/ProjectData.h/.cpp`
- `src/gui/main_window/MainWindow.h/.cpp/.ui`
- `src/gui/main_window/MenuWorkflowController.h/.cpp`
- `src/gui/cmake/GuiSources.cmake`
- `tests/CMakeLists.txt`

---

### Task 1: Establish the typed marker domain model

**Files:**
- Create: `src/core/control_points/CMakeLists.txt`
- Create: `src/core/control_points/model/MarkerTypes.h`
- Create: `src/core/control_points/model/MarkerSet.h`
- Create: `src/core/control_points/model/MarkerSet.cpp`
- Create: `src/core/control_points/model/MarkerSetValidator.h`
- Create: `src/core/control_points/model/MarkerSetValidator.cpp`
- Create: `src/core/control_points/tests/test_marker_set.cpp`
- Modify: `src/core/CMakeLists.txt`

**Interfaces:**
- Produces: `MarkerSet::addMarker`, `MarkerSet::upsertProjection`, `MarkerSet::removeProjection`, `MarkerSetValidator::validate`.
- Enforces: immutable UUID, unique non-empty label, one projection per marker/image UUID, finite original-pixel coordinates.

- [ ] **Step 1: Write the failing domain tests**

```cpp
TEST(MarkerSetTest, RejectsDuplicateLabelsAndProjectionPerImage)
{
    MarkerSet set;
    const MarkerId first = set.addMarker(QStringLiteral("GCP001"), MarkerRole::ControlPoint);
    EXPECT_THROW(set.addMarker(QStringLiteral("GCP001"), MarkerRole::TieMarker), MarkerModelError);

    MarkerProjection projection;
    projection.imageId = QStringLiteral("image-uuid-1");
    projection.xy = QPointF(120.25, 330.75);
    projection.state = ProjectionState::ManualPinned;
    set.upsertProjection(first, projection);
    set.upsertProjection(first, projection);

    ASSERT_EQ(set.marker(first).projections.size(), 1u);
}

TEST(MarkerSetTest, KeepsControlCheckAndTieRolesDistinct)
{
    EXPECT_TRUE(markerRoleUsesReferenceConstraint(MarkerRole::ControlPoint));
    EXPECT_FALSE(markerRoleUsesReferenceConstraint(MarkerRole::CheckPoint));
    EXPECT_FALSE(markerRoleUsesReferenceConstraint(MarkerRole::TieMarker));
}
```

- [ ] **Step 2: Configure and run the test to verify RED**

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release --target test_marker_set --config Release --parallel 24
```

Expected: compilation fails because `MarkerSet` and marker types do not exist.

- [ ] **Step 3: Implement the minimal strong model**

Use these public types and signatures:

```cpp
enum class MarkerRole { TieMarker, ControlPoint, CheckPoint };
enum class ProjectionState { ManualPinned, AutoDetected, Predicted, Blocked, Disabled };
enum class ScaleBarRole { Control, Check };

struct MarkerProjection
{
    QString imageId;
    QString imagePathSnapshot;
    QPointF xy;
    ProjectionState state = ProjectionState::Predicted;
    double sigmaPx = 1.0;
    double confidence = 0.0;
    double residualPx = std::numeric_limits<double>::quiet_NaN();
    QString source;
    QString imageContentSignature;
};

class MarkerSet
{
public:
    MarkerId addMarker(const QString &label, MarkerRole role);
    void removeMarker(const MarkerId &id);
    void renameMarker(const MarkerId &id, const QString &label);
    void upsertProjection(const MarkerId &id, const MarkerProjection &projection);
    void removeProjection(const MarkerId &id, const QString &imageId);
    const Marker &marker(const MarkerId &id) const;
};
```

- [ ] **Step 4: Run model tests**

Run `build/windows-vcpkg-cuda-release/src/core/control_points/test_marker_set.exe`.

Expected: all MarkerSet tests pass.

- [ ] **Step 5: Commit the domain model**

```powershell
git add src/core/CMakeLists.txt src/core/control_points
git commit -m "feat: add typed marker domain model"
```

---

### Task 2: Add atomic sidecar persistence and stable project image IDs

**Files:**
- Create: `src/core/control_points/io/MarkerSetJson.h/.cpp`
- Create: `src/core/control_points/io/MarkerSetStore.h/.cpp`
- Create: `src/core/control_points/tests/test_marker_set_store.cpp`
- Create: `src/gui/markers/ProjectMarkerRepository.h/.cpp`
- Modify: `src/gui/project/data/ProjectData.h/.cpp`
- Modify: `src/gui/project/io/ProjectIO.h/.cpp`
- Modify: `tests/test_project_data.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`

**Interfaces:**
- Consumes: Task 1 `MarkerSet`.
- Produces: `MarkerSetStore::load/save`, stable `image_uuid` for every project image, `ProjectMarkerRepository::open/save`.

- [ ] **Step 1: Write failing persistence and identity tests**

```cpp
TEST(MarkerSetStoreTest, SavesAtomicallyAndRoundTripsAllProjectionStates)
{
    QTemporaryDir dir;
    MarkerSet expected = makeMarkerSetWithEveryProjectionState();
    MarkerSetStore store(dir.filePath(QStringLiteral("marker_set.json")));
    ASSERT_TRUE(store.save(expected).ok);
    const auto loaded = store.load();
    ASSERT_TRUE(loaded.ok);
    EXPECT_EQ(loaded.markerSet, expected);
}

TEST(ProjectDataTest, AssignsStableUuidToImportedImages)
{
    ProjectData project;
    project.createProject(_tempProjectPath);
    project.addImages({QStringLiteral("E:/images/a.jpg")});
    const QString firstId = project.coreFilesMeta()["images"].toArray()[0].toObject()["image_uuid"].toString();
    project.saveProject();
    project.openProject(_tempProjectPath);
    EXPECT_EQ(project.coreFilesMeta()["images"].toArray()[0].toObject()["image_uuid"].toString(), firstId);
}
```

- [ ] **Step 2: Run tests and verify RED**

Expected: missing store/repository APIs and `image_uuid`.

- [ ] **Step 3: Implement schema v1 and atomic store**

Required store result:

```cpp
struct MarkerSetIoResult
{
    bool ok = false;
    MarkerSet markerSet;
    QString error;
    QString backupPath;
};

class MarkerSetStore
{
public:
    explicit MarkerSetStore(QString path);
    MarkerSetIoResult load() const;
    MarkerSetIoResult save(const MarkerSet &markerSet) const;
};
```

Use `QSaveFile`; save to `assets/control_points/marker_set.json`; metadata stores only `path`, `schema_version`, counts and `updated_at`. On corrupt JSON, return a read-only error and never overwrite the source.

- [ ] **Step 4: Run store and ProjectData tests**

Run:

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_marker_set_store test_project_data --config Release --parallel 24
ctest --test-dir build/windows-vcpkg-cuda-release -C Release -R "MarkerSetStore|ProjectData" --output-on-failure
```

- [ ] **Step 5: Commit persistence**

```powershell
git add src/core/control_points src/gui/markers src/gui/project src/gui/cmake tests/test_project_data.cpp
git commit -m "feat: persist marker sets in project assets"
```

---

### Task 3: Migrate legacy survey control and remove the old runtime schema

**Files:**
- Create: `src/core/control_points/io/SurveyControlMigration.h/.cpp`
- Create: `src/core/control_points/io/MarkerCsv.h/.cpp`
- Create: `src/core/control_points/tests/test_survey_control_migration.cpp`
- Modify: `src/core/qc/SurveyControlImport.h/.cpp`
- Modify: `src/gui/project/support/ProjectSurveyControl.h/.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

**Interfaces:**
- Consumes: MarkerSet store and image UUID mapping.
- Produces: one-shot `migrateSurveyControl`, CSV import/export directly against MarkerSet.

- [ ] **Step 1: Write migration tests**

```cpp
TEST(SurveyControlMigrationTest, ConvertsPinnedObservationsAndScaleBarIds)
{
    const QJsonObject legacy = makeLegacySurveyControlWithControlCheckAndScaleBar();
    const SurveyControlMigrationResult result = migrateSurveyControl(legacy, imageIdentityMap());
    ASSERT_TRUE(result.ok);
    EXPECT_EQ(result.markerSet.markers().size(), 2u);
    EXPECT_EQ(result.markerSet.scaleBars().size(), 1u);
    EXPECT_EQ(result.markerSet.markers()[0].projections[0].state, ProjectionState::ManualPinned);
}
```

Also test that metadata is unchanged when sidecar verification fails.

- [ ] **Step 2: Verify RED**

Expected: migration API missing.

- [ ] **Step 3: Implement transactional migration**

```cpp
struct SurveyControlMigrationResult
{
    bool ok = false;
    MarkerSet markerSet;
    QString error;
    int migratedMarkers = 0;
    int migratedProjections = 0;
    int migratedScaleBars = 0;
};
```

Project flow: convert -> save sidecar -> reload and validate -> update metadata -> remove `survey_control`. Do not retain read/write compatibility branches after successful migration.

- [ ] **Step 4: Run migration and existing import tests**

Run `ctest ... -R "SurveyControlMigration|SurveyControlImport|ProjectSurveyControl"`.

- [ ] **Step 5: Commit migration**

```powershell
git add src/core/control_points src/core/qc src/gui/project/support tests
git commit -m "feat: migrate survey controls to marker sets"
```

---

### Task 4: Add reversible marker edit commands

**Files:**
- Create: `src/core/control_points/commands/MarkerChangeSet.h/.cpp`
- Create: `src/core/control_points/tests/test_marker_change_set.cpp`
- Create: `src/gui/markers/MarkerUndoCommand.h/.cpp`
- Modify: `src/gui/markers/ProjectMarkerRepository.h/.cpp`

**Interfaces:**
- Produces: immutable before/after changes and a GUI `QUndoCommand` adapter.

- [ ] **Step 1: Write failing undo tests**

```cpp
TEST(MarkerChangeSetTest, RevertsProjectionReplacementExactly)
{
    MarkerSet set = makeMarkerWithProjection(QPointF(10.0, 20.0));
    const MarkerChangeSet change = MarkerChangeSet::replaceProjection(
        set, markerId(), imageId(), pinnedProjection(30.0, 40.0));
    change.apply(&set);
    EXPECT_EQ(set.marker(markerId()).projection(imageId()).xy, QPointF(30.0, 40.0));
    change.revert(&set);
    EXPECT_EQ(set.marker(markerId()).projection(imageId()).xy, QPointF(10.0, 20.0));
}
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement change sets and GUI adapter**

```cpp
class MarkerChangeSet
{
public:
    void apply(MarkerSet *set) const;
    void revert(MarkerSet *set) const;
    QString description() const;
};
```

Repository emits `markerSetChanged(revision, affectedMarkerIds)` after apply/revert and coalesces drag movement into one undo command.

- [ ] **Step 4: Run command tests**

- [ ] **Step 5: Commit commands**

```powershell
git add src/core/control_points/commands src/core/control_points/tests src/gui/markers
git commit -m "feat: add undoable marker edits"
```

---

### Task 5: Integrate photo right-click placement and marker overlays

**Files:**
- Create: `src/gui/markers/MarkerOverlayItems.h/.cpp`
- Create: `src/gui/markers/MarkerWorkspaceController.h/.cpp`
- Create: `tests/test_marker_canvas_interaction.cpp`
- Modify: `src/gui/widgets/CanvasWidget.h/.cpp`
- Modify: `src/gui/views/LayerRenderer.h/.cpp`
- Modify: `src/gui/main_window/MainWindow.h/.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: repository and change sets.
- Produces: `CanvasWidget::imageContextRequested(imagePath, originalPixel)`, marker overlay rendering and drag events.

- [ ] **Step 1: Write GUI interaction tests**

```cpp
TEST(MarkerCanvasInteractionTest, RightClickUsesOriginalPixelAfterViewRotation)
{
    CanvasWidget canvas;
    canvas.showImage(testImagePath());
    canvas.setViewRotationDegrees(90);
    QSignalSpy spy(&canvas, &CanvasWidget::imageContextRequested);
    QTest::mouseClick(canvas.viewport(), Qt::RightButton, Qt::NoModifier, QPoint(250, 180));
    ASSERT_EQ(spy.count(), 1);
    EXPECT_TRUE(pointInsideOriginalImage(spy.takeFirst().at(1).toPointF()));
}
```

Add tests for “new marker”, “place existing marker”, one projection per image, and drag coalescing.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement the confirmed context menu**

`CanvasWidget` maps viewport coordinates through `mapToScene`; because view rotation is applied to the QGraphicsView transform, scene coordinates remain original-image coordinates. Reject points outside `LayerRenderer::imageBounds()`.

Menu commands:

```cpp
enum class MarkerPhotoCommand
{
    AddNewMarker,
    PlaceExistingMarker,
    RemoveProjection,
    BlockProjection,
    UnblockProjection,
    OpenFocusMeasurement
};
```

Overlay colors: green manual, blue auto, gray predicted, red quality failure. Use a stable item size in screen pixels so zoom does not resize labels incoherently.

- [ ] **Step 4: Build and run GUI tests**

```powershell
cmake --build build/windows-vcpkg-cuda-release --target test_marker_canvas_interaction plascan_gui --config Release --parallel 24
build/windows-vcpkg-cuda-release/tests/test_marker_canvas_interaction.exe
```

- [ ] **Step 5: Commit right-click interaction**

```powershell
git add src/gui/markers src/gui/widgets/CanvasWidget.* src/gui/views/LayerRenderer.* src/gui/main_window src/gui/cmake tests
git commit -m "feat: add photo marker placement workflow"
```

---

### Task 6: Build marker/reference panels and focused measurement UI

**Files:**
- Create: `src/gui/markers/MarkerReferencePanel.h/.cpp`
- Create: `src/gui/markers/MarkerProjectionPanel.h/.cpp`
- Create: `src/gui/markers/MarkerFocusMeasurementDialog.h/.cpp`
- Create: `tests/test_marker_panels.cpp`
- Modify: `src/gui/widgets/DualImageViewer.h/.cpp`
- Modify: `src/gui/main_window/MainWindow.ui`
- Modify: `src/gui/main_window/MainWindow.h/.cpp`

**Interfaces:**
- Produces: compact marker table, editable reference fields, projection table, scale bars and candidate-image review dialog.

- [ ] **Step 1: Write panel state tests**

Verify role switching, XY/Z sigma editing, projection status badges, candidate ordering, confirm/block/disable actions, and repository revision refresh.

```cpp
TEST(MarkerFocusMeasurementDialogTest, ConfirmingPredictionCreatesPinnedProjection)
{
    MarkerFocusMeasurementDialog dialog;
    dialog.setContext(makePredictedCandidateContext());
    clickNamedButton(&dialog, "confirmMarkerProjectionButton");
    ASSERT_EQ(emittedChanges(dialog).size(), 1);
    EXPECT_EQ(emittedChanges(dialog)[0].afterState(), ProjectionState::ManualPinned);
}
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement panels without nested card layout**

Use tables/tabs and icon buttons. Double-clicking a marker opens focused measurement. `DualImageViewer` gains a marker mode that draws one reliable projection and one candidate without parsing `.match` files.

- [ ] **Step 4: Run panel tests and Playwright/computer-use screenshot QA**

Check 1920x1080, 1366x768 and 125% scaling; verify menus and labels do not overlap.

- [ ] **Step 5: Commit marker UI**

```powershell
git add src/gui/markers src/gui/widgets/DualImageViewer.* src/gui/main_window tests
git commit -m "feat: add marker reference and focus panels"
```

---

### Task 7: Implement marker triangulation, epipolar guidance and prediction

**Files:**
- Create: `src/core/control_points/geometry/MarkerGeometry.h/.cpp`
- Create: `src/core/control_points/geometry/MarkerProjectionPredictor.h/.cpp`
- Create: `src/core/control_points/tests/test_marker_geometry.cpp`
- Modify: `src/gui/markers/MarkerFocusMeasurementDialog.cpp`

**Interfaces:**
- Produces: `triangulateMarker`, `epipolarSearchBand`, `predictMarkerProjections`, `refineMarkerProjection`.

- [ ] **Step 1: Write synthetic camera tests**

```cpp
TEST(MarkerGeometryTest, PredictsOnlyPositiveDepthUnmaskedImages)
{
    const MarkerTriangulation triangulated = triangulateMarker(twoViewPinnedMarker(), cameras());
    ASSERT_TRUE(triangulated.valid);
    const auto predictions = predictMarkerProjections(triangulated.xyz, cameras(), masks());
    EXPECT_TRUE(containsImage(predictions, "visible-image"));
    EXPECT_FALSE(containsImage(predictions, "behind-camera"));
    EXPECT_FALSE(containsImage(predictions, "masked-image"));
}
```

Also cover weak intersection angle, one-observation epipolar band, boundary and uncertainty rejection.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement robust geometry**

```cpp
struct MarkerTriangulation
{
    bool valid = false;
    std::array<double, 3> xyz{};
    double rmsPx = 0.0;
    double minimumAngleDeg = 0.0;
    std::array<double, 9> covariance{};
    QString rejectionReason;
};
```

Use existing camera projection conventions and triangulation utilities. Prediction results remain `Predicted`; local refinement may create `AutoDetected` only after residual and mask gates pass.

- [ ] **Step 4: Run geometry tests**

- [ ] **Step 5: Commit geometry assistance**

```powershell
git add src/core/control_points/geometry src/core/control_points/tests src/gui/markers
git commit -m "feat: add marker projection guidance"
```

---

### Task 8: Add CRS-aware control point import/export

**Files:**
- Create: `src/core/control_points/reference/CoordinateReference.h/.cpp`
- Create: `src/core/control_points/io/MarkerCsv.h/.cpp`
- Create: `src/core/control_points/tests/test_coordinate_reference.cpp`
- Create: `src/core/control_points/tests/test_marker_csv.cpp`
- Modify: `src/core/control_points/CMakeLists.txt`
- Modify: `src/gui/markers/MarkerReferencePanel.cpp`

**Interfaces:**
- Produces: GDAL/OGR coordinate transformation, explicit axis/unit metadata, CSV import/export.

- [ ] **Step 1: Write CRS tests**

Cover EPSG:4326 axis order, EPSG:3857 conversion, WKT round trip, invalid CRS, meters/feet, separate XY/Z sigma and unknown vertical datum.

```cpp
TEST(CoordinateReferenceTest, ConvertsTraditionalGisAxisOrderExplicitly)
{
    CoordinateReference source = CoordinateReference::fromEpsg(4326, AxisOrder::LongitudeLatitude);
    CoordinateReference target = CoordinateReference::fromEpsg(3857);
    const auto transformed = transformCoordinate({116.391, 39.907, 50.0}, source, target);
    ASSERT_TRUE(transformed.ok);
    EXPECT_NEAR(transformed.xyz[0], 12956586.0, 100.0);
}
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement OGR transformation and validation**

Link `${PLASCAN_GDAL_TARGET}`. If CRS or units are unresolved, preserve Marker data but set `referenceUsable=false` and block BA inclusion with an explicit error.

- [ ] **Step 4: Run CRS and CSV tests**

- [ ] **Step 5: Commit reference coordinates**

```powershell
git add src/core/control_points src/gui/markers
git commit -m "feat: add marker coordinate reference support"
```

---

### Task 9: Feed manual and detected marker tracks into incremental SfM

**Files:**
- Create: `src/core/control_points/registration/PriorTrack.h`
- Create: `tests/test_sfm_prior_tracks.cpp`
- Modify: `src/core/sfm/pipeline/IncrementalSfm.h/.cpp`
- Modify: `src/core/sfm/graph/CorrespondenceGraph.h/.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.h/.cpp`
- Modify: `src/gui/CMakeLists.txt`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: valid `ManualPinned` and `AutoDetected` projections.
- Produces: `IncrementalSfm::addPriorTrack(const PriorTrack&)` and service option `markerSetPath`.

- [ ] **Step 1: Write prior-track registration tests**

```cpp
TEST(SfmPriorTrackTest, RegistersWeaklyTexturedThirdImageWithPinnedMarkerTrack)
{
    IncrementalSfm sfm(testOptions());
    addSyntheticImagesAndOrdinaryMatches(&sfm);
    sfm.addPriorTrack(makeThreeViewPinnedTrack());
    const IncrementalSfmResult result = sfm.run();
    EXPECT_EQ(result.registeredImages, 3);
    EXPECT_EQ(result.priorTracksAccepted, 1);
}
```

Test that `Predicted`, `Disabled`, `Blocked`, stale and duplicate-image projections are rejected.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement a separate prior-track graph path**

```cpp
struct PriorTrack
{
    std::string markerId;
    std::vector<PriorObservation> observations;
    double confidence = 1.0;
};

void IncrementalSfm::addPriorTrack(const PriorTrack &track);
```

Do not modify `.sift` or `.match` caches. Convert observations to graph-owned synthetic feature indices only in memory, tag their source, preserve observation uniqueness and include acceptance/rejection diagnostics.

- [ ] **Step 4: Run prior-track, SfM and aerial workflow tests**

Run `ctest ... -R "SfmPriorTrack|SfmPipeline|AerialTriangulation"`.

- [ ] **Step 5: Commit SfM integration**

```powershell
git add src/core/control_points/registration src/core/sfm src/core/aerial_triangulation tests
git commit -m "feat: use marker tracks during sfm"
```

---

### Task 10: Implement absolute orientation, control/check roles and CPU BA

**Files:**
- Create: `src/core/control_points/registration/ControlNetworkSolver.h/.cpp`
- Create: `src/core/control_points/quality/MarkerQualityReport.h/.cpp`
- Create: `src/core/control_points/tests/test_control_network_solver.cpp`
- Create: `src/core/control_points/tests/test_marker_quality_report.cpp`
- Modify: `src/core/sfm/BaInputBuilder.h/.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/gui/project/services/BundleAdjustService.cpp`

**Interfaces:**
- Produces: weighted RANSAC + Umeyama similarity transform, role-aware BA input and residual report.

- [ ] **Step 1: Write control-network tests**

Cover three non-collinear controls, collinear rejection, one gross outlier, check points excluded from fitting, control/check scale bars and Ceres backend selection.

```cpp
TEST(ControlNetworkSolverTest, CheckPointsDoNotInfluenceSimilarityTransform)
{
    ControlNetworkInput input = exactFourControlNetwork();
    input.checkPoints.push_back(deliberatelyWrongCheckPoint());
    const ControlNetworkResult result = solveControlNetwork(input);
    ASSERT_TRUE(result.ok);
    EXPECT_NEAR(result.transform.scale, expectedScale(), 1e-9);
    EXPECT_GT(result.checkPointResiduals.front().total, 10.0);
}
```

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement absolute orientation and role-aware BA**

```cpp
struct ControlNetworkResult
{
    bool ok = false;
    SimilarityTransform3D transform;
    QVector<MarkerResidual> controlResiduals;
    QVector<MarkerResidual> checkPointResiduals;
    QString error;
};
```

Transform cameras, sparse points and estimated markers once, then run Ceres BA. Existing `BAControlPointConstraint` receives only controls; `BAScaleBarConstraint` receives only control scale bars. Report actual backend and fallback reason.

- [ ] **Step 4: Run BA/control network tests**

Run `ctest ... -R "ControlNetwork|MarkerQuality|BaInputBuilder|BundleAdjustCeres"`.

- [ ] **Step 5: Commit CPU control network**

```powershell
git add src/core/control_points src/core/sfm src/core/aerial_triangulation src/gui/project/services
git commit -m "feat: solve control networks during aerial triangulation"
```

---

### Task 11: Add AprilTag dependency and detector

**Files:**
- Modify: `vcpkg.json`
- Modify: `cmake/PlascanPackages.cmake`
- Create: `src/core/control_points/detection/MarkerDetector.h`
- Create: `src/core/control_points/detection/AprilTagDetector.h/.cpp`
- Create: `src/core/control_points/detection/DetectionMerger.h/.cpp`
- Create: `src/core/control_points/tests/test_apriltag_detector.cpp`
- Modify: `src/core/control_points/CMakeLists.txt`

**Interfaces:**
- Produces: cancellable detector interface and AprilTag family/ID compatible observations.

- [ ] **Step 1: Add failing detector tests using generated tags**

```cpp
TEST(AprilTagDetectorTest, DecodesSupportedFamiliesAcrossRotations)
{
    for (const AprilTagFamily family : supportedAprilTagFamilies())
    {
        for (const int rotation : {0, 90, 180, 270})
        {
            const QImage image = renderAprilTag(family, 7, rotation, 160);
            const auto detections = detector(family).detect(image, emptyMask(), {});
            ASSERT_EQ(detections.size(), 1u);
            EXPECT_EQ(detections.front().targetId, 7);
        }
    }
}
```

- [ ] **Step 2: Add `apriltag` manifest dependency and verify RED before linking**

Use:

```cmake
find_package(apriltag REQUIRED)
target_link_libraries(control_points PRIVATE apriltag::apriltag)
```

- [ ] **Step 3: Implement detector ownership and family adapters**

The detector must destroy all C resources, copy detections before worker completion, apply mask rejection at the fitted center/corners, and return subpixel center, decision margin and hamming distance.

- [ ] **Step 4: Build and run all AprilTag tests on Windows Release**

- [ ] **Step 5: Commit AprilTag support**

```powershell
git add vcpkg.json cmake/PlascanPackages.cmake src/core/control_points
git commit -m "feat: detect supported apriltag markers"
```

---

### Task 12: Implement Metashape-compatible circular coded targets

**Files:**
- Create: `src/core/control_points/detection/CircularTargetCodebook.h/.cpp`
- Create: `src/core/control_points/detection/CircularTargetDetector.h/.cpp`
- Create: `src/core/control_points/tests/test_circular_target_codebook.cpp`
- Create: `src/core/control_points/tests/test_circular_target_detector.cpp`
- Create: `scripts/marker_targets/import_metashape_target_corpus.py`
- Create: `testData/photogrammetry_benchmarks/marker_targets/README.md`
- Modify: `scripts/env/setup_python_runtime.py`

**Interfaces:**
- Produces: compatible 12/14/16/20-bit ID decoding, parity option, ellipse center fitting and detector confidence.

- [ ] **Step 1: Establish the compatibility corpus before decoder code**

Generate target PDFs with a licensed local Metashape installation for every supported circular family. Import rasterized targets plus `family,id,page,sha256` manifest; do not copy Metashape binaries or undocumented program files.

Add `pymupdf` to the repository `.venv` setup so corpus rasterization is reproducible on Windows and CI.

The importer command is:

```powershell
python scripts/marker_targets/import_metashape_target_corpus.py `
  --input E:/code/test/metashape_marker_exports `
  --output testData/photogrammetry_benchmarks/marker_targets/metashape_generated
```

- [ ] **Step 2: Write exhaustive failing codebook tests**

For every manifest entry, decode 0/90/180/270 rotations and require exact ID. Add parity enabled/disabled cases and one-bit corruption rejection.

- [ ] **Step 3: Implement candidate extraction and codebook decoding**

Pipeline: grayscale normalization -> adaptive threshold -> contour hierarchy -> concentric ellipse fit -> polar sector sampling -> rotation canonicalization -> family codebook lookup -> parity -> center refinement -> NMS.

Public result:

```cpp
struct MarkerDetection
{
    TargetFamily family;
    int targetId = -1;
    QPointF center;
    QPolygonF corners;
    double confidence = 0.0;
    double centerSigmaPx = 1.0;
};
```

- [ ] **Step 4: Verify compatibility and localization thresholds**

Requirements: every golden ID/rotation exact; synthetic median center error <= 0.15 px, P95 <= 0.35 px; masked targets produce no result.

- [ ] **Step 5: Commit circular target compatibility**

```powershell
git add src/core/control_points scripts/marker_targets testData/photogrammetry_benchmarks/marker_targets
git add scripts/env/setup_python_runtime.py
git commit -m "feat: detect compatible circular coded targets"
```

---

### Task 13: Add non-coded detection and asynchronous detection workflow

**Files:**
- Create: `src/core/control_points/detection/NonCodedTargetDetector.h/.cpp`
- Create: `src/core/control_points/tests/test_non_coded_target_detector.cpp`
- Create: `src/gui/markers/MarkerTaskRunner.h/.cpp`
- Create: `src/gui/markers/DetectMarkersDialog.h/.cpp`
- Create: `src/cli/marker_detect_cli.cpp`
- Create: `tests/test_detect_markers_workflow.cpp`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `src/gui/main_window/MenuWorkflowController.h/.cpp`
- Modify: `src/gui/menu/MainMenu.h/.cpp`

**Interfaces:**
- Produces: post-alignment non-coded association and cancellable project-wide detection.

- [ ] **Step 1: Write detector and workflow tests**

Test that unaligned non-coded candidates remain unassociated, aligned candidates merge only under epipolar/reprojection/scale gates, masks are respected, duplicate coded IDs enter conflict review, cancellation stops before save, and user edits win revision conflicts.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement background runner**

```cpp
struct MarkerDetectionProgress
{
    int imagesCompleted = 0;
    int imageCount = 0;
    int candidatesDetected = 0;
    int markersMerged = 0;
    QString currentImage;
};
```

Use bounded workers and per-worker detector instances. Merge results on the owning thread as a `MarkerChangeSet`; compare repository revision before apply and route conflicts to review.

`marker_detect_cli` consumes image list、mask list、target family and detector thresholds, then writes the same `MarkerSet` change payload used by the GUI. It is the deterministic entry point for corpus and PDF round-trip verification.

- [ ] **Step 4: Run workflow tests and GUI cancellation smoke test**

- [ ] **Step 5: Commit detection workflow**

```powershell
git add src/core/control_points src/gui/markers src/gui/main_window src/gui/menu src/cli tests
git commit -m "feat: add project marker detection workflow"
```

---

### Task 14: Generate compatible printable marker PDFs

**Files:**
- Create: `src/core/control_points/print/MarkerPdfWriter.h/.cpp`
- Create: `src/core/control_points/print/MarkerSheetRenderer.h/.cpp`
- Create: `src/core/control_points/tests/test_marker_pdf_writer.cpp`
- Create: `src/gui/markers/PrintMarkersDialog.h/.cpp`
- Create: `scripts/marker_targets/verify_marker_pdf.py`
- Modify: `src/gui/main_window/MenuWorkflowController.h/.cpp`
- Modify: `src/gui/menu/MainMenu.h/.cpp`

**Interfaces:**
- Produces: QPdfWriter output for supported circular and AprilTag families with physical-size metadata.

- [ ] **Step 1: Write PDF layout and detector round-trip tests**

```cpp
TEST(MarkerPdfWriterTest, MarkerSheetRasterDecodesToRequestedIds)
{
    const MarkerPrintRequest request = circular12BitRequest({1, 2, 3});
    const QVector<QImage> pages = renderMarkerSheets(request, 600);
    EXPECT_EQ(decodedIds(pages), QSet<int>({1, 2, 3}));
}
```

Add a second test that writes the PDF and verifies the `%PDF` signature, page count and requested physical page size.

- [ ] **Step 2: Verify RED**

- [ ] **Step 3: Implement PDF writer and print dialog**

Options: family, ID start/count, target diameter in mm, page size, margins, label visibility and spacing. Reject physically overlapping layouts. `MarkerPdfWriter` and detector tests must both consume `MarkerSheetRenderer`, so the raster and PDF layouts cannot diverge. Include a metadata footer with family and scale; do not add decorative graphics near target borders.

- [ ] **Step 4: Run PDF round-trip tests**

Use the Task 12 `.venv` `pymupdf` dependency to rasterize the actual PDF at 600 DPI, then invoke the Task 13 `marker_detect_cli` on each page:

```powershell
python scripts/env/setup_python_runtime.py --device cpu
.\.venv\Scripts\python.exe scripts/marker_targets/verify_marker_pdf.py `
  --pdf build/marker_tests/circular12.pdf `
  --detector build/windows-vcpkg-cuda-release/bin/marker_detect_cli.exe `
  --family circular12 `
  --expected-ids 1,2,3
```

Expected: IDs `1,2,3` each occur exactly once and no extra ID is reported.

- [ ] **Step 5: Commit marker printing**

```powershell
git add src/core/control_points/print src/core/control_points/tests src/gui/markers src/gui/main_window src/gui/menu scripts
git commit -m "feat: print compatible coded markers"
```

---

### Task 15: Add native CUDA control-point and scale-bar residuals, then complete end-to-end validation

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.h/.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaDeviceTypes.cuh`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaKernels.cuh`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cu/.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp`
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_control.cpp`
- Create: `src/cli/marker_control_cli.cpp`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/sfm/README.md`
- Modify: `src/core/aerial_triangulation/README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 10 control network and existing BA constraints.
- Produces: CUDA residual/Jacobian support, CPU/CUDA quality gate and reproducible CLI validation.

- [ ] **Step 1: Replace current fallback expectation with failing CUDA parity tests**

Build a synthetic 12-camera network with controls, checks and scale bars. Run Ceres CPU and native CUDA from identical initial values. Require:

- camera center max difference <= `1e-5` scene units,
- point max difference <= `1e-5`,
- final RMS relative difference <= `1e-4`,
- control/check role behavior identical,
- disabled constraints contribute zero residual.

- [ ] **Step 2: Verify RED against the current explicit unsupported path**

Expected: workset reports that native CUDA does not support control point constraints.

- [ ] **Step 3: Implement CUDA residuals and quality-gated fallback**

Add one weighted 3D residual per `BAControlPointConstraint` and one scalar distance residual per `BAScaleBarConstraint`. Include robust loss and sigma in both normal-equation accumulation and cost reporting. Preserve existing camera/point Schur structure.

Auto mode behavior:

```cpp
if (cudaResult.success && passesCpuParityGate(cudaResult, sampledCpuResult))
{
    return cudaResult;
}
return runCeresCpuWithFallbackReason(problem, cudaResult.message);
```

- [ ] **Step 4: Run full verification**

Focused tests:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --parallel 24
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure `
  -R "Marker|ControlNetwork|SurveyControl|SfmPriorTrack|AerialTriangulation|BundleAdjust"
```

Real-data validation:

```powershell
build/windows-vcpkg-cuda-release/bin/marker_control_cli.exe `
  --project E:/code/test/marker_control/marker_control.plascan `
  --detect-markers circular12 `
  --run-aerial-triangulation `
  --ba-device auto `
  --report E:/code/test/marker_control/reports/marker_e2e.json
```

Verify manually in GUI:

1. Right-click creates and places markers at the selected pixel.
2. Reopen preserves marker/projection/CRS/scale-bar state.
3. Detection progress is real and cancellation leaves the project valid.
4. At least three controls produce absolute orientation.
5. Check points do not change the solution when toggled.
6. CPU/CUDA backend and fallback reason are visible in reports.

- [ ] **Step 5: Remove obsolete files and update documentation**

Delete `SurveyControlDialog.h/.cpp` and old runtime-only wiring after all replacements compile. Update architecture and README to describe the single marker system, sidecar schema and CLI. Record the historical terrain test failure separately if full CTest still reports it.

- [ ] **Step 6: Commit the completed product**

```powershell
git add src/core/bundle_adjust src/cli src/gui src/core/control_points docs CHANGELOG.md tests vcpkg.json cmake
git commit -m "feat: complete metashape-style marker workflow"
```

---

## Execution Order and Gates

| Gate | Tasks | Required result before continuing |
|---|---:|---|
| A: usable manual marker MVP | 1-6 | Right-click/manual placement, persistence, undo and focused review work without SfM changes |
| B: photogrammetric control network | 7-10 | Predictions, CRS, PriorTrack, absolute orientation and Ceres BA pass synthetic and real smoke tests |
| C: automatic marker product | 11-14 | AprilTag/circular/non-coded detection and printable PDFs pass compatibility corpus |
| D: accelerated production release | 15 | CUDA parity, full GUI/CLI validation, migration cleanup and docs complete |

Do not start Gate C until Gate B is stable on a real control-point dataset. Do not delete legacy survey-control files until Gate D verification proves migration and GUI replacement are complete.

## Plan Self-Review

- **Spec coverage:** All design sections map to Tasks 1-15: data/persistence (1-4), GUI (5-6), geometry/CRS (7-8), SfM/control network (9-10), detection/printing (11-14), CUDA/product validation (15).
- **Placeholder scan:** No unresolved placeholders or unspecified test steps remain; every task names concrete tests, APIs and commands.
- **Type consistency:** `MarkerSet`, `MarkerProjection`, `MarkerChangeSet`, `PriorTrack`, `ControlNetworkResult` and `MarkerDetection` are introduced once and consumed by later tasks under the same names.
- **Risk isolation:** Circular code compatibility and CUDA constraints are late gates; they cannot block the manually usable control-point workflow.
