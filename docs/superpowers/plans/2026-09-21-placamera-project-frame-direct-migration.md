# PlaCamera Project and Frame Direct Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy camera objects from the PlaScan project-loading and frame-pinhole production path so these boundaries consume PlaCamera directly.

**Architecture:** A new PlaScan project camera store will decode the existing `camera_definitions` and `camera_instances` JSON fields directly into `placamera::CameraInstanceSet`, using PlaCamera's registry and state factories. Tsai input will construct `placamera::FramePinholeModel` directly. The frame-pinhole numerical path will then replace legacy `FramePinholeNumericState` at the aerial-triangulation/SfM/intersection boundary, with solver-specific packets remaining explicit and non-legacy.

**Tech Stack:** C++20, CMake, Qt6 Core for project JSON, PlaCamera 0.2, PlaCoordinate types, GDAL only at RPC/IO boundaries, GTest, existing PlaBundle/SfM/MVS targets.

**Spec:** `docs/superpowers/specs/2026-09-21-placamera-direct-migration-design.md`

## Global Constraints

- PlaCamera is the only production camera model; do not add a Legacy/Shadow/PlaCamera backend switch.
- Do not add bidirectional converters between legacy camera classes and PlaCamera.
- Existing `.plascan` JSON field names remain readable/writable where they can represent PlaCamera state; decoding must produce PlaCamera objects directly.
- `src/core/placamera_adapter` is transitional work only; migrated production targets must not link it as a compatibility layer.
- PlaCamera core remains free of Qt, OpenCV, GDAL, and project JSON types.
- Keep `CameraOperationPlan`, reference observations, solver priors, image conversion, and GPU packets in PlaScan application/solver layers.
- Do not modify dirty submodules or unrelated in-flight PlaBundle, GUI, task-runtime, and coordinate-system changes.
- Do not commit, push, or create branches unless the user explicitly requests it.

## Review Focus

- A project record with duplicate or dangling definition/instance/image identities must fail before any partial `CameraInstanceSet` is published; Task 1 adds atomicity tests.
- Frame-pinhole unit and pixel-convention conversion must preserve the existing project semantics exactly; Task 1 and Task 2 add round-trip projection tests.
- External Tsai input must bind stable definition/instance/image IDs without deriving identity from a path or vector index; Task 2 adds identity tests.
- The aerial-triangulation/SfM boundary must not retain `camera_models::frame_pinhole::FramePinholeNumericState`; Task 3 adds compile-boundary and numerical regression tests.
- MVS/GPU preparation must receive an explicitly named numeric packet derived from PlaCamera, not a renamed legacy type; Task 4 adds packet construction and dimension tests.

---

### Task 1: Direct PlaCamera project store

**Files:**
- Create: `src/core/placamera_runtime/CMakeLists.txt`
- Create: `src/core/placamera_runtime/ProjectCameraStore.h`
- Create: `src/core/placamera_runtime/ProjectCameraStore.cpp`
- Create: `src/core/placamera_runtime/tests/test_project_camera_store.cpp`
- Modify: `src/core/CMakeLists.txt`
- Modify: `src/core/placamera_adapter/CMakeLists.txt`
- Modify: `src/core/camera/project/CameraProjectRecords.cpp` only to extract model-independent JSON validation/helpers; do not add new legacy model construction.

**Interfaces:**
- Produces `xjw::placamera_runtime::ProjectCameraLoadResult` with `placamera::CameraInstanceSet instances` and `QStringList errors`.
- Produces `xjw::placamera_runtime::ProjectCameraWriteResult` with `updatedCount`, `clearedCount`, and `QStringList errors`.
- Exposes:

```cpp
ProjectCameraLoadResult loadProjectCameras(const QJsonObject& projectFiles);
ProjectCameraWriteResult writeProjectCameras(QJsonObject* projectFiles,
                                             const placamera::CameraInstanceSet& instances);
```

- Consumes only canonical project JSON and PlaCamera state/registry APIs; it must not include `camera/project/CameraProjectRuntime.h`, `camera/models/CameraModelFactories.h`, or any legacy model header.

- [ ] **Step 1: Write the failing tests**

Add tests named:

```cpp
TEST(ProjectCameraStoreTest, LoadsFrameRpcAndLineScanDirectlyIntoPlaCameraSet);
TEST(ProjectCameraStoreTest, RejectsDuplicateIdentityWithoutPublishingPartialSet);
TEST(ProjectCameraStoreTest, WritesUpdatedPlaCameraStateAtomically);
TEST(ProjectCameraStoreTest, RejectsLegacyEmbeddedCameraFields);
```

The first test constructs the same canonical `camera_definitions` and `camera_instances` arrays used by current project tests, calls `loadProjectCameras`, and asserts that returned models are `FramePinholeModel`, `RpcModel`, and `LineScanModel`. The second test inserts a duplicate instance identity and asserts `instances.empty()` with errors. The third test mutates a frame model, calls `writeProjectCameras`, reloads it, and compares projection output. The fourth test adds `images[*].camera` and asserts a structured rejection.

- [ ] **Step 2: Run the focused tests and verify the expected red state**

Run:

```bash
cmake --build build/linux-source-release --target test_placamera_project_store -j 4
ctest --test-dir build/linux-source-release --output-on-failure -R 'ProjectCameraStoreTest'
```

Expected: configuration/build fails because `placamera_runtime` and `loadProjectCameras` do not exist yet. Do not implement production code before recording this failure.

- [ ] **Step 3: Implement the direct load path**

Create one PlaCamera `ModelRegistry`, register built-in JSON factories, and convert each project definition/instance record into `CameraDefinitionState` and `CameraInstanceState` without constructing any legacy object. Decode model parameters into canonical PlaCamera state envelopes, create the definition first, then create each instance and add it to a temporary `CameraInstanceSet`. Publish the set only after all records and identity checks succeed; homogeneous operations call `requireCommonGroundFrame()` on their selected subset.

Use the existing canonical project keys (`model`, `world_frame`, `image_uuid`, `instance_id`, `definition_id`, `image_width`, `image_height`, and model-specific state) but keep the conversion code in `ProjectCameraStore.cpp`; do not route through `CameraProjectRuntime` or `CameraModelRegistry`.

- [ ] **Step 4: Implement atomic direct writeback**

Encode each PlaCamera definition and instance with `encodeCameraDefinitionJson` and `encodeCameraInstanceJson`, map the canonical envelope fields back to the existing project arrays, and validate identities before replacing the document. Build a new `QJsonObject` first; assign it to `*projectFiles` only when every model encodes and every identity/frame check passes.

- [ ] **Step 5: Run the focused tests and the existing project tests**

Run:

```bash
cmake --build build/linux-source-release --target test_placamera_project_store test_camera_project_store -j 4
ctest --test-dir build/linux-source-release --output-on-failure -R 'ProjectCameraStoreTest|CameraProjectStoreTest'
```

Expected: all new direct-store tests and existing JSON validation tests pass; no test invokes the old runtime as part of the direct-store path.

### Task 2: Direct external camera loaders

**Files:**
- Modify: `src/core/camera/FramePinholeTsaiIO.h`
- Modify: `src/core/camera/FramePinholeTsaiIO.cpp`
- Modify: `src/core/camera/RpcRasterIO.h`
- Modify: `src/core/camera/RpcRasterIO.cpp`
- Modify: `src/core/camera/PlanetaryLineScanIsdIO.h`
- Modify: `src/core/camera/PlanetaryLineScanIsdIO.cpp`
- Modify: `src/core/camera/test/test_tsai_loader.cpp`
- Modify: `src/core/camera/test/RpcRasterIO_tests.cpp`
- Modify: `src/core/camera/test/PlanetaryLineScanIsdIO_tests.cpp`
- Modify: `src/core/camera/CMakeLists.txt`

**Interfaces:**
- Replace `loadFramePinholeNumericStateFromTsaiFile` with:

```cpp
bool loadFramePinholeModelFromTsaiFile(const std::string& path,
                                       placamera::FramePinholeModel* model,
                                       std::string* error = nullptr);
```

- Replace `importRpcRasterInstance` with a direct `placamera::RpcModel` result that requires an explicit Cartesian `placamera::FrameId`.
- Change `PlanetaryLineScanIsdImport::instance` to `std::optional<placamera::LineScanModel> model`.
- No loader may include a legacy model header after this task.

- [ ] **Step 1: Write failing direct-loader tests**

Add direct-return tests that call the new function names, assert model identity and image size, and project known samples. Keep the tracked RPC raster pair and existing line-scan ISD fixtures. Remove tests that only verify construction of legacy instances.

- [ ] **Step 2: Run the loader tests and verify red state**

Run:

```bash
cmake --build build/linux-source-release --target test_camera_tsai test_camera_unit test_planetary_line_scan_camera -j 4
```

Expected: compile failures for the new direct function names and result types.

- [ ] **Step 3: Implement direct construction**

Move the existing parsing/math code behind the new APIs. Construct PlaCamera definitions and models with explicit IDs supplied by the caller. Preserve pixel conventions, image dimensions, RPC ellipsoid/frame metadata, line timing, detector geometry, trajectory, and capture time. Do not create a legacy object as an intermediate.

- [ ] **Step 4: Run focused loader and real-data tests**

Run:

```bash
ctest --test-dir build/linux-source-release --output-on-failure -R 'CameraTsai|RpcRasterIO|PlanetaryLineScanIsdIO|PlaCameraRpcRasterParity'
```

Expected: all direct loader tests pass, including tracked RPC raster parity and existing line-scan fixture checks.

### Task 3: Direct frame-pinhole aerial-triangulation and SfM boundary

**Files:**
- Modify: `src/core/aerial_triangulation/engine/PinholeEngine.h`
- Modify: `src/core/aerial_triangulation/engine/PinholeEngine.cpp` if present in the target
- Modify: `src/core/aerial_triangulation/reconstruction/SfmAttemptRunner.cpp`
- Modify: `src/core/sfm/geometry/ProjectionGeometry.*`
- Modify: `src/core/sfm/triangulation/Triangulator.*`
- Modify: `src/core/intersection/Intersection.*`
- Create: `src/core/intersection/tests/test_intersection_placamera.cpp`
- Modify: `src/core/intersection/CMakeLists.txt`
- Modify: `src/core/camera/reference/geometry/ReferenceCameraGeometry.*`
- Modify: `src/core/plabundle_adapter/FrameCameraAdapter.*`
- Modify: affected CMake files and tests discovered by the first compile-boundary run.

**Interfaces:**
- `aerial_triangulation::engine::PinholeImage::camera` becomes `placamera::FramePinholeNumericState`.
- Geometry functions consume `placamera::FramePinholeNumericState` or `placamera::FramePinholeModel`; no overload accepting the legacy type is added.
- `ReferenceCameraGeometry` stores a PlaCamera frame state or a named `PlaCameraFrameGeometry` value owned by the reference module.
- `plabundle_adapter` converts directly from PlaCamera frame state to `plabundle::Camera` numeric data.

- [ ] **Step 1: Add failing compile-boundary and numerical tests**

Update the focused tests to construct `placamera::FramePinholeNumericState` and call the new signatures. Add a repository test that scans migrated production targets for `camera/models/frame_pinhole/FramePinholeNumericState.h` and fails if it is included by `aerial_triangulation`, `sfm`, `intersection`, or `plabundle_adapter`.

- [ ] **Step 2: Run the affected tests and record the expected red state**

Run:

```bash
cmake --build build/linux-source-release --target \
    test_sfm_attempt_runner test_aerial_triangulation_pipeline \
    test_sfm_pipeline test_projection_geometry test_intersection_placamera \
    test_plabundle_adapter -j 4
```

Expected: compile failures identify each remaining legacy frame type at the boundary.

- [ ] **Step 3: Replace the boundary types and preserve numerical semantics**

Use PlaCamera's `groundToImage`, `imageToImagingLocus`, `applyPoseDelta`, `toModel`, and `triangulatePair`. Preserve camera-center convention, positive-depth handling, Brown-Conrady distortion, pixel-center convention, stable IDs, and common ground frame validation. Update `SfmAttemptRunner` binding code to bind by `ImageId`, never by path or vector index.

- [ ] **Step 4: Run focused numerical and integration tests**

Run:

```bash
cmake --build build/linux-source-release --target \
    test_sfm_attempt_runner test_aerial_triangulation_pipeline \
    test_sfm_pipeline test_projection_geometry test_intersection_placamera \
    test_plabundle_adapter -j 4
ctest --test-dir build/linux-source-release --output-on-failure -R 'Aerial|Sfm|ProjectionGeometry|Intersection|PlaBundle|FramePinhole'
```

Expected: all affected numerical regressions pass and the compile-boundary scan reports no legacy frame-pinhole include in migrated targets.

### Task 4: Direct frame-pinhole MVS, CLI, and GUI consumers

**Files:**
- Modify: `src/core/mvs/MvsImagePreprocessor.*`
- Modify: `src/core/mvs/DenseCloudBuilder.*`
- Modify: `src/core/mvs/DisparityTriangulator.*`
- Modify: `src/core/mvs/PatchMatchHostUtils.*`
- Modify: `src/core/mvs/RecoveredDepthScene.*`
- Modify: `src/core/mvs/DepthPoseRefinementStage.*`
- Modify: `src/cli/common/cli_photogrammetry_common.*`
- Modify: `src/cli/common/FinalBaCameraExporter.cpp`
- Modify: `src/common/project/ProjectSession.*`
- Modify: `src/gui/project/services/ProjectSession.*`
- Modify: `src/gui/project/services/BundleAdjustService.*`
- Modify: affected MVS/CLI/GUI tests and CMake targets.

**Interfaces:**
- Introduce an explicitly named `PlaCameraFramePacket` in the MVS or common geometry boundary containing only finite arrays/scalars needed by CPU/GPU kernels.
- The packet is created from `placamera::FramePinholeNumericState`; no legacy camera state appears in its declaration.
- Project and GUI services return PlaCamera model/state objects or `PlaCameraFramePacket`, never old camera classes.

- [ ] **Step 1: Add packet and consumer tests first**

Test that packet construction preserves image size, camera center, rotation, intrinsics, distortion, and positive-depth orientation. Add MVS/CLI/GUI tests that load one project through `loadProjectCameras` and construct the packet by `ImageId`.

- [ ] **Step 2: Run the affected tests and verify the red state**

Run:

```bash
cmake --build build/linux-source-release --target \
    test_mvs_pipeline test_mvs_image_preprocessor test_workflow_cli \
    test_project_session test_project_session_facade \
    test_project_bundle_adjust_controller -j 4
```

Expected: compile failures from old frame-state parameters and missing packet construction.

- [ ] **Step 3: Migrate consumers and GPU preparation**

Replace old frame-state parameters with PlaCamera state or `PlaCameraFramePacket`. Keep OpenCV/Qt/GPU code at existing application boundaries. Ensure prepared-resolution scaling creates a new PlaCamera definition/model or a new packet with explicit scaled intrinsics; never mutate a shared definition silently.

- [ ] **Step 4: Run the frame migration gate**

Run:

```bash
cmake --build build/linux-source-release --target \
    test_mvs_pipeline test_mvs_image_preprocessor test_workflow_cli \
    test_project_session test_project_session_facade \
    test_project_bundle_adjust_controller -j 4
ctest --test-dir build/linux-source-release --output-on-failure -R 'Mvs|Workflow|ProjectSession|BundleAdjust|FramePinhole|PlaCamera'
rg -n 'camera_models::frame_pinhole::FramePinholeNumericState|camera/models/frame_pinhole/FramePinholeNumericState.h' src/core/aerial_triangulation src/core/sfm src/core/intersection src/core/mvs src/cli src/common src/gui
```

Expected: affected tests pass and the repository search returns no migrated production include or symbol reference.

## Follow-up plans

After this plan passes review and its frame migration gate is green, create separate plans for:

1. direct RPC raster/triangulation/DEM/DOM migration;
2. direct line-scan ISD/planetary BA migration;
3. reference-camera and solver-prior ownership cleanup;
4. deletion of the remaining legacy camera CMake targets and model sources;
5. optional direct `placamera_csm` integration.

## Completion contract for this plan

- The direct project store and frame-pinhole production path build without the legacy frame model headers.
- Focused Linux/GCC tests pass with zero failures.
- No new compatibility target, backend switch, or legacy-to-PlaCamera converter is introduced.
- Existing transitional files are either removed from migrated target link interfaces or explicitly listed in the next follow-up plan.
