# PlaScan Camera Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mixed camera/pose/reference design with extensible camera definitions, image instances, capability-based services, and shared coordinate/reference types for frame pinhole, RPC, pushbroom, and future models.

**Architecture:** Add a Qt-independent `camera_core` domain layer containing IDs, frames, poses, time, uncertainty, model definitions, instances, and capability contracts. Concrete frame-pinhole, RPC, and pushbroom implementations become separate model modules. Project persistence stores `camera_definitions` and `camera_instances`; reference observations use the same pose/frame types and are resolved through an explicit resolver. Existing `images[*].camera`, old `ProjectCameraIO` overloads, and compatibility readers are deleted.

**Tech Stack:** C++20, CMake, Qt6 JSON/GUI at the project and GUI edges, OpenCV/GDAL for existing model implementations, GoogleTest, existing Linux GCC release preset.

**Spec:** `docs/superpowers/specs/2026-09-18-camera-architecture-design.md`

**实施状态（2026-09-19）：** Tasks 1–5 的核心类型、模型注册表、三类模型、规范化工程存储和严格工厂校验已完成；Task 6 已完成：
SfM、BA、MVS 的数值入口现在只接收经过 `Projection + StaticPose + Optimization` 校验后生成的
`FramePinholeNumericState`，能力边界不再保留 `StaticPinholeView` 兼容对象。Task 9 已完成共享参考
observation/resolver/comparator/store 的核心实现。旧工程字段不会被读取或双写；`src/core/camera` 中的
`FramePinholeCamera` 仅保留为遗留文件/显示边界，数值内核不再依赖它。

## Global Constraints

- Old `.plascan` schemas and `images[*].camera`/`images[*].camera_file` are rejected; no legacy project reader, writer, or dual-write path is added. Field aliases are allowed only while normalizing an explicitly supported external camera import into the canonical schema.
- `CameraDefinition` contains model/calibration data; `CameraInstance` contains image-specific state; external references use shared frame/pose/time/uncertainty types.
- Models expose only capabilities they implement; unsupported capabilities fail with a structured error naming image, definition, model, and frame.
- RPC never receives a fabricated static camera center; pushbroom never receives a fabricated static pose.
- New core domain types stay independent of Qt; Qt JSON remains at `camera_project`/GUI edges.
- Every production change follows a red-green-refactor loop: add a focused failing test, run it, implement the minimum, rerun the focused test, then run affected tests.
- Only files owned by the current task may be modified; preserve unrelated existing working-tree changes and do not reset submodules.
- Temporary artifacts belong under `build/tmp/camera-architecture/` and are removed after the task unless needed to reproduce a failure.
- No commit, branch switch, push, or tag is performed without an explicit user request.

## File Map

### New core files

- `src/core/camera_core/CMakeLists.txt`: Qt-independent camera domain target.
- `src/core/camera_core/types/CameraIds.h`, `CameraFrames.h`, `CameraPose.h`, `CameraTime.h`, `CameraUncertainty.h`, `CameraErrors.h`: validated value types.
- `src/core/camera_core/model/CameraDefinition.h`, `CameraInstance.h`, `CameraModelRegistry.h`, `CameraModelRegistry.cpp`: type-erased definition/instance boundary and string-keyed model factory registry.
- `src/core/camera_core/capabilities/CameraCapabilities.h`, `CameraCapabilities.cpp`, `CapabilityRequirements.h`, `CapabilityRequirements.cpp`: capability contracts and fail-closed checks.
- `src/core/camera_core/transform/CoordinateTransformService.h`, `CoordinateTransformService.cpp`: explicit frame/unit/attitude conversion boundary.

### New model files

- `src/core/camera_models/frame_pinhole/FramePinholeDefinition.h/.cpp`
- `src/core/camera_models/frame_pinhole/FramePinholeInstance.h/.cpp`
- `src/core/camera_models/frame_pinhole/FramePinholeProjection.h/.cpp`
- `src/core/camera_models/frame_pinhole/FramePinholeOptimization.h/.cpp`
- `src/core/camera_models/rpc/RpcDefinition.h/.cpp`
- `src/core/camera_models/rpc/RpcInstance.h/.cpp`
- `src/core/camera_models/rpc/RpcProjection.h/.cpp`
- `src/core/camera_models/rpc/RpcIntersectionService.h/.cpp`
- `src/core/camera_models/linescan/LineScanDefinition.h/.cpp`
- `src/core/camera_models/linescan/LineScanInstance.h/.cpp`
- `src/core/camera_models/linescan/LineScanProjection.h/.cpp`
- `src/core/camera_models/linescan/LineScanTrajectory.h/.cpp`
- `src/core/camera_models/linescan/LineScanOptimization.h/.cpp`

### New project/reference files

- `src/core/camera_project/CMakeLists.txt`
- `src/core/camera_project/CameraProjectStore.h/.cpp`
- `src/core/camera_project/CameraProjectValidation.h/.cpp`
- `src/core/camera_reference/model/CameraReferenceObservation.h/.cpp`
- `src/core/camera_reference/resolve/CameraReferenceResolver.h/.cpp`
- `src/core/camera_reference/compare/CameraReferenceComparator.h/.cpp`
- `src/core/camera_reference/io/CameraReferenceSetStore.h/.cpp`

### Primary existing files to migrate or delete

- `src/core/CMakeLists.txt`, `src/gui/cmake/GuiCoreLinking.cmake`, relevant test CMake files.
- `src/core/camera/FramePinholeCamera.*`, `CameraModel.*`, `ProjectCameraIO.*`, `ProjectRpcCameraIO.cpp`, `RpcCameraModel.*`, `RpcStereoIntersection.*`, `PlanetaryLineScanCamera.*`.
- `src/core/sfm/project/ProjectMatchInputReader.*`, `BaInputBuilder.*`, `BaTrackBuilder.*`, `MarkerBaAdapter.*`, `src/core/sfm/reconstruction/SfmReconstruction.*`.
- `src/gui/project/services/BundleAdjustService.*` and all MVS public interfaces that accept `FramePinholeCamera`.
- `src/common/project/ProjectDocumentModel.*`, `ProjectSessionModel.*`, `ProjectIO.*`, `ProjectMetadata.*` where camera fields are read or written.
- `src/gui/reference/MetashapeCameraReferenceSetBuilder.*`, `ProjectCameraReferenceRepository.*`, `CameraReferenceTreeModel.*`, `CameraReferenceController.*`.
- `docs/PROJECT_ARCHITECTURE.md`, `docs/project/PLASCAN_PROJECT_FORMAT.md`, `src/core/camera/README.md`, and affected tests.

---

### Task 1: Establish the new domain target and validated value types

**Files:**
- Create: `src/core/camera_core/CMakeLists.txt`
- Create: `src/core/camera_core/types/CameraIds.h`
- Create: `src/core/camera_core/types/CameraFrames.h`
- Create: `src/core/camera_core/types/CameraPose.h`
- Create: `src/core/camera_core/types/CameraTime.h`
- Create: `src/core/camera_core/types/CameraUncertainty.h`
- Create: `src/core/camera_core/types/CameraErrors.h`
- Modify: `src/core/CMakeLists.txt`
- Test: `tests/test_camera_core_types.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- `CameraDefinitionId`, `CameraInstanceId`, `ImageId`, and `ReferenceSourceId` wrap non-empty strings and expose `value()`, equality, and hashing.
- `CoordinateFrame` carries id, kind, linear unit, angle unit, optional parent, and a validated rigid transform.
- `Pose` carries a frame id, center, and camera-to-world rotation; construction rejects non-finite values, non-orthogonal matrices, and non-positive determinant.
- `TimeReference` carries a scale and finite seconds.
- `PoseCovariance` validates diagonal size 6 or symmetric 6x6 size 21 and exposes a stable layout enum.

- [x] **Step 1: Add failing type tests**

  Add tests for empty IDs, invalid rotations, invalid covariance lengths, valid local/ECEF frame values, and time-scale round trips. Use real constructors and assert exact error categories.

- [x] **Step 2: Run the focused test and verify the expected failure**

  Run `cmake --build build/linux-source-release --target test_camera_core_types -j2` after registering the target. Expected result before implementation: compilation failure because the new headers and target do not exist.

- [x] **Step 3: Implement the value types and target**

  Keep headers free of Qt/OpenCV/GDAL. Add `camera_core` as a static library with C++20 and public include directory. Register it before model modules in `src/core/CMakeLists.txt`.

- [x] **Step 4: Run the focused test and affected source checks**

  Run `cmake --build build/linux-source-release --target test_camera_core_types -j2` and `ctest --test-dir build/linux-source-release -R CameraCoreTypes --output-on-failure`. Expected result: all type validation tests pass.

- [x] **Step 5: Refactor only after green**

  Remove duplicated local pose/rotation validation helpers from the new files, rerun the focused test, and run `git diff --check`.

### Task 2: Add model definitions, instances, registry, and capability contracts

**Files:**
- Create: `src/core/camera_core/model/CameraDefinition.h/.cpp`
- Create: `src/core/camera_core/model/CameraInstance.h/.cpp`
- Create: `src/core/camera_core/model/CameraModelRegistry.h/.cpp`
- Create: `src/core/camera_core/capabilities/CameraCapabilities.h/.cpp`
- Create: `src/core/camera_core/capabilities/CapabilityRequirements.h/.cpp`
- Test: `tests/test_camera_core_registry.cpp`
- Modify: `src/core/camera_core/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- `CameraDefinition` exposes `definitionId()`, `modelType()`, `parameterSchemaVersion()`, `worldFrame()`, and `capabilities()`; it is immutable after construction.
- `CameraInstance` exposes `imageId()`, `definition()`, `imageSize()`, `captureTime()`, and `capabilities()`; model-specific state is held by the concrete instance.
- `CameraModelRegistry::registerFactory(modelType, factory)` rejects duplicate keys; `createDefinition()` and `createInstance()` return `CameraResult<T>` with structured errors.
- `CapabilityKind` includes projection, inverse projection, ray, static pose, trajectory, image correction, and optimization.
- `requireCapabilities(instance, requirements)` returns a `CapabilityError` containing all missing capabilities.

- [x] **Step 1: Write failing registry and capability tests**

  Test duplicate registration rejection, unknown model rejection, immutable definition identity, and an instance that deliberately lacks static pose causing `requireCapabilities()` to report `static_pose`.

- [x] **Step 2: Run the focused test to observe the missing API failure**

  Run `cmake --build build/linux-source-release --target test_camera_core_registry -j2`. Expected result: compilation failure before the new interfaces exist.

- [x] **Step 3: Implement the type-erased registry and contracts**

  Use string-keyed factories and `std::unique_ptr`; do not use a central `std::variant` of all camera types. Keep the capability query independent of Qt JSON.

- [x] **Step 4: Run focused tests**

  Run `ctest --test-dir build/linux-source-release -R CameraCoreRegistry --output-on-failure`. Expected result: all registry and capability tests pass.

- [x] **Step 5: Refactor capability errors**

  Ensure errors include model type and definition/instance IDs, rerun the focused tests, and run `git diff --check`.

### Task 3: Introduce explicit coordinate transforms and reference primitives

**Files:**
- Create: `src/core/camera_core/transform/CoordinateTransformService.h/.cpp`
- Create: `src/core/camera_reference/model/CameraReferenceObservation.h/.cpp`
- Create: `src/core/camera_reference/resolve/CameraReferenceResolver.h/.cpp`
- Create: `tests/test_camera_reference_resolver.cpp`
- Modify: `src/core/camera_reference/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- `CoordinateTransformService::registerFrame()`, `resolve(frameId)`, and `transformPoint(source, target, point)` require explicit frames and return structured transform errors.
- `CameraReferenceObservation` stores image id, source id, source frame, optional time, optional position/orientation, covariance, lever arm, and enabled state.
- `CameraReferenceResolver::resolve(observation, targetFrame, options)` returns `ResolvedCameraReference` with target pose, covariance, transform-chain hash, lever-arm-applied flag, and an explicit unresolved reason.

- [x] **Step 1: Add failing tests**

  Cover local ENU to ECEF point round trip, missing frame rejection, missing orientation convention rejection, lever-arm direction rejection, and successful resolved pose with a deterministic transform hash.

- [x] **Step 2: Run `test_camera_reference_resolver` and verify RED**

  Run `cmake --build build/linux-source-release --target test_camera_reference_resolver -j2`; expected result is a missing-type or missing-symbol failure.

- [x] **Step 3: Implement frame graph and resolver**

  Reuse existing GDAL/geodesy math only behind this service. Do not copy the old `RawCameraReference`/`ResolvedCameraReference` structs into the new implementation. Preserve source values and return `Unresolved` when metadata is incomplete.

- [x] **Step 4: Run resolver tests**

  Run `ctest --test-dir build/linux-source-release -R CameraReferenceResolver --output-on-failure`; expected result is green.

- [x] **Step 5: Refactor and check**

  Consolidate frame validation in `camera_core`, rerun the test, and run `git diff --check`.

### Task 4: Split frame-pinhole definition, instance, projection, and optimization

**Files:**
- Create: `src/core/camera_models/frame_pinhole/FramePinholeDefinition.h/.cpp`
- Create: `src/core/camera_models/frame_pinhole/FramePinholeInstance.h/.cpp`
- Create: `src/core/camera_models/frame_pinhole/FramePinholeProjection.h/.cpp`
- Create: `src/core/camera_models/frame_pinhole/FramePinholeOptimization.h/.cpp`
- Create: `src/core/camera_models/frame_pinhole/CMakeLists.txt`
- Test: `src/core/camera_models/frame_pinhole/tests/test_frame_pinhole_model.cpp`
- Modify: `src/core/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Delete after consumers migrate: `src/core/camera/FramePinholeCamera.h`, `src/core/camera/FramePinholeCamera.cpp`, `src/core/camera/FramePinholeCameraModel.cpp`

**Interfaces:**
- `FramePinholeDefinition::create(intrinsics, distortion, pixelConvention, frame)` returns an immutable validated definition.
- `FramePinholeInstance::create(imageId, definition, imageSize, pose)` returns a validated instance.
- `FramePinholeProjection` implements projection, inverse projection, ray, static pose, and image-correction capabilities against a definition/instance pair.
- `FramePinholeOptimization` exposes separate pose and optional calibration parameter blocks; updates produce a new instance/definition value rather than mutating a shared definition.

- [x] **Step 1: Port existing mathematical tests into new API as failing tests**

  Cover Tsai load values, Brown-Conrady projection, negative-depth handling, undistortion, scaled intrinsics, positive-depth derived view, and pose update. Do not include the old `FramePinholeCamera` header in the new test.

- [x] **Step 2: Run the new test and verify RED**

  Run `ctest --test-dir build/linux-source-release -R FramePinholeModel --output-on-failure`; expected result is failure because the split classes are absent.

- [x] **Step 3: Implement by moving the existing formulas behind the split boundary**

  Preserve numerical behavior and explicit camera-to-world convention. Move Tsai file parsing into `camera_io/tsai` in the next task; this task only implements in-memory model behavior.

- [x] **Step 4: Run new tests and old pinhole tests for comparison**

  Run `ctest --test-dir build/linux-source-release -R 'FramePinholeModel|CameraModel|CameraBaseline' --output-on-failure`. Record any numerical difference under `build/tmp/camera-architecture/frame-pinhole-diff.txt`.

- [x] **Step 5: Refactor only with green tests**

  Remove duplicated pose conversion helpers, then rerun all three test groups and `git diff --check`.

### Task 5: Move Tsai/project camera IO to definition/instance stores

**Files:**
- Create: `src/core/camera_io/tsai/TsaiCameraReader.h/.cpp`
- Create: `src/core/camera_project/CameraProjectStore.h/.cpp`
- Create: `src/core/camera_project/CameraProjectValidation.h/.cpp`
- Create: `src/core/camera_project/CMakeLists.txt`
- Test: `src/core/camera_project/tests/test_camera_project_store.cpp`
- Modify: `src/common/project/ProjectDocumentModel.*`
- Modify: `src/common/project/ProjectSessionModel.*`
- Modify: `src/common/project/ProjectIO.*`
- Modify: `src/core/CMakeLists.txt`, `tests/CMakeLists.txt`
- Delete after migration: `src/core/camera/ProjectCameraIO.h`, `src/core/camera/ProjectCameraIO.cpp`

**Interfaces:**
- `CameraProjectStore::load(document)` loads `project_files.camera_definitions` and `project_files.camera_instances` and rejects unknown schema versions, duplicate ids, dangling references, and duplicate image bindings.
- `CameraProjectStore::save(document, cameraData)` writes those two collections atomically.
- `CameraProjectValidation::validate(definitions, instances, images)` returns all errors in one result.
- `TsaiCameraReader` returns a `FramePinholeDefinition` plus source metadata; it never attaches a pose implicitly.

- [ ] **Step 1: Add failing schema tests**

  Test a valid definition/instance document, duplicate definition id, dangling instance reference, duplicate image instance, missing frame, and explicit rejection of a document containing `images[*].camera`.

- [ ] **Step 2: Run `test_camera_project_store` and verify RED**

  Run `cmake --build build/linux-source-release --target test_camera_project_store -j2`; expected result is failure before the store exists.

- [ ] **Step 3: Implement the new store and wire it into `ProjectDocumentModel`**

  Store camera collections exactly at `project_files.camera_definitions` and `project_files.camera_instances`. Remove `setImageCamera`, `setImageCameras`, `replaceImageCameras`, and `clearImageCameras`; replace them with definition/instance operations that validate the complete graph before committing.

- [ ] **Step 4: Run project tests**

  Run `ctest --test-dir build/linux-source-release -R 'CameraProjectStore|ProjectData|ProjectIO' --output-on-failure`. Expected result: new schema tests and updated project tests pass.

- [ ] **Step 5: Refactor persistence paths**

  Remove camera-specific path and metadata comments that refer to `images[*].camera`, rerun tests, and run `git diff --check`.

### Task 6: Migrate SfM, BA, and MVS to capability-based camera instances

**Files:**
- Modify: `src/core/sfm/project/ProjectMatchInputReader.*`
- Modify: `src/core/sfm/project/BaInputBuilder.*`
- Modify: `src/core/sfm/project/BaTrackBuilder.*`
- Modify: `src/core/sfm/project/MarkerBaAdapter.*`
- Modify: `src/core/sfm/reconstruction/SfmReconstruction.*`
- Modify: `src/gui/project/services/BundleAdjustService.*`
- Modify: `src/core/mvs/MvsPipelineService.*`, `src/core/mvs/PatchMatchHostUtils.*`, and public MVS input headers that accept `FramePinholeCamera`
- Test: affected SfM/BA/MVS tests plus `tests/test_camera_capability_requirements.cpp`

**Interfaces:**
- SfM/BA/MVS receive solver-owned `FramePinholeNumericState` values created from a typed instance only after the
  `Projection + StaticPose + Optimization` capability check.
- The capability-checked conversion is the numerical boundary; no `StaticPinholeView` compatibility object is retained.
- `CameraInstanceSet::forImage(imageId)` returns a typed instance or a structured missing-binding error.
- GUI services no longer parse camera JSON or construct `FramePinholeCamera` directly.

- [x] **Step 1: Add failing capability-boundary tests**

  Test that pinhole instances enter BA/MVS, RPC instances are rejected by MVS with `missing static_pose`, and a mixed set reports the exact image id that failed.

- [x] **Step 2: Run focused SfM/BA/MVS tests and observe RED**

  Run `ctest --test-dir build/linux-source-release -R 'CameraCapabilityRequirements|Sfm|BundleAdjust|Mvs' --output-on-failure`; expected result is compile or assertion failure until consumers use the new instances.

- [x] **Step 3: Migrate input builders and service signatures**

  Replace vectors of `FramePinholeCamera` with numeric states. Keep project JSON and typed camera instances at the IO boundary;
  numerical kernels never decode them or call a legacy camera adapter.

- [x] **Step 4: Run affected tests**

  Run the filtered SfM/BA/MVS command and build the affected camera, SfM, BA, MVS, aerial-triangulation, CLI and GUI targets.
  The Linux/GCC run completed with all 684 selected tests passing and the affected targets building successfully.

- [x] **Step 5: Remove direct concrete-camera includes from services**

  The search `rg -n 'FramePinholeCamera|cameraFromJson|imageCameraFromEntry' src/core/sfm src/core/mvs src/gui/project/services`
  returns no matches, and `git diff --check` passes.

### Task 7: Migrate RPC definition/instance and RPC services

**Files:**
- Create: `src/core/camera_models/rpc/RpcDefinition.*`, `RpcInstance.*`, `RpcProjection.*`, `RpcIntersectionService.*`, and model CMake/tests.
- Modify: `src/core/aerial_triangulation/engine/RpcEngine.*`, `RpcPointIntersection.*`, `RpcCoordinates.*`, and RPC tests.
- Modify: `src/core/camera/RpcCameraIO.*`, `RpcBiasAdjustment.*` while moving IO/adjustment to the new module.
- Delete after migration: `src/core/camera/RpcCameraModel.*`, `src/core/camera/RpcStereoIntersection.*`, old RPC project IO.

**Interfaces:**
- `RpcDefinition` contains RPC00B coefficients, geodetic/ECEF frame metadata, height domain, and validation.
- `RpcInstance` contains image size and image correction only; it does not expose static pose.
- `RpcProjection` implements projection, height-constrained inverse projection, and explicitly marked approximate chord rays.
- `RpcIntersectionService` accepts RPC instances and returns ECEF/geodetic points with residual diagnostics.

- [ ] **Step 1: Port RPC tests to the new types and add static-pose rejection**
- [ ] **Step 2: Run `ctest --test-dir build/linux-source-release -R 'Rpc|Aerial' --output-on-failure` and verify the new tests fail**
- [ ] **Step 3: Implement the split RPC types and migrate aerial engine inputs**
- [ ] **Step 4: Run the filtered RPC/aerial tests and the target build**
- [ ] **Step 5: Remove fixed-center helpers and old RPC class files; rerun `rg -n 'RpcCameraModel|RpcStereoIntersection'`**

### Task 8: Migrate pushbroom line-scan definition, trajectory, and optimization

**Files:**
- Create: `src/core/camera_models/linescan/LineScanDefinition.*`, `LineScanInstance.*`, `LineScanProjection.*`, `LineScanTrajectory.*`, `LineScanOptimization.*`, and CMake/tests.
- Modify: existing line-scan tests and any aerial/sparse consumers that include `PlanetaryLineScanCamera.h`.
- Delete after migration: `src/core/camera/PlanetaryLineScanCamera.*`, `PlanetaryLineScanProjection.cpp`, and old line-scan-only serialization helpers.

**Interfaces:**
- `LineScanDefinition` owns optical and detector parameters.
- `LineScanInstance` owns trajectory samples, line-time mapping, target/body-fixed frame, and instance bias.
- `LineScanTrajectory` implements time/line interpolation and returns explicit `TimeReference` values.
- `LineScanProjection` implements trajectory-based projection and ray capabilities; it never returns a static pose capability.
- `LineScanOptimization` exposes trajectory/attitude/time parameter blocks through the generic optimization contract.

- [ ] **Step 1: Add new API tests for line-time, ray, observed-line projection, and no-static-pose capability**
- [ ] **Step 2: Run the line-scan test target and verify RED**
- [ ] **Step 3: Move existing interpolation and distortion code behind the split API**
- [ ] **Step 4: Run line-scan and aerial tests plus `plascan_core` build**
- [ ] **Step 5: Delete the old line-scan class and check for stale includes**

### Task 9: Rebuild camera reference storage, resolver, comparison, and GUI adapters

**Files:**
- Modify: `src/core/camera_reference/CMakeLists.txt` and all old model/JSON files.
- Create/modify: `src/core/camera_reference/io/CameraReferenceSetStore.*`, `resolve/*`, `compare/*`.
- Modify: `src/gui/reference/MetashapeCameraReferenceSetBuilder.*`, `ProjectCameraReferenceRepository.*`, `CameraReferenceTreeModel.*`, `CameraReferenceController.*`, `CameraReferenceCsvExporter.*`.
- Modify: `tests/test_reference_models.cpp`, `tests/test_metashape_camera_reference_importer.cpp`.

**Interfaces:**
- Sidecar records contain `CameraReferenceObservation` plus resolved status; no duplicate `Vector3d`, matrix, or pose structs remain in GUI/core reference code.
- Builder maps source records to `ImageId` and stores unresolved source frame metadata.
- Repository validates the image-set fingerprint and loads/saves only the new reference schema.
- Comparator accepts `CameraInstance` and a resolved reference; static-pose comparison is available only when the instance advertises it. RPC comparison reports `region_model_without_static_pose`.
- GUI tree model consumes comparator results and does not read `images[*].camera` or manually decode `C/R`.

- [ ] **Step 1: Add failing reference-schema and comparator tests**
- [ ] **Step 2: Run `ctest --test-dir build/linux-source-release -R 'Reference|Metashape' --output-on-failure` and verify RED**
- [ ] **Step 3: Implement the new sidecar/resolver/comparator adapters**
- [ ] **Step 4: Run reference tests and build the GUI target**
- [ ] **Step 5: Remove old GUI-local reference structs and JSON decoding helpers**

### Task 10: Remove the aggregate camera module and update project/documentation contracts

**Files:**
- Modify: `src/core/CMakeLists.txt`, all camera/model/project/test CMake files, `src/gui/cmake/GuiCoreLinking.cmake`.
- Delete: the old aggregate `src/core/camera` files after their consumers are migrated. Move `CameraBaseline.*` to `src/core/camera_services/baseline/`, and move `CameraFormatConverter.*` plus `ColmapImageUndistorter.*` to `src/core/camera_io/external_formats/` before deleting the originals. `CameraModel.*` and `ProjectCameraIO.*` are deleted after their replacement interfaces are wired.
- Modify: `docs/PROJECT_ARCHITECTURE.md`, `docs/project/PLASCAN_PROJECT_FORMAT.md`, `src/core/camera/README.md` (replace with module READMEs), `README.md`, and affected source-contract tests.
- Test: `tests/test_repo_hygiene.py`, `tests/test_source_contracts.cpp`, project format tests.

**Interfaces:**
- CMake exposes `camera_core`, `camera_models_*`, `camera_io`, `camera_project`, `camera_reference`, and `camera_services` with the dependency direction in the spec.
- No target named `camera` is used by production consumers after this task.
- Documentation describes the new `camera_definitions`/`camera_instances` schema and capability requirements.

- [ ] **Step 1: Add source-contract tests that reject stale fields and targets**

  Assert that production source has no `images[*].camera`, `cameraFromJson`, `imageCameraFromEntry`, or old target links, while the new schema names and capability checks are present.

- [ ] **Step 2: Run the source-contract tests and verify RED**

  Run `ctest --test-dir build/linux-source-release -R 'SourceContracts|RepoHygiene' --output-on-failure`; expected result is failure while old code remains.

- [ ] **Step 3: Remove old files/targets and update all documentation**

  Do not leave forwarding headers or compatibility aliases. Update architecture diagrams and model docs to reflect actual file locations and behavior.

- [ ] **Step 4: Run affected tests and static checks**

  Run `python3 -m py_compile` on any changed Python validation scripts, `ctest --test-dir build/linux-source-release --output-on-failure`, and `git diff --check`.

- [ ] **Step 5: Inspect the final dependency graph**

  Run `cmake --build build/linux-source-release --target help`, inspect `compile_commands.json` for stale `src/core/camera` includes, and record any intentional test-only references in `build/tmp/camera-architecture/final-scan.txt`.

### Task 11: Full native verification and cleanup

**Files:**
- Modify only files required by failing verification, if any.
- Preserve: `build/tmp/camera-architecture/` only when a failure artifact is needed for reproduction.

- [ ] **Step 1: Configure with the project-standard entry point**

  Run `python3 scripts/env/configure_with_env.py --source-deps --build --test` and capture output under `build/tmp/camera-architecture/configure.log`.

- [ ] **Step 2: Run targeted suites**

  Run `python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure -R 'Camera|Reference|Sfm|BundleAdjust|Mvs|Aerial|Project'`.

- [ ] **Step 3: Run the full native test suite**

  Run `python3 scripts/env/run_tests.py --test-dir build/linux-source-release --output-on-failure`.

- [ ] **Step 4: Run repository checks**

  Run `git diff --check`, `python3 -m py_compile tests/test_repo_hygiene.py` when that Python file changes (the source-contract test is C++), and `git status --short`.

- [ ] **Step 5: Clean temporary artifacts and report limitations**

  Remove only temporary files created under `build/tmp/camera-architecture/` that are no longer needed. Keep no generated files in `testData/`, `resources/models/`, or the repository root.
