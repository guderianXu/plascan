# Camera Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** Tasks 1–9 and the v3 sidecar boundary are implemented on 2026-09-19; affected Linux/GCC targets pass. Task 8 closes the stale numeric-camera handoff between SFM writeback and the CLI MVS stage. Task 9 resolves matching/aerial reference geometry by `ImageId`, keeps paths as locators only, and the GUI/MVS sidecar reader accepts only `plascan.sfm_sparse_points.v3`. The project matching reader now rejects duplicate/unknown image locators, ambiguous normalized aliases, and any selected non-pinhole camera instead of silently shrinking the solve set; marker and survey observations prefer canonical IDs. Remaining camera work is limited to external resolver production wiring and non-pinhole solver paths.

**Goal:** Close the two correctness gaps left after the SfM/BA/MVS migration: malformed numeric camera state must fail at the project boundary, and solver inputs must reject mixed coordinate frames before geometry is evaluated.

**Architecture:** Keep the canonical `camera_core`/`camera_models`/`camera_project` layering. Add validation as an explicit numerical-state invariant without introducing a legacy compatibility path. Add a frame-consistency contract to `CameraInstanceSet`, enforce it in `ProjectMatchInputReader`, and make pairwise numeric geometry reject mixed frames as a second defensive boundary; frame conversion remains an explicit future operation through `CoordinateTransformService` rather than an implicit fallback.

**Tech Stack:** C++20, Qt6 JSON at the project edge, GoogleTest, existing Linux GCC release preset.

**Spec:** `docs/superpowers/specs/2026-09-18-camera-architecture-design.md`

## Global Constraints

- Old `.plascan` schemas and `images[*].camera` are rejected; no compatibility reader or dual-write path is added.
- Numerical code receives validated typed camera instances or solver-owned numeric state; it never silently interprets malformed JSON.
- A camera collection used by one geometric solve must have one explicit world coordinate frame; mixed frames fail with image and frame identifiers until an explicit transform is supplied.
- Core model headers remain Qt-independent.
- Every production change follows a red-green-refactor loop.
- Preserve unrelated working-tree changes; do not reset submodules or create commits/pushes.
- Temporary artifacts go under `build/tmp/camera-architecture/` and are removed when no longer needed.

## Review Focus

- Missing, zero, negative, or non-finite pinhole intrinsics at the legacy JSON edge must be rejected before a state is considered usable; tests belong to Task 1.
- A non-orthonormal or non-finite pose in a decoded numeric state must be rejected; tests belong to Task 1.
- Two valid camera instances with different world-frame IDs must fail closed with both image IDs/frames, and pairwise triangulation must reject the same mismatch; tests belong to Task 2.
- A single-frame instance set and an empty set must preserve their current successful/empty semantics; tests belong to Task 2.
- Existing pinhole project input and RPC capability rejection must remain explicit; a static solve that selects an RPC or pushbroom image now fails as a whole instead of silently dropping that image. Affected SfM tests belong to Task 2.

### Task 1: Make numeric-state validation explicit at the project boundary

**Files:**
- Modify: `src/core/camera_models/frame_pinhole/FramePinholeNumericState.h`
- Modify: `src/core/camera_models/frame_pinhole/FramePinholeNumericState.cpp`
- Modify: `src/core/camera/ProjectCameraIO.cpp`
- Test: `src/common/project/test/test_project_camera_io.cpp`

**Interfaces:**
- Add `bool validateNumericalState(std::string* error = nullptr) const noexcept` to `FramePinholeNumericState`. It validates finite positive focal lengths and pixel pitch, finite principal point and distortion, signed axes, finite camera center, and a proper orthonormal camera-to-world rotation. It does not require an image size because legacy camera metadata has no image-size field.
- `decodeFramePinholeNumericState()` must call this validator after decoding and return `false` for invalid values; it must not silently accept `QJsonValue::toDouble()` defaults.
- `toInstance()` must return `nullptr` when the numeric state fails the same invariant instead of allowing a model constructor exception to escape.

- [x] **Step 1: Write failing tests**

  Extend `ProjectCameraIOTest` with: a valid serialized state still decodes; zero focal length is rejected; a negative pixel pitch is rejected; and a rotation containing a non-finite/degenerate value is rejected. Add one test that manually builds an invalid numeric state and verifies `toInstance()` returns `nullptr`.

- [x] **Step 2: Run the focused tests and observe RED**

  Run `cmake --build build/linux-source-release --target test_common_project_camera_io -j2` followed by `ctest --test-dir build/linux-source-release -R 'ProjectCameraIOTest' --output-on-failure`. The new rejection tests must fail before the implementation changes.

- [x] **Step 3: Implement the validator and boundary call**

  Reuse the existing pinhole definition validation rules for intrinsics/distortion, add finite/orthonormal/determinant checks for the numeric pose, produce a short field-specific error string, call it from `decodeFramePinholeNumericState()` and `toInstance()`, and keep solver setters source-compatible for existing synthetic tests.

- [x] **Step 4: Run the focused and affected tests**

  Re-run `ctest --test-dir build/linux-source-release -R 'ProjectCameraIOTest|FramePinholeModel|CameraProjectStore|CameraCapabilityRequirements' --output-on-failure` and rebuild the `camera` and `camera_models_frame_pinhole` targets.

- [x] **Step 5: Refactor and inspect**

  Remove duplicated finite/rotation checks introduced by the task, run `git diff --check`, and confirm no compatibility field or alternate decode path was added.

### Task 2: Enforce one world frame at the numerical input boundary

**Files:**
- Modify: `src/core/camera_core/model/CameraInstanceSet.h`
- Modify: `src/core/camera_core/model/CameraInstanceSet.cpp`
- Modify: `src/core/camera_models/frame_pinhole/FramePinholeNumericState.cpp`
- Modify: `src/core/sfm/project/ProjectMatchInputReader.cpp`
- Test: `tests/test_camera_core_registry.cpp`
- Test: `src/core/sfm/test/test_ba_input_builder.cpp`
- Test: `src/core/sfm/test/test_triangulation_service.cpp`

**Interfaces:**
- Add `CameraInstanceSetFrameResult requireCommonWorldFrame() const` with an optional common frame and per-image failures containing the image ID, observed frame, and reason.
- The method returns success for an empty set and for a set whose definitions all carry the same frame; it returns failure for any mixed frame without attempting an implicit conversion.
- `readProjectMatchInput()` must call this check immediately after `CameraProjectRuntime::load()` and report the first structured frame error through `ProjectMatchInputDiagnostics::firstCameraError`.
- `FramePinholeNumericState::triangulatePair()` must return an invalid intersection when the two states carry different `worldFrame()` values, before calculating a midpoint or reprojection error.

- [x] **Step 1: Write failing tests**

  Add registry tests for empty, single-frame, and mixed-frame sets. Add a project-input regression that constructs two otherwise valid pinhole instances with different definition frames and asserts `readProjectMatchInput()` returns `false` with both the image and frame names in the diagnostic. Add a triangulation regression that gives two otherwise valid numeric states different frame IDs and asserts the returned `PairIntersection.valid` is `false`.

- [x] **Step 2: Run the focused tests and observe RED**

  Run `cmake --build build/linux-source-release --target test_camera_core_registry test_ba_input_builder test_ba_track_builder test_triangulation_service -j2` and `ctest --test-dir build/linux-source-release -R 'CameraCoreRegistry|ProjectMatchInputReader|Triangulation' --output-on-failure`. The mixed-frame assertions must fail before the new contract is wired.

- [x] **Step 3: Implement the set contract and input check**

  Compare `definition().worldFrame()` values by ID, preserve the first frame as the common frame, collect every conflicting image/frame pair, and stop before numeric-state construction. Add the same-frame guard at the start of `triangulatePair()`, before any ray midpoint is formed. Do not call `CoordinateTransformService` implicitly; callers that need mixed frames must normalize them explicitly first.

- [x] **Step 4: Run affected tests and build**

  Run `ctest --test-dir build/linux-source-release --output-on-failure -R 'CameraCoreRegistry|ProjectMatchInputReader|Sfm|BundleAdjust|Mvs|CameraProjectStore'` and build `camera_core sfm_core bundle_adjust mvs_backend mvs_pipeline`.

- [x] **Step 5: Refactor and inspect**

  Keep the diagnostic wording stable, run `git diff --check`, and audit that no solver or project reader creates an implicit frame conversion.

### Task 3: Record the external-reference integration boundary

**Files:**
- Modify: `docs/superpowers/plans/2026-09-19-camera-boundary-hardening.md`
- Inspect only: `src/core/camera_reference/resolve/CameraReferenceResolver.*`, `src/gui/reference/MetashapeCameraReferenceSetBuilder.*`, `src/core/matchphototask/*`, `src/core/aerial_triangulation/*`

**Interfaces:**
- Document the next migration seam: imported external observations are resolved into `ResolvedCameraReference` in the target solver frame, then converted to a typed pose prior; matching geometry continues to require a typed camera model with intrinsics and must not receive a pose-only reference.
- No code is changed in this task; its deliverable is a precise follow-up task list after Tasks 1–2 expose stable frame and validation contracts.

- [x] **Step 1: Verify current call graph**

  Confirm that `CameraReferenceResolver` has no production caller and that current guided matching still accepts `FramePinholeCamera` maps.

- [x] **Step 2: Record the smallest next migration**

  Add a short note to this plan naming the adapter type, owning module, and tests needed to connect the resolver without adding a compatibility wrapper.

#### Follow-up seam recorded after inspection (2026-09-19)

`CameraReferenceResolver::resolve()` currently has no production caller. The GUI Metashape builder still stores a Qt `CameraReferenceSet` whose orientation and lever-arm semantics are explicitly unresolved, while guided matching and aerial triangulation continue to accept `QMap`/`QHash<QString, FramePinholeCamera>` and feed `ReferencePoseEpipolarGeometry`. The resolver result therefore cannot be passed to matching as a camera: `ResolvedCameraReference` contains a resolved pose and covariance, but no focal lengths, principal point, distortion, or pixel convention.

The smallest direct migration is a typed adapter at the solver boundary:

1. `camera_reference` owns a value such as `ResolvedCameraPosePrior`, keyed by `ImageId`, carrying the resolved target-frame `Pose`, covariance, transform hash, and lever-arm provenance. It is constructed only from a `ResolvedCameraReference` whose status is `Resolved`; unresolved observations are rejected with their reason. The adapter must make the lever-arm vector frame and orientation convention explicit before the GUI source can feed it.
2. `sfm`/`bundle_adjust` owns the index-alignment step that converts those image-keyed pose priors to the existing `BACameraPosePrior` vector after checking that every prior target frame equals the numerical camera set frame. This is a direct typed conversion, not a second legacy map.
3. `matchphototask` and `aerial_triangulation` each get a separate `ReferenceCameraGeometry` adapter that joins a resolved pose prior with the canonical `FramePinholeInstance`/`FramePinholeNumericState` for the same `ImageId`. It must return validated intrinsics-plus-pose geometry and fail on a missing model or frame mismatch; a pose-only reference must never be substituted for a pinhole camera.
4. Replace the old path-keyed `FramePinholeCamera` parameters at those API boundaries once the adapters land, then remove the old fields. Do not add a dual-read/dual-write compatibility wrapper.

The follow-up tests should cover resolver success/failure and lever-arm frame conversion in `tests/test_camera_reference_resolver.cpp`, image/frame alignment and covariance-to-prior mapping in a new `src/core/sfm/test/test_camera_reference_pose_prior.cpp`, and the intrinsic-plus-pose requirement in `src/core/matchphototask/tests/test_match_photos_task.cpp` and `src/core/aerial_triangulation/tests/test_aerial_triangulation_workflow.cpp`. The existing GUI reference builder tests should also assert that unresolved source conventions cannot enter the typed adapter.

### Task 4: Connect typed external pose priors to BA and incremental SfM

- [x] Add `ResolvedCameraPosePrior` with explicit target frame, covariance, transform provenance, and observation-resolution fingerprint.
- [x] Add `CameraReferencePosePriorAdapter` at the SfM/BA boundary; reject unresolved, unbound, duplicate, mixed-frame, or mixed-provenance references.
- [x] Wire the typed prior vector through project BA and incremental SfM; preserve matched/ignored diagnostics and prevent mixing with manual pose-prior sources.

### Task 5: Make aerial camera identity/frame propagation explicit

- [x] Add `SolverCameraBinding` to GUI/CLI-facing and prepared aerial inputs.
- [x] Validate binding count, uniqueness, common frame, and external-reference requirements before reading the tie-point graph.
- [x] Bind every loaded/estimated pinhole numeric state through the explicit boundary; never derive identity from a path, file name, or numeric index.
- [x] Resolve complete canonical project bindings from validated `camera_instances` by `image_uuid`; reject explicit bindings that disagree with the canonical collection.

### Task 6: Gate every pinhole numerical consumer with one invariant

- [x] Validate numerical intrinsics, distortion, pose, and frame at the shared BA boundary and the SfM/aerial entry points.
- [x] Validate MVS pipeline and depth fusion inputs before raster, CUDA, workspace, or fusion side effects.
- [x] Prevent an unbound solver state from being promoted back into a typed project instance through `toInstance()`.
- [x] Add regressions for unbound promotion, canonical identity propagation, explicit identity mismatch, and missing external bindings.

### Task 7: Close the MVS workspace replay decoder bypass

- [x] Reject missing or non-numeric required replay camera fields instead of accepting `QJsonValue::toDouble()` defaults.
- [x] Treat optional Brown distortion fields as zero only when absent; reject malformed values when present.
- [x] Reuse `FramePinholeNumericState::validateNumericalState()` so replay rotation, center, intrinsics, and distortion obey the same invariant as project/SfM/BA/MVS inputs.
- [x] Add regressions for missing principal point, malformed distortion, negative focal length, and non-orthonormal pose.

### Task 8: Re-resolve canonical frame-pinhole cameras after SFM writeback

**Files:**

- Modify: `src/cli/workflows/ReconstructionPipelineRunner.cpp`
- Test: `src/cli/workflows/tests/test_workflow_cli.cpp`
- Document: `src/core/mvs/README.md`, `docs/PROJECT_ARCHITECTURE.md`

**Interfaces and boundary:**

- After successful SFM, `updateCameraInstances()` writes the refined camera instances into the active Chunk. The CLI must refresh its project snapshot and load `project_files.camera_definitions`/`camera_instances` through `CameraProjectRuntime::load()` before constructing any MVS `CameraView`.
- Resolve each registered image path through the canonical image table (`image_uuid`) and call `CameraProjectRuntimeResult::framePinholeStateForImage()`. Require a bound `instanceId`/`imageId` and one common `worldFrame`; missing, non-pinhole, incomplete, or mixed-frame entries fail before MVS image preparation.
- Do not reconstruct MVS identity from `pendingCamUpdates`, path keys, list order, or the generic `serializeFramePinholeNumericState()` output. That serializer is a numeric geometry snapshot (it may carry the world-frame field) and does not transmit the canonical instance or image identity.

- [x] **Step 1: Refresh the post-SFM project snapshot**

  Re-read `projectSession.mergedMetadata()` immediately after the SFM camera-instance transaction so later stages cannot consume the pre-writeback camera graph.

- [x] **Step 2: Resolve the canonical MVS camera set**

  Map normalized registered paths to canonical image UUIDs, load the runtime registry, resolve frame-pinhole numeric states by image ID, and reject missing identity or mixed frames before creating MVS views.

- [x] **Step 3: Keep generic numeric serialization identity-free**

  Continue using the numeric serializer for reports/exports only. It must not become a second project-camera transport or a source of solver identity.

- [x] **Step 4: Add the CLI contract regression**

  Keep a source-level contract test asserting the runtime load and identity-aware resolver are used and the old path-keyed numeric decode path is absent.

- [x] **Step 5: Close the replay identity boundary**

  MVS-specific camera artifacts now persist `instance_id`, `image_id`, and `world_frame` when the source state is
  bound. Replay requires the complete binding, rejects duplicate identities and mixed frames, and binds the
  decoded numeric state before opening a raster. The depth-input hash includes the same identifiers and the
  algorithm revision advances so pre-binding workspaces are stale rather than implicitly compatible.
  Stored-depth fusion applies the same strict binding and compares the artifact identity with the current canonical
  camera before consuming prepared/grid depth data.

  Derived MVS raster cameras must preserve the source binding as well. The epipolar rectifier now copies the source
  numeric state before changing the rectified pose/intrinsics, so a transformed camera cannot silently become an
  unbound solver state.

**Known follow-up risk:** external reference resolver output still has no production GUI/CLI adapter in every import path,
and RPC/line-scan numerical consumers intentionally remain model-specific. They must continue to fail at the pinhole
geometry boundary unless an explicit model adapter is supplied; a resolved pose-only external reference must never be
substituted for camera geometry.

### Task 9: Replace path-keyed reference geometry at matching and aerial boundaries

**Status:** complete for the current matching/aerial boundary; external resolver wiring remains a follow-up seam

`ReferenceCameraGeometry` and `ReferenceCameraPosition` now live in `camera_reference_core`. Both carry an explicit
`ImageId` and `CoordinateFrameId`; projection geometry additionally requires a validated, bound
`FramePinholeNumericState`. Matching and aerial options use ImageId-keyed maps, while paths are retained only as input
locators paired with an explicit ordered `imageIds` vector. Position-only references cannot be passed to an epipolar
projector, and mixed reference frames fail before pair planning.

The remaining migration replaces the GUI and CLI path-map builders with the typed maps, converts external Tsai input
only after joining it to a canonical project instance/frame, and removes the old fields from guided matching and aerial
workflow contexts. No dual fields or legacy read/write compatibility layer is introduced.

## Verification

After Tasks 1–2 and Task 8, run `git diff --check`, the focused tests above, and the existing camera/SfM/BA/MVS filter. Do not claim the full camera migration complete: RPC, line-scan, GUI reference adapters, guided matching's old path-keyed maps, and old post-processing consumers remain separate follow-up work.
