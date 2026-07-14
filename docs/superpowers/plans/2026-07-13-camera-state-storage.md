# Camera Structured State Storage Implementation Plan

> **For agentic workers:** Execute inline in the current workspace. Do not create subagents or commits; the repository contains unrelated user changes that must be preserved.

**Goal:** Make `Camera::Intrinsics`, `Camera::Distortion`, and `Camera::Pose` the only persistent camera parameter state without changing camera math or file-format behavior.

**Architecture:** Replace the private scalar parameter fields with `_intrinsics`, `_distortion`, and `_pose`. Existing public operations read and update those structures directly; no alias fields, deprecated wrappers, or synchronized compatibility state are introduced. Neighboring code may consume structured snapshots where that makes the new state boundary clearer.

**Tech Stack:** C++17, CMake, GoogleTest, OpenCV-dependent PlaScan camera core.

## Global Constraints

- Preserve all existing Tsai parsing, projection, distortion, unit-conversion, and depth-axis semantics.
- Preserve the user's existing comments and moved `src/core/camera/test/` layout.
- Do not modify unrelated dirty files and do not create a Git commit.
- Keep C++17, four-space indentation, Allman braces, and lines near 120 characters.

---

### Task 1: Establish Camera behavior baseline

**Files:**
- Read: `src/core/camera/Camera.h`
- Read: `src/core/camera/Camera.cpp`
- Test: `src/core/camera/test/Camera_tests.cpp`

**Interfaces:**
- Consumes: existing `Camera` public API and configured `build` directory.
- Produces: baseline result for `test_camera_unit` before the storage refactor.

- [ ] **Step 1: Configure tests**

Run:

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
```

Expected: configuration completes and the moved camera test sources are accepted.

- [ ] **Step 2: Build and run the synthetic camera tests**

Run:

```powershell
cmake --build build --target test_camera_unit -j 4
ctest --test-dir build --output-on-failure -R "CameraBasic|CameraIntrinsics|CameraProjection|CameraToPositiveDepthModel"
```

Expected: all discovered synthetic Camera tests pass. Any pre-existing failure is recorded before code changes.

---

### Task 2: Replace duplicated private fields with structured state

**Files:**
- Modify: `src/core/camera/Camera.h`
- Modify: `src/core/camera/Camera.cpp`
- Modify: `src/core/camera/PositiveDepthCameraModel.cpp`
- Test: `src/core/camera/test/Camera_tests.cpp`

**Interfaces:**
- Consumes: `Camera::Intrinsics`, `Camera::Distortion`, and `Camera::Pose` with their current field names and defaults.
- Produces: one `_intrinsics`, one `_distortion`, and one `_pose` member as the only persistent parameter state.

- [ ] **Step 1: Add a structured-state characterization test**

Add a test that initializes every parameter group through existing controlled setters, reads `intrinsics()`, `distortion()`, and `pose()`, verifies all fields, and verifies that modifying a returned snapshot does not mutate the camera.

- [ ] **Step 2: Run the characterization test before refactoring**

Run:

```powershell
cmake --build build --target test_camera_unit -j 4
ctest --test-dir build --output-on-failure -R "CameraStructuredState"
```

Expected: PASS, establishing unchanged public behavior for this storage-only refactor.

- [ ] **Step 3: Replace private scalar storage in `Camera.h`**

Use:

```cpp
Intrinsics _intrinsics;
Distortion _distortion;
Pose _pose;
bool _isLoaded = false;
```

Update inline getters and setters to access these members directly. Do not retain any of `_focalX`, `_radialK1`, `_cameraCenter`, or their sibling fields.

- [ ] **Step 4: Update `Camera.cpp` to use the unique structures**

Return the three structures from their grouped getters and replace all internal scalar-field access with the corresponding structured field. Preserve the special `loadFromFile()` behavior that does not reset `depthAxisFlipped` unless `w_direction` is present.

- [ ] **Step 5: Update the nearest consumer**

In `PositiveDepthCameraModel.cpp`, read one `Intrinsics` snapshot and one `Pose` snapshot, then populate the positive-depth model from those values. This exercises the intended grouped API without adding a compatibility layer.

- [ ] **Step 6: Confirm duplicate fields are gone**

Run:

```powershell
rg -n "_(focalX|focalY|principalX|principalY|pixelPitch|uAxisSign|vAxisSign|radialK[123]|tangentialP[12]|cameraCenter|cameraToWorldRotation|depthAxisFlipped)\b" src/core/camera/Camera.h src/core/camera/Camera.cpp
```

Expected: no matches.

---

### Task 3: Verify camera compatibility and regressions

**Files:**
- Verify: `src/core/camera/Camera.h`
- Verify: `src/core/camera/Camera.cpp`
- Verify: `src/core/camera/PositiveDepthCameraModel.cpp`
- Verify: `src/core/camera/test/Camera_tests.cpp`

**Interfaces:**
- Consumes: refactored Camera implementation.
- Produces: compiled camera library and passing Camera/Tsai/format-conversion tests.

- [ ] **Step 1: Build all camera targets**

Run:

```powershell
cmake --build build --target camera test_camera_unit test_camera_tsai test_camera_format_converter -j 4
```

Expected: all targets compile and link.

- [ ] **Step 2: Run all camera tests**

Run:

```powershell
ctest --test-dir build --output-on-failure -R "Camera|camera"
```

Expected: all camera-related CTest cases pass. Dataset-dependent skips or failures are reported by exact test name.

- [ ] **Step 3: Inspect the final diff**

Run:

```powershell
git diff --check -- src/core/camera/Camera.h src/core/camera/Camera.cpp src/core/camera/PositiveDepthCameraModel.cpp src/core/camera/test/Camera_tests.cpp
git diff -- src/core/camera/Camera.h src/core/camera/Camera.cpp src/core/camera/PositiveDepthCameraModel.cpp src/core/camera/test/Camera_tests.cpp
```

Expected: no whitespace errors; diff contains only the structured-state refactor, its focused test, and preserved existing user comments.

---

### Task 4: Separate Camera declarations from multi-line definitions

**Files:**
- Modify: `src/core/camera/Camera.h`
- Modify: `src/core/camera/Camera.cpp`
- Test: `src/core/camera/test/Camera_tests.cpp`

**Interfaces:**
- Consumes: the existing public setter signatures.
- Produces: a declaration-focused header and one implementation location for every nontrivial setter.

- [ ] **Step 1: Replace multi-line setter bodies with declarations**

Keep the existing signatures for `setPose`, `setCameraCenter`, `setIntrinsics`,
`setIntrinsicsMillimeters`, `setPixelPitch`, `setAxisDirections`,
`setDepthAxisFlipped`, and both `setDistortion` overloads, followed by semicolons.

- [ ] **Step 2: Add the definitions to `Camera.cpp`**

Place the definitions after the grouped state getters and before coordinate conversion methods. Copy each existing body without changing validation, `_isLoaded`, units, or axis normalization behavior.

- [ ] **Step 3: Verify the header boundary**

Run:

```powershell
rg -n -U "void set[^;]*\n\s*\{" src/core/camera/Camera.h
```

Expected: no matches.

- [ ] **Step 4: Rebuild and run all Camera tests**

Run the three camera test executables after building `camera`, `test_camera_unit`,
`test_camera_tsai`, and `test_camera_format_converter`.

Expected: 23/23 tests pass with no behavior changes.

---

### Task 5: Document Camera parameters and Tsai usage

**Files:**
- Modify: `src/core/camera/Camera.h`
- Modify: `src/core/camera/Camera.cpp`
- Modify: `src/core/mvs/MvsTypes.h`
- Modify: `src/gui/markers/MarkerFocusMeasurementDialog.cpp`
- Create: `src/core/camera/README.md`

**Interfaces:**
- Consumes: current `Camera` parser, saver, projection API, and `PositiveDepthCameraModel`.
- Produces: field-level Doxygen comments, direct positive-depth type usage, and module documentation.

- [ ] **Step 1: Add field comments**

Document pixel/mm units, row-major `R_cw`, world-coordinate camera center, normalized-coordinate distortion coefficients, and physical forward-depth behavior.

- [ ] **Step 2: Remove the nested positive-depth alias**

Change `toPositiveDepthModel()` to return `PositiveDepthCameraModel` directly and update explicit `Camera::PositiveDepthModel` references. Do not add a replacement alias.

- [ ] **Step 3: Write the module README**

Document required and optional Tsai fields, parser defaults and validation, projection conventions, manual initialization, load/project/undistort/scale/save examples, and positive-depth conversion.

- [ ] **Step 4: Verify code and documentation**

Build Camera and affected MVS/GUI targets, run all three Camera test binaries, scan for `Camera::PositiveDepthModel`, and run `git diff --check`.

Expected: no obsolete alias references, affected targets compile, and 23/23 Camera tests pass.
