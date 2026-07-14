# Camera Single Model Implementation Plan

> **For agentic workers:** Execute inline in the current dirty workspace. Do not create subagents, compatibility wrappers, commits, or worktrees. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `PositiveDepthCameraModel` with `Camera` everywhere and make MVS explicitly undistort images before using zero-distortion pinhole geometry.

**Architecture:** `Camera` remains the only public model. It can return a positive-depth-normalized `Camera`, project with positive depth, and unproject distorted pixels. `MvsImagePreprocessor` transforms actual images and returns a zero-distortion `Camera`; CUDA keeps only translation-unit-private float parameter structs.

**Tech Stack:** C++17, CMake, GoogleTest, OpenCV, Qt6, optional CUDA.

---

### Task 1: Establish the baseline and desired Camera API

**Files:**
- Modify: `src/core/camera/test/Camera_tests.cpp`
- Modify: `src/core/camera/Camera.h`
- Modify: `src/core/camera/Camera.cpp`

- [ ] **Step 1: Run the existing camera tests before production edits**

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON
cmake --build build/windows-vcpkg-cuda-release --target test_camera_unit -j 8
ctest --test-dir build/windows-vcpkg-cuda-release/src/core/camera --output-on-failure -R "Camera"
```

Record any pre-existing failure by exact name.

- [ ] **Step 2: Add failing positive-depth Camera tests**

Tests call the desired API:

```cpp
const Camera normalized = camera.normalizedForPositiveDepth();
EXPECT_TRUE(normalized.projectWorldPoint(world, normalizedPixel));
EXPECT_TRUE(camera.projectWorldPointWithDepth(world, originalPixel, depth));
EXPECT_TRUE(camera.unprojectPixel(originalPixel, depth, restoredWorld));
```

Cover nonzero `p1/p2`, negative u/v directions, flipped depth and round-trip world coordinates.

- [ ] **Step 3: Run the new tests and verify RED**

Build `test_camera_unit`; expected compile failure is missing `normalizedForPositiveDepth`, `projectWorldPointWithDepth`, and `unprojectPixel`.

- [ ] **Step 4: Implement the minimal Camera behavior**

Add declarations to `Camera.h` and definitions to `Camera.cpp`. `normalizedForPositiveDepth()` returns `Camera`, adjusts pose and tangential coefficients, resets axis signs/depth flip, and keeps all other state. Projection uses existing distortion code; unprojection uses `undistortPixel()` and `R_cw`.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run `test_camera_unit`; all old and new camera tests must pass.

### Task 2: Add explicit MVS image preprocessing

**Files:**
- Create: `src/core/mvs/MvsImagePreprocessor.h`
- Create: `src/core/mvs/MvsImagePreprocessor.cpp`
- Create: `src/core/mvs/tests/test_mvs_image_preprocessor.cpp`
- Modify: `src/core/mvs/CMakeLists.txt`

- [ ] **Step 1: Write the failing preprocessing tests**

Desired interface:

```cpp
bool prepareMvsImage(const cv::Mat &source,
                     const Camera &sourceCamera,
                     cv::Mat *prepared,
                     Camera *preparedCamera,
                     std::string *errorMessage = nullptr);
```

Test empty input rejection, invalid camera rejection, zero-distortion no-remap behavior, and nonzero Brown distortion remapping with an output camera whose distortion is exactly zero and whose axes use positive-depth convention.

- [ ] **Step 2: Run and verify RED**

Configure/build `test_mvs_image_preprocessor`; expected failure is the missing implementation/target.

- [ ] **Step 3: Implement the preprocessor**

Normalize the `Camera`, build OpenCV `K` and `[k1,k2,p1,p2,k3]`, use `cv::initUndistortRectifyMap` plus `cv::remap`, then clear distortion on the returned `Camera`. Preserve source size and channel count.

- [ ] **Step 4: Run and verify GREEN**

Run `test_mvs_image_preprocessor` and `test_camera_unit`.

### Task 3: Migrate MVS public contracts to Camera

**Files:**
- Modify: `src/core/mvs/MvsTypes.h`
- Modify: `src/core/mvs/DepthMapGenerator.h`
- Modify: `src/core/mvs/DepthFrameUtils.*`
- Modify: `src/core/mvs/DepthPyramidEstimator.h`
- Modify: `src/core/mvs/PatchMatchCUDA.*`
- Modify: `src/core/mvs/PatchMatchNoCUDA.cpp`
- Modify: `src/core/mvs/DenseCloudBuilder.*`
- Modify: `src/core/mvs/DensePointCloudCUDA.*`

- [ ] **Step 1: Replace public types**

Change all model fields and parameters from `PositiveDepthCameraModel` to `Camera`; remove `CameraView::positiveDepthModel()`.

- [ ] **Step 2: Replace field access**

Use `intrinsics()`, `worldToCameraRotation()`, `worldToCameraTranslation()`, `cameraCenter()`, `projectWorldPointWithDepth()` and `unprojectPixel()`. CUDA `.cu` files convert `Camera` to private float structs immediately before kernel launch.

- [ ] **Step 3: Compile the MVS target**

Run `cmake --build ... --target mvs -j 8`; use compiler errors to locate remaining contract mismatches, without reintroducing aliases.

### Task 4: Integrate prepared images and Camera values

**Files:**
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/DepthMapFusion.*`
- Modify: `src/core/mvs/EpipolarRectifier.*`
- Modify: `src/core/mvs/StereoDenseCloudPipelinePaths.cpp`
- Modify: `src/core/mvs/DisparityTriangulator.*`
- Modify: `src/core/mvs/MvsSceneClassifier.cpp`
- Test: `src/core/mvs/EpipolarRectifier_tests.cpp`

- [ ] **Step 1: Add a failing rectifier contract test**

Pass a `Camera` with nonzero distortion directly to `rectify()` and expect a clear failure. Then preprocess both images and expect rectification success.

- [ ] **Step 2: Verify RED**

Run the focused rectifier test; expected initial failure is that the rectifier silently accepts distortion.

- [ ] **Step 3: Prepare every reference/source image**

In `DepthMapGenerator`, call `prepareMvsImage()` before PatchMatch. Persist the returned working `Camera` in `DepthFrameResult`; scale it whenever the image/depth grid is resized.

- [ ] **Step 4: Keep color and depth geometry aligned**

Store the original `Camera` alongside the working `Camera` in `FusionFrameInput`. `ColorImageCache` applies `prepareMvsImage()` to the original color image and resizes it to the depth grid before sampling.

- [ ] **Step 5: Make rectification operate only on zero-distortion Camera**

Use Camera getters to build `K/R/t`; output a zero-distortion `Camera` with updated intrinsics and camera-to-world pose. Replace all direct `R_cw/T/C` access in depth consistency and fusion.

- [ ] **Step 6: Verify GREEN**

Build/run `test_mvs_rectifier_unit`, `test_mvs_image_preprocessor`, `test_mvs_pipeline` and `test_mvs_types`.

### Task 5: Migrate mesh, QC, CLI, GUI and tests

**Files:**
- Modify every remaining file reported by `rg "PositiveDepthCameraModel|toPositiveDepthModel|positiveDepthModel" src tests`
- Modify: `src/core/mesh/DepthMapMeshBuilder.*`
- Modify: `src/core/mesh/VisualHullReconstructor.h`
- Modify: `src/core/qc/ModelImageQualityTypes.h`
- Modify: `src/core/qc/ModelImageQualityEvaluator.cpp`
- Modify: `src/core/qc/ModelMeshRenderer.*`
- Modify: `src/cli/cli_epipolar_rectify.cpp`
- Modify: `src/cli/cli_model_quality.cpp`
- Modify: `src/gui/markers/MarkerFocusMeasurementDialog.cpp`

- [ ] **Step 1: Replace construction and serialization**

Use `Camera` setters for tests/JSON parsing and getters for serialization. Parsed depth-camera artifacts create a zero-distortion, positive-depth `Camera`.

- [ ] **Step 2: Replace math calls**

Use `Camera` projection/unprojection and explicit rotation/translation snapshots; do not add public field aliases.

- [ ] **Step 3: Build affected targets**

Build mesh, QC, CLI, GUI-related test targets available in the configured build and fix only migration errors.

### Task 6: Delete the old model and update documentation

**Files:**
- Delete: `src/core/camera/PositiveDepthCameraModel.h`
- Delete: `src/core/camera/PositiveDepthCameraModel.cpp`
- Modify: `src/core/camera/CMakeLists.txt`
- Modify: `src/core/camera/README.md`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify source-contract tests that asserted the deleted compatibility type

- [ ] **Step 1: Remove source files and CMake entries**

Delete both files without a forwarding header or type alias.

- [ ] **Step 2: Update docs and source contracts**

Document Camera as the sole model and MVS preprocessing contract. Replace obsolete string assertions with checks for `Camera`, explicit image remapping and absence of the deleted symbol.

- [ ] **Step 3: Confirm removal**

Run:

```powershell
rg -n "PositiveDepthCameraModel|toPositiveDepthModel|positiveDepthModel" src tests
```

Expected: no matches.

### Task 7: Final verification

- [ ] **Step 1: Configure and build**

Run CMake configure with `BUILD_TESTS=ON`, then build `camera`, `mvs`, targeted mesh/QC/CLI targets, followed by the normal project build if resource limits allow.

- [ ] **Step 2: Run focused tests**

Run Camera, MVS, mesh, QC and source-contract tests with `--output-on-failure`.

- [ ] **Step 3: Run full CTest**

Run `ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure`; report the exact result and separately identify the documented historical terrain DOM failure if present.

- [ ] **Step 4: Inspect diffs**

Run `git diff --check` and `git status --short`; confirm only task-related portions were changed and no unrelated user modifications were reverted.
