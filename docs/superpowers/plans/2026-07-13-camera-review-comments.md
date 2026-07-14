# Camera Review Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 camera 模块的生产代码、测试代码和 CMake 补充面向 review 的中文注释，同时保持所有可执行行为不变。

**Architecture:** 注释按“文件职责—类型/API 契约—关键数学与格式分支—测试意图”四层组织。只编辑注释和必要空行；不拆分既有大文件，不改符号、表达式、字符串、路径、目标名或测试断言。

**Tech Stack:** C++17、CMake、GoogleTest、CTest、ASP Tsai、Middlebury、EPFL、COLMAP、Metashape

---

### Task 1: Public camera API contracts

**Files:**
- Modify: `src/core/camera/Camera.h`
- Modify: `src/core/camera/PositiveDepthCameraModel.h`
- Modify: `src/core/camera/CameraFormatConverter.h`

- [ ] **Step 1: Complete `Camera` contract comments**

  Keep the existing file overview and Doxygen blocks, then add missing review context for axis-sign accessors,
  `depthAxisFlipped()`, `setDistortion(const Distortion &)`, and validity semantics. Use comments such as:

  ```cpp
  /// 返回像素 u 轴相对相机 x 轴的方向符号（仅为 +1 或 -1）。
  int uAxisSign() const { return _uAxisSign; }

  /// 返回是否使用 `Z_cam < 0` 作为物理前向；该标志来自 Tsai `w_direction`。
  bool depthAxisFlipped() const { return _depthAxisFlipped; }
  ```

- [ ] **Step 2: Document the positive-depth snapshot layout and math**

  Add a file overview and Doxygen blocks explaining that the public unions are compatibility aliases over the
  same storage, `rotationWorldToCamera` and `translationWorldToCamera` implement `Xc = R_wc Xw + t_wc`,
  `cameraCenter` supports inverse projection, and the optional pixel homographies are applied after projection
  and before unprojection. Include parameter, return, positive-depth, and invalid-input semantics for every method.

- [ ] **Step 3: Document conversion options and result ownership**

  Add a file overview and Doxygen comments for `CameraFormat`, `CameraConversionOptions`,
  `CameraConversionResult`, format-name helpers, and `convertCameraDataset()`. State that the function reports
  expected failures through `success/errorMessage`, writes `cameras/*.tsai`, `image_camera.lis`, and
  `summary.json`, and may remove a non-empty output directory only when `overwrite` is true.

### Task 2: Camera projection implementation review notes

**Files:**
- Modify: `src/core/camera/Camera.cpp`
- Modify: `src/core/camera/PositiveDepthCameraModel.cpp`

- [ ] **Step 1: Align the `Camera.cpp` overview with current symbols**

  Replace stale method names in the file header with the current public/private method names and explicitly state
  that runtime intrinsics use pixels while Tsai I/O uses millimetres through `pitch`.

- [ ] **Step 2: Explain state reset, key matching, and validation**

  Add comments around `loadFromFile()` explaining why state is reset before parsing, why `startsWithKey()` checks
  a delimiter, which six fields are mandatory, why `k1/p1` parsing starts after `=` or `:`, and why determinant
  deviation is warning-only while non-positive pitch/focal length is fatal.

- [ ] **Step 3: Explain numerical projection paths**

  Preserve existing formulas and add review notes for signed Z division, the `1e-9` depth guard, Rodrigues small-
  angle fallback, numerical Jacobian, singular determinant handling, and the fact that `undistortPixel()` returns
  the latest iterate even when the loop exits by iteration limit or singularity.

- [ ] **Step 4: Explain positive-depth canonicalization**

  Add comments showing how ASP axis signs are folded into the rows of `R_wc` and `t_wc`, why focal magnitudes are
  made positive, why projection rejects `Z <= 1e-6`, and why unprojection multiplies by `R_wc^T` before adding
  `C`. Explain the homogeneous divide fallback and paired forward/inverse pixel transforms.

### Task 3: External format conversion implementation review notes

**Files:**
- Modify: `src/core/camera/CameraFormatConverter.cpp`

- [ ] **Step 1: Add file and section-level navigation**

  Add a file overview plus section comments for the unified intermediate record, text/JSON/shell helpers, linear
  algebra, ZIP/XML access, format-specific parsing, source discovery, output generation, and public API.

- [ ] **Step 2: Document the unified record and loss reporting**

  Explain the row-major `K`, Brown-Conrady coefficient order `[k1,k2,k3,p1,p2]`, camera-to-world rotation,
  world-space center, per-camera warnings, and why unsupported skew/distortion is reported instead of silently
  represented in Tsai output.

- [ ] **Step 3: Document pose conversion for every format**

  Add the exact relation `C = -R_wc^T t` for Middlebury/COLMAP inputs, note that EPFL already supplies
  camera-to-world rotation and center, and note that Metashape's 4x4 transform stores camera-to-world rotation in
  its upper-left block and the center in the last column.

- [ ] **Step 4: Document discovery and Metashape edge cases**

  Explain COLMAP sparse-directory candidates, complete-image-root acceptance, Metashape `.psx`/`.files`/
  `doc.xml`/`chunk.zip` variants, adjusted calibration preference, principal-point offset conversion, label suffix
  removal, case-insensitive image fallback, and the all-images-required rule.

- [ ] **Step 5: Document output safety and result flow**

  Explain shell quoting in `image_camera.lis`, relative-path fallback, `pitch = 1` preserving pixel-valued K,
  output-directory equivalence/non-empty checks, warning aggregation, and exception-to-result conversion.

### Task 4: Test and build review notes

**Files:**
- Modify: `src/core/camera/test/Camera_tests.cpp`
- Modify: `src/core/camera/test/CameraFormatConverter_tests.cpp`
- Modify: `src/core/camera/test/test_tsai_loader.cpp`
- Modify: `src/core/camera/test/test.cpp`
- Modify: `src/core/camera/CMakeLists.txt`

- [ ] **Step 1: Add test file overviews and scenario comments**

  Describe each test file's responsibility. In projection tests explain front/back and depth-flip expectations; in
  converter tests explain synthetic dataset layout, pose expectations, ordering, warnings and cleanup; in Tsai
  loader tests explain external `testData/tsai` dependency and geometric invariants.

- [ ] **Step 2: Clarify the diagnostic executable**

  State that `test/test.cpp` is a manual `camera_test` diagnostic tool rather than a GTest suite, document its two
  positional arguments, rotation diagnostics, stdin sentinel `END`, and generated optical-axis self-check.

- [ ] **Step 3: Clarify CMake target roles**

  Add comments distinguishing the production static library, always-built manual diagnostic target, and the three
  GTest targets gated by `BUILD_TESTS` and `GTest_FOUND`. Document why all tests keep the camera root as an include
  directory and why `test_camera_tsai` receives the repository-level `testData` path.

### Task 5: Comment-only and behavior verification

**Files:**
- Verify: `src/core/camera/*.h`
- Verify: `src/core/camera/*.cpp`
- Verify: `src/core/camera/test/*.cpp`
- Verify: `src/core/camera/CMakeLists.txt`

- [ ] **Step 1: Inspect the diff for behavior changes**

  Run: `git diff -- src/core/camera`

  Expected: aside from the previously approved test-file moves and CMake path updates, this task adds only comments
  and whitespace. No executable token, string literal, target property, assertion, or path changes.

- [ ] **Step 2: Reconfigure the existing Windows build**

  Run: `cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON`

  Expected: configuration and generation complete successfully.

- [ ] **Step 3: Build all camera targets**

  Run: `cmake --build build/windows-vcpkg-cuda-release --target camera_test test_camera_unit test_camera_tsai test_camera_format_converter -j 8`

  Expected: all four targets build successfully.

- [ ] **Step 4: Run all camera tests**

  Run: `ctest --test-dir build/windows-vcpkg-cuda-release/src/core/camera --output-on-failure`

  Expected: 22 of 22 tests pass.

- [ ] **Step 5: Check whitespace and task status**

  Run: `git diff --check -- src/core/camera`

  Run: `git status --short -- src/core/camera docs/superpowers/specs/2026-07-13-camera-review-comments-design.md docs/superpowers/plans/2026-07-13-camera-review-comments.md`

  Expected: no whitespace errors. Preserve all unrelated working-tree changes and do not commit or push.
