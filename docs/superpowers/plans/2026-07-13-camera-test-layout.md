# Camera Test Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/core/camera` 根目录中的测试与诊断源文件集中到 `src/core/camera/test`，保持现有目标和测试行为不变。

**Architecture:** 只调整源文件物理位置；`camera` 库源码、`testdata`、目标名和链接依赖保持不变。父级 `CMakeLists.txt` 直接引用新路径，架构文档同步展示新的目录层级。

**Tech Stack:** C++17、CMake、GoogleTest、CTest

---

### Task 1: Relocate camera test sources

**Files:**
- Move: `src/core/camera/Camera_tests.cpp` -> `src/core/camera/test/Camera_tests.cpp`
- Move: `src/core/camera/CameraFormatConverter_tests.cpp` -> `src/core/camera/test/CameraFormatConverter_tests.cpp`
- Move: `src/core/camera/test_tsai_loader.cpp` -> `src/core/camera/test/test_tsai_loader.cpp`
- Move: `src/core/camera/test.cpp` -> `src/core/camera/test/test.cpp`
- Modify: `src/core/camera/CMakeLists.txt`

- [ ] **Step 1: Move the four source files without changing their contents**

  Preserve file contents byte-for-byte and leave `src/core/camera/testdata` in place because it is test data, not test code.

- [ ] **Step 2: Update every camera test target to use the new paths**

  Use `test/test.cpp`, `test/Camera_tests.cpp`, `test/test_tsai_loader.cpp`, and
  `test/CameraFormatConverter_tests.cpp` as the executable sources. Keep target names, link libraries,
  include directories, compile definitions, and `gtest_discover_tests` calls unchanged.

- [ ] **Step 3: Confirm stale source paths are gone**

  Run: `rg -n "Camera_tests|CameraFormatConverter_tests|test_tsai_loader|add_executable\\(camera_test" src/core/camera`

  Expected: all CMake source references include the `test/` prefix, and all four `.cpp` files are under
  `src/core/camera/test`.

### Task 2: Synchronize architecture documentation

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: Replace the flat camera test entry with a nested test directory**

  Document the four moved source files under `camera/test/`; do not alter unrelated existing documentation edits.

- [ ] **Step 2: Verify the documented tree matches the filesystem**

  Run: `rg --files src/core/camera | Sort-Object`

  Expected: production sources remain at the module root, test sources appear under `test/`, and `testdata/1.tsai`
  plus `testdata/2.tsai` remain unchanged.

### Task 3: Configure, build, and run camera tests

**Files:**
- Verify: `src/core/camera/CMakeLists.txt`
- Verify: `src/core/camera/test/*.cpp`

- [ ] **Step 1: Reconfigure the existing test-enabled Windows build**

  Run: `cmake -S . -B build/windows-vcpkg-cuda-release -DBUILD_TESTS=ON`

  Expected: configuration and generation complete without missing-source errors.

- [ ] **Step 2: Build the camera targets**

  Run: `cmake --build build/windows-vcpkg-cuda-release --target camera_test test_camera_unit test_camera_tsai test_camera_format_converter -j 8`

  Expected: all four targets build successfully from their new source paths.

- [ ] **Step 3: Run the camera tests**

  Run: `ctest --test-dir build/windows-vcpkg-cuda-release --output-on-failure -R "Camera(Basic|Intrinsics|Projection|ToPositiveDepthModel|Test|FormatConverter)"`

  Expected: discovered camera tests pass.

- [ ] **Step 4: Inspect only task-related changes**

  Run: `git status --short -- src/core/camera docs/PROJECT_ARCHITECTURE.md docs/superpowers/plans/2026-07-13-camera-test-layout.md`

  Run: `git diff --check -- src/core/camera docs/PROJECT_ARCHITECTURE.md docs/superpowers/plans/2026-07-13-camera-test-layout.md`

  Expected: four renames, two small reference/documentation edits, this plan, and no whitespace errors. Do not commit.
