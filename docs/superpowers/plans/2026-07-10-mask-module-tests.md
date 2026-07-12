# Mask Module Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the core mask test source and CMake registration into `src/core/mask/tests/` while preserving the existing target and CTest behavior.

**Architecture:** The mask library remains defined by `src/core/mask/CMakeLists.txt`, which conditionally adds a dedicated test subdirectory. The new test-local CMake file owns only the core mask test executable and explicitly supplies its test-data path.

**Tech Stack:** C++17, CMake, GoogleTest, OpenCV, LibTorch

---

### Task 1: Relocate the core mask test source

**Files:**
- Move: `tests/test_mask_generation.cpp` to `src/core/mask/tests/test_mask_generation.cpp`

- [ ] **Step 1: Move the existing source without changing test logic**

Use an `apply_patch` file move so all existing `MaskGenerator`, `Sam21MaskGenerator`, and
`U2NetMaskGenerator` tests remain byte-for-byte equivalent.

- [ ] **Step 2: Verify the old source path is gone and the new path exists**

Run:

```powershell
Test-Path tests/test_mask_generation.cpp
Test-Path src/core/mask/tests/test_mask_generation.cpp
```

Expected: `False`, then `True`.

### Task 2: Transfer CMake ownership to the mask module

**Files:**
- Create: `src/core/mask/tests/CMakeLists.txt`
- Modify: `src/core/mask/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Define the test in the module test directory**

Create `src/core/mask/tests/CMakeLists.txt` with:

```cmake
include(PlascanTestRuntime)

add_executable(test_mask_generation
    test_mask_generation.cpp
)

target_link_libraries(test_mask_generation PRIVATE
    mask
    ${OpenCV_LIBS}
    GTest::gtest_main
)

target_include_directories(test_mask_generation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(test_mask_generation PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/testData"
)

plascan_gtest_discover_tests(test_mask_generation)
```

- [ ] **Step 2: Add the test subdirectory from the mask module**

Append to `src/core/mask/CMakeLists.txt`:

```cmake
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

- [ ] **Step 3: Remove only the old mask target block**

Delete the `add_executable(test_mask_generation ...)` through
`gtest_discover_tests(test_mask_generation)` block from `tests/CMakeLists.txt`, leaving
all unrelated dirty-worktree changes untouched.

### Task 3: Configure, build, and run the relocated tests

**Files:**
- Test: `src/core/mask/tests/test_mask_generation.cpp`

- [ ] **Step 1: Reconfigure the existing Windows build**

Run:

```powershell
cmake -S . -B build/windows-vcpkg-cuda-release
```

Expected: configuration and generation complete without duplicate-target errors.

- [ ] **Step 2: Build the relocated target**

Run from a Visual Studio developer environment:

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_mask_generation --parallel 16
```

Expected: `test_mask_generation` builds successfully.

- [ ] **Step 3: Confirm CTest discovery is preserved**

Run:

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release -C Release -N -R "MaskGenerator|MaskComposer|Sam21MaskGenerator|U2NetMaskGenerator"
```

Expected: Windows lists the aggregate `test_mask_generation` test with runtime-path
injection; non-Windows builds list the individual GoogleTest cases.

- [ ] **Step 4: Run the core mask tests**

Run:

```powershell
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "MaskGenerator|MaskComposer|Sam21MaskGenerator|U2NetMaskGenerator"
```

Expected: all available core mask tests pass; model-dependent tests may report GoogleTest skips when optional model resources or CUDA support are absent.

No commit is included because repository instructions require an explicit user request before committing.
