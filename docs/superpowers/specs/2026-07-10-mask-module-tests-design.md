# Mask Module Tests Design

## Goal

Move the core mask test source and its CMake ownership from the repository-level
`tests/` directory into `src/core/mask/tests/` without changing test behavior,
target names, or CTest case names.

## Structure

- `src/core/mask/tests/test_mask_generation.cpp` contains the existing core mask tests.
- `src/core/mask/tests/CMakeLists.txt` defines and registers `test_mask_generation`.
- `src/core/mask/CMakeLists.txt` adds the test subdirectory only when `BUILD_TESTS` is enabled.
- `tests/CMakeLists.txt` no longer owns the core mask test target.

GUI-level mask tests remain in the repository-level GUI test targets because moving
them into the core module would introduce GUI dependencies into `src/core/mask`.

## Compatibility

The executable target remains named `test_mask_generation`. Registration uses the
project's `plascan_gtest_discover_tests` helper so module-local tests receive the required
OpenCV, LibTorch, CUDA, and vcpkg runtime paths on Windows; non-Windows builds retain
normal GoogleTest discovery. The new test CMake file defines `TEST_DATA_DIR` explicitly
because the repository-level test directory's inherited compile definition will no longer
apply after the move.

## Verification

Reconfigure the existing Windows build, build `test_mask_generation`, list its discovered
CTest cases, and run the core mask tests with failure output enabled.
