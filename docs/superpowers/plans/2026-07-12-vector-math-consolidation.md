# Vector Math Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PlaMatrix the single owner of reusable three-dimensional vector arithmetic, migrate PlaPoint and PlaScan to it, and remove PlaScan's unused common math module.

**Architecture:** Add a header-only `plamatrix::Vec3<T>` arithmetic API while preserving existing PlaMatrix point-cloud API compatibility. PlaPoint consumes that API internally; PlaScan keeps its `std::array<double, 3>` public contracts and converts only at arithmetic call sites.

**Tech Stack:** C++17, CMake, GoogleTest, PlaMatrix, PlaPoint, PlaScan, optional CUDA 13.1.

---

### Task 1: Add the PlaMatrix vector API with TDD

**Files:**
- Create: `3rdparty/plamatrix/test/unit/ops/vector_test.cpp`
- Create: `3rdparty/plamatrix/include/plamatrix/ops/vector.h`
- Modify: `3rdparty/plamatrix/test/CMakeLists.txt`
- Modify: `3rdparty/plamatrix/include/plamatrix/ops/point_cloud.h`
- Modify: `3rdparty/plamatrix/include/plamatrix/plamatrix.h`

- [ ] **Step 1: Write the failing vector API tests**

Add tests that include `<plamatrix/ops/vector.h>` and verify three-value and `std::array` construction, `toArray()`, arithmetic operators, `dot()`, `cross()`, `squaredNorm()`, `norm()`, near-zero `normalized()`, and component-wise `isFinite()` for NaN and infinity.

```cpp
#include <gtest/gtest.h>

#include <array>
#include <limits>

#include <plamatrix/ops/vector.h>

TEST(Vector3, ConvertsArraysAndAppliesArithmetic)
{
    const plamatrix::Vec3<double> a(std::array<double, 3>{1.0, 2.0, 3.0});
    const plamatrix::Vec3<double> b{4.0, -1.0, 2.0};
    const auto sum = a + b;
    const auto difference = a - b;
    const auto scaled = 2.0 * a;

    EXPECT_EQ(sum.toArray(), (std::array<double, 3>{5.0, 1.0, 5.0}));
    EXPECT_EQ(difference.toArray(), (std::array<double, 3>{-3.0, 3.0, 1.0}));
    EXPECT_EQ(scaled.toArray(), (std::array<double, 3>{2.0, 4.0, 6.0}));
    EXPECT_EQ((a / 2.0).toArray(), (std::array<double, 3>{0.5, 1.0, 1.5}));
}

TEST(Vector3, ComputesProductsAndNorms)
{
    const plamatrix::Vec3<double> x{1.0, 0.0, 0.0};
    const plamatrix::Vec3<double> y{0.0, 1.0, 0.0};
    EXPECT_DOUBLE_EQ(plamatrix::dot(x, y), 0.0);
    EXPECT_EQ(plamatrix::cross(x, y).toArray(), (std::array<double, 3>{0.0, 0.0, 1.0}));

    const plamatrix::Vec3<double> v{2.0, 3.0, 6.0};
    EXPECT_DOUBLE_EQ(plamatrix::squaredNorm(v), 49.0);
    EXPECT_DOUBLE_EQ(plamatrix::norm(v), 7.0);
}

TEST(Vector3, NormalizesAndPreservesNearZeroVectors)
{
    const auto unit = plamatrix::normalized(plamatrix::Vec3<double>{0.0, 3.0, 4.0});
    EXPECT_NEAR(unit.y, 0.6, 1.0e-12);
    EXPECT_NEAR(unit.z, 0.8, 1.0e-12);

    const plamatrix::Vec3<double> tiny{1.0e-15, 0.0, 0.0};
    EXPECT_EQ(plamatrix::normalized(tiny, 1.0e-12).toArray(), tiny.toArray());
}

TEST(Vector3, DetectsNonFiniteComponents)
{
    EXPECT_TRUE(plamatrix::isFinite(plamatrix::Vec3<double>{1.0, 2.0, 3.0}));
    EXPECT_FALSE(plamatrix::isFinite(plamatrix::Vec3<double>{
        std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}));
    EXPECT_FALSE(plamatrix::isFinite(plamatrix::Vec3<double>{
        0.0, std::numeric_limits<double>::infinity(), 0.0}));
}
```

- [ ] **Step 2: Register and run the tests to verify RED**

Add `unit/ops/vector_test.cpp` to `plamatrix_tests`, then run:

```powershell
cmake -S . -B build-codex-cpu -DPLAMATRIX_WITH_CUDA=OFF -DPLAMATRIX_BUILD_TESTS=ON
cmake --build build-codex-cpu --target plamatrix_tests --config Release -- /m
```

Expected: compilation fails because `plamatrix/ops/vector.h` does not exist.

- [ ] **Step 3: Implement the minimal header-only API**

Create `ops/vector.h` with a value-initialized `Vec3<T>`, explicit `std::array` conversion, `toArray()`, arithmetic operators, and the tested free functions. Move the existing `Vec3<T>` declaration out of `ops/point_cloud.h`, include `ops/vector.h` there, and expose it from `plamatrix/plamatrix.h`.

```cpp
template <typename Scalar>
struct Vec3
{
    Scalar x{};
    Scalar y{};
    Scalar z{};

    constexpr Vec3() = default;
    constexpr Vec3(Scalar x_value, Scalar y_value, Scalar z_value)
        : x(x_value), y(y_value), z(z_value) {}
    explicit constexpr Vec3(const std::array<Scalar, 3> &values)
        : x(values[0]), y(values[1]), z(values[2]) {}

    constexpr std::array<Scalar, 3> toArray() const { return {x, y, z}; }
};
```

- [ ] **Step 4: Verify GREEN and the complete PlaMatrix CPU suite**

```powershell
cmake --build build-codex-cpu --target plamatrix_tests --config Release -- /m
ctest --test-dir build-codex-cpu -C Release --output-on-failure -R "plamatrix.Vector3"
ctest --test-dir build-codex-cpu -C Release --output-on-failure
```

Expected: vector tests and all PlaMatrix CPU tests pass.

### Task 2: Refactor PlaPoint mesh processing onto PlaMatrix vectors

**Files:**
- Modify: `3rdparty/plapoint/include/plapoint/mesh/mesh_processing.h`
- Test: `3rdparty/plapoint/test/unit/mesh/mesh_processing_test.cpp`

- [ ] **Step 1: Establish the green characterization baseline**

Run the existing mesh-processing tests before changing production code:

```powershell
ctest --test-dir build-codex-cpu-verify -C Release --output-on-failure -R "plapoint.*MeshProcessing"
```

Expected: existing mesh-processing behavior passes.

- [ ] **Step 2: Replace the private vector implementation**

Delete `detail::Vec3`, `detail::cross()`, and `detail::norm()`. Use `plamatrix::Vec3<long double>`, `plamatrix::cross()`, and `plamatrix::norm()` in `pointAt()`, `triangleArea()`, and vertex-normal accumulation. Keep all public PlaPoint signatures unchanged.

- [ ] **Step 3: Build against the modified PlaMatrix and verify PlaPoint**

Install the CPU PlaMatrix build to a temporary prefix, configure PlaPoint against it, then run focused and complete tests:

```powershell
cmake --install build-codex-cpu --config Release --prefix "$env:TEMP\codex-plamatrix-vector-cpu"
cmake -S . -B build-codex-cpu-verify -DPLAPOINT_WITH_CUDA=OFF -DPLAPOINT_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$env:TEMP\codex-plamatrix-vector-cpu"
cmake --build build-codex-cpu-verify --target plapoint_tests --config Release -- /m
ctest --test-dir build-codex-cpu-verify -C Release --output-on-failure -R "plapoint.*MeshProcessing"
ctest --test-dir build-codex-cpu-verify -C Release --output-on-failure
```

Expected: focused mesh tests and the full PlaPoint CPU suite pass.

### Task 3: Migrate PlaScan and remove common math

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
- Modify: `src/core/bundle_adjust/CMakeLists.txt`
- Modify: `src/core/sfm/triangulation/InitialSparsePointCloudTriangulator.cpp`
- Modify: `src/core/sfm/filtering/SparsePointCloudProcessor.cpp`
- Modify: `src/core/sfm/CMakeLists.txt`
- Modify: `src/common/CMakeLists.txt`
- Delete: `src/common/math/Vec.h`
- Delete: `src/common/math/Vec3Ops.h`
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: Replace `Vec3Ops` call sites**

Include `<plamatrix/ops/vector.h>`. Convert existing arrays at arithmetic call sites:

```cpp
plamatrix::isFinite(plamatrix::Vec3<double>(point));
plamatrix::norm(plamatrix::Vec3<double>(point));
plamatrix::dot(plamatrix::Vec3<double>(left), plamatrix::Vec3<double>(right));
```

In local normal estimation, subtract converted vectors, calculate the cross product, and return `normal.toArray()` after normalization.

- [ ] **Step 2: Declare direct dependencies and remove obsolete common math files**

Link `bundle_adjust` and `sfm` privately to `plamatrix::plamatrix`. Remove the unused `plascan_common_math` target and delete both headers under `src/common/math`.

- [ ] **Step 3: Update architecture documentation**

Remove the `common/math` tree entries and stop describing `src/common` as owning generic mathematics. Preserve all unrelated existing documentation edits.

- [ ] **Step 4: Verify no stale references remain**

```powershell
rg -n "math/Vec|Vec3Ops|plascan_common_math|xjw::Point3f|xjw::Point2f|xjw::ColorRGBA" src tests docs/PROJECT_ARCHITECTURE.md
```

Expected: no matches.

- [ ] **Step 5: Build and run focused PlaScan tests**

```powershell
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target bundle_adjust sfm --config Release -- /m
ctest --test-dir build -C Release --output-on-failure -R "BundleAdjust|Sfm|SparsePointCloud"
```

Expected: targets build and focused BA/SfM tests pass.

### Task 4: Verify the CUDA dependency chain

**Files:**
- No additional source files expected.

- [ ] **Step 1: Build and test PlaMatrix with CUDA**

```powershell
cmake -S . -B build-task-c-cuda -DPLAMATRIX_WITH_CUDA=ON -DPLAMATRIX_BUILD_TESTS=ON
cmake --build build-task-c-cuda --target plamatrix_tests --config Release -- /m
ctest --test-dir build-task-c-cuda -C Release --output-on-failure
cmake --install build-task-c-cuda --config Release --prefix "$env:TEMP\codex-plamatrix-vector-cuda"
```

Expected: CUDA compilation succeeds and all runnable tests pass; tests without a CUDA device may report the project's configured skip code.

- [ ] **Step 2: Build and test PlaPoint with CUDA 13.1**

```powershell
cmake -S . -B build-codex-cuda-verify -DPLAPOINT_WITH_CUDA=ON -DPLAPOINT_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$env:TEMP\codex-plamatrix-vector-cuda"
cmake --build build-codex-cuda-verify --target plapoint_tests --config Release -- /m
ctest --test-dir build-codex-cuda-verify -C Release --output-on-failure
```

Expected: PlaPoint compiles against the new PlaMatrix vector header and the full CUDA-enabled test suite passes.

### Task 5: Final scope audit

**Files:**
- Review only.

- [ ] **Step 1: Inspect the three worktrees**

```powershell
git status --short
git -C 3rdparty/plamatrix status --short
git -C 3rdparty/plapoint status --short
git diff --stat
```

Confirm that only vector consolidation files and the already-existing user changes are present. Do not stage or commit because the user has not requested a commit.

- [ ] **Step 2: Summarize verification and residual warnings**

Report exact test commands and outcomes, including CUDA skips, compiler warnings, or pre-existing PlaScan test failures separately.
