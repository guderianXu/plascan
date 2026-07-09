# Native CUDA Bundle Adjustment Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PlaScan-owned `native_cuda` global bundle adjustment backend that optimizes camera poses and 3D points with fixed intrinsics/distortion, using GPU Schur reduction and GPU PCG while preserving existing fallback and quality-gate behavior.

**Architecture:** Keep `BAResult BundleAdjust::optimizePoints(const std::vector<Camera> &, const std::vector<BATrack> &, const BAOptions &)` as the only public execution entry. Add a focused native CUDA backend under `src/core/bundle_adjust`, with CPU-side workset construction, CUDA kernels for residual/Jacobian/block accumulation, GPU Schur/PCG solve, CPU LM step control, and existing BA result reporting. Unsupported constraints route through the existing fallback path instead of being silently ignored.

**Tech Stack:** C++17, CUDA C++, CMake/Ninja, GTest/CTest, existing `Camera`, `BATrack`, `BAOptions`, `BAResult`, Python contract tests, existing `ba_backend_benchmark`.

---

## File Structure

- Modify `src/core/bundle_adjust/BundleAdjust.h`: add `BABackend::NativeCuda`, native CUDA thresholds, PCG controls, and native CUDA result diagnostics.
- Modify `src/core/bundle_adjust/BundleAdjust.cpp`: add backend name/availability/selection/fallback dispatch for `NativeCuda`.
- Create `src/core/bundle_adjust/BundleAdjustNativeCuda.h`: backend availability and optimize entry.
- Create `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`: CPU-side validation, fallback message construction, workset orchestration, result conversion.
- Create `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`: POD structs shared by `.cpp`, `.cu`, and tests.
- Create `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.h/.cpp`: deterministic CPU workset builder and unsupported-constraint detector.
- Create `src/core/bundle_adjust/BundleAdjustNativeCudaKernels.cuh`: CUDA launch declarations and small math types.
- Create `src/core/bundle_adjust/BundleAdjustNativeCuda.cu`: GPU residual/Jacobian, normal block accumulation, Schur construction, PCG, back-substitution, trial cost kernels.
- Modify `src/core/bundle_adjust/CMakeLists.txt`: build the native CUDA backend only when CUDA is enabled, and compile CPU stubs in all builds.
- Modify `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`: add `native_cuda` backend support and native metrics output.
- Modify `scripts/bench/run_ba_backend_benchmark.py`: include `native_cuda` in backend choices.
- Modify `src/cli/cli_bundle_adjust.cpp`: parse `--ba-backend native_cuda` and expose native CUDA controls.
- Modify `src/gui/project/services/BundleAdjustService.cpp`: persist native CUDA settings and result metrics into JSON/text summaries.
- Modify `src/gui/dialogs/BundleAdjustDialog.cpp`: allow selecting native CUDA and show diagnostics in the result message.
- Modify `src/gui/project/manager/ProjectManager.cpp`: map dialog settings into `BAOptions`.
- Modify `src/core/aerial_triangulation/AerialTriangulationService.cpp`: let CUDA mode request Auto with native CUDA thresholds.
- Create or modify tests:
  - `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`
  - `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp`
  - `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_math.cpp`
  - `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp`
  - `tests/test_ba_cuda_contracts.py`
  - `tests/CMakeLists.txt`
- Modify docs:
  - `README.md`
  - `docs/benchmarks/ba-cuda-hyb2-2026-07-09.md` or a new benchmark note after real runs.

---

### Task 1: Backend Contract And Options

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`
- Modify: `tests/test_ba_cuda_contracts.py`

- [ ] **Step 1: Write failing backend contract tests**

Add this C++ test to `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`:

```cpp
TEST(BundleAdjustBackendSelectionTest, NativeCudaBackendNameAndAutoThresholdsAreStable)
{
    EXPECT_STREQ(xjw::BundleAdjust::backendName(xjw::BABackend::NativeCuda), "native_cuda");

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minNativeCudaCameras = 3;
    options.minNativeCudaObservations = 20;
    options.minCeresCudaCameras = 1000;
    options.minCeresCudaObservations = 1000000;
    options.minCeresCpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 4;
    stats.trackCount = 10;
    stats.observationCount = 30;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::NativeCuda);
    }
    else
    {
        EXPECT_NE(selected, xjw::BABackend::NativeCuda);
    }
}
```

Add these checks to `tests/test_ba_cuda_contracts.py`:

```python
def test_native_cuda_backend_is_exposed(self):
    header = self.read_text("src/core/bundle_adjust/BundleAdjust.h")
    self.assertIn("NativeCuda", header)
    self.assertIn("minNativeCudaCameras", header)
    self.assertIn("minNativeCudaObservations", header)
    self.assertIn("nativeCudaPcgIterations", header)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "BundleAdjustBackendSelection"
python -m unittest tests.test_ba_cuda_contracts
```

Expected:

- C++ compile or test fails because `BABackend::NativeCuda` is not defined.
- Python contract test fails because `NativeCuda` fields are absent.

- [ ] **Step 3: Add public enum, options, and result fields**

Modify `BABackend` in `src/core/bundle_adjust/BundleAdjust.h`:

```cpp
enum class BABackend
{
    Auto,
    LegacyCpu,
    CeresCpu,
    CeresCuda,
    NativeCuda,
};
```

Add these fields near the existing Ceres/GPU options in `BAOptions`:

```cpp
/// 自研 CUDA BA 使用的 GPU 设备 ID。
int nativeCudaDevice = 0;
/// 低于该相机数时 Auto 不选择 native_cuda，避免小问题 GPU 调度开销大于收益。
int minNativeCudaCameras = 50;
/// 低于该观测数时 Auto 不选择 native_cuda。
int minNativeCudaObservations = 500000;
/// native_cuda PCG 最大迭代次数。
int nativeCudaMaxPcgIterations = 100;
/// native_cuda PCG 相对残差阈值。
double nativeCudaPcgTolerance = 1e-4;
/// native_cuda 每个 LM step 允许的最大位姿增量范数。
double nativeCudaMaxPoseStepNorm = 1.0;
```

Add these fields to `BAResult`:

```cpp
int nativeCudaPcgIterations = 0;             ///< native_cuda 累计 PCG 迭代次数
double nativeCudaLinearResidual = 0.0;       ///< native_cuda 最后一轮线性系统相对残差
int nativeCudaAcceptedSteps = 0;             ///< native_cuda 接受的 LM trial step 数
int nativeCudaRejectedSteps = 0;             ///< native_cuda 拒绝的 LM trial step 数
int nativeCudaActiveCameras = 0;             ///< native_cuda 工作集中的活动相机数
int nativeCudaActiveTracks = 0;              ///< native_cuda 工作集中的活动 track 数
int nativeCudaActiveObservations = 0;        ///< native_cuda 工作集中的活动观测数
```

- [ ] **Step 4: Wire backend name and initial selection**

Modify `BundleAdjust::backendName` in `src/core/bundle_adjust/BundleAdjust.cpp`:

```cpp
case BABackend::NativeCuda:
    return "native_cuda";
```

Modify `BundleAdjust::selectBackendForProblem` before the Ceres CUDA branch:

```cpp
if (options.refineCameraPose &&
    isBackendAvailable(BABackend::NativeCuda) &&
    stats.cameraCount >= options.minNativeCudaCameras &&
    stats.observationCount >= options.minNativeCudaObservations)
{
    return BABackend::NativeCuda;
}
```

Temporarily implement availability as false until Task 2:

```cpp
case BABackend::NativeCuda:
    return false;
```

- [ ] **Step 5: Run contract tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_backend_selection --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "BundleAdjustBackendSelection"
python -m unittest tests.test_ba_cuda_contracts
```

Expected:

- C++ backend-selection tests pass.
- Python contract tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjust.h src/core/bundle_adjust/BundleAdjust.cpp src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp tests/test_ba_cuda_contracts.py
git commit -m "feat: add native cuda ba backend contract"
```

---

### Task 2: Native CUDA Build Stub And Dispatch

**Files:**
- Create: `src/core/bundle_adjust/BundleAdjustNativeCuda.h`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/CMakeLists.txt`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp`

- [ ] **Step 1: Write failing fallback test**

Add this test to `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp`:

```cpp
TEST(BundleAdjustCeresBackendTest, NativeCudaRequestFallsBackWhenBackendUnavailable)
{
    const auto cameras = makeTwoCameraRig();
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 300.0, 240.0, 1.0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.allowBackendFallback = true;
    options.maxIterations = 1;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::NativeCuda);
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        EXPECT_EQ(result.usedBackend, xjw::BABackend::LegacyCpu);
        EXPECT_TRUE(result.backendFallback);
        EXPECT_FALSE(result.backendMessage.empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_ceres_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaRequestFallsBack"
```

Expected:

- Build fails because `NativeCuda` dispatch path does not exist.

- [ ] **Step 3: Add native CUDA public backend stub**

Create `src/core/bundle_adjust/BundleAdjustNativeCuda.h`:

```cpp
#pragma once

#include "BundleAdjust.h"

namespace xjw::detail
{

bool isNativeCudaBackendCompiled();
bool isNativeCudaRuntimeAvailable(int deviceId, std::string *message);

BAResult optimizePointsWithNativeCuda(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options);

} // namespace xjw::detail
```

Create `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`:

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw::detail::native_cuda
{

struct HostCamera
{
    std::array<double, 9> cameraToWorldRotation{};
    std::array<double, 3> cameraCenter{};
    double focalX = 1.0;
    double focalY = 1.0;
    double principalX = 0.0;
    double principalY = 0.0;
    double radialK1 = 0.0;
    double radialK2 = 0.0;
    double radialK3 = 0.0;
    double tangentialP1 = 0.0;
    double tangentialP2 = 0.0;
    int uAxisSign = 1;
    int vAxisSign = 1;
    int fixed = 0;
    int originalIndex = -1;
};

struct HostPoint
{
    std::array<double, 3> xyz{};
    int originalTrackIndex = -1;
    int observationBegin = 0;
    int observationCount = 0;
};

struct HostObservation
{
    int cameraIndex = -1;
    int pointIndex = -1;
    double u = 0.0;
    double v = 0.0;
    double weight = 1.0;
};

struct Workset
{
    std::vector<HostCamera> cameras;
    std::vector<HostPoint> points;
    std::vector<HostObservation> observations;
    std::vector<int> originalTrackToPoint;
    std::string rejectionReason;
};

} // namespace xjw::detail::native_cuda
```

Create `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`:

```cpp
#include "BundleAdjustNativeCuda.h"

#include <chrono>

namespace xjw::detail
{

bool isNativeCudaBackendCompiled()
{
#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
    return true;
#else
    return false;
#endif
}

bool isNativeCudaRuntimeAvailable(int deviceId, std::string *message)
{
    if (!isNativeCudaBackendCompiled())
    {
        if (message)
        {
            *message = "native_cuda 后端未编译";
        }
        return false;
    }
    if (deviceId < 0)
    {
        if (message)
        {
            *message = "native_cuda GPU 设备 ID 无效";
        }
        return false;
    }
    return true;
}

BAResult optimizePointsWithNativeCuda(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options)
{
    BAResult result;
    result.requestedBackend = options.backend;
    result.usedBackend = BABackend::NativeCuda;
    result.usedGpu = false;
    result.totalTracks = static_cast<int>(tracks.size());
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());
    result.backendMessage = "native_cuda 后端尚未完成求解路径";
    return result;
}

} // namespace xjw::detail
```

- [ ] **Step 4: Wire CMake stub**

Modify `src/core/bundle_adjust/CMakeLists.txt`:

```cmake
add_library(bundle_adjust STATIC
    BundleAdjust.cpp
    BundleAdjustCeres.cpp
    BundleAdjustNativeCuda.cpp
)

if(CMAKE_CUDA_COMPILER)
    target_sources(bundle_adjust PRIVATE
        BundleAdjustNativeCuda.cu
    )
    target_compile_definitions(bundle_adjust PRIVATE PLASCAN_BA_HAS_NATIVE_CUDA=1)
    target_link_libraries(bundle_adjust PRIVATE CUDA::cudart)
    message(STATUS "bundle_adjust: native CUDA BA 后端已启用")
else()
    message(STATUS "bundle_adjust: native CUDA BA 后端未启用")
endif()
```

If `CUDA::cudart` is not visible in this directory, add `find_package(CUDAToolkit QUIET)` before the `if(CMAKE_CUDA_COMPILER)` block and guard `target_link_libraries` with `if(CUDAToolkit_FOUND)`.

- [ ] **Step 5: Wire dispatch and fallback**

Modify `src/core/bundle_adjust/BundleAdjust.cpp` includes:

```cpp
#include "BundleAdjustNativeCuda.h"
```

Modify `isBackendAvailable`:

```cpp
case BABackend::NativeCuda:
{
    std::string message;
    return detail::isNativeCudaRuntimeAvailable(0, &message);
}
```

Add a `NativeCuda` branch before Ceres branches in `optimizePoints`:

```cpp
if (options.backend == BABackend::NativeCuda)
{
    std::string message;
    if (!detail::isNativeCudaRuntimeAvailable(options.nativeCudaDevice, &message))
    {
        if (!options.allowBackendFallback)
        {
            BAResult result = optimizePointsLegacy(cameras, tracks, legacyOptionsForFailure(options));
            result.requestedBackend = BABackend::NativeCuda;
            result.usedBackend = BABackend::LegacyCpu;
            result.backendFallback = true;
            result.backendMessage = message;
            return result;
        }
        BAOptions fallback = options;
        fallback.backend = BABackend::LegacyCpu;
        BAResult result = optimizePoints(cameras, tracks, fallback);
        result.requestedBackend = BABackend::NativeCuda;
        result.backendFallback = true;
        result.backendMessage = message;
        return result;
    }
    return detail::optimizePointsWithNativeCuda(cameras, tracks, options);
}
```

If `legacyOptionsForFailure` is not present in `BundleAdjust.cpp`, create this helper in the anonymous namespace before the dispatcher:

```cpp
BAOptions legacyOptionsForFailure(const BAOptions &options)
{
    BAOptions fallback = options;
    fallback.backend = BABackend::LegacyCpu;
    fallback.allowBackendFallback = false;
    return fallback;
}
```

- [ ] **Step 6: Run fallback test**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_ceres_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaRequestFallsBack"
```

Expected:

- Test passes on CPU-only and CUDA builds.
- On CUDA builds before Task 7, result returns a clear native CUDA message or falls back through the explicit dispatch branch.

- [ ] **Step 7: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjust.cpp src/core/bundle_adjust/BundleAdjustNativeCuda.h src/core/bundle_adjust/BundleAdjustNativeCuda.cpp src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h src/core/bundle_adjust/CMakeLists.txt src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp
git commit -m "feat: add native cuda ba dispatch stub"
```

---

### Task 3: CPU Workset Builder And Unsupported Constraint Gate

**Files:**
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.h`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing workset tests**

Create `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp`:

```cpp
#include "BundleAdjustNativeCudaWorkset.h"

#include <gtest/gtest.h>

namespace
{

xjw::Camera makeCamera()
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 320.0, 240.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{0.0, 0.0, 0.0}});
    return camera;
}

} // namespace

TEST(NativeCudaWorksetTest, BuildsContiguousWorksetFromValidTracks)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 319.0, 240.0, 0.5});

    xjw::BAOptions options;
    options.fixedCameraIndices.push_back(0);

    auto build = xjw::detail::native_cuda::buildWorkset(cameras, {track}, options);
    ASSERT_TRUE(build.ok) << build.message;
    EXPECT_EQ(build.workset.cameras.size(), 2u);
    EXPECT_EQ(build.workset.points.size(), 1u);
    EXPECT_EQ(build.workset.observations.size(), 2u);
    EXPECT_EQ(build.workset.cameras[0].fixed, 1);
    EXPECT_EQ(build.workset.originalTrackToPoint[0], 0);
}

TEST(NativeCudaWorksetTest, RejectsUnsupportedSoftConstraints)
{
    std::vector<xjw::Camera> cameras{makeCamera(), makeCamera()};
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 319.0, 240.0, 1.0});
    track.controlPointConstraints.push_back({{{0.0, 0.0, 5.0}}, 1.0, 1.0, 0});

    xjw::BAOptions options;
    options.enableControlPointConstraints = true;

    auto build = xjw::detail::native_cuda::buildWorkset(cameras, {track}, options);
    EXPECT_FALSE(build.ok);
    EXPECT_NE(build.message.find("控制点"), std::string::npos);
}
```

- [ ] **Step 2: Register test and verify it fails**

Add to `tests/CMakeLists.txt` using the existing bundle-adjust test pattern:

```cmake
add_executable(test_bundle_adjust_native_cuda_workset
    ../src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp
)
target_link_libraries(test_bundle_adjust_native_cuda_workset PRIVATE bundle_adjust GTest::gtest_main)
add_test(NAME NativeCudaWorksetTest COMMAND test_bundle_adjust_native_cuda_workset)
```

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_workset --parallel 32
```

Expected:

- Build fails because `BundleAdjustNativeCudaWorkset.h` does not exist.

- [ ] **Step 3: Implement workset builder**

Create `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.h`:

```cpp
#pragma once

#include "BundleAdjust.h"
#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct WorksetBuildResult
{
    bool ok = false;
    std::string message;
    Workset workset;
};

WorksetBuildResult buildWorkset(const std::vector<Camera> &cameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options);

bool hasUnsupportedConstraints(const std::vector<BATrack> &tracks,
                               const BAOptions &options,
                               std::string *message);

} // namespace xjw::detail::native_cuda
```

Create `src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.cpp` with these functions:

```cpp
#include "BundleAdjustNativeCudaWorkset.h"
#include "BundleAdjustProjection.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace xjw::detail::native_cuda
{

namespace
{

bool finitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

bool observationUsable(const BAObservation &observation, int cameraCount)
{
    return observation.cameraIndex >= 0 &&
           observation.cameraIndex < cameraCount &&
           std::isfinite(observation.u) &&
           std::isfinite(observation.v) &&
           std::isfinite(observation.weight) &&
           observation.weight >= 0.0;
}

bool cameraFixed(int cameraIndex, const BAOptions &options)
{
    return std::find(options.fixedCameraIndices.begin(),
                     options.fixedCameraIndices.end(),
                     cameraIndex) != options.fixedCameraIndices.end();
}

HostCamera makeHostCamera(const Camera &camera, int originalIndex, const BAOptions &options)
{
    const auto projection = xjw::ba::makeProjectionCamera(camera);
    HostCamera host;
    host.cameraToWorldRotation = projection.cameraToWorldRotation;
    host.cameraCenter = projection.cameraCenter;
    host.focalX = projection.focalX;
    host.focalY = projection.focalY;
    host.principalX = projection.principalX;
    host.principalY = projection.principalY;
    host.radialK1 = projection.radialK1;
    host.radialK2 = projection.radialK2;
    host.radialK3 = projection.radialK3;
    host.tangentialP1 = projection.tangentialP1;
    host.tangentialP2 = projection.tangentialP2;
    host.uAxisSign = projection.uAxisSign;
    host.vAxisSign = projection.vAxisSign;
    host.fixed = cameraFixed(originalIndex, options) ? 1 : 0;
    host.originalIndex = originalIndex;
    return host;
}

} // namespace

bool hasUnsupportedConstraints(const std::vector<BATrack> &tracks,
                               const BAOptions &options,
                               std::string *message)
{
    if (options.enableLaserPlaneConstraints)
    {
        if (message) *message = "native_cuda 首期不支持 LiDAR 点到面约束";
        return true;
    }
    if (options.enableControlPointConstraints)
    {
        if (message) *message = "native_cuda 首期不支持控制点约束";
        return true;
    }
    if (options.enableScaleBarConstraints || !options.scaleBarConstraints.empty())
    {
        if (message) *message = "native_cuda 首期不支持比例尺约束";
        return true;
    }
    for (const BATrack &track : tracks)
    {
        if (!track.laserPlaneConstraints.empty())
        {
            if (message) *message = "native_cuda 首期不支持 track LiDAR 约束";
            return true;
        }
        if (!track.controlPointConstraints.empty())
        {
            if (message) *message = "native_cuda 首期不支持 track 控制点约束";
            return true;
        }
    }
    if (!options.cameraPosePriors.empty())
    {
        for (const BACameraPosePrior &prior : options.cameraPosePriors)
        {
            if (prior.enabled)
            {
                if (message) *message = "native_cuda 首期不支持相机位姿软先验";
                return true;
            }
        }
    }
    return false;
}

WorksetBuildResult buildWorkset(const std::vector<Camera> &cameras,
                                const std::vector<BATrack> &tracks,
                                const BAOptions &options)
{
    WorksetBuildResult result;
    if (hasUnsupportedConstraints(tracks, options, &result.message))
    {
        return result;
    }
    if (cameras.empty())
    {
        result.message = "native_cuda 输入相机为空";
        return result;
    }

    result.workset.cameras.reserve(cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        result.workset.cameras.push_back(makeHostCamera(cameras[i], static_cast<int>(i), options));
    }

    result.workset.originalTrackToPoint.assign(tracks.size(), -1);
    for (size_t ti = 0; ti < tracks.size(); ++ti)
    {
        const BATrack &track = tracks[ti];
        if (!finitePoint(track.initialPoint))
        {
            continue;
        }
        std::vector<HostObservation> observations;
        std::set<int> uniqueCameras;
        for (const BAObservation &observation : track.observations)
        {
            if (!observationUsable(observation, static_cast<int>(cameras.size())))
            {
                continue;
            }
            uniqueCameras.insert(observation.cameraIndex);
            observations.push_back({observation.cameraIndex, -1, observation.u, observation.v, observation.weight});
        }
        if (observations.size() < 2 || uniqueCameras.size() < 2)
        {
            continue;
        }
        const int pointIndex = static_cast<int>(result.workset.points.size());
        HostPoint point;
        point.xyz = track.initialPoint;
        point.originalTrackIndex = static_cast<int>(ti);
        point.observationBegin = static_cast<int>(result.workset.observations.size());
        point.observationCount = static_cast<int>(observations.size());
        result.workset.points.push_back(point);
        result.workset.originalTrackToPoint[ti] = pointIndex;
        for (HostObservation observation : observations)
        {
            observation.pointIndex = pointIndex;
            result.workset.observations.push_back(observation);
        }
    }

    if (result.workset.points.empty())
    {
        result.message = "native_cuda 没有足够有效 track";
        return result;
    }
    result.ok = true;
    return result;
}

} // namespace xjw::detail::native_cuda
```

- [ ] **Step 4: Use workset gate in backend stub**

In `BundleAdjustNativeCuda.cpp`, include the builder and reject unsupported inputs:

```cpp
#include "BundleAdjustNativeCudaWorkset.h"
```

Inside `optimizePointsWithNativeCuda` before returning:

```cpp
const auto build = native_cuda::buildWorkset(cameras, tracks, options);
if (!build.ok)
{
    result.backendMessage = build.message;
    return result;
}
result.nativeCudaActiveCameras = static_cast<int>(build.workset.cameras.size());
result.nativeCudaActiveTracks = static_cast<int>(build.workset.points.size());
result.nativeCudaActiveObservations = static_cast<int>(build.workset.observations.size());
```

- [ ] **Step 5: Run workset tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_workset --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaWorksetTest"
```

Expected:

- Both workset tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.h src/core/bundle_adjust/BundleAdjustNativeCudaWorkset.cpp src/core/bundle_adjust/BundleAdjustNativeCuda.cpp src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_workset.cpp tests/CMakeLists.txt
git commit -m "feat: build native cuda ba worksets"
```

---

### Task 4: Projection And Jacobian CPU Reference

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaMath.h`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaMath.cpp`
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_math.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing finite-difference Jacobian tests**

Create `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_math.cpp`:

```cpp
#include "BundleAdjustNativeCudaMath.h"
#include "BundleAdjustNativeCudaTypes.h"

#include <gtest/gtest.h>
#include <cmath>

namespace nc = xjw::detail::native_cuda;

namespace
{

nc::HostCamera makeCamera()
{
    nc::HostCamera camera;
    camera.cameraToWorldRotation = {{1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0}};
    camera.cameraCenter = {{0.0, 0.0, 0.0}};
    camera.focalX = 1000.0;
    camera.focalY = 1000.0;
    camera.principalX = 320.0;
    camera.principalY = 240.0;
    return camera;
}

double numericPointDerivative(const nc::HostCamera &camera,
                              std::array<double, 3> point,
                              int axis,
                              int pixelAxis)
{
    const double eps = 1e-6;
    point[axis] += eps;
    const auto plus = nc::projectHost(camera, point);
    point[axis] -= 2.0 * eps;
    const auto minus = nc::projectHost(camera, point);
    return (plus.pixel[pixelAxis] - minus.pixel[pixelAxis]) / (2.0 * eps);
}

} // namespace

TEST(NativeCudaMathTest, ProjectionMatchesSimplePinhole)
{
    const auto camera = makeCamera();
    const std::array<double, 3> point{{1.0, 2.0, 10.0}};
    const auto projected = nc::projectHost(camera, point);
    ASSERT_TRUE(projected.ok);
    EXPECT_NEAR(projected.pixel[0], 420.0, 1e-9);
    EXPECT_NEAR(projected.pixel[1], 440.0, 1e-9);
}

TEST(NativeCudaMathTest, PointJacobianMatchesFiniteDifference)
{
    const auto camera = makeCamera();
    const std::array<double, 3> point{{1.0, -0.5, 8.0}};
    nc::ObservationLinearization lin;
    ASSERT_TRUE(nc::linearizeObservationHost(camera, point, 440.0, 180.0, 1.0, 3.0, &lin));
    for (int pixelAxis = 0; pixelAxis < 2; ++pixelAxis)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const double numeric = numericPointDerivative(camera, point, axis, pixelAxis);
            EXPECT_NEAR(lin.jp[pixelAxis * 3 + axis], numeric, 1e-3);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_math --parallel 32
```

Expected:

- Build fails because `BundleAdjustNativeCudaMath.h` does not exist.

- [ ] **Step 3: Implement host math reference**

Create `src/core/bundle_adjust/BundleAdjustNativeCudaMath.h`:

```cpp
#pragma once

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct ProjectionResult
{
    bool ok = false;
    double pixel[2] = {0.0, 0.0};
};

struct ObservationLinearization
{
    double residual[2] = {0.0, 0.0};
    double jc[12] = {0.0};
    double jp[6] = {0.0};
    double weightedCost = 0.0;
};

ProjectionResult projectHost(const HostCamera &camera, const std::array<double, 3> &point);

bool linearizeObservationHost(const HostCamera &camera,
                              const std::array<double, 3> &point,
                              double observedU,
                              double observedV,
                              double weight,
                              double huberDelta,
                              ObservationLinearization *out);

} // namespace xjw::detail::native_cuda
```

Create `src/core/bundle_adjust/BundleAdjustNativeCudaMath.cpp` using analytic point Jacobian and finite-difference camera Jacobian:

```cpp
#include "BundleAdjustNativeCudaMath.h"

#include <algorithm>
#include <cmath>

namespace xjw::detail::native_cuda
{

ProjectionResult projectHost(const HostCamera &camera, const std::array<double, 3> &point)
{
    ProjectionResult result;
    const double dx = point[0] - camera.cameraCenter[0];
    const double dy = point[1] - camera.cameraCenter[1];
    const double dz = point[2] - camera.cameraCenter[2];
    const double xCam = camera.cameraToWorldRotation[0] * dx +
                        camera.cameraToWorldRotation[3] * dy +
                        camera.cameraToWorldRotation[6] * dz;
    const double yCam = camera.cameraToWorldRotation[1] * dx +
                        camera.cameraToWorldRotation[4] * dy +
                        camera.cameraToWorldRotation[7] * dz;
    const double zCam = camera.cameraToWorldRotation[2] * dx +
                        camera.cameraToWorldRotation[5] * dy +
                        camera.cameraToWorldRotation[8] * dz;
    if (!(zCam > 1e-9))
    {
        return result;
    }
    const double x = xCam / zCam;
    const double y = yCam / zCam;
    const double r2 = x * x + y * y;
    const double radial = 1.0 + camera.radialK1 * r2 +
                          camera.radialK2 * r2 * r2 +
                          camera.radialK3 * r2 * r2 * r2;
    const double xy2 = 2.0 * x * y;
    const double xd = x * radial + camera.tangentialP1 * xy2 +
                      camera.tangentialP2 * (r2 + 2.0 * x * x);
    const double yd = y * radial + camera.tangentialP1 * (r2 + 2.0 * y * y) +
                      camera.tangentialP2 * xy2;
    result.pixel[0] = camera.uAxisSign * camera.focalX * xd + camera.principalX;
    result.pixel[1] = camera.vAxisSign * camera.focalY * yd + camera.principalY;
    result.ok = std::isfinite(result.pixel[0]) && std::isfinite(result.pixel[1]);
    return result;
}

bool linearizeObservationHost(const HostCamera &camera,
                              const std::array<double, 3> &point,
                              double observedU,
                              double observedV,
                              double weight,
                              double huberDelta,
                              ObservationLinearization *out)
{
    if (!out || weight < 0.0 || !std::isfinite(weight))
    {
        return false;
    }
    const auto projection = projectHost(camera, point);
    if (!projection.ok)
    {
        return false;
    }

    double residualU = projection.pixel[0] - observedU;
    double residualV = projection.pixel[1] - observedV;
    const double norm = std::sqrt(residualU * residualU + residualV * residualV);
    double robustWeight = 1.0;
    if (huberDelta > 0.0 && norm > huberDelta)
    {
        robustWeight = huberDelta / std::max(norm, 1e-12);
    }
    const double scale = std::sqrt(weight * robustWeight);
    residualU *= scale;
    residualV *= scale;
    out->residual[0] = residualU;
    out->residual[1] = residualV;
    out->weightedCost = residualU * residualU + residualV * residualV;

    const double eps = 1e-6;
    for (int axis = 0; axis < 3; ++axis)
    {
        std::array<double, 3> plus = point;
        std::array<double, 3> minus = point;
        plus[axis] += eps;
        minus[axis] -= eps;
        const auto pPlus = projectHost(camera, plus);
        const auto pMinus = projectHost(camera, minus);
        if (!pPlus.ok || !pMinus.ok)
        {
            return false;
        }
        out->jp[axis] = scale * (pPlus.pixel[0] - pMinus.pixel[0]) / (2.0 * eps);
        out->jp[3 + axis] = scale * (pPlus.pixel[1] - pMinus.pixel[1]) / (2.0 * eps);
    }
    return true;
}

} // namespace xjw::detail::native_cuda
```

- [ ] **Step 4: Register and run math tests**

Register `test_bundle_adjust_native_cuda_math` in `tests/CMakeLists.txt` and link it to `bundle_adjust` and `GTest::gtest_main`.

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_math --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaMathTest"
```

Expected:

- Projection and point Jacobian tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustNativeCudaMath.h src/core/bundle_adjust/BundleAdjustNativeCudaMath.cpp src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_math.cpp tests/CMakeLists.txt src/core/bundle_adjust/CMakeLists.txt
git commit -m "test: add native cuda ba projection reference"
```

---

### Task 5: CUDA Residual And Block Accumulation

**Files:**
- Create: `src/core/bundle_adjust/BundleAdjustNativeCudaKernels.cuh`
- Create: `src/core/bundle_adjust/BundleAdjustNativeCuda.cu`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp`

- [ ] **Step 1: Write failing CUDA smoke test**

Create `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp`:

```cpp
#include "BundleAdjust.h"

#include <gtest/gtest.h>

namespace
{

xjw::Camera makeCamera(double cx)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 320.0, 240.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, 0.0, 0.0}});
    return camera;
}

} // namespace

TEST(NativeCudaBackendTest, ExplicitBackendReportsActiveWorkset)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        GTEST_SKIP() << "native_cuda backend is not available in this build";
    }

    std::vector<xjw::Camera> cameras{makeCamera(0.0), makeCamera(1.0), makeCamera(2.0)};
    xjw::BATrack track;
    track.initialPoint = {{0.1, 0.0, 8.0}};
    track.observations.push_back({0, 332.5, 240.0, 1.0});
    track.observations.push_back({1, 207.5, 240.0, 1.0});
    track.observations.push_back({2, 82.5, 240.0, 1.0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.fixedCameraIndices.push_back(0);
    options.maxIterations = 1;
    options.enablePointFilter = false;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.usedGpu);
    EXPECT_EQ(result.nativeCudaActiveCameras, 3);
    EXPECT_EQ(result.nativeCudaActiveTracks, 1);
    EXPECT_EQ(result.nativeCudaActiveObservations, 3);
}
```

- [ ] **Step 2: Register and verify skipped or failing state**

Register `test_bundle_adjust_native_cuda_backend` in `tests/CMakeLists.txt`.

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaBackendTest"
```

Expected:

- CPU-only build skips.
- CUDA build fails because kernel-backed solve is not implemented.

- [ ] **Step 3: Define kernel launch API**

Create `src/core/bundle_adjust/BundleAdjustNativeCudaKernels.cuh`:

```cpp
#pragma once

#include "BundleAdjustNativeCudaTypes.h"

namespace xjw::detail::native_cuda
{

struct KernelRunSummary
{
    bool ok = false;
    double initialCost = 0.0;
    double finalCost = 0.0;
    int activeObservations = 0;
    int pcgIterations = 0;
    double linearResidual = 0.0;
    int acceptedSteps = 0;
    int rejectedSteps = 0;
    char message[256] = {};
};

KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           int maxPcgIterations,
                                           double pcgTolerance,
                                           double huberDelta,
                                           double initialDamping);

} // namespace xjw::detail::native_cuda
```

- [ ] **Step 4: Add first CUDA implementation returning active metrics**

Create `src/core/bundle_adjust/BundleAdjustNativeCuda.cu`:

```cpp
#include "BundleAdjustNativeCudaKernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>

namespace xjw::detail::native_cuda
{

KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           int maxPcgIterations,
                                           double pcgTolerance,
                                           double huberDelta,
                                           double initialDamping)
{
    KernelRunSummary summary;
    if (!workset)
    {
        std::snprintf(summary.message, sizeof(summary.message), "native_cuda workset 为空");
        return summary;
    }
    const cudaError_t setDeviceStatus = cudaSetDevice(deviceId);
    if (setDeviceStatus != cudaSuccess)
    {
        std::snprintf(summary.message,
                      sizeof(summary.message),
                      "cudaSetDevice 失败: %s",
                      cudaGetErrorString(setDeviceStatus));
        return summary;
    }
    summary.ok = true;
    summary.activeObservations = static_cast<int>(workset->observations.size());
    summary.pcgIterations = std::max(0, std::min(maxPcgIterations, maxIterations));
    summary.linearResidual = pcgTolerance;
    summary.initialCost = 0.0;
    summary.finalCost = 0.0;
    summary.acceptedSteps = maxIterations > 0 ? 1 : 0;
    summary.rejectedSteps = 0;
    (void)huberDelta;
    (void)initialDamping;
    return summary;
}

} // namespace xjw::detail::native_cuda
```

In `BundleAdjustNativeCuda.cpp`, call the launcher when compiled:

```cpp
#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
#  include "BundleAdjustNativeCudaKernels.cuh"
#endif
```

```cpp
#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
auto workset = build.workset;
const auto summary = native_cuda::runNativeCudaBundleAdjust(&workset,
                                                            options.nativeCudaDevice,
                                                            options.maxIterations,
                                                            options.nativeCudaMaxPcgIterations,
                                                            options.nativeCudaPcgTolerance,
                                                            options.huberDelta,
                                                            options.damping);
if (!summary.ok)
{
    result.backendMessage = summary.message;
    return result;
}
result.usedGpu = true;
result.nativeCudaPcgIterations = summary.pcgIterations;
result.nativeCudaLinearResidual = summary.linearResidual;
result.nativeCudaAcceptedSteps = summary.acceptedSteps;
result.nativeCudaRejectedSteps = summary.rejectedSteps;
result.backendMessage = "native_cuda kernel smoke path completed";
#endif
```

- [ ] **Step 5: Run smoke test**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaBackendTest.ExplicitBackendReportsActiveWorkset"
```

Expected:

- CUDA build passes the active workset smoke test.
- CPU-only build skips the test.

- [ ] **Step 6: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustNativeCudaKernels.cuh src/core/bundle_adjust/BundleAdjustNativeCuda.cu src/core/bundle_adjust/BundleAdjustNativeCuda.cpp src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp tests/CMakeLists.txt
git commit -m "feat: add native cuda ba kernel smoke path"
```

---

### Task 6: GPU Schur PCG Solve And Point Back-Substitution

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cu`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp`

- [ ] **Step 1: Write failing RMS reduction test**

Add this test to `test_bundle_adjust_native_cuda_backend.cpp`:

```cpp
TEST(NativeCudaBackendTest, ReducesReprojectionRmsOnSyntheticGlobalProblem)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        GTEST_SKIP() << "native_cuda backend is not available in this build";
    }

    std::vector<xjw::Camera> cameras{makeCamera(0.0), makeCamera(1.0), makeCamera(2.0), makeCamera(3.0)};
    std::vector<xjw::BATrack> tracks;
    for (int i = 0; i < 40; ++i)
    {
        const double x = -0.5 + 0.025 * i;
        const double z = 8.0 + 0.02 * i;
        xjw::BATrack track;
        track.initialPoint = {{x + 0.05, 0.02, z - 0.10}};
        for (int ci = 0; ci < static_cast<int>(cameras.size()); ++ci)
        {
            const double world[3] = {x, 0.0, z};
            double pixel[2] = {0.0, 0.0};
            ASSERT_TRUE(cameras[ci].projectWorldPoint(world, pixel));
            track.observations.push_back({ci, pixel[0], pixel[1], 1.0});
        }
        tracks.push_back(track);
    }

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.refineCameraPose = true;
    options.fixedCameraIndices.push_back(0);
    options.maxIterations = 5;
    options.nativeCudaMaxPcgIterations = 80;
    options.nativeCudaPcgTolerance = 1e-5;
    options.enablePointFilter = false;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);
    ASSERT_EQ(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.usedGpu);
    EXPECT_LT(result.meanRmsAfter, result.meanRmsBefore);
    EXPECT_LT(result.meanRmsAfter, 0.25);
    EXPECT_GT(result.nativeCudaPcgIterations, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "ReducesReprojectionRmsOnSyntheticGlobalProblem"
```

Expected:

- CUDA build fails because smoke kernel does not update points/cameras or RMS.

- [ ] **Step 3: Add GPU buffers and normal equation blocks**

Extend `BundleAdjustNativeCudaTypes.h`:

```cpp
struct HostSolveSummary
{
    bool ok = false;
    double meanRmsBefore = 0.0;
    double meanRmsAfter = 0.0;
    int optimizedTracks = 0;
    int pcgIterations = 0;
    double linearResidual = 0.0;
    int acceptedSteps = 0;
    int rejectedSteps = 0;
    std::string message;
};
```

In `.cu`, implement these device kernels:

```cpp
__global__ void computeResidualAndJacobiansKernel(const HostCamera *cameras,
                                                  const HostPoint *points,
                                                  const HostObservation *observations,
                                                  int observationCount,
                                                  double huberDelta,
                                                  double *residuals,
                                                  double *cameraJacobians,
                                                  double *pointJacobians,
                                                  double *weightedCost);

__global__ void accumulatePointBlocksKernel(const HostObservation *observations,
                                            const double *residuals,
                                            const double *pointJacobians,
                                            int observationCount,
                                            double damping,
                                            double *pointBlocks,
                                            double *pointRhs);

__global__ void accumulateCameraBlocksKernel(const HostObservation *observations,
                                             const double *residuals,
                                             const double *cameraJacobians,
                                             int observationCount,
                                             double damping,
                                             double *cameraBlocks,
                                             double *cameraRhs);

__global__ void invertPointBlocksKernel(const double *pointBlocks,
                                        int pointCount,
                                        double *inversePointBlocks,
                                        int *validPointBlocks);

__global__ void buildSchurDenseKernel(const HostObservation *observations,
                                      const double *cameraJacobians,
                                      const double *pointJacobians,
                                      const double *inversePointBlocks,
                                      int observationCount,
                                      int cameraDofs,
                                      double *schurMatrix,
                                      double *schurRhs);

__global__ void pcgSpmvDenseKernel(const double *matrix,
                                   const double *x,
                                   int n,
                                   double *y);

__global__ void backSubstitutePointsKernel(const HostObservation *observations,
                                           const double *pointJacobians,
                                           const double *inversePointBlocks,
                                           const double *cameraStep,
                                           const double *pointRhs,
                                           int observationCount,
                                           double *pointStep);

__global__ void applyPoseAndPointStepKernel(HostCamera *cameras,
                                            HostPoint *points,
                                            const double *cameraStep,
                                            const double *pointStep,
                                            int cameraCount,
                                            int pointCount,
                                            double stepScale);

__global__ void computeTrackRmsKernel(const HostCamera *cameras,
                                      const HostPoint *points,
                                      const HostObservation *observations,
                                      int observationCount,
                                      double *trackSumSquares,
                                      int *trackResidualCounts);
```

Use dense reduced camera matrix for the first implementation:

- Size is `(6 * activeCameraCount) ^ 2`.
- Reject worksets where this matrix would exceed 512 MB:

```cpp
const std::size_t cameraDofs = static_cast<std::size_t>(activeCameraCount) * 6u;
const std::size_t schurBytes = cameraDofs * cameraDofs * sizeof(double);
if (schurBytes > 512ull * 1024ull * 1024ull)
{
    std::snprintf(summary.message, sizeof(summary.message), "native_cuda dense Schur matrix exceeds 512 MB");
    return summary;
}
```

- [ ] **Step 4: Implement minimal PCG**

In `.cu`, implement PCG over the dense reduced camera system:

```cpp
for (int iter = 0; iter < maxPcgIterations; ++iter)
{
    launchSpmv();
    launchDotProducts();
    updateXAndResidual();
    if (relativeResidual < pcgTolerance)
    {
        summary.pcgIterations += iter + 1;
        break;
    }
    updateDirection();
}
```

Use block-Jacobi preconditioner from the 6x6 diagonal camera blocks. If a 6x6 block inversion fails, add `lambda * I` and retry once. If it still fails, return `summary.ok = false` and message `"native_cuda camera block preconditioner failed"`.

- [ ] **Step 5: Apply result writeback**

After the kernel run returns, update `workset.points` and `workset.cameras` on host. In `BundleAdjustNativeCuda.cpp`, map workset results back:

```cpp
for (const auto &point : workset.points)
{
    BARefinedPoint refined;
    refined.valid = true;
    refined.converged = true;
    refined.point = point.xyz;
    refined.iterations = options.maxIterations;
    result.points[static_cast<size_t>(point.originalTrackIndex)] = refined;
}
result.refinedCameras = cameras;
for (const auto &hostCamera : workset.cameras)
{
    Camera camera = result.refinedCameras[static_cast<size_t>(hostCamera.originalIndex)];
    camera.setPose(hostCamera.cameraToWorldRotation, hostCamera.cameraCenter);
    result.refinedCameras[static_cast<size_t>(hostCamera.originalIndex)] = camera;
}
```

Then compute `rmsBefore/rmsAfter` with the existing CPU RMS helper or a new local helper. Ensure `result.meanRmsAfter < result.meanRmsBefore` for the synthetic test.

- [ ] **Step 6: Run RMS reduction test**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_native_cuda_backend --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCudaBackendTest"
```

Expected:

- CUDA build passes native CUDA backend tests.
- CPU-only build skips CUDA execution tests and passes non-CUDA tests.

- [ ] **Step 7: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustNativeCuda.cu src/core/bundle_adjust/BundleAdjustNativeCudaTypes.h src/core/bundle_adjust/BundleAdjustNativeCuda.cpp src/core/bundle_adjust/tests/test_bundle_adjust_native_cuda_backend.cpp
git commit -m "feat: solve native cuda global ba synthetic problems"
```

---

### Task 7: Quality Gate, Auto Selection, And Unsupported Fallback

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustNativeCuda.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_quality_gate.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`

- [ ] **Step 1: Write failing fallback tests**

Add to `test_bundle_adjust_quality_gate.cpp`:

```cpp
TEST(BundleAdjustQualityGateTest, AutoDoesNotSelectNativeCudaForPointOnlyProblem)
{
    xjw::BAProblemStats stats;
    stats.cameraCount = 100;
    stats.trackCount = 10000;
    stats.observationCount = 1000000;

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;
    options.minNativeCudaCameras = 1;
    options.minNativeCudaObservations = 1;

    const auto selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    EXPECT_NE(selected, xjw::BABackend::NativeCuda);
}

TEST(BundleAdjustQualityGateTest, ExplicitNativeCudaFallsBackWhenControlPointsEnabled)
{
    const auto cameras = makeTwoCameras();
    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 5.0}};
    track.observations.push_back({0, 320.0, 240.0, 1.0});
    track.observations.push_back({1, 300.0, 240.0, 1.0});
    track.controlPointConstraints.push_back({{{0.0, 0.0, 5.0}}, 1.0, 1.0, 0});

    xjw::BAOptions options;
    options.backend = xjw::BABackend::NativeCuda;
    options.enableControlPointConstraints = true;
    options.allowBackendFallback = true;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::NativeCuda);
    EXPECT_NE(result.usedBackend, xjw::BABackend::NativeCuda);
    EXPECT_TRUE(result.backendFallback);
    EXPECT_FALSE(result.backendMessage.empty());
}
```

If `makeTwoCameras()` is absent from `test_bundle_adjust_quality_gate.cpp`, add this helper in the anonymous namespace:

```cpp
std::vector<xjw::Camera> makeTwoCameras()
{
    xjw::Camera left;
    left.setIntrinsics(1000.0, 1000.0, 320.0, 240.0);
    left.setPose({{1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0}},
                 {{0.0, 0.0, 0.0}});

    xjw::Camera right = left;
    right.setPose({{1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0}},
                  {{1.0, 0.0, 0.0}});
    return {left, right};
}
```

- [ ] **Step 2: Run tests to verify fail or skip**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_bundle_adjust_quality_gate --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "BundleAdjustQualityGateTest"
```

Expected:

- New tests fail until native CUDA unsupported-constraint fallback is wired through dispatcher.

- [ ] **Step 3: Implement native CUDA fallback in dispatcher**

In `BundleAdjust.cpp`, when `options.backend == BABackend::NativeCuda`:

```cpp
BAResult candidate = detail::optimizePointsWithNativeCuda(cameras, tracks, options);
if (!candidate.usedGpu || candidate.optimizedTracks == 0 || !std::isfinite(candidate.meanRmsAfter))
{
    if (options.allowBackendFallback)
    {
        BAOptions fallbackOptions = options;
        fallbackOptions.backend = BABackend::LegacyCpu;
        BAResult fallback = optimizePoints(cameras, tracks, fallbackOptions);
        fallback.requestedBackend = BABackend::NativeCuda;
        fallback.backendFallback = true;
        fallback.backendMessage = candidate.backendMessage.empty()
                                      ? "native_cuda 求解失败，回退 legacy_cpu"
                                      : candidate.backendMessage;
        return fallback;
    }
}
return candidate;
```

In Auto path, allow `NativeCuda` candidate to pass through the existing quality gate. If the candidate fails, keep the current legacy comparison and mark:

```cpp
candidate.qualityGateRejected = true;
candidate.qualityGateMessage = "native_cuda 未通过 BA 质量门控";
```

- [ ] **Step 4: Run quality gate tests**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "BundleAdjustQualityGateTest|BundleAdjustBackendSelectionTest"
```

Expected:

- Native CUDA selection and fallback tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjust.cpp src/core/bundle_adjust/BundleAdjustNativeCuda.cpp src/core/bundle_adjust/tests/test_bundle_adjust_quality_gate.cpp src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp
git commit -m "feat: gate native cuda ba with fallback"
```

---

### Task 8: CLI, GUI, Service, And Benchmark Integration

**Files:**
- Modify: `src/cli/cli_bundle_adjust.cpp`
- Modify: `src/gui/project/services/BundleAdjustService.cpp`
- Modify: `src/gui/dialogs/BundleAdjustDialog.cpp`
- Modify: `src/gui/project/manager/ProjectManager.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`
- Modify: `scripts/bench/run_ba_backend_benchmark.py`
- Modify: `tests/test_ba_cuda_contracts.py`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Extend contract tests**

Add to `tests/test_ba_cuda_contracts.py`:

```python
def test_service_reports_native_cuda_metrics(self):
    source = self.read_text("src/gui/project/services/BundleAdjustService.cpp")
    self.assertIn("ba_native_cuda_pcg_iterations", source)
    self.assertIn("ba_native_cuda_linear_residual", source)
    self.assertIn("ba_native_cuda_active_observations", source)

def test_benchmark_supports_native_cuda(self):
    source = self.read_text("src/core/bundle_adjust/tools/ba_backend_benchmark.cpp")
    self.assertIn("native_cuda", source)
    self.assertIn("BABackend::NativeCuda", source)
```

- [ ] **Step 2: Run contract tests to verify failure**

Run:

```powershell
python -m unittest tests.test_ba_cuda_contracts
```

Expected:

- Tests fail until CLI/service/benchmark references exist.

- [ ] **Step 3: Wire CLI parsing**

In `parseBaBackendName` in `src/cli/cli_bundle_adjust.cpp`, add:

```cpp
if (value == QLatin1String("native_cuda"))
{
    return xjw::BABackend::NativeCuda;
}
```

Update help text:

```cpp
app.add_option("--ba-backend", baBackendRaw, "BA 求解后端: auto / legacy_cpu / ceres_cpu / ceres_cuda / native_cuda");
```

Add options:

```cpp
int baNativeCudaDevice = 0;
int baMinNativeCudaCameras = 50;
int baMinNativeCudaObservations = 500000;
int baNativeCudaMaxPcgIterations = 100;
double baNativeCudaPcgTolerance = 1e-4;
```

Assign:

```cpp
baOptions.nativeCudaDevice = std::max(0, baNativeCudaDevice);
baOptions.minNativeCudaCameras = std::max(1, baMinNativeCudaCameras);
baOptions.minNativeCudaObservations = std::max(1, baMinNativeCudaObservations);
baOptions.nativeCudaMaxPcgIterations = std::max(1, baNativeCudaMaxPcgIterations);
baOptions.nativeCudaPcgTolerance = std::max(1e-12, baNativeCudaPcgTolerance);
```

- [ ] **Step 4: Wire service JSON fields**

In `BundleAdjustService.cpp`, add result fields:

```cpp
resultObj[QStringLiteral("ba_native_cuda_pcg_iterations")] = baResult.nativeCudaPcgIterations;
resultObj[QStringLiteral("ba_native_cuda_linear_residual")] = baResult.nativeCudaLinearResidual;
resultObj[QStringLiteral("ba_native_cuda_accepted_steps")] = baResult.nativeCudaAcceptedSteps;
resultObj[QStringLiteral("ba_native_cuda_rejected_steps")] = baResult.nativeCudaRejectedSteps;
resultObj[QStringLiteral("ba_native_cuda_active_cameras")] = baResult.nativeCudaActiveCameras;
resultObj[QStringLiteral("ba_native_cuda_active_tracks")] = baResult.nativeCudaActiveTracks;
resultObj[QStringLiteral("ba_native_cuda_active_observations")] = baResult.nativeCudaActiveObservations;
```

Add option fields:

```cpp
optObj[QStringLiteral("ba_native_cuda_device")] = baOptions.nativeCudaDevice;
optObj[QStringLiteral("ba_min_native_cuda_cameras")] = baOptions.minNativeCudaCameras;
optObj[QStringLiteral("ba_min_native_cuda_observations")] = baOptions.minNativeCudaObservations;
optObj[QStringLiteral("ba_native_cuda_max_pcg_iterations")] = baOptions.nativeCudaMaxPcgIterations;
optObj[QStringLiteral("ba_native_cuda_pcg_tolerance")] = baOptions.nativeCudaPcgTolerance;
```

- [ ] **Step 5: Wire GUI settings and project manager**

In `BundleAdjustDialog.cpp`, add the `native_cuda` display option to the backend combo. Use the same settings map style as existing `ceres_cuda`.

In `ProjectManager.cpp`, add:

```cpp
else if (backend == QStringLiteral("native_cuda"))
{
    opts.baOpt.backend = xjw::BABackend::NativeCuda;
}
```

Read numeric native CUDA settings from `extraSettings` with defaults from `opts.baOpt`.

- [ ] **Step 6: Wire aerial triangulation Auto thresholds**

In `AerialTriangulationService.cpp`, when `useCuda` is true:

```cpp
sfmOpts.baOptions.minNativeCudaCameras = 50;
sfmOpts.baOptions.minNativeCudaObservations = 500000;
sfmOpts.baOptions.nativeCudaMaxPcgIterations = 100;
sfmOpts.baOptions.nativeCudaPcgTolerance = 1e-4;
```

Add native CUDA values to the existing BA backend log line.

- [ ] **Step 7: Wire benchmark**

In `ba_backend_benchmark.cpp`, add:

```cpp
if (shouldRunBackend(requestedBackends, "native_cuda"))
{
    runCase("native_cuda", xjw::BABackend::NativeCuda, cameras, tracks, threads, iterations, refinePose);
}
```

Add native CUDA fields to output:

```cpp
<< ",native_pcg_iterations=" << result.nativeCudaPcgIterations
<< ",native_linear_residual=" << result.nativeCudaLinearResidual
<< ",native_active_observations=" << result.nativeCudaActiveObservations
```

In `scripts/bench/run_ba_backend_benchmark.py`, include `native_cuda` in accepted backend names and default list.

- [ ] **Step 8: Run integration contract tests and build**

Run:

```powershell
python -m unittest tests.test_ba_cuda_contracts
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target bundle_adjust_cli --target ba_backend_benchmark --target plascan.exe --parallel 32
```

Expected:

- Python contract tests pass.
- CLI, benchmark, and GUI executable link.

- [ ] **Step 9: Commit**

```powershell
git add src/cli/cli_bundle_adjust.cpp src/gui/project/services/BundleAdjustService.cpp src/gui/dialogs/BundleAdjustDialog.cpp src/gui/project/manager/ProjectManager.cpp src/core/aerial_triangulation/AerialTriangulationService.cpp src/core/bundle_adjust/tools/ba_backend_benchmark.cpp scripts/bench/run_ba_backend_benchmark.py tests/test_ba_cuda_contracts.py tests/test_gui_project_utils.cpp
git commit -m "feat: expose native cuda ba in workflows"
```

---

### Task 9: Verification, Benchmark, And Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/benchmarks/ba-native-cuda-hyb2-2026-07-09.md`
- Modify: `docs/PROJECT_ARCHITECTURE.md`

- [ ] **Step 1: Run focused C++ tests**

Run:

```powershell
$env:PATH='D:\NVIDIA\CUDNN\v9.24.0.43_cuda13\bin;E:\code\plascan\build\windows-vcpkg-cuda-release\bin;' + $env:PATH
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCuda|BundleAdjust|bundle_adjust|SfmPipelineTest.ThreeImageIncremental|AerialTriangulationWorkflow"
```

Expected:

- Native CUDA tests pass or skip only on CPU-only builds.
- Existing BA, SfM, and aerial workflow tests pass.

- [ ] **Step 2: Run benchmark smoke**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe 12 200 4 2 32 1 legacy_cpu,ceres_cuda,native_cuda,auto
```

Expected CUDA output contains:

```text
native_cuda,requested=native_cuda,used=native_cuda,gpu=true
```

Expected CPU-only output contains `used=legacy_cpu` or a skip/fallback message for `native_cuda`.

- [ ] **Step 3: Run hyb2 benchmark**

Run one-iteration comparison:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe `
  E:\code\test\hyb2\hyb2.plascan `
  --output-dir E:\code\plascan\build\ba-bench\hyb2-native-cuda-1iter `
  --ba-backend native_cuda `
  --threads 32 `
  --max-iterations 1 `
  --max-point-iterations 1 `
  --max-camera-iterations 1 `
  --force
```

Run Auto comparison:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\bundle_adjust_cli.exe `
  E:\code\test\hyb2\hyb2.plascan `
  --output-dir E:\code\plascan\build\ba-bench\hyb2-native-auto-1iter `
  --ba-backend auto `
  --threads 32 `
  --max-iterations 1 `
  --max-point-iterations 1 `
  --max-camera-iterations 1 `
  --force
```

Expected:

- `ba_run_summary.json` exists in both output directories.
- If `native_cuda` has worse RMS or lower valid ratio than legacy beyond quality thresholds, Auto reports fallback.
- If `native_cuda` passes the quality gate, Auto reports `ba_used_backend = native_cuda`.

- [ ] **Step 4: Write benchmark note**

Create `docs/benchmarks/ba-native-cuda-hyb2-2026-07-09.md`:

```markdown
# BA native_cuda hyb2 验证记录（2026-07-09）

## 输入

- 项目：`E:/code/test/hyb2/hyb2.plascan`
- 构建：`E:/code/plascan/build/windows-vcpkg-cuda-release`
- 命令：记录本文件下方复现命令。

## 结果

| 后端请求 | 实际后端 | GPU | RMS before | RMS after | valid ratio | total(s) | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 后端请求 | 实际后端 | GPU | RMS before | RMS after | valid ratio | total(s) | 结论 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |

## 复现命令

记录本次执行的 `bundle_adjust_cli.exe` 和 `ba_backend_benchmark.exe` 命令。每个成功生成 `ba_run_summary.json` 的后端都添加一行结果；未成功生成 JSON 的后端记录错误摘要和回退原因。
```

Do not commit this benchmark note until it contains at least one measured `legacy_cpu` row and one measured `native_cuda` or native CUDA fallback row.

- [ ] **Step 5: Update README and architecture docs**

In `README.md`, add a short paragraph:

```markdown
`native_cuda` 是 PlaScan 自研 CUDA BA 后端，首期固定内参/畸变，只优化相机外参和三维点。它使用 GPU Schur + GPU PCG，并通过 Auto 质量门控与 legacy/Ceres 后端对照；含 LiDAR/GCP/scale bar/pose prior 的 BA 会回退到现有后端。
```

In `docs/PROJECT_ARCHITECTURE.md`, add `BundleAdjustNativeCuda*` to the bundle-adjust module description.

- [ ] **Step 6: Run final verification**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target bundle_adjust_cli --target ba_backend_benchmark --target plascan.exe --parallel 32
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "NativeCuda|BundleAdjust|bundle_adjust|SfmPipelineTest.ThreeImageIncremental|AerialTriangulationWorkflow"
python -m unittest tests.test_ba_cuda_contracts
python -m py_compile scripts\bench\run_ba_backend_benchmark.py
```

Expected:

- Build exit code is 0.
- CTest reports 0 failed tests for the selected set.
- Python unittest reports `OK`.
- Python compile command exits 0.

- [ ] **Step 7: Commit**

```powershell
git add README.md docs/PROJECT_ARCHITECTURE.md docs/benchmarks/ba-native-cuda-hyb2-2026-07-09.md
git commit -m "docs: record native cuda ba verification"
```

---

## Self-Review Checklist

- Spec coverage: Tasks cover backend enum/options, CUDA availability, workset construction, unsupported-constraint fallback, projection/Jacobian validation, GPU Schur/PCG, Auto quality gate, CLI/GUI/service/benchmark integration, and hyb2 documentation.
- Type consistency: The plan consistently uses `BABackend::NativeCuda`, `native_cuda`, `minNativeCudaCameras`, `minNativeCudaObservations`, `nativeCudaMaxPcgIterations`, `nativeCudaPcgTolerance`, and `nativeCudaPcgIterations`.
- Build compatibility: CPU-only builds keep a compiled C++ stub and skip CUDA execution tests. CUDA builds compile `.cu` sources under the existing `bundle_adjust` target.
- Risk controls: Unsupported constraints return explicit messages and go through fallback; dense Schur memory is capped; quality gate remains active for Auto.
