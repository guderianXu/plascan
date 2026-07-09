# BA CUDA Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前“可启用但未证明更快”的 BA CUDA 后端优化成可观测、可回归、可按规模自动选择 CPU/GPU 求解路径的光束法平差实现。

**Architecture:** 先建立可重复 benchmark 和结果记录，避免性能优化靠主观判断。随后把 Ceres BA 从 `NumericDiffCostFunction + DENSE_SCHUR` 拆成明确的投影/残差/求解器选择层，优先用 AutoDiff 或解析雅可比降低 CPU 残差开销，再根据问题规模选择 legacy OpenMP、Ceres CPU、Ceres CUDA。GUI、CLI、空三流程只消费统一的 `BAOptions` 和 `BAResult` 元数据。

**Tech Stack:** C++17, CMake, GTest, Ceres Solver, CUDA Toolkit, OpenMP, Qt6, Python unittest/benchmark runner.

---

## File Structure

- Modify: `src/core/bundle_adjust/BundleAdjust.h`
  - 增加 Ceres 求解策略、性能统计字段和后端选择配置。
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
  - 保留 legacy OpenMP 路径，补充自动后端选择入口和统计填充。
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.h`
  - 拆出 Ceres 配置结构、可测试的后端能力判断接口。
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
  - 替换数值差分残差，加入 AutoDiff/解析投影残差，完善 Ceres CPU/CUDA solver 选择。
- Create: `src/core/bundle_adjust/BundleAdjustProjection.h`
  - 提供 BA 专用、可模板化的相机投影模型，供 Ceres AutoDiff 使用。
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp`
  - 验证 BA 投影模型与 `Camera::projectWorldPoint` 数值一致。
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp`
  - 扩展 CUDA 命中、fallback、统计字段和数值质量测试。
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`
  - 验证自动后端选择策略。
- Create: `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`
  - 可重复 synthetic benchmark，输出 CSV/JSON。
- Create: `scripts/bench/run_ba_backend_benchmark.py`
  - 运行 benchmark、采集结果、比较 legacy/Ceres CPU/Ceres CUDA。
- Modify: `src/core/bundle_adjust/CMakeLists.txt`
  - 注册投影测试和 benchmark 工具。
- Modify: `tests/CMakeLists.txt`
  - 注册新增测试目标。
- Modify: `src/cli/cli_bundle_adjust.cpp`
  - 暴露求解策略和 benchmark 相关输出字段。
- Modify: `src/gui/dialogs/BundleAdjustDialog.cpp`
  - 显示实际后端、是否 GPU、fallback、耗时和求解器类型。
- Modify: `src/gui/project/services/BundleAdjustService.cpp`
  - 把 BA 后端统计写入 JSON 和文本报告。
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
  - 空三 BA 默认使用自动策略，而不是无条件请求 CUDA。
- Modify: `README.md`
  - 更新 BA CUDA 的真实能力边界、推荐参数和 benchmark 命令。
- Modify: `docs/PROJECT_ARCHITECTURE.md`
  - 更新 bundle_adjust 模块边界。

---

## Task 1: Add Reproducible BA Benchmark Target

**Files:**
- Create: `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp`
- Modify: `src/core/bundle_adjust/CMakeLists.txt`
- Create: `scripts/bench/run_ba_backend_benchmark.py`

- [ ] **Step 1: Write the benchmark source**

Create `src/core/bundle_adjust/tools/ba_backend_benchmark.cpp` with a deterministic synthetic BA scene:

```cpp
#include "BundleAdjust.h"
#include "Camera.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
xjw::Camera makeCamera(double cx, double cy, double cz)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{cx, cy, cz}});
    return camera;
}

bool projectPoint(const xjw::Camera &camera, const std::array<double, 3> &point, double *u, double *v)
{
    const double world[3] = {point[0], point[1], point[2]};
    double pixel[2] = {0.0, 0.0};
    if (!camera.projectWorldPoint(world, pixel))
    {
        return false;
    }
    *u = pixel[0];
    *v = pixel[1];
    return true;
}

std::vector<xjw::Camera> makeCameras(int count)
{
    std::vector<xjw::Camera> cameras;
    cameras.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        const double t = (static_cast<double>(i) / std::max(1, count - 1) - 0.5) * 18.0;
        cameras.push_back(makeCamera(t, std::sin(i * 0.45) * 3.0, 0.0));
    }
    return cameras;
}

std::vector<xjw::BATrack> makeTracks(const std::vector<xjw::Camera> &cameras, int trackCount, int viewsPerTrack)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> xy(-4.0, 4.0);
    std::uniform_real_distribution<double> z(32.0, 52.0);
    std::normal_distribution<double> initNoise(0.0, 0.30);
    std::normal_distribution<double> imageNoise(0.0, 0.05);

    std::vector<xjw::BATrack> tracks;
    tracks.reserve(static_cast<size_t>(trackCount));
    for (int i = 0; i < trackCount; ++i)
    {
        const std::array<double, 3> truth{{xy(rng), xy(rng), z(rng)}};
        xjw::BATrack track;
        track.initialPoint = {{truth[0] + initNoise(rng), truth[1] + initNoise(rng), truth[2] + initNoise(rng)}};
        const int start = i % static_cast<int>(cameras.size());
        for (int k = 0; k < viewsPerTrack; ++k)
        {
            const int ci = (start + k * 3) % static_cast<int>(cameras.size());
            double u = 0.0;
            double v = 0.0;
            if (projectPoint(cameras[static_cast<size_t>(ci)], truth, &u, &v))
            {
                track.observations.push_back(xjw::BAObservation{ci, u + imageNoise(rng), v + imageNoise(rng), 1.0});
            }
        }
        if (track.observations.size() >= 2)
        {
            tracks.push_back(track);
        }
    }
    return tracks;
}

void runCase(const char *name,
             xjw::BABackend backend,
             const std::vector<xjw::Camera> &cameras,
             const std::vector<xjw::BATrack> &tracks,
             int threads,
             int iterations)
{
    xjw::BAOptions options;
    options.backend = backend;
    options.numThreads = threads;
    options.maxIterations = iterations;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.minCeresCudaCameras = 1;
    options.allowBackendFallback = true;

    const auto t0 = std::chrono::steady_clock::now();
    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, tracks, options);
    const auto t1 = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    std::cout << name
              << ",requested=" << xjw::BundleAdjust::backendName(result.requestedBackend)
              << ",used=" << xjw::BundleAdjust::backendName(result.usedBackend)
              << ",gpu=" << (result.usedGpu ? "true" : "false")
              << ",fallback=" << (result.backendFallback ? "true" : "false")
              << ",tracks=" << result.totalTracks
              << ",optimized=" << result.optimizedTracks
              << ",rms_before=" << result.meanRmsBefore
              << ",rms_after=" << result.meanRmsAfter
              << ",seconds=" << seconds
              << "\n";
}
} // namespace

int main(int argc, char **argv)
{
    const int cameraCount = argc > 1 ? std::max(2, std::atoi(argv[1])) : 80;
    const int trackCount = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3000;
    const int viewsPerTrack = argc > 3 ? std::max(2, std::atoi(argv[3])) : 8;
    const int iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 8;
    const int threads = argc > 5 ? std::max(1, std::atoi(argv[5])) : 32;

    const auto cameras = makeCameras(cameraCount);
    const auto tracks = makeTracks(cameras, trackCount, viewsPerTrack);
    std::cout << "dataset,cameras=" << cameras.size()
              << ",tracks=" << tracks.size()
              << ",views_per_track=" << viewsPerTrack
              << ",iterations=" << iterations
              << ",threads=" << threads
              << "\n";
    runCase("legacy_cpu", xjw::BABackend::LegacyCpu, cameras, tracks, threads, iterations);
    runCase("ceres_cpu", xjw::BABackend::CeresCpu, cameras, tracks, threads, iterations);
    runCase("ceres_cuda", xjw::BABackend::CeresCuda, cameras, tracks, threads, iterations);
    return 0;
}
```

- [ ] **Step 2: Register the benchmark target**

Modify `src/core/bundle_adjust/CMakeLists.txt`:

```cmake
add_executable(ba_backend_benchmark
    tools/ba_backend_benchmark.cpp
)
target_link_libraries(ba_backend_benchmark PRIVATE
    bundle_adjust
    camera
)
target_include_directories(ba_backend_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../camera
)
set_target_properties(ba_backend_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
```

- [ ] **Step 3: Build the target**

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target ba_backend_benchmark --parallel 32"
```

Expected: exit `0`.

- [ ] **Step 4: Add benchmark runner**

Create `scripts/bench/run_ba_backend_benchmark.py`:

```python
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_line(line: str) -> dict[str, str]:
    parts = [part.strip() for part in line.split(",")]
    row = {"case": parts[0]}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            row[key] = value
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PlaScan BA backend benchmark.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cameras", default=80, type=int)
    parser.add_argument("--tracks", default=3000, type=int)
    parser.add_argument("--views", default=8, type=int)
    parser.add_argument("--iterations", default=8, type=int)
    parser.add_argument("--threads", default=32, type=int)
    args = parser.parse_args()

    completed = subprocess.run(
        [
            str(args.exe),
            str(args.cameras),
            str(args.tracks),
            str(args.views),
            str(args.iterations),
            str(args.threads),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [parse_line(line) for line in completed.stdout.splitlines() if line and not line.startswith("[")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(completed.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the benchmark**

Run:

```powershell
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\synthetic-80x3000.csv --cameras 80 --tracks 3000 --views 8 --iterations 8 --threads 32
```

Expected: output includes `legacy_cpu`, `ceres_cpu`, `ceres_cuda`, and `ceres_cuda` reports `gpu=true` when CUDA is available.

- [ ] **Step 6: Commit**

```powershell
git add src/core/bundle_adjust/tools/ba_backend_benchmark.cpp src/core/bundle_adjust/CMakeLists.txt scripts/bench/run_ba_backend_benchmark.py
git commit -m "test: add bundle adjustment backend benchmark"
```

---

## Task 2: Add BA Projection Model Tests Before Replacing Numeric Diff

**Files:**
- Create: `src/core/bundle_adjust/BundleAdjustProjection.h`
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add the projection header**

Create `src/core/bundle_adjust/BundleAdjustProjection.h`:

```cpp
#pragma once

#include "Camera.h"

#include <array>

namespace xjw::ba
{

struct ProjectionCamera
{
    std::array<double, 9> rotation{};
    std::array<double, 3> center{};
    double fx = 1.0;
    double fy = 1.0;
    double cx = 0.0;
    double cy = 0.0;
};

ProjectionCamera makeProjectionCamera(const Camera &camera);

template <typename T>
bool projectPinhole(const ProjectionCamera &camera, const T *point, T *pixel)
{
    const T dx = point[0] - T(camera.center[0]);
    const T dy = point[1] - T(camera.center[1]);
    const T dz = point[2] - T(camera.center[2]);
    const T x = T(camera.rotation[0]) * dx + T(camera.rotation[3]) * dy + T(camera.rotation[6]) * dz;
    const T y = T(camera.rotation[1]) * dx + T(camera.rotation[4]) * dy + T(camera.rotation[7]) * dz;
    const T z = T(camera.rotation[2]) * dx + T(camera.rotation[5]) * dy + T(camera.rotation[8]) * dz;
    if (z <= T(1e-12))
    {
        return false;
    }
    pixel[0] = T(camera.fx) * x / z + T(camera.cx);
    pixel[1] = T(camera.fy) * y / z + T(camera.cy);
    return true;
}

} // namespace xjw::ba
```

- [ ] **Step 2: Implement `makeProjectionCamera` in `BundleAdjustCeres.cpp`**

Add near the top of `src/core/bundle_adjust/BundleAdjustCeres.cpp`:

```cpp
#include "BundleAdjustProjection.h"
```

Add implementation:

```cpp
namespace xjw::ba
{
ProjectionCamera makeProjectionCamera(const Camera &camera)
{
    ProjectionCamera out;
    out.rotation = camera.cameraToWorldRotation();
    out.center = camera.cameraCenter();
    out.fx = camera.fx();
    out.fy = camera.fy();
    out.cx = camera.cx();
    out.cy = camera.cy();
    return out;
}
} // namespace xjw::ba
```

If `Camera` does not currently expose `fx()`, `fy()`, `cx()`, `cy()`, add const getters to `src/core/camera/Camera.h` and `src/core/camera/Camera.cpp`.

- [ ] **Step 3: Write projection consistency test**

Create `src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp`:

```cpp
#include <gtest/gtest.h>

#include "BundleAdjustProjection.h"
#include "Camera.h"

TEST(BundleAdjustProjectionTest, MatchesCameraProjectWorldPointForPinholeCase)
{
    xjw::Camera camera;
    camera.setIntrinsics(1000.0, 980.0, 512.0, 384.0);
    camera.setPose({{1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0}},
                   {{-2.0, 1.0, 0.0}});

    const double world[3] = {0.3, -0.2, 40.0};
    double cameraPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(camera.projectWorldPoint(world, cameraPixel));

    const auto projectionCamera = xjw::ba::makeProjectionCamera(camera);
    double projectionPixel[2] = {0.0, 0.0};
    ASSERT_TRUE(xjw::ba::projectPinhole(projectionCamera, world, projectionPixel));

    EXPECT_NEAR(projectionPixel[0], cameraPixel[0], 1e-9);
    EXPECT_NEAR(projectionPixel[1], cameraPixel[1], 1e-9);
}
```

- [ ] **Step 4: Register and run the test**

Modify `tests/CMakeLists.txt`:

```cmake
add_executable(test_bundle_adjust_projection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp
)
target_link_libraries(test_bundle_adjust_projection PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_projection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_projection)
```

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_bundle_adjust_projection --parallel 32"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R BundleAdjustProjection
```

Expected: projection test passes.

- [ ] **Step 5: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustProjection.h src/core/bundle_adjust/BundleAdjustCeres.cpp src/core/bundle_adjust/tests/test_bundle_adjust_projection.cpp tests/CMakeLists.txt src/core/camera/Camera.h src/core/camera/Camera.cpp
git commit -m "test: cover bundle adjustment projection model"
```

---

## Task 3: Replace Ceres Numeric Diff Reprojection Residual

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
- Modify: `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp`

- [ ] **Step 1: Add a test that protects backend quality**

Extend `src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp` with:

```cpp
TEST(BundleAdjustCeresBackendTest, CeresCudaAndCpuReachComparableRms)
{
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu));

    const std::vector<xjw::Camera> cameras{
        makeCamera(-8.0, 0.0, 0.0),
        makeCamera(8.0, 0.0, 0.0),
        makeCamera(0.0, 8.0, 0.0),
        makeCamera(0.0, -8.0, 0.0),
    };
    const std::array<double, 3> truth{{0.4, -0.3, 38.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, {{1.5, -1.2, 43.0}});

    xjw::BAOptions cpuOptions;
    cpuOptions.backend = xjw::BABackend::CeresCpu;
    cpuOptions.refineCameraPose = false;
    cpuOptions.enablePointFilter = false;
    cpuOptions.maxIterations = 20;
    const xjw::BAResult cpu = xjw::BundleAdjust::optimizePoints(cameras, {track}, cpuOptions);

    xjw::BAOptions cudaOptions = cpuOptions;
    cudaOptions.backend = xjw::BABackend::CeresCuda;
    cudaOptions.minCeresCudaCameras = 1;
    const xjw::BAResult gpu = xjw::BundleAdjust::optimizePoints(cameras, {track}, cudaOptions);

    ASSERT_EQ(cpu.points.size(), 1u);
    ASSERT_EQ(gpu.points.size(), 1u);
    ASSERT_TRUE(cpu.points.front().valid);
    ASSERT_TRUE(gpu.points.front().valid);
    EXPECT_NEAR(gpu.meanRmsAfter, cpu.meanRmsAfter, 1e-6);
}
```

- [ ] **Step 2: Run the test before implementation**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R BundleAdjustCeres
```

Expected: current tests pass; new quality guard may pass before replacement, but it must remain green after replacement.

- [ ] **Step 3: Replace reprojection numeric diff with AutoDiff**

In `src/core/bundle_adjust/BundleAdjustCeres.cpp`, replace `ReprojectionResidual` with a templated functor using `xjw::ba::projectPinhole`:

```cpp
struct ReprojectionResidual
{
    xjw::ba::ProjectionCamera camera;
    BAObservation observation;

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        T pixel[2] = {T(0), T(0)};
        const T sqrtWeight = sqrt(T(safeObservationWeight(observation)));
        if (!xjw::ba::projectPinhole(camera, point, pixel))
        {
            residuals[0] = sqrtWeight * T(1.0e6);
            residuals[1] = sqrtWeight * T(1.0e6);
            return true;
        }
        residuals[0] = sqrtWeight * (pixel[0] - T(observation.u));
        residuals[1] = sqrtWeight * (pixel[1] - T(observation.v));
        return true;
    }
};
```

When `options.refineCameraPose == false`, add residual blocks as:

```cpp
auto *cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 3>(
    new ReprojectionResidual{xjw::ba::makeProjectionCamera(cameras[static_cast<size_t>(observation.cameraIndex)]),
                             observation});
problem.AddResidualBlock(cost,
                         makeHuberLoss(options.huberDelta),
                         pointParams[ti].data());
```

Keep the existing numeric-diff residual only for `options.refineCameraPose == true` until Task 4 adds pose AutoDiff.

- [ ] **Step 4: Run BA tests**

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_bundle_adjust_ceres_backend ba_backend_benchmark --parallel 32"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R BundleAdjustCeres
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\after-autodiff-points.csv --cameras 80 --tracks 3000 --views 8 --iterations 8 --threads 32
```

Expected: Ceres CPU and CUDA RMS remain comparable to previous output; Ceres CPU/CUDA time should improve relative to the pre-change benchmark.

- [ ] **Step 5: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjustCeres.cpp src/core/bundle_adjust/tests/test_bundle_adjust_ceres_backend.cpp
git commit -m "perf: use autodiff reprojection for fixed-camera BA"
```

---

## Task 4: Add Solver Policy and Automatic Backend Selection

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.h`
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
- Create: `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add failing selection tests**

Create `src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp`:

```cpp
#include <gtest/gtest.h>

#include "BundleAdjust.h"

TEST(BundleAdjustBackendSelectionTest, SmallFixedCameraProblemUsesLegacyCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.minCeresCudaCameras = 50;

    const xjw::BAProblemStats stats;
    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    EXPECT_EQ(selected, xjw::BABackend::LegacyCpu);
}

TEST(BundleAdjustBackendSelectionTest, LargeProblemCanSelectCudaWhenAvailable)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.minCeresCudaCameras = 50;

    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 50000;
    stats.observationCount = 300000;
    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::CeresCuda);
    }
    else
    {
        EXPECT_NE(selected, xjw::BABackend::CeresCuda);
    }
}
```

- [ ] **Step 2: Add types to `BundleAdjust.h`**

Add:

```cpp
enum class BABackend
{
    Auto,
    LegacyCpu,
    CeresCpu,
    CeresCuda
};

enum class BACeresLinearSolver
{
    Auto,
    DenseSchurCpu,
    DenseSchurCuda,
    SparseSchurCpu
};

struct BAProblemStats
{
    int cameraCount = 0;
    int trackCount = 0;
    int observationCount = 0;
};
```

Add fields:

```cpp
BACeresLinearSolver ceresLinearSolver = BACeresLinearSolver::Auto;
int minCeresCudaObservations = 500000;
```

Add API:

```cpp
static BAProblemStats summarizeProblem(const std::vector<Camera> &cameras,
                                       const std::vector<BATrack> &tracks);
static BABackend selectBackendForProblem(const BAProblemStats &stats,
                                         const BAOptions &options);
```

- [ ] **Step 3: Implement selection**

In `src/core/bundle_adjust/BundleAdjust.cpp`:

```cpp
BAProblemStats BundleAdjust::summarizeProblem(const std::vector<Camera> &cameras,
                                              const std::vector<BATrack> &tracks)
{
    BAProblemStats stats;
    stats.cameraCount = static_cast<int>(cameras.size());
    stats.trackCount = static_cast<int>(tracks.size());
    for (const BATrack &track : tracks)
    {
        stats.observationCount += static_cast<int>(track.observations.size());
    }
    return stats;
}

BABackend BundleAdjust::selectBackendForProblem(const BAProblemStats &stats,
                                                const BAOptions &options)
{
    if (options.backend != BABackend::Auto)
    {
        return options.backend;
    }
    if (isBackendAvailable(BABackend::CeresCuda) &&
        stats.cameraCount >= options.minCeresCudaCameras &&
        stats.observationCount >= options.minCeresCudaObservations)
    {
        return BABackend::CeresCuda;
    }
    if (isBackendAvailable(BABackend::CeresCpu) && stats.observationCount >= 50000)
    {
        return BABackend::CeresCpu;
    }
    return BABackend::LegacyCpu;
}
```

Use `selectBackendForProblem` at the start of `BundleAdjust::optimizePoints`.

- [ ] **Step 4: Register and run tests**

Modify `tests/CMakeLists.txt`:

```cmake
add_executable(test_bundle_adjust_backend_selection
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp
)
target_link_libraries(test_bundle_adjust_backend_selection PRIVATE
    bundle_adjust
    camera
    GTest::gtest_main
)
target_include_directories(test_bundle_adjust_backend_selection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/bundle_adjust
    ${CMAKE_SOURCE_DIR}/src/core/camera
)
gtest_discover_tests(test_bundle_adjust_backend_selection)
```

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_bundle_adjust_backend_selection --parallel 32"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R BundleAdjustBackendSelection
```

Expected: backend selection tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjust.h src/core/bundle_adjust/BundleAdjust.cpp src/core/bundle_adjust/BundleAdjustCeres.h src/core/bundle_adjust/BundleAdjustCeres.cpp src/core/bundle_adjust/tests/test_bundle_adjust_backend_selection.cpp tests/CMakeLists.txt
git commit -m "feat: add automatic BA backend selection"
```

---

## Task 5: Record Solver Timing and Backend Details

**Files:**
- Modify: `src/core/bundle_adjust/BundleAdjust.h`
- Modify: `src/core/bundle_adjust/BundleAdjust.cpp`
- Modify: `src/core/bundle_adjust/BundleAdjustCeres.cpp`
- Modify: `src/gui/project/services/BundleAdjustService.cpp`
- Modify: `src/gui/dialogs/BundleAdjustDialog.cpp`
- Modify: `src/cli/cli_bundle_adjust.cpp`

- [ ] **Step 1: Add result fields**

In `BAResult`:

```cpp
double setupSeconds = 0.0;
double solveSeconds = 0.0;
double totalSeconds = 0.0;
int observationCount = 0;
const char *ceresLinearSolverName = "";
```

- [ ] **Step 2: Fill timing in Ceres backend**

In `optimizePointsWithCeres`, measure problem construction and solve:

```cpp
const auto setupStart = std::chrono::steady_clock::now();
// build ceres::Problem
const auto setupEnd = std::chrono::steady_clock::now();
ceres::Solve(solverOptions, &problem, &summary);
const auto solveEnd = std::chrono::steady_clock::now();

result.setupSeconds = std::chrono::duration<double>(setupEnd - setupStart).count();
result.solveSeconds = std::chrono::duration<double>(solveEnd - setupEnd).count();
result.totalSeconds = std::chrono::duration<double>(solveEnd - setupStart).count();
```

- [ ] **Step 3: Fill timing in legacy backend**

In legacy `optimizePointsLegacyImpl`, wrap the existing optimization body and set:

```cpp
result.setupSeconds = 0.0;
result.solveSeconds = result.totalSeconds;
result.ceresLinearSolverName = "none";
```

- [ ] **Step 4: Expose in service reports**

In `BundleAdjustService.cpp`, add JSON keys:

```cpp
resultJson["ba_setup_seconds"] = baResult.setupSeconds;
resultJson["ba_solve_seconds"] = baResult.solveSeconds;
resultJson["ba_total_seconds"] = baResult.totalSeconds;
resultJson["ba_observation_count"] = baResult.observationCount;
resultJson["ba_ceres_linear_solver"] = QString::fromLatin1(baResult.ceresLinearSolverName);
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target test_bundle_adjust_ceres_backend test_bundle_adjust_service_lidar bundle_adjust_cli plascan_gui --parallel 32"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R "BundleAdjustCeres|BundleAdjustServiceLidar"
```

Expected: tests pass and reports contain timing fields.

- [ ] **Step 6: Commit**

```powershell
git add src/core/bundle_adjust/BundleAdjust.h src/core/bundle_adjust/BundleAdjust.cpp src/core/bundle_adjust/BundleAdjustCeres.cpp src/gui/project/services/BundleAdjustService.cpp src/gui/dialogs/BundleAdjustDialog.cpp src/cli/cli_bundle_adjust.cpp
git commit -m "feat: report bundle adjustment backend timings"
```

---

## Task 6: Integrate Auto Backend Into Aerial Triangulation and CLI

**Files:**
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/gui/project/manager/ProjectManager.cpp`
- Modify: `src/gui/dialogs/BundleAdjustDialog.cpp`
- Modify: `src/cli/cli_bundle_adjust.cpp`
- Modify: `tests/test_ba_cuda_contracts.py`

- [ ] **Step 1: Update contract tests**

Extend `tests/test_ba_cuda_contracts.py`:

```python
def test_aerial_triangulation_uses_auto_backend_not_unconditional_cuda(self):
    source = (ROOT / "src/core/aerial_triangulation/AerialTriangulationService.cpp").read_text(encoding="utf-8")
    self.assertIn("BABackend::Auto", source)
    self.assertNotIn("sfmOpts.baOptions.backend = xjw::BABackend::CeresCuda;", source)

def test_cli_accepts_auto_backend(self):
    source = (ROOT / "src/cli/cli_bundle_adjust.cpp").read_text(encoding="utf-8")
    self.assertIn("auto|legacy_cpu|ceres_cpu|ceres_cuda", source)
    self.assertIn("BABackend::Auto", source)
```

- [ ] **Step 2: Change aerial triangulation default**

In `AerialTriangulationService.cpp`, set:

```cpp
sfmOpts.baOptions.backend = xjw::BABackend::Auto;
sfmOpts.baOptions.allowBackendFallback = true;
sfmOpts.baOptions.minCeresCudaCameras = 50;
sfmOpts.baOptions.minCeresCudaObservations = 500000;
```

Keep CUDA availability as a capability, not a forced backend.

- [ ] **Step 3: Change CLI parser**

In `cli_bundle_adjust.cpp`, accept:

```cpp
if (name == "auto")
{
    return xjw::BABackend::Auto;
}
```

Default `baBackendRaw` should become `"auto"`.

- [ ] **Step 4: Run contract and build**

Run:

```powershell
python tests\test_ba_cuda_contracts.py
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --target bundle_adjust_cli plascan_gui --parallel 32"
```

Expected: contract tests and build pass.

- [ ] **Step 5: Commit**

```powershell
git add src/core/aerial_triangulation/AerialTriangulationService.cpp src/gui/project/manager/ProjectManager.cpp src/gui/dialogs/BundleAdjustDialog.cpp src/cli/cli_bundle_adjust.cpp tests/test_ba_cuda_contracts.py
git commit -m "feat: use automatic BA backend selection in workflows"
```

---

## Task 7: Real Data Validation on hyb2

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Optional create: `docs/benchmarks/ba-cuda-hyb2-2026-07-08.md`

- [ ] **Step 1: Run synthetic benchmarks at three scales**

Run:

```powershell
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\synthetic-small.csv --cameras 30 --tracks 3000 --views 6 --iterations 8 --threads 32
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\synthetic-medium.csv --cameras 120 --tracks 50000 --views 8 --iterations 8 --threads 32
python scripts\bench\run_ba_backend_benchmark.py --exe E:\code\plascan\build\windows-vcpkg-cuda-release\bin\ba_backend_benchmark.exe --out E:\code\plascan\build\ba-bench\synthetic-large.csv --cameras 200 --tracks 150000 --views 8 --iterations 8 --threads 32
```

Expected: each CSV contains all backends, actual backend, GPU flag, time, RMS.

- [ ] **Step 2: Run hyb2 aerial triangulation**

Run the existing CLI workflow that matches GUI aerial triangulation. Use the actual CLI target name available in the build:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\reconstruct_pipeline_cli.exe --project E:\code\test\hyb2 --aerial-triangulation-only --ba-backend auto
```

If the CLI does not expose exactly these flags, inspect:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\bin\reconstruct_pipeline_cli.exe --help
```

and record the exact working command in `docs/benchmarks/ba-cuda-hyb2-2026-07-08.md`.

- [ ] **Step 3: Check result metadata**

Verify the output project/report includes:

```text
ba_requested_backend
ba_used_backend
ba_used_gpu
ba_backend_fallback
ba_setup_seconds
ba_solve_seconds
ba_total_seconds
ba_observation_count
```

- [ ] **Step 4: Document benchmark result**

Create `docs/benchmarks/ba-cuda-hyb2-2026-07-08.md` with:

```markdown
# BA CUDA hyb2 Benchmark - 2026-07-08

## Environment

- GPU: NVIDIA GeForce RTX 5080
- Build: `build/windows-vcpkg-cuda-release`
- Threads: 32
- Dataset: `E:/code/test/hyb2`

## Commands

```powershell
<exact command used>
```

## Results

| Case | Used Backend | GPU | Fallback | Observations | Setup Seconds | Solve Seconds | Total Seconds | RMS Before | RMS After |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hyb2 auto |  |  |  |  |  |  |  |  |  |

## Conclusion

- CUDA is considered beneficial only if it reduces total BA time without increasing RMS or reducing registered cameras.
- If CUDA is slower on hyb2, keep `auto` thresholds conservative and prefer legacy/Ceres CPU for this project size.
```

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
cmd.exe /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --parallel 32"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure -R "BundleAdjust|Sfm|Aerial"
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release --output-on-failure
```

Expected: focused tests pass. If full `ctest` hits the known `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` `dom_png not found` historical failure, record it separately and do not claim full pass.

- [ ] **Step 6: Commit**

```powershell
git add README.md docs/PROJECT_ARCHITECTURE.md docs/benchmarks/ba-cuda-hyb2-2026-07-08.md
git commit -m "docs: record bundle adjustment cuda benchmark results"
```

---

## Success Criteria

- `ctest -R "BundleAdjust|Sfm|Aerial"` passes.
- `test_bundle_adjust_ceres_backend` proves CUDA requests actually use `CeresCuda` with `usedGpu=true` on CUDA builds.
- Benchmark CSV exists for small, medium, and large synthetic cases.
- hyb2 validation records actual backend, GPU usage, fallback state, RMS, camera/track stats, and timing.
- `auto` backend does not force CUDA when problem size is too small.
- CUDA is only recommended when measured `totalSeconds` improves without worse RMS or fewer registered cameras.

## Known Risks

- Current Ceres CUDA path accelerates solver linear algebra, not all residual evaluation.
- If camera pose refinement still uses numeric diff, CUDA benefit may remain limited.
- `DENSE_SCHUR` can be inappropriate for very large SfM problems; if benchmark shows memory pressure, add a follow-up plan for sparse Schur/CUDA sparse after confirming Ceres build support.
- The legacy OpenMP backend may remain faster for fixed-camera point-only BA; this is acceptable if `Auto` selects it.
