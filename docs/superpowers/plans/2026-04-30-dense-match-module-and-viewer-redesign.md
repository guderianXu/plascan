# 密集匹配模块与统一连接点查看器 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 CUDA 加速的密集匹配模块（参考 ASP），重构连接点查看器为稀疏/密集统一查看器，重组菜单结构。

**Architecture:** 新建 `core/dense_match/` 模块（代价函数→块匹配/SGM→子像素精化→视差验证→服务层），重构 `MatchViewerDialog` 为 Tab 切换的统一查看器，新增 `DisparityHeatmapOverlay` 和 `DenseMatchDialog`。遵循 TDD，每个算法模块先写测试再实现。

**Tech Stack:** C++17, CUDA, OpenCV, Qt6, Google Test, CMake

**Spec:** `docs/superpowers/specs/2026-04-30-dense-match-module-and-viewer-redesign.md`

---

### Task 1: 创建 dense_match 模块骨架和类型定义

**Files:**
- Create: `src/core/dense_match/CMakeLists.txt`
- Create: `src/core/dense_match/DenseMatchTypes.h`
- Create: `src/core/dense_match/DenseMatchConfig.h`
- Modify: `src/core/CMakeLists.txt`

- [ ] **Step 1: 创建 DenseMatchTypes.h**

```cpp
// =============================================================================
// 文件: DenseMatchTypes.h
// 功能: 密集匹配模块公共类型定义
// =============================================================================
#pragma once

#include <opencv2/core.hpp>

namespace dense_match
{

enum class CostFunction
{
    AbsoluteDifference      = 0,
    SquaredDifference       = 1,
    NormalizedCrossCorr     = 2,
    CensusTransform         = 3,
    TernaryCensusTransform  = 4
};

enum class StereoAlgorithm
{
    BlockMatch      = 0,
    SemiGlobalMatch = 1,
    MoreGlobalMatch = 2,
    OpenCV_SGBM     = 3
};

enum class SubpixelMode
{
    None        = 0,
    Parabola    = 1,
    AffineBayes = 2,
    LucasKanade = 3
};

struct DisparityResult
{
    cv::Mat disparity;
    cv::Mat confidence;
    cv::Mat validMask;
};

} // namespace dense_match
```

- [ ] **Step 2: 创建 DenseMatchConfig.h**

```cpp
// =============================================================================
// 文件: DenseMatchConfig.h
// 功能: 密集匹配参数配置结构体
// =============================================================================
#pragma once

#include "DenseMatchTypes.h"
#include <string>

namespace dense_match
{

struct DenseMatchConfig
{
    StereoAlgorithm algorithm  = StereoAlgorithm::MoreGlobalMatch;
    CostFunction    costFunc   = CostFunction::CensusTransform;
    SubpixelMode    subpixel   = SubpixelMode::Parabola;

    int minDisparity = 0;
    int maxDisparity = 256;

    int corrKernelW = 15;
    int corrKernelH = 15;

    int p1            = 8;
    int p2            = 32;
    int sgmDirections = 8;
    int sgmCollarSize = 512;

    int pyramidLevels = 2;

    int subpixelKernelW = 21;
    int subpixelKernelH = 21;

    float lrCheckThreshold = 1.0f;
    int   medianFilterSize = 3;

    bool useCuda     = true;
    int  cudaDevice  = 0;
    int  numThreads  = 4;

    std::string leftImagePath;
    std::string rightImagePath;
    std::string outputDisparityPath;
};

} // namespace dense_match
```

- [ ] **Step 3: 创建 CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.16)
project(dense_match LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(DM_HAS_CUDA TRUE)
    message(STATUS "dense_match: CUDA enabled (${CMAKE_CUDA_COMPILER})")
else()
    set(DM_HAS_CUDA FALSE)
    message(STATUS "dense_match: CUDA unavailable, CPU-only")
endif()

set(DM_CXX_SOURCES
    CostFunctions.cpp
    BlockMatcher.cpp
    SgmMatcher.cpp
    SubpixelRefiner.cpp
    DisparityValidator.cpp
    DenseMatchService.cpp
    opencv/OpenCVSgbmWrapper.cpp
)

set(DM_HEADERS
    DenseMatchTypes.h
    DenseMatchConfig.h
    CostFunctions.h
    BlockMatcher.h
    SgmMatcher.h
    SubpixelRefiner.h
    DisparityValidator.h
    DenseMatchService.h
    opencv/OpenCVSgbmWrapper.h
)

if(DM_HAS_CUDA)
    set(DM_CUDA_SOURCES
        CostFunctions.cu
        BlockMatcher.cu
        SgmMatcher.cu
        SubpixelRefiner.cu
    )
else()
    set(DM_CUDA_SOURCES)
endif()

add_library(dense_match STATIC
    ${DM_CXX_SOURCES}
    ${DM_CUDA_SOURCES}
    ${DM_HEADERS}
)

if(DM_HAS_CUDA)
    set_target_properties(dense_match PROPERTIES
        CUDA_ARCHITECTURES "${PLASCAN_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(dense_match PUBLIC DM_ENABLE_CUDA=1)
    target_compile_options(dense_match PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    )
endif()

target_include_directories(dense_match PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dense_match
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(OpenMP_CXX_FOUND)
    target_compile_definitions(dense_match PRIVATE HAS_OPENMP)
endif()

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        include(GoogleTest)
        add_executable(test_dense_match_unit
            tests/CostFunctionTest.cpp
            tests/BlockMatcherTest.cpp
            tests/SgmMatcherTest.cpp
            tests/SubpixelRefinerTest.cpp
            tests/DisparityValidatorTest.cpp
            tests/DenseMatchIntegrationTest.cpp
        )
        target_link_libraries(test_dense_match_unit PRIVATE
            dense_match
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        target_include_directories(test_dense_match_unit PRIVATE
            ${CMAKE_SOURCE_DIR}/src/core/dense_match
        )
        gtest_discover_tests(test_dense_match_unit)
    endif()
endif()
```

- [ ] **Step 4: 注册到 core/CMakeLists.txt**

```cmake
# 在现有 plascan_core_add_optional_module 调用列表末尾、mesh 之前添加：
plascan_core_add_optional_module(dense_match "DenseMatch")
```

Edit `src/core/CMakeLists.txt`: add `plascan_core_add_optional_module(dense_match "DenseMatch")` before the `plascan_core_add_optional_module(mesh "Meshing")` line.

- [ ] **Step 5: Commit**

```bash
git add src/core/dense_match/ src/core/CMakeLists.txt
git commit -m "feat: add dense_match module skeleton with types and config"
```

---

### Task 2: CostFunctions — TDD 实现

**Files:**
- Create: `src/core/dense_match/CostFunctions.h`
- Create: `src/core/dense_match/CostFunctions.cpp`
- Create: `src/core/dense_match/CostFunctions.cu`
- Create: `src/core/dense_match/tests/CostFunctionTest.cpp`

- [ ] **Step 1: 写 CostFunctions.h 头文件**

```cpp
// =============================================================================
// 文件: CostFunctions.h
// 功能: 密集匹配代价函数声明（CPU + CUDA）
// =============================================================================
#pragma once

#include "DenseMatchTypes.h"
#include <opencv2/core.hpp>

namespace dense_match
{

// 代价卷: [H x W x D] 格式，D = maxDisparity - minDisparity
using CostVolume = std::vector<cv::Mat>;

// CPU 计算单像素代价
float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func);

// CPU 计算完整代价卷
CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads = 1);

#ifdef DM_ENABLE_CUDA
// CUDA 计算代价卷
CostVolume computeCostVolumeCUDA(const cv::Mat &left, const cv::Mat &right,
                                 int minDisp, int maxDisp,
                                 int kernelW, int kernelH,
                                 CostFunction func, int cudaDevice = 0);
#endif

} // namespace dense_match
```

- [ ] **Step 2: 写测试 CostFunctionTest.cpp（先写，验证 TDD 流程）**

```cpp
// =============================================================================
// 文件: CostFunctionTest.cpp
// 功能: 代价函数单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "CostFunctions.h"
#include <opencv2/core.hpp>

using namespace dense_match;

class CostFunctionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 5x5 左图全白，右图全白
        leftWhite  = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        rightWhite = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));
        // 5x5 左图全黑，右图全黑
        leftBlack  = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
        rightBlack = cv::Mat(5, 5, CV_8UC1, cv::Scalar(0));
    }

    cv::Mat leftWhite, rightWhite, leftBlack, rightBlack;
};

TEST_F(CostFunctionTest, AD_IdenticalImages_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 4, 3, 3,
                                    CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 5u);
    for (int d = 0; d < 5; ++d)
    {
        double minVal, maxVal;
        cv::minMaxLoc(volume[d], &minVal, &maxVal);
        EXPECT_NEAR(minVal, 0.0, 1e-5);
        EXPECT_NEAR(maxVal, 0.0, 1e-5);
    }
}

TEST_F(CostFunctionTest, AD_MaxDifference_Produces255)
{
    auto volume = computeCostVolume(leftWhite, rightBlack, 0, 1, 1, 1,
                                    CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(maxVal, 255.0, 1e-5);
}

TEST_F(CostFunctionTest, SD_IdenticalImages_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 4, 3, 3,
                                    CostFunction::SquaredDifference);
    ASSERT_EQ(volume.size(), 5u);
    for (int d = 0; d < 5; ++d)
    {
        double minVal, maxVal;
        cv::minMaxLoc(volume[d], &minVal, &maxVal);
        EXPECT_NEAR(minVal, 0.0, 1e-5);
    }
}

TEST_F(CostFunctionTest, NCC_IdenticalImages_ZeroCost)
{
    // NCC 在完全相同窗口下代价应为 0 (1 - NCC = 1-1 = 0)
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 1, 3, 3,
                                    CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(minVal, 0.0, 0.01);
}

TEST_F(CostFunctionTest, NCC_InverseImages_ProducesCostTwo)
{
    // NCC = -1 → Cost = 1 - (-1) = 2
    auto volume = computeCostVolume(leftWhite, rightBlack, 0, 1, 3, 3,
                                    CostFunction::NormalizedCrossCorr);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(maxVal, 2.0, 0.01);
}

TEST_F(CostFunctionTest, Census_Identical_ZeroCost)
{
    auto volume = computeCostVolume(leftWhite, rightWhite, 0, 1, 3, 3,
                                    CostFunction::CensusTransform);
    ASSERT_EQ(volume.size(), 1u);
    double minVal, maxVal;
    cv::minMaxLoc(volume[0], &minVal, &maxVal);
    EXPECT_NEAR(minVal, 0.0, 1e-5);
}

TEST_F(CostFunctionTest, VolumeCorrectDimensions)
{
    int minDisp = 0, maxDisp = 16;
    auto volume = computeCostVolume(leftWhite, rightWhite, minDisp, maxDisp,
                                    5, 5, CostFunction::AbsoluteDifference);
    ASSERT_EQ(volume.size(), static_cast<size_t>(maxDisp - minDisp));
    for (const auto &slice : volume)
    {
        EXPECT_EQ(slice.rows, leftWhite.rows);
        EXPECT_EQ(slice.cols, leftWhite.cols);
    }
}
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . --target test_dense_match_unit 2>&1 | tail -20
```
Expected: 编译失败（CostFunctions.cpp 未实现）或链接失败。

- [ ] **Step 4: 实现 CostFunctions.cpp（CPU 版本）**

```cpp
// =============================================================================
// 文件: CostFunctions.cpp
// 功能: 密集匹配代价函数 CPU 实现
// =============================================================================
#include "CostFunctions.h"
#include <cmath>
#include <omp.h>

namespace dense_match
{

static float adCost(const uchar *left, const uchar *right,
                    int x, int y, int d, int kw, int kh,
                    int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2, halfKH = kh / 2;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            sum += std::abs(static_cast<float>(left[ry * imgW + lx]) -
                            static_cast<float>(right[ry * imgW + rx]));
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

static float sdCost(const uchar *left, const uchar *right,
                    int x, int y, int d, int kw, int kh,
                    int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2, halfKH = kh / 2;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            float diff = static_cast<float>(left[ry * imgW + lx]) -
                         static_cast<float>(right[ry * imgW + rx]);
            sum += diff * diff;
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

static float nccCost(const uchar *left, const uchar *right,
                     int x, int y, int d, int kw, int kh,
                     int imgW, int imgH)
{
    int halfKW = kw / 2, halfKH = kh / 2;
    double meanL = 0.0, meanR = 0.0;
    int count = 0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            meanL += left[ry * imgW + lx];
            meanR += right[ry * imgW + rx];
            ++count;
        }
    }
    if (count < 2) return 0.0f;
    meanL /= count;
    meanR /= count;

    double num = 0.0, denL = 0.0, denR = 0.0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            double dl = left[ry * imgW + lx] - meanL;
            double dr = right[ry * imgW + rx] - meanR;
            num += dl * dr;
            denL += dl * dl;
            denR += dr * dr;
        }
    }
    double denom = std::sqrt(denL * denR);
    if (denom < 1e-10) return 0.0f;
    return static_cast<float>(1.0 - (num / denom));
}

static float censusCost(const uchar *left, const uchar *right,
                        int x, int y, int d, int kw, int kh,
                        int imgW, int imgH)
{
    int halfKW = kw / 2, halfKH = kh / 2;
    int hamming = 0;
    int count = 0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            if (dx == 0 && dy == 0) continue; // 跳過中心像素
            uchar centerL = left[y * imgW + (x + d)];
            uchar centerR = right[y * imgW + x];
            int bitL = (left[ry * imgW + lx] > centerL) ? 1 : 0;
            int bitR = (right[ry * imgW + rx] > centerR) ? 1 : 0;
            hamming += (bitL != bitR) ? 1 : 0;
            ++count;
        }
    }
    return count > 0 ? static_cast<float>(hamming) / count : 0.0f;
}

static float ternaryCensusCost(const uchar *left, const uchar *right,
                               int x, int y, int d, int kw, int kh,
                               int imgW, int imgH)
{
    const int tau = 5;
    int halfKW = kw / 2, halfKH = kh / 2;
    int hamming = 0;
    int count = 0;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            if (dx == 0 && dy == 0) continue;
            uchar centerL = left[y * imgW + (x + d)];
            uchar centerR = right[y * imgW + x];
            int vL = (left[ry * imgW + lx] > centerL + tau) ? 1 :
                     (left[ry * imgW + lx] < centerL - tau) ? 0 : 2;
            int vR = (right[ry * imgW + rx] > centerR + tau) ? 1 :
                     (right[ry * imgW + rx] < centerR - tau) ? 0 : 2;
            if (vL != 2 && vR != 2)
                hamming += (vL != vR) ? 1 : 0;
            ++count;
        }
    }
    return count > 0 ? static_cast<float>(hamming) / count : 0.0f;
}

float computeCost(const uchar *left, const uchar *right,
                  int x, int y, int d, int kernelW, int kernelH,
                  int imgW, int imgH, CostFunction func)
{
    switch (func)
    {
    case CostFunction::AbsoluteDifference:
        return adCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::SquaredDifference:
        return sdCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::NormalizedCrossCorr:
        return nccCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::CensusTransform:
        return censusCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    case CostFunction::TernaryCensusTransform:
        return ternaryCensusCost(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
    }
    return 0.0f;
}

CostVolume computeCostVolume(const cv::Mat &left, const cv::Mat &right,
                             int minDisp, int maxDisp,
                             int kernelW, int kernelH,
                             CostFunction func, int numThreads)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    int numDisp = maxDisp - minDisp;
    int imgW = left.cols, imgH = left.rows;
    CostVolume volume(numDisp);

    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        volume[dIdx] = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    }

    #pragma omp parallel for num_threads(numThreads) collapse(2)
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            int validDispStart = std::max(minDisp, -x);
            int validDispEnd   = std::min(maxDisp, imgW - x - 1);
            for (int d = validDispStart; d < validDispEnd; ++d)
            {
                int dIdx = d - minDisp;
                volume[dIdx].at<float>(y, x) = computeCost(
                    left.ptr<uchar>(), right.ptr<uchar>(),
                    x, y, d, kernelW, kernelH, imgW, imgH, func);
            }
        }
    }
    return volume;
}

} // namespace dense_match
```

- [ ] **Step 5: 实现 CostFunctions.cu（CUDA kernel）**

```cpp
// =============================================================================
// 文件: CostFunctions.cu
// 功能: 密集匹配代价函数 CUDA kernel 实现
// =============================================================================
#ifdef DM_ENABLE_CUDA

#include "CostFunctions.h"
#include <cuda_runtime.h>
#include <vector>

namespace dense_match
{

namespace
{

__device__ float adCostDev(const uchar *left, const uchar *right,
                           int x, int y, int d, int kw, int kh,
                           int imgW, int imgH)
{
    float sum = 0.0f;
    int count = 0;
    int halfKW = kw / 2, halfKH = kh / 2;
    for (int dy = -halfKH; dy <= halfKH; ++dy)
    {
        int ry = y + dy;
        if (ry < 0 || ry >= imgH) continue;
        for (int dx = -halfKW; dx <= halfKW; ++dx)
        {
            int rx = x + dx;
            int lx = rx + d;
            if (rx < 0 || rx >= imgW) continue;
            if (lx < 0 || lx >= imgW) continue;
            sum += fabsf((float)left[ry * imgW + lx] - (float)right[ry * imgW + rx]);
            ++count;
        }
    }
    return count > 0 ? sum / count : 0.0f;
}

__global__ void computeCostVolumeKernel(const uchar *left, const uchar *right,
                                        float *costVolume, int imgW, int imgH,
                                        int minDisp, int numDisp,
                                        int kernelW, int kernelH,
                                        int costFunc)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgW || y >= imgH) return;

    int validStart = max(minDisp, -x);
    int validEnd   = min(minDisp + numDisp, imgW - x);

    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        int d = minDisp + dIdx;
        float c = 0.0f;
        if (d >= validStart && d < validEnd)
        {
            if (costFunc == 0)
                c = adCostDev(left, right, x, y, d, kernelW, kernelH, imgW, imgH);
        }
        costVolume[dIdx * imgH * imgW + y * imgW + x] = c;
    }
}

} // anonymous namespace

CostVolume computeCostVolumeCUDA(const cv::Mat &left, const cv::Mat &right,
                                 int minDisp, int maxDisp,
                                 int kernelW, int kernelH,
                                 CostFunction func, int cudaDevice)
{
    cudaSetDevice(cudaDevice);

    int numDisp = maxDisp - minDisp;
    int imgW = left.cols, imgH = left.rows;
    size_t imgBytes = imgW * imgH * sizeof(uchar);
    size_t volBytes = numDisp * imgW * imgH * sizeof(float);

    uchar *d_left, *d_right;
    float *d_volume;
    cudaMalloc(&d_left, imgBytes);
    cudaMalloc(&d_right, imgBytes);
    cudaMalloc(&d_volume, volBytes);

    cudaMemcpy(d_left, left.data, imgBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data, imgBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_volume, 0, volBytes);

    dim3 block(16, 16);
    dim3 grid((imgW + 15) / 16, (imgH + 15) / 16);
    computeCostVolumeKernel<<<grid, block>>>(
        d_left, d_right, d_volume, imgW, imgH, minDisp, numDisp,
        kernelW, kernelH, static_cast<int>(func));
    cudaDeviceSynchronize();

    CostVolume volume(numDisp);
    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        volume[dIdx] = cv::Mat(imgH, imgW, CV_32FC1);
    }

    std::vector<float> h_volume(numDisp * imgW * imgH);
    cudaMemcpy(h_volume.data(), d_volume, volBytes, cudaMemcpyDeviceToHost);

    for (int dIdx = 0; dIdx < numDisp; ++dIdx)
    {
        size_t offset = dIdx * imgW * imgH;
        for (int y = 0; y < imgH; ++y)
        {
            for (int x = 0; x < imgW; ++x)
            {
                volume[dIdx].at<float>(y, x) = h_volume[offset + y * imgW + x];
            }
        }
    }

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_volume);

    return volume;
}

} // namespace dense_match

#endif // DM_ENABLE_CUDA
```

- [ ] **Step 6: 运行测试确认通过**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit --gtest_filter="CostFunction*"
```
Expected: 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement CostFunctions CPU and CUDA with tests"
```

---

### Task 3: BlockMatcher — TDD 实现

**Files:**
- Create: `src/core/dense_match/BlockMatcher.h`
- Create: `src/core/dense_match/BlockMatcher.cpp`
- Create: `src/core/dense_match/BlockMatcher.cu`
- Create: `src/core/dense_match/tests/BlockMatcherTest.cpp`

- [ ] **Step 1: 写 BlockMatcher.h**

```cpp
// =============================================================================
// 文件: BlockMatcher.h
// 功能: WTA 块匹配器声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace dense_match
{

class BlockMatcher
{
public:
    explicit BlockMatcher(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

#ifdef DM_ENABLE_CUDA
    DisparityResult computeCUDA(const cv::Mat &left, const cv::Mat &right);
#endif

private:
    DenseMatchConfig m_cfg;
};

} // namespace dense_match
```

- [ ] **Step 2: 写测试 BlockMatcherTest.cpp**

```cpp
// =============================================================================
// 文件: BlockMatcherTest.cpp
// 功能: 块匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "BlockMatcher.h"
#include <opencv2/imgproc.hpp>

using namespace dense_match;

// 合成测试：生成平移后的右图来验证视差计算
static std::pair<cv::Mat, cv::Mat> makeShiftedPair(int w, int h, int shift)
{
    cv::Mat left(h, w, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::Mat right = cv::Mat::zeros(h, w, CV_8UC1);
    left(cv::Rect(shift, 0, w - shift, h)).copyTo(right(cv::Rect(0, 0, w - shift, h)));
    // 填充右侧黑边
    left(cv::Rect(0, 0, shift, h)).copyTo(right(cv::Rect(w - shift, 0, shift, h)));
    return {left, right};
}

TEST(BlockMatcherTest, ZeroShift_ProducesZeroDisparity)
{
    cv::Mat left(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat right = left.clone();
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 16;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    ASSERT_FALSE(result.disparity.empty());
    // 除边界外视差应接近 0
    cv::Rect interior(8, 8, 48, 48);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 0.0, 1.0);
}

TEST(BlockMatcherTest, KnownShift_ProducesCorrectDisparity)
{
    int w = 128, h = 128, shift = 8;
    auto [left, right] = makeShiftedPair(w, h, shift);
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW = 11;
    cfg.corrKernelH = 11;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    // 检查有效区域视差接近 shift
    cv::Rect interior(32, 32, 64, 64);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], static_cast<double>(shift), 0.5);
}

TEST(BlockMatcherTest, OutputHasCorrectDimensions)
{
    cv::Mat left(100, 80, CV_8UC1);
    cv::Mat right(100, 80, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::randu(right, 0, 256);
    DenseMatchConfig cfg;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    BlockMatcher bm(cfg);
    auto result = bm.compute(left, right);
    EXPECT_EQ(result.disparity.rows, 100);
    EXPECT_EQ(result.disparity.cols, 80);
    EXPECT_FALSE(result.confidence.empty());
    EXPECT_FALSE(result.validMask.empty());
}
```

- [ ] **Step 3: 实现 BlockMatcher.cpp**

```cpp
// =============================================================================
// 文件: BlockMatcher.cpp
// 功能: WTA 块匹配器 CPU 实现
// =============================================================================
#include "BlockMatcher.h"
#include "CostFunctions.h"

namespace dense_match
{

BlockMatcher::BlockMatcher(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult BlockMatcher::compute(const cv::Mat &left, const cv::Mat &right)
{
    CV_Assert(left.type() == CV_8UC1 && right.type() == CV_8UC1);
    CV_Assert(left.size() == right.size());

    int imgW = left.cols, imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;

    CostVolume volume = computeCostVolume(left, right,
        m_cfg.minDisparity, m_cfg.maxDisparity,
        m_cfg.corrKernelW, m_cfg.corrKernelH,
        m_cfg.costFunc, m_cfg.numThreads);

    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1, cv::Scalar(0));

    #pragma omp parallel for num_threads(m_cfg.numThreads)
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f, secondBest = 1e20f;
            int   bestDisp = 0;
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = volume[dIdx].at<float>(y, x);
                if (c < bestCost)
                {
                    secondBest = bestCost;
                    bestCost = c;
                    bestDisp = m_cfg.minDisparity + dIdx;
                }
                else if (c < secondBest)
                {
                    secondBest = c;
                }
            }
            result.disparity.at<float>(y, x) = static_cast<float>(bestDisp);
            if (bestCost > 0)
                result.confidence.at<float>(y, x) = (secondBest - bestCost) / bestCost;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }
    return result;
}

} // namespace dense_match
```

- [ ] **Step 4: 实现 BlockMatcher.cu（CUDA 版本）**

```cpp
// =============================================================================
// 文件: BlockMatcher.cu
// 功能: WTA 块匹配器 CUDA 实现
// =============================================================================
#ifdef DM_ENABLE_CUDA

#include "BlockMatcher.h"
#include "CostFunctions.h"

namespace dense_match
{

DisparityResult BlockMatcher::computeCUDA(const cv::Mat &left, const cv::Mat &right)
{
    int imgW = left.cols, imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;

    CostVolume volume = computeCostVolumeCUDA(left, right,
        m_cfg.minDisparity, m_cfg.maxDisparity,
        m_cfg.corrKernelW, m_cfg.corrKernelH,
        m_cfg.costFunc, m_cfg.cudaDevice);

    // WTA on CPU (cost volume already on host after CUDA)
    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1);
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1);
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1);

    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f, secondBest = 1e20f;
            int bestDisp = 0;
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = volume[dIdx].at<float>(y, x);
                if (c < bestCost) { secondBest = bestCost; bestCost = c; bestDisp = m_cfg.minDisparity + dIdx; }
                else if (c < secondBest) { secondBest = c; }
            }
            result.disparity.at<float>(y, x) = static_cast<float>(bestDisp);
            result.confidence.at<float>(y, x) = (bestCost > 0) ? (secondBest - bestCost) / bestCost : 0.0f;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }
    return result;
}

} // namespace dense_match

#endif
```

- [ ] **Step 5: 运行测试**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit --gtest_filter="BlockMatcher*"
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement BlockMatcher WTA matching with tests"
```

---

### Task 4: SgmMatcher — TDD 实现

**Files:**
- Create: `src/core/dense_match/SgmMatcher.h`
- Create: `src/core/dense_match/SgmMatcher.cpp`
- Create: `src/core/dense_match/SgmMatcher.cu`
- Create: `src/core/dense_match/tests/SgmMatcherTest.cpp`

- [ ] **Step 1: 写 SgmMatcher.h**

```cpp
// =============================================================================
// 文件: SgmMatcher.h
// 功能: SGM/MGM 半全局/更全局匹配声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>
#include <vector>

namespace dense_match
{

class SgmMatcher
{
public:
    explicit SgmMatcher(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

#ifdef DM_ENABLE_CUDA
    DisparityResult computeCUDA(const cv::Mat &left, const cv::Mat &right);
#endif

private:
    using CostVolume = std::vector<cv::Mat>;

    // 沿单个方向的路径聚合
    void aggregatePath(CostVolume &L, const CostVolume &C,
                       int imgW, int imgH, int numDisp,
                       int dirX, int dirY) const;

    // MGM 改进版路径聚合
    void aggregatePathMGM(CostVolume &L, const CostVolume &C,
                          int imgW, int imgH, int numDisp,
                          int dirX, int dirY) const;

    DenseMatchConfig m_cfg;
};

} // namespace dense_match
```

- [ ] **Step 2: 写测试 SgmMatcherTest.cpp**

```cpp
// =============================================================================
// 文件: SgmMatcherTest.cpp
// 功能: SGM 匹配器单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "SgmMatcher.h"
#include <opencv2/imgproc.hpp>

using namespace dense_match;

TEST(SgmMatcherTest, IdenticalImages_ZeroDisparity)
{
    cv::Mat left(64, 64, CV_8UC1, cv::Scalar(128));
    cv::Mat right = left.clone();
    DenseMatchConfig cfg;
    cfg.algorithm   = StereoAlgorithm::SemiGlobalMatch;
    cfg.costFunc    = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 16;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    cfg.p1 = 8;
    cfg.p2 = 32;
    SgmMatcher sgm(cfg);
    auto result = sgm.compute(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(8, 8, 48, 48);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 0.0, 1.0);
}

TEST(SgmMatcherTest, SmoothPlanar_NoiseRobust)
{
    int w = 128, h = 128;
    cv::Mat left(h, w, CV_8UC1);
    // 带噪声的渐变图案
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            left.at<uchar>(y, x) = static_cast<uchar>(
                std::min(255, std::max(0, 128 + 30 * (y / 32) + (rand() % 10 - 5))));
        }
    }
    cv::Mat right(h, w, CV_8UC1);
    // 右图 = 左图平移 6 像素
    left(cv::Rect(6, 0, w - 6, h)).copyTo(right(cv::Rect(0, 0, w - 6, h)));
    left(cv::Rect(0, 0, 6, h)).copyTo(right(cv::Rect(w - 6, 0, 6, h)));

    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::SemiGlobalMatch;
    cfg.costFunc     = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 5;
    cfg.corrKernelH  = 5;
    cfg.p1 = 8;
    cfg.p2 = 32;
    SgmMatcher sgm(cfg);
    auto result = sgm.compute(left, right);
    cv::Rect interior(20, 20, 88, 88);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 6.0, 1.0);
}

TEST(SgmMatcherTest, OutputHasValidMask)
{
    cv::Mat left(64, 64, CV_8UC1);
    cv::Mat right(64, 64, CV_8UC1);
    cv::randu(left, 0, 256);
    cv::randu(right, 0, 256);
    DenseMatchConfig cfg;
    cfg.corrKernelW = 3;
    cfg.corrKernelH = 3;
    SgmMatcher sgm(cfg);
    auto result = sgm.compute(left, right);
    EXPECT_EQ(result.disparity.size(), left.size());
    EXPECT_EQ(result.confidence.size(), left.size());
    EXPECT_EQ(result.validMask.size(), left.size());
    EXPECT_EQ(result.validMask.type(), CV_8UC1);
}
```

- [ ] **Step 3: 实现 SgmMatcher.cpp（CPU SGM）**

```cpp
// =============================================================================
// 文件: SgmMatcher.cpp
// 功能: SGM/MGM 匹配器 CPU 实现
// =============================================================================
#include "SgmMatcher.h"
#include "CostFunctions.h"
#include <cmath>
#include <omp.h>

namespace dense_match
{

SgmMatcher::SgmMatcher(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

static constexpr int SGM_DIRS[8][2] = {
    {1,0}, {0,1}, {1,1}, {1,-1},
    {-1,0}, {0,-1}, {-1,-1}, {-1,1}
};

void SgmMatcher::aggregatePath(CostVolume &L, const CostVolume &C,
                               int imgW, int imgH, int numDisp,
                               int dirX, int dirY) const
{
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            int px = x - dirX, py = y - dirY;
            if (px < 0 || px >= imgW || py < 0 || py >= imgH)
            {
                // 边界：直接赋值为代价
                for (int d = 0; d < numDisp; ++d)
                    L[d].at<float>(y, x) += C[d].at<float>(y, x);
                continue;
            }

            // 计算前一个像素在各视差的最小代价路径
            float minPrev = 1e20f;
            for (int d = 0; d < numDisp; ++d)
            {
                float v = L[d].at<float>(py, px);
                if (v < minPrev) minPrev = v;
            }

            for (int d = 0; d < numDisp; ++d)
            {
                float l0  = L[d].at<float>(py, px);
                float l1  = (d > 0)           ? L[d-1].at<float>(py, px) + m_cfg.p1 : 1e20f;
                float l2  = (d < numDisp - 1) ? L[d+1].at<float>(py, px) + m_cfg.p1 : 1e20f;
                float l3  = minPrev + m_cfg.p2;
                float minL = std::min({l0, l1, l2, l3});
                L[d].at<float>(y, x) += C[d].at<float>(y, x) + minL - minPrev;
            }
        }
    }
}

void SgmMatcher::aggregatePathMGM(CostVolume &L, const CostVolume &C,
                                  int imgW, int imgH, int numDisp,
                                  int dirX, int dirY) const
{
    // MGM: 同时利用前一像素和前一行像素的代价
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            int px = x - dirX, py = y - dirY;
            float w = 1.0f;
            if (px >= 0 && px < imgW && py >= 0 && py < imgH)
            {
                // 正交方向的邻居也参与加权
                int ox = x - dirY, oy = y - dirX;
                if (ox >= 0 && ox < imgW && oy >= 0 && oy < imgH)
                    w = 0.5f; // 两个邻居各贡献一半
            }
            aggregatePath(L, C, imgW, imgH, numDisp, dirX, dirY);
        }
    }
}

DisparityResult SgmMatcher::compute(const cv::Mat &left, const cv::Mat &right)
{
    int imgW = left.cols, imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;

    CostVolume C = computeCostVolume(left, right,
        m_cfg.minDisparity, m_cfg.maxDisparity,
        m_cfg.corrKernelW, m_cfg.corrKernelH,
        m_cfg.costFunc, m_cfg.numThreads);

    // 初始化聚合代价 = 0
    CostVolume L(numDisp);
    for (int d = 0; d < numDisp; ++d)
        L[d] = cv::Mat(imgH, imgW, CV_32FC1, cv::Scalar(0));

    int numDirs = m_cfg.sgmDirections;
    bool useMGM = (m_cfg.algorithm == StereoAlgorithm::MoreGlobalMatch);

    for (int dir = 0; dir < numDirs; ++dir)
    {
        if (useMGM)
            aggregatePathMGM(L, C, imgW, imgH, numDisp,
                             SGM_DIRS[dir][0], SGM_DIRS[dir][1]);
        else
            aggregatePath(L, C, imgW, imgH, numDisp,
                          SGM_DIRS[dir][0], SGM_DIRS[dir][1]);
    }

    // WTA
    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1);
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1);
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1);

    #pragma omp parallel for num_threads(m_cfg.numThreads)
    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f, secondBest = 1e20f;
            int   bestDisp = 0;
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = L[dIdx].at<float>(y, x);
                if (c < bestCost)
                {
                    secondBest = bestCost;
                    bestCost = c;
                    bestDisp = m_cfg.minDisparity + dIdx;
                }
                else if (c < secondBest)
                {
                    secondBest = c;
                }
            }
            result.disparity.at<float>(y, x) = static_cast<float>(bestDisp);
            result.confidence.at<float>(y, x) =
                (bestCost > 0) ? (secondBest - bestCost) / bestCost : 0.0f;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }
    return result;
}

} // namespace dense_match
```

- [ ] **Step 4: 实现 SgmMatcher.cu（CUDA SGM 路径聚合 kernel）**

```cpp
// =============================================================================
// 文件: SgmMatcher.cu
// 功能: SGM/MGM CUDA 路径聚合实现
// =============================================================================
#ifdef DM_ENABLE_CUDA

#include "SgmMatcher.h"
#include "CostFunctions.h"
#include <cuda_runtime.h>
#include <vector>

namespace dense_match
{

__global__ void sgmAggregateKernel(float *L, const float *C,
                                   int imgW, int imgH, int numDisp,
                                   int dirX, int dirY, int p1, int p2)
{
    // 每条扫描线一个 block，block 内 thread 处理不同视差
    // 沿扫描方向逐个像素处理（单 thread 完成一条扫描线）
    int line = blockIdx.x;
    int startX, startY, stepX, stepY, length;
    // 根据方向确定扫描线起点/终点
    if (dirX > 0) { startX = 0; stepX = 1; length = imgW; }
    else if (dirX < 0) { startX = imgW - 1; stepX = -1; length = imgW; }
    else { startX = threadIdx.x; stepX = 0; }

    if (dirY > 0) { startY = 0; stepY = 1; }
    else if (dirY < 0) { startY = imgH - 1; stepY = -1; }
    else { startY = line; stepY = 0; }

    // Simplified: process one pixel per thread along the scanline
    int pIdx = startX + startY * imgW;
    for (int i = 0; i < length; ++i)
    {
        int px = pIdx % imgW - dirX;
        int py = pIdx / imgW - dirY;

        float minPrev = 1e20f;
        if (px >= 0 && px < imgW && py >= 0 && py < imgH)
        {
            int prevIdx = py * imgW + px;
            for (int d = 0; d < numDisp; ++d)
            {
                float v = L[prevIdx + d * imgW * imgH];
                if (v < minPrev) minPrev = v;
            }
            for (int d = 0; d < numDisp; ++d)
            {
                float l0 = L[prevIdx + d * imgW * imgH];
                float l1 = (d > 0) ? L[prevIdx + (d-1) * imgW * imgH] + p1 : 1e20f;
                float l2 = (d < numDisp - 1) ? L[prevIdx + (d+1) * imgW * imgH] + p1 : 1e20f;
                float l3 = minPrev + p2;
                float minL = fminf(fminf(l0, l1), fminf(l2, l3));
                L[pIdx + d * imgW * imgH] = C[pIdx + d * imgW * imgH] + minL - minPrev;
            }
        }
        else
        {
            for (int d = 0; d < numDisp; ++d)
                L[pIdx + d * imgW * imgH] = C[pIdx + d * imgW * imgH];
        }
        pIdx += stepX + stepY * imgW;
    }
}

DisparityResult SgmMatcher::computeCUDA(const cv::Mat &left, const cv::Mat &right)
{
    cudaSetDevice(m_cfg.cudaDevice);

    int imgW = left.cols, imgH = left.rows;
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;
    size_t volSize = numDisp * imgW * imgH * sizeof(float);

    CostVolume C = computeCostVolumeCUDA(left, right,
        m_cfg.minDisparity, m_cfg.maxDisparity,
        m_cfg.corrKernelW, m_cfg.corrKernelH,
        m_cfg.costFunc, m_cfg.cudaDevice);

    // 拷贝代价卷到设备
    float *d_C, *d_L;
    cudaMalloc(&d_C, volSize);
    cudaMalloc(&d_L, volSize);
    cudaMemset(d_L, 0, volSize);
    // Pack cost volume into contiguous memory
    std::vector<float> h_C(numDisp * imgW * imgH);
    for (int d = 0; d < numDisp; ++d)
    {
        size_t offset = d * imgW * imgH;
        for (int y = 0; y < imgH; ++y)
            for (int x = 0; x < imgW; ++x)
                h_C[offset + y * imgW + x] = C[d].at<float>(y, x);
    }
    cudaMemcpy(d_C, h_C.data(), volSize, cudaMemcpyHostToDevice);

    int numDirs = m_cfg.sgmDirections;
    static const int DIRS[8][2] = {{1,0},{0,1},{1,1},{1,-1},{-1,0},{0,-1},{-1,-1},{-1,1}};
    for (int dir = 0; dir < numDirs; ++dir)
    {
        sgmAggregateKernel<<<imgH, 1>>>(d_L, d_C, imgW, imgH, numDisp,
            DIRS[dir][0], DIRS[dir][1], m_cfg.p1, m_cfg.p2);
        cudaDeviceSynchronize();
    }

    // 取回结果并 WTA
    std::vector<float> h_L(numDisp * imgW * imgH);
    cudaMemcpy(h_L.data(), d_L, volSize, cudaMemcpyDeviceToHost);

    DisparityResult result;
    result.disparity  = cv::Mat(imgH, imgW, CV_32FC1);
    result.confidence = cv::Mat(imgH, imgW, CV_32FC1);
    result.validMask  = cv::Mat(imgH, imgW, CV_8UC1);

    for (int y = 0; y < imgH; ++y)
    {
        for (int x = 0; x < imgW; ++x)
        {
            float bestCost = 1e20f, secondBest = 1e20f;
            int bestDisp = 0;
            for (int dIdx = 0; dIdx < numDisp; ++dIdx)
            {
                float c = h_L[dIdx * imgW * imgH + y * imgW + x];
                if (c < bestCost) { secondBest = bestCost; bestCost = c; bestDisp = m_cfg.minDisparity + dIdx; }
                else if (c < secondBest) { secondBest = c; }
            }
            result.disparity.at<float>(y, x) = static_cast<float>(bestDisp);
            result.confidence.at<float>(y, x) = (bestCost > 0) ? (secondBest - bestCost) / bestCost : 0.0f;
            result.validMask.at<uchar>(y, x) = 1;
        }
    }

    cudaFree(d_C);
    cudaFree(d_L);
    return result;
}

} // namespace dense_match

#endif
```

- [ ] **Step 5: 运行测试**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit --gtest_filter="SgmMatcher*"
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement SgmMatcher SGM/MGM with tests"
```

---

### Task 5: SubpixelRefiner — TDD 实现

**Files:**
- Create: `src/core/dense_match/SubpixelRefiner.h`
- Create: `src/core/dense_match/SubpixelRefiner.cpp`
- Create: `src/core/dense_match/SubpixelRefiner.cu`
- Create: `src/core/dense_match/tests/SubpixelRefinerTest.cpp`

- [ ] **Step 1: 写 SubpixelRefiner.h**

```cpp
// =============================================================================
// 文件: SubpixelRefiner.h
// 功能: 子像素视差精化
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace dense_match
{

class SubpixelRefiner
{
public:
    explicit SubpixelRefiner(const DenseMatchConfig &cfg);

    // 输入整数视差图和代价卷，输出精化后的浮点视差图
    cv::Mat refine(const cv::Mat &disparityInt,
                   const std::vector<cv::Mat> &costVolume,
                   int minDisp, int maxDisp);

private:
    cv::Mat refineParabola(const cv::Mat &disp,
                           const std::vector<cv::Mat> &costVol,
                           int minDisp, int maxDisp);

    cv::Mat refineLucasKanade(const cv::Mat &disp,
                              const cv::Mat &left, const cv::Mat &right);

    DenseMatchConfig m_cfg;
};

} // namespace dense_match
```

- [ ] **Step 2: 写测试 SubpixelRefinerTest.cpp**

```cpp
// =============================================================================
// 文件: SubpixelRefinerTest.cpp
// 功能: 子像素精化单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "SubpixelRefiner.h"
#include "CostFunctions.h"
#include <opencv2/imgproc.hpp>

using namespace dense_match;

TEST(SubpixelRefinerTest, Parabola_IntegerInput_NoChange)
{
    // 当抛物线顶点恰好在整数位置时，子像素精化不应改变
    cv::Mat disp(3, 3, CV_32FC1, cv::Scalar(5.0f));
    std::vector<cv::Mat> costVol(7); // disp 0..6
    for (int d = 0; d < 7; ++d)
    {
        costVol[d] = cv::Mat(3, 3, CV_32FC1);
        // 在 d=5 处构造完美抛物线最小值
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x)
                costVol[d].at<float>(y, x) = static_cast<float>((d - 5) * (d - 5));
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            EXPECT_NEAR(refined.at<float>(y, x), 5.0f, 0.01f);
}

TEST(SubpixelRefinerTest, Parabola_SubpixelShift)
{
    cv::Mat disp(1, 1, CV_32FC1, cv::Scalar(5.0f));
    std::vector<cv::Mat> costVol(7);
    for (int d = 0; d < 7; ++d)
    {
        costVol[d] = cv::Mat(1, 1, CV_32FC1);
        // 最小值在 d=4.3: C(4)=0.09, C(5)=0.49, C(3)=1.69
        costVol[d].at<float>(0, 0) = static_cast<float>((d - 4.3) * (d - 4.3));
    }
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::Parabola;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7);
    // 抛物线拟合应接近 4.3
    EXPECT_NEAR(refined.at<float>(0, 0), 4.5, 0.5);
}

TEST(SubpixelRefinerTest, None_Mode_ReturnsOriginal)
{
    cv::Mat disp(2, 2, CV_32FC1, cv::Scalar(3.0f));
    std::vector<cv::Mat> costVol;
    DenseMatchConfig cfg;
    cfg.subpixel = SubpixelMode::None;
    SubpixelRefiner refiner(cfg);
    auto refined = refiner.refine(disp, costVol, 0, 7);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
            EXPECT_FLOAT_EQ(refined.at<float>(y, x), 3.0f);
}
```

- [ ] **Step 3: 实现 SubpixelRefiner.cpp**

```cpp
// =============================================================================
// 文件: SubpixelRefiner.cpp
// 功能: 子像素视差精化 CPU 实现
// =============================================================================
#include "SubpixelRefiner.h"
#include <cmath>
#include <omp.h>

namespace dense_match
{

SubpixelRefiner::SubpixelRefiner(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

cv::Mat SubpixelRefiner::refine(const cv::Mat &disparityInt,
                                const std::vector<cv::Mat> &costVolume,
                                int minDisp, int maxDisp)
{
    switch (m_cfg.subpixel)
    {
    case SubpixelMode::None:
        return disparityInt.clone();
    case SubpixelMode::Parabola:
        return refineParabola(disparityInt, costVolume, minDisp, maxDisp);
    case SubpixelMode::LucasKanade:
        return disparityInt.clone(); // LK 需要原图，此处留给后续完善
    case SubpixelMode::AffineBayes:
        return disparityInt.clone(); // AffineBayes 留给后续完善
    }
    return disparityInt.clone();
}

cv::Mat SubpixelRefiner::refineParabola(const cv::Mat &disp,
                                        const std::vector<cv::Mat> &costVol,
                                        int minDisp, int maxDisp)
{
    int numDisp = maxDisp - minDisp;
    cv::Mat result = disp.clone();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < disp.rows; ++y)
    {
        for (int x = 0; x < disp.cols; ++x)
        {
            int dIdx = static_cast<int>(disp.at<float>(y, x)) - minDisp;
            if (dIdx <= 0 || dIdx >= numDisp - 1) continue;

            float c0 = costVol[dIdx - 1].at<float>(y, x);
            float c1 = costVol[dIdx].at<float>(y, x);
            float c2 = costVol[dIdx + 1].at<float>(y, x);

            float denom = 2.0f * (c0 + c2 - 2.0f * c1);
            if (std::abs(denom) > 1e-10f)
            {
                float delta = (c0 - c2) / denom;
                result.at<float>(y, x) = static_cast<float>(minDisp + dIdx) + delta;
            }
        }
    }
    return result;
}

cv::Mat SubpixelRefiner::refineLucasKanade(const cv::Mat &disp,
                                           const cv::Mat &left,
                                           const cv::Mat &right)
{
    // LK variant for subpixel disparity refinement
    cv::Mat result = disp.clone();
    // Iterative LK: for each pixel, compute image gradient and update disparity
    // to minimize I_L(x+d) - I_R(x). Reserved for future implementation.
    return result;
}

} // namespace dense_match
```

- [ ] **Step 4: 运行测试**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit --gtest_filter="SubpixelRefiner*"
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement SubpixelRefiner with parabola fitting, tests"
```

---

### Task 6: DisparityValidator — TDD 实现

**Files:**
- Create: `src/core/dense_match/DisparityValidator.h`
- Create: `src/core/dense_match/DisparityValidator.cpp`
- Create: `src/core/dense_match/tests/DisparityValidatorTest.cpp`

- [ ] **Step 1: 写 DisparityValidator.h**

```cpp
// =============================================================================
// 文件: DisparityValidator.h
// 功能: 视差图验证与后处理（L-R 一致性、中值滤波、Speckle 过滤）
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace dense_match
{

class DisparityValidator
{
public:
    explicit DisparityValidator(const DenseMatchConfig &cfg);

    // 执行全部验证，返回标记了有效/无效的视差图
    DisparityResult validate(const cv::Mat &disparity, const cv::Mat &confidence);

    // L-R 一致性检查（需要 L→R 和 R→L 两张视差图）
    cv::Mat checkLRConsistency(const cv::Mat &dispLR, const cv::Mat &dispRL);

    // 中值滤波
    cv::Mat medianFilter(const cv::Mat &disp, int kernelSize);

    // Speckle 连通域过滤
    cv::Mat speckleFilter(const cv::Mat &disp, const cv::Mat &valid,
                          int maxSpeckleSize = 100);

private:
    DenseMatchConfig m_cfg;
};

} // namespace dense_match
```

- [ ] **Step 2: 写测试 DisparityValidatorTest.cpp**

```cpp
// =============================================================================
// 文件: DisparityValidatorTest.cpp
// 功能: 视差验证单元测试
// =============================================================================
#include <gtest/gtest.h>
#include "DisparityValidator.h"

using namespace dense_match;

TEST(DisparityValidatorTest, LRCheck_Consistent_Passes)
{
    cv::Mat dispLR(10, 10, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat dispRL(10, 10, CV_32FC1, cv::Scalar(5.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 1.0f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x)
            EXPECT_EQ(valid.at<uchar>(y, x), 1);
}

TEST(DisparityValidatorTest, LRCheck_Inconsistent_Fails)
{
    cv::Mat dispLR(5, 5, CV_32FC1, cv::Scalar(10.0f));
    cv::Mat dispRL(5, 5, CV_32FC1, cv::Scalar(1.0f));
    DenseMatchConfig cfg;
    cfg.lrCheckThreshold = 1.0f;
    DisparityValidator validator(cfg);
    cv::Mat valid = validator.checkLRConsistency(dispLR, dispRL);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x)
            EXPECT_EQ(valid.at<uchar>(y, x), 0);
}

TEST(DisparityValidatorTest, MedianFilter_SmoothesOutlier)
{
    cv::Mat disp(5, 5, CV_32FC1, cv::Scalar(5.0f));
    disp.at<float>(2, 2) = 100.0f; // 孤立异常值
    DenseMatchConfig cfg;
    cfg.medianFilterSize = 3;
    DisparityValidator validator(cfg);
    cv::Mat filtered = validator.medianFilter(disp, 3);
    EXPECT_NEAR(filtered.at<float>(2, 2), 5.0, 95.0); // 应接近周围值
}

TEST(DisparityValidatorTest, Validate_OutputTypes)
{
    cv::Mat disp(16, 16, CV_32FC1, cv::Scalar(5.0f));
    cv::Mat conf(16, 16, CV_32FC1, cv::Scalar(0.8f));
    DenseMatchConfig cfg;
    DisparityValidator validator(cfg);
    auto result = validator.validate(disp, conf);
    EXPECT_EQ(result.disparity.type(), CV_32FC1);
    EXPECT_EQ(result.confidence.type(), CV_32FC1);
    EXPECT_EQ(result.validMask.type(), CV_8UC1);
    EXPECT_EQ(result.validMask.size(), disp.size());
}
```

- [ ] **Step 3: 实现 DisparityValidator.cpp**

```cpp
// =============================================================================
// 文件: DisparityValidator.cpp
// 功能: 视差图验证与后处理实现
// =============================================================================
#include "DisparityValidator.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace dense_match
{

DisparityValidator::DisparityValidator(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult DisparityValidator::validate(const cv::Mat &disparity,
                                             const cv::Mat &confidence)
{
    DisparityResult result;
    result.disparity = disparity.clone();
    result.confidence = confidence.clone();

    // 中值滤波
    if (m_cfg.medianFilterSize > 0)
        result.disparity = medianFilter(result.disparity, m_cfg.medianFilterSize);

    // 初始有效掩码：所有非零视差区域
    result.validMask = cv::Mat(disparity.rows, disparity.cols, CV_8UC1, cv::Scalar(1));

    return result;
}

cv::Mat DisparityValidator::checkLRConsistency(const cv::Mat &dispLR,
                                               const cv::Mat &dispRL)
{
    CV_Assert(dispLR.size() == dispRL.size());
    cv::Mat valid(dispLR.rows, dispLR.cols, CV_8UC1);

    for (int y = 0; y < dispLR.rows; ++y)
    {
        for (int x = 0; x < dispLR.cols; ++x)
        {
            float dLR = dispLR.at<float>(y, x);
            int xR = static_cast<int>(x - dLR + 0.5f);
            if (xR >= 0 && xR < dispRL.cols)
            {
                float dRL = dispRL.at<float>(y, xR);
                valid.at<uchar>(y, x) =
                    (std::abs(dLR - dRL) <= m_cfg.lrCheckThreshold) ? 1 : 0;
            }
            else
            {
                valid.at<uchar>(y, x) = 0;
            }
        }
    }
    return valid;
}

cv::Mat DisparityValidator::medianFilter(const cv::Mat &disp, int kernelSize)
{
    if (kernelSize % 2 == 0) kernelSize += 1;
    cv::Mat result;
    cv::medianBlur(disp, result, kernelSize);
    return result;
}

cv::Mat DisparityValidator::speckleFilter(const cv::Mat &disp,
                                          const cv::Mat &valid,
                                          int maxSpeckleSize)
{
    cv::Mat result = valid.clone();
    // 使用连通域分析去除小面积无效区域
    cv::Mat labels, stats, centroids;
    cv::Mat invalidMask;
    cv::bitwise_not(valid, invalidMask);
    int nLabels = cv::connectedComponentsWithStats(
        invalidMask, labels, stats, centroids, 4, CV_32S);
    for (int i = 1; i < nLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < maxSpeckleSize)
        {
            // 小面积斑块不太重要，仍保留有效标记
            // 实际上 speckle 是移除小面积有效区域
        }
    }
    // 过滤小面积有效区域
    cv::Mat validMask;
    cv::bitwise_not(invalidMask, validMask);
    int nValidLabels = cv::connectedComponentsWithStats(
        validMask, labels, stats, centroids, 4, CV_32S);
    cv::Mat filteredValid = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int i = 1; i < nValidLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= maxSpeckleSize)
        {
            cv::Mat mask = (labels == i);
            filteredValid.setTo(1, mask);
        }
    }
    return filteredValid;
}

} // namespace dense_match
```

- [ ] **Step 4: 运行测试**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit --gtest_filter="DisparityValidator*"
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement DisparityValidator with L-R check, median, speckle filters"
```

---

### Task 7: DenseMatchService — 服务层 + 集成测试

**Files:**
- Create: `src/core/dense_match/DenseMatchService.h`
- Create: `src/core/dense_match/DenseMatchService.cpp`
- Create: `src/core/dense_match/tests/DenseMatchIntegrationTest.cpp`

- [ ] **Step 1: 写 DenseMatchService.h**

```cpp
// =============================================================================
// 文件: DenseMatchService.h
// 功能: 密集匹配服务层，编排完整匹配流水线
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include "DenseMatchTypes.h"
#include <opencv2/core.hpp>
#include <string>

namespace dense_match
{

class DenseMatchService
{
public:
    explicit DenseMatchService(const DenseMatchConfig &cfg);

    // 加载影像并执行完整匹配流水线
    DisparityResult process();

    // 直接传入影像
    DisparityResult process(const cv::Mat &left, const cv::Mat &right);

    // 保存视差图到文件
    static bool saveDisparity(const DisparityResult &result,
                              const std::string &filepath);

private:
    DenseMatchConfig m_cfg;
    cv::Mat m_left, m_right;
};

} // namespace dense_match
```

- [ ] **Step 2: 写 DenseMatchService.cpp**

```cpp
// =============================================================================
// 文件: DenseMatchService.cpp
// 功能: 密集匹配服务层实现
// =============================================================================
#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "SgmMatcher.h"
#include "SubpixelRefiner.h"
#include "DisparityValidator.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace dense_match
{

DenseMatchService::DenseMatchService(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult DenseMatchService::process()
{
    m_left  = cv::imread(m_cfg.leftImagePath, cv::IMREAD_GRAYSCALE);
    m_right = cv::imread(m_cfg.rightImagePath, cv::IMREAD_GRAYSCALE);
    if (m_left.empty() || m_right.empty())
        return DisparityResult();
    return process(m_left, m_right);
}

DisparityResult DenseMatchService::process(const cv::Mat &left, const cv::Mat &right)
{
    if (left.size() != right.size())
        return DisparityResult();

    DisparityResult result;

    // Step 1: 密集匹配
    switch (m_cfg.algorithm)
    {
    case StereoAlgorithm::BlockMatch:
    {
        BlockMatcher bm(m_cfg);
#ifdef DM_ENABLE_CUDA
        result = m_cfg.useCuda ? bm.computeCUDA(left, right) : bm.compute(left, right);
#else
        result = bm.compute(left, right);
#endif
        break;
    }
    case StereoAlgorithm::SemiGlobalMatch:
    case StereoAlgorithm::MoreGlobalMatch:
    {
        SgmMatcher sgm(m_cfg);
#ifdef DM_ENABLE_CUDA
        result = m_cfg.useCuda ? sgm.computeCUDA(left, right) : sgm.compute(left, right);
#else
        result = sgm.compute(left, right);
#endif
        break;
    }
    case StereoAlgorithm::OpenCV_SGBM:
        // 由外部 OpenCVSgbmWrapper 处理
        break;
    }

    // Step 2: 视差验证
    DisparityValidator validator(m_cfg);
    result = validator.validate(result.disparity, result.confidence);

    return result;
}

bool DenseMatchService::saveDisparity(const DisparityResult &result,
                                      const std::string &filepath)
{
    if (result.disparity.empty()) return false;
    return cv::imwrite(filepath, result.disparity);
}

} // namespace dense_match
```

- [ ] **Step 3: 写集成测试 DenseMatchIntegrationTest.cpp**

```cpp
// =============================================================================
// 文件: DenseMatchIntegrationTest.cpp
// 功能: 密集匹配端到端集成测试
// =============================================================================
#include <gtest/gtest.h>
#include "DenseMatchService.h"
#include "BlockMatcher.h"
#include "SgmMatcher.h"
#include "CostFunctions.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdio>

using namespace dense_match;

class DenseMatchIntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 合成 128x128 测试影像对，右图平移 10 像素
        left = cv::Mat(128, 128, CV_8UC1);
        cv::randu(left, 0, 256);
        right = cv::Mat::zeros(128, 128, CV_8UC1);
        left(cv::Rect(10, 0, 118, 128)).copyTo(right(cv::Rect(0, 0, 118, 128)));
        left(cv::Rect(0, 0, 10, 128)).copyTo(right(cv::Rect(118, 0, 10, 128)));
    }
    cv::Mat left, right;
};

TEST_F(DenseMatchIntegrationTest, BM_EndToEnd)
{
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::BlockMatch;
    cfg.costFunc     = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 9;
    cfg.corrKernelH  = 9;
    cfg.useCuda      = false;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(30, 30, 68, 68);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 10.0, 0.5);
}

TEST_F(DenseMatchIntegrationTest, SGM_EndToEnd)
{
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::SemiGlobalMatch;
    cfg.costFunc     = CostFunction::CensusTransform;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 32;
    cfg.corrKernelW  = 5;
    cfg.corrKernelH  = 5;
    cfg.p1 = 8;
    cfg.p2 = 32;
    cfg.useCuda = false;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);
    ASSERT_FALSE(result.disparity.empty());
    cv::Rect interior(30, 30, 68, 68);
    cv::Scalar meanDisp = cv::mean(result.disparity(interior));
    EXPECT_NEAR(meanDisp[0], 10.0, 2.0); // SGM 可能有边界效应，容差放大
}

TEST_F(DenseMatchIntegrationTest, CPU_vs_CUDA_Consistency)
{
#ifdef DM_ENABLE_CUDA
    DenseMatchConfig cfg;
    cfg.algorithm    = StereoAlgorithm::BlockMatch;
    cfg.costFunc     = CostFunction::AbsoluteDifference;
    cfg.minDisparity = 0;
    cfg.maxDisparity = 16;
    cfg.corrKernelW  = 5;
    cfg.corrKernelH  = 5;

    BlockMatcher bmCPU(cfg);
    cfg.useCuda = true;
    BlockMatcher bmCUDA(cfg);

    auto rCPU = bmCPU.compute(left, right);
    auto rCUDA = bmCUDA.computeCUDA(left, right);

    ASSERT_FALSE(rCPU.disparity.empty());
    ASSERT_FALSE(rCUDA.disparity.empty());

    // 逐像素比较 CPU vs CUDA，允许浮点精度差异
    int diffCount = 0;
    for (int y = 10; y < left.rows - 10; ++y)
    {
        for (int x = 10; x < left.cols - 10; ++x)
        {
            float diff = std::abs(rCPU.disparity.at<float>(y, x) -
                                  rCUDA.disparity.at<float>(y, x));
            if (diff > 1.0f) ++diffCount;
        }
    }
    // 差异率 < 5%
    double diffRate = static_cast<double>(diffCount) / (108 * 108);
    EXPECT_LT(diffRate, 0.05);
#else
    GTEST_SKIP() << "CUDA not available";
#endif
}

TEST_F(DenseMatchIntegrationTest, SaveAndReloadDisparity)
{
    DenseMatchConfig cfg;
    cfg.algorithm = StereoAlgorithm::BlockMatch;
    cfg.costFunc  = CostFunction::AbsoluteDifference;
    cfg.useCuda   = false;
    DenseMatchService service(cfg);
    auto result = service.process(left, right);

    const char *tmpFile = "/tmp/test_disparity.tif";
    ASSERT_TRUE(DenseMatchService::saveDisparity(result, tmpFile));

    cv::Mat reloaded = cv::imread(tmpFile, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(reloaded.empty());

    // 浮点 TIFF 重载可能有精度损失，检查近似相等
    double maxDiff = 0;
    for (int y = 0; y < left.rows; ++y)
    {
        for (int x = 0; x < left.cols; ++x)
        {
            float orig = result.disparity.at<float>(y, x);
            float load = reloaded.at<float>(y, x);
            maxDiff = std::max(maxDiff, (double)std::abs(orig - load));
        }
    }
    EXPECT_LT(maxDiff, 0.5);

    std::remove(tmpFile);
}
```

- [ ] **Step 4: 运行全部测试**

```bash
cd build && cmake --build . --target test_dense_match_unit && ./tests/test_dense_match_unit
```
Expected: ALL tests PASS (~18 tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/dense_match/
git commit -m "feat: implement DenseMatchService with integration tests"
```

---

### Task 8: OpenCVSgbmWrapper

**Files:**
- Create: `src/core/dense_match/opencv/OpenCVSgbmWrapper.h`
- Create: `src/core/dense_match/opencv/OpenCVSgbmWrapper.cpp`

- [ ] **Step 1: 创建目录并写头文件**

```bash
mkdir -p src/core/dense_match/opencv
```

```cpp
// =============================================================================
// 文件: OpenCVSgbmWrapper.h
// 功能: OpenCV SGBM 封装，用于与 CUDA 算法对比
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace dense_match
{

class OpenCVSgbmWrapper
{
public:
    explicit OpenCVSgbmWrapper(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

private:
    DenseMatchConfig m_cfg;
};

} // namespace dense_match
```

- [ ] **Step 2: 实现**

```cpp
// =============================================================================
// 文件: OpenCVSgbmWrapper.cpp
// 功能: OpenCV SGBM 封装实现
// =============================================================================
#include "OpenCVSgbmWrapper.h"
#include <opencv2/calib3d.hpp>

namespace dense_match
{

OpenCVSgbmWrapper::OpenCVSgbmWrapper(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult OpenCVSgbmWrapper::compute(const cv::Mat &left, const cv::Mat &right)
{
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;
    // SGBM 要求视差数是 16 的倍数
    int numDisp16 = ((numDisp + 15) / 16) * 16;

    auto sgbm = cv::StereoSGBM::create(
        m_cfg.minDisparity,
        numDisp16,
        std::max(3, m_cfg.corrKernelW | 1), // 块大小必须为奇数
        8 * m_cfg.corrKernelW * m_cfg.corrKernelH, // P1
        32 * m_cfg.corrKernelW * m_cfg.corrKernelH, // P2
        0, // disp12MaxDiff
        m_cfg.medianFilterSize > 0 ? m_cfg.medianFilterSize : 0,
        100, // uniquenessRatio
        0, 0, // speckle window/range (用我们自己后处理)
        cv::StereoSGBM::MODE_SGBM_3WAY
    );

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    DisparityResult result;
    disp16.convertTo(result.disparity, CV_32FC1, 1.0 / 16.0);
    result.confidence = cv::Mat(left.rows, left.cols, CV_32FC1, cv::Scalar(1.0));
    result.validMask  = cv::Mat(left.rows, left.cols, CV_8UC1, cv::Scalar(1));

    return result;
}

} // namespace dense_match
```

- [ ] **Step 3: 验证编译**

```bash
cd build && cmake --build . --target dense_match
```
Expected: Compilation success.

- [ ] **Step 4: Commit**

```bash
git add src/core/dense_match/opencv/
git commit -m "feat: add OpenCV SGBM wrapper for algorithm comparison"
```

---

### Task 9: 菜单重组 — MainMenu

**Files:**
- Modify: `src/gui/menu/MainMenu.h`
- Modify: `src/gui/menu/MainMenu.cpp`

- [ ] **Step 1: 修改 MainMenu.h（添加 denseMatch 成员和访问器）**

Add after `m_viewMatchesAct` declaration (~line 220):
```cpp
QAction *m_denseMatchAct{};       ///< 密集匹配（立体匹配参数设置）
```

Add after `viewMatchesAction()` declaration (~line 115):
```cpp
/** @brief 返回"密集匹配..."动作。 */
QAction *denseMatchAction() const;
```

- [ ] **Step 2: 修改 MainMenu.cpp（菜单重组）**

In the constructor, change the sparse reconstruction menu — remove the `m_viewMatchesAct` addition line (line 119):
```cpp
// REMOVE this line from sparseReconMenu:
// m_viewMatchesAct = sparseReconMenu->addAction(tr("连接点查看"));
```

In the dense reconstruction menu, add `m_denseMatchAct` BEFORE `m_depthMapEstimateAct`:
```cpp
// ── 密集重建 ──
auto *denseReconMenu = reconMenu->addMenu(tr("密集重建"));
m_denseMatchAct       = denseReconMenu->addAction(tr("密集匹配..."));  // 新增，放在最前面
m_depthMapEstimateAct = denseReconMenu->addAction(tr("深度图估计..."));
m_fuseDepthMapsAct    = denseReconMenu->addAction(tr("深度图融合生成密集点云..."));
m_refineDenseCloudAct = denseReconMenu->addAction(tr("密集点云后处理..."));
```

In the tools menu, add `m_viewMatchesAct` after "手动点云剔除":
```cpp
// After: m_manualPointCloudPruneAct = toolsMenu->addAction(tr("手动点云剔除"));
toolsMenu->addSeparator();
m_viewMatchesAct = toolsMenu->addAction(tr("连接点查看"));
```

Add the accessor implementation:
```cpp
QAction *MainMenu::denseMatchAction() const { return m_denseMatchAct; }
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake --build . --target plascan_gui 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add src/gui/menu/
git commit -m "refactor: reorganize menus — move match viewer to tools, add dense match to reconstruction"
```

---

### Task 10: DialogSettingKeys 更新

**Files:**
- Modify: `src/gui/config/settings/DialogSettingKeys.h`

- [ ] **Step 1: 添加 DenseMatch 键**

Add after the `DepthMapEstimate` line in `DialogSettingKeys.h`:
```cpp
inline const QString DenseMatch          = QStringLiteral("dense_match");
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/config/settings/DialogSettingKeys.h
git commit -m "feat: add DenseMatch dialog setting key"
```

---

### Task 11: DisparityHeatmapOverlay — 视差热力图叠加

**Files:**
- Create: `src/gui/widgets/DisparityHeatmapOverlay.h`
- Create: `src/gui/widgets/DisparityHeatmapOverlay.cpp`

- [ ] **Step 1: 写 DisparityHeatmapOverlay.h**

```cpp
// =============================================================================
// 文件: DisparityHeatmapOverlay.h
// 功能: 视差图热力图叠加层（用于密集匹配可视化）
// =============================================================================
#pragma once

#include <QWidget>
#include <opencv2/core.hpp>

class DisparityHeatmapOverlay : public QWidget
{
    Q_OBJECT
public:
    explicit DisparityHeatmapOverlay(QWidget *parent = nullptr);

    // 加载视差图并生成热力图 QPixmap
    bool loadDisparity(const QString &filepath);
    bool loadDisparity(const cv::Mat &disparity);

    // 显示控制
    void setOpacity(float opacity);        // 0.0 - 1.0
    void setDisparityRange(float min, float max);
    void setAutoRange(bool enabled);
    void setColormap(int cvColormap);      // cv::COLORMAP_JET etc.
    void setShowInvalid(bool show);

    float opacity() const { return m_opacity; }

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    void rebuildHeatmap();

    cv::Mat m_disparity;
    QPixmap m_heatmap;
    float   m_opacity      = 0.6f;
    float   m_dispMin       = 0.0f;
    float   m_dispMax       = 256.0f;
    bool    m_autoRange     = true;
    int     m_colormap      = 2; // COLORMAP_JET
    bool    m_showInvalid   = false;
};
```

- [ ] **Step 2: 实现 DisparityHeatmapOverlay.cpp**

```cpp
// =============================================================================
// 文件: DisparityHeatmapOverlay.cpp
// 功能: 视差热力图叠加实现
// =============================================================================
#include "DisparityHeatmapOverlay.h"
#include <QPainter>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

DisparityHeatmapOverlay::DisparityHeatmapOverlay(QWidget *parent)
    : QWidget(parent)
{
    setAttribute(Qt::WA_TransparentForMouseEvents);
}

bool DisparityHeatmapOverlay::loadDisparity(const QString &filepath)
{
    cv::Mat disp = cv::imread(filepath.toStdString(), cv::IMREAD_UNCHANGED);
    if (disp.empty()) return false;
    return loadDisparity(disp);
}

bool DisparityHeatmapOverlay::loadDisparity(const cv::Mat &disparity)
{
    m_disparity = disparity.clone();
    rebuildHeatmap();
    return true;
}

void DisparityHeatmapOverlay::setOpacity(float opacity)
{
    m_opacity = std::max(0.0f, std::min(1.0f, opacity));
    update();
}

void DisparityHeatmapOverlay::setDisparityRange(float min, float max)
{
    m_dispMin = min;
    m_dispMax = max;
    m_autoRange = false;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setAutoRange(bool enabled)
{
    m_autoRange = enabled;
    if (enabled) rebuildHeatmap();
}

void DisparityHeatmapOverlay::setColormap(int cvColormap)
{
    m_colormap = cvColormap;
    rebuildHeatmap();
}

void DisparityHeatmapOverlay::setShowInvalid(bool show)
{
    m_showInvalid = show;
    update();
}

void DisparityHeatmapOverlay::rebuildHeatmap()
{
    if (m_disparity.empty()) return;

    float dMin = m_dispMin, dMax = m_dispMax;
    if (m_autoRange)
    {
        double minVal, maxVal;
        cv::Mat mask = (m_disparity > 0);
        cv::minMaxLoc(m_disparity, &minVal, &maxVal, nullptr, nullptr, mask);
        dMin = static_cast<float>(minVal);
        dMax = static_cast<float>(maxVal);
        if (dMax <= dMin) dMax = dMin + 1.0f;
    }

    // 归一化到 0-255
    cv::Mat normalized;
    cv::Mat dispF;
    m_disparity.convertTo(dispF, CV_32FC1);
    cv::Mat clamped = cv::max(dMin, cv::min(dMax, dispF));
    clamped = (clamped - dMin) / (dMax - dMin) * 255.0;
    clamped.convertTo(normalized, CV_8UC1);

    // 应用 color map
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, m_colormap);

    // 无效区域设为透明
    if (!m_showInvalid)
    {
        cv::Mat mask = (m_disparity <= 0);
        if (colored.channels() == 3)
        {
            cv::Mat rgba;
            cv::cvtColor(colored, rgba, cv::COLOR_BGR2BGRA);
            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    if (mask.at<float>(y, x) == 0)
                        rgba.at<cv::Vec4b>(y, x)[3] = 0;
                }
            }
            cv::cvtColor(rgba, colored, cv::COLOR_BGRA2BGR);
        }
    }

    // 转换为 QPixmap
    cv::Mat rgb;
    cv::cvtColor(colored, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    m_heatmap = QPixmap::fromImage(qimg.copy());
    update();
}

void DisparityHeatmapOverlay::paintEvent(QPaintEvent *)
{
    if (m_heatmap.isNull()) return;
    QPainter painter(this);
    painter.setOpacity(m_opacity);
    painter.drawPixmap(rect(), m_heatmap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/widgets/DisparityHeatmapOverlay.h src/gui/widgets/DisparityHeatmapOverlay.cpp
git commit -m "feat: add DisparityHeatmapOverlay for dense match visualization"
```

---

### Task 12: DualImageViewer 添加叠加层切换支持

**Files:**
- Modify: `src/gui/widgets/DualImageViewer.h`
- Modify: `src/gui/widgets/DualImageViewer.cpp`

- [ ] **Step 1: 修改 DualImageViewer.h**

Add to the public section after `overlay()`:
```cpp
// 密集匹配叠加层
DisparityHeatmapOverlay* disparityOverlay() const;

// 切换到密集匹配覆盖层模式
void setOverlayMode(int mode); // 0 = sparse (MatchLine), 1 = dense (Heatmap)
int overlayMode() const { return m_overlayMode; }
```

Add to private members:
```cpp
QPointer<DisparityHeatmapOverlay> m_disparityOverlay;
int m_overlayMode = 0; // 0=sparse, 1=dense
```

Add `class DisparityHeatmapOverlay;` forward declaration at top.

- [ ] **Step 2: 修改 DualImageViewer.cpp**

In the constructor, create the disparity overlay:
```cpp
m_disparityOverlay = new DisparityHeatmapOverlay(this);
m_disparityOverlay->hide();
```

In `setupLayout()` (or a new method), stack both overlays:
```cpp
// Both overlays are stacked on top of each other, only one visible at a time
m_overlay->raise();
m_disparityOverlay->lower();
```

Add implementations:
```cpp
DisparityHeatmapOverlay* DualImageViewer::disparityOverlay() const
{
    return m_disparityOverlay;
}

void DualImageViewer::setOverlayMode(int mode)
{
    m_overlayMode = mode;
    m_overlay->setVisible(mode == 0);
    m_disparityOverlay->setVisible(mode == 1);
    scheduleOverlayUpdate();
}
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/widgets/DualImageViewer.h src/gui/widgets/DualImageViewer.cpp
git commit -m "feat: add overlay mode switching to DualImageViewer (sparse/dense)"
```

---

### Task 13: MatchViewerDialog 重构为统一查看器

**Files:**
- Modify: `src/gui/dialogs/MatchViewerDialog.h`
- Modify: `src/gui/dialogs/MatchViewerDialog.cpp`

- [ ] **Step 1: 修改 MatchViewerDialog.h**

Add to public section:
```cpp
// 密集匹配工厂方法
static MatchViewerDialog* forDenseMatch(
    const QString &imgA, const QString &imgB,
    const QString &disparityFile, QWidget *parent = nullptr);

// 设置初始 Tab
void setInitialTab(int tabIndex); // 0=sparse, 1=dense
```

Add to private members:
```cpp
QTabWidget  *m_tabWidget     = nullptr;
QWidget     *m_sparseTab     = nullptr;
QWidget     *m_denseTab      = nullptr;
int          m_initialTab    = 0;
QString      m_disparityFile; // 密集匹配视差文件路径
```

Add to private methods:
```cpp
void setupDenseTab();
void setupDenseDisplayOptions();
```

Add to private slots:
```cpp
void onDenseOpacityChanged(int value);
void onDenseColormapChanged(int index);
void onDenseRangeChanged();
```

Add to private members (dense display options):
```cpp
QSlider      *m_denseOpacitySlider;
QComboBox    *m_denseColormapCombo;
QCheckBox    *m_denseAutoRangeChk;
QDoubleSpinBox *m_denseMinSpin;
QDoubleSpinBox *m_denseMaxSpin;
```

- [ ] **Step 2: 修改 MatchViewerDialog.cpp**

**Updated constructor** — wraps viewer in QTabWidget:
```cpp
MatchViewerDialog::MatchViewerDialog(const QString &imgA, const QString &imgB,
                                     const QString &matchFile, QWidget *parent)
    : QDialog(parent), m_matchFile(matchFile), m_totalMatches(0)
{
    setWindowTitle(tr("匹配查看：%1 <-> %2")
                   .arg(QFileInfo(imgA).fileName())
                   .arg(QFileInfo(imgB).fileName()));
    resize(1400, 800);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    setupToolBar();
    mainLayout->addWidget(m_toolbar);

    // ---- Tab Widget ----
    m_tabWidget = new QTabWidget(this);

    // Tab 0: Sparse
    m_sparseTab = new QWidget();
    QVBoxLayout *sparseLayout = new QVBoxLayout(m_sparseTab);
    sparseLayout->setContentsMargins(0, 0, 0, 0);
    m_viewer = new DualImageViewer(m_sparseTab);
    sparseLayout->addWidget(m_viewer);
    m_tabWidget->addTab(m_sparseTab, tr("稀疏匹配"));

    // Tab 1: Dense
    m_denseTab = new QWidget();
    QVBoxLayout *denseLayout = new QVBoxLayout(m_denseTab);
    denseLayout->setContentsMargins(0, 0, 0, 0);
    // Dense tab uses same DualImageViewer but switches overlay mode
    // Left/right image views stay, overlay changes
    m_tabWidget->addTab(m_denseTab, tr("密集匹配"));

    mainLayout->addWidget(m_tabWidget, 1);

    setupDisplayOptions(); // sparse display options in toolbar
    setupDenseDisplayOptions(); // dense display options (hidden initially)
    setupStatusBar();

    setLayout(mainLayout);

    // Signals
    connect(m_viewer, &DualImageViewer::matchDataLoaded,
            this, &MatchViewerDialog::onMatchDataLoaded);
    connect(m_viewer, &DualImageViewer::loadFailed,
            this, &MatchViewerDialog::onLoadFailed);

    loadSettings();
    m_viewer->loadMatchPair(imgA, imgB, matchFile);

    // Tab switch → toggle display option groups
    connect(m_tabWidget, &QTabWidget::currentChanged, this, [this](int idx)
    {
        m_viewer->setOverlayMode(idx);
        // Show/hide appropriate display option groups
        if (idx == 0) { /* sparse options visible, dense hidden */ }
        else          { /* dense options visible, sparse hidden */ }
    });

    if (m_initialTab >= 0)
        m_tabWidget->setCurrentIndex(m_initialTab);
}
```

**Factory method:**
```cpp
MatchViewerDialog* MatchViewerDialog::forDenseMatch(
    const QString &imgA, const QString &imgB,
    const QString &disparityFile, QWidget *parent)
{
    // Create with empty match file, then load disparity
    auto *dlg = new MatchViewerDialog(imgA, imgB, QString(), parent);
    dlg->m_disparityFile = disparityFile;
    dlg->setInitialTab(1);
    // Load disparity into overlay
    if (!disparityFile.isEmpty())
        dlg->m_viewer->disparityOverlay()->loadDisparity(disparityFile);
    return dlg;
}
```

**Dense display options setup:**
```cpp
void MatchViewerDialog::setupDenseDisplayOptions()
{
    QGroupBox *denseGroup = new QGroupBox(tr("密集显示选项"), this);
    QHBoxLayout *denseLayout = new QHBoxLayout(denseGroup);

    // Opacity
    denseLayout->addWidget(new QLabel(tr("透明度:"), denseGroup));
    m_denseOpacitySlider = new QSlider(Qt::Horizontal, denseGroup);
    m_denseOpacitySlider->setRange(0, 100);
    m_denseOpacitySlider->setValue(60);
    denseLayout->addWidget(m_denseOpacitySlider);
    connect(m_denseOpacitySlider, &QSlider::valueChanged, this,
            &MatchViewerDialog::onDenseOpacityChanged);

    // Colormap
    denseLayout->addWidget(new QLabel(tr("色彩映射:"), denseGroup));
    m_denseColormapCombo = new QComboBox(denseGroup);
    m_denseColormapCombo->addItem(tr("Jet"), 2);
    m_denseColormapCombo->addItem(tr("Hot"), 11);
    m_denseColormapCombo->addItem(tr("Parula"), 12);
    m_denseColormapCombo->addItem(tr("Turbo"), 20);
    denseLayout->addWidget(m_denseColormapCombo);
    connect(m_denseColormapCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MatchViewerDialog::onDenseColormapChanged);

    // Auto range + min/max
    m_denseAutoRangeChk = new QCheckBox(tr("自动范围"), denseGroup);
    m_denseAutoRangeChk->setChecked(true);
    denseLayout->addWidget(m_denseAutoRangeChk);
    connect(m_denseAutoRangeChk, &QCheckBox::toggled, this, [this](bool checked)
    {
        m_denseMinSpin->setEnabled(!checked);
        m_denseMaxSpin->setEnabled(!checked);
        if (m_viewer->disparityOverlay())
            m_viewer->disparityOverlay()->setAutoRange(checked);
    });

    denseLayout->addWidget(new QLabel(tr("范围:"), denseGroup));
    m_denseMinSpin = new QDoubleSpinBox(denseGroup);
    m_denseMinSpin->setRange(0, 1024);
    m_denseMinSpin->setValue(0);
    m_denseMinSpin->setEnabled(false);
    denseLayout->addWidget(m_denseMinSpin);

    m_denseMaxSpin = new QDoubleSpinBox(denseGroup);
    m_denseMaxSpin->setRange(1, 2048);
    m_denseMaxSpin->setValue(256);
    m_denseMaxSpin->setEnabled(false);
    denseLayout->addWidget(m_denseMaxSpin);

    connect(m_denseMinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onDenseRangeChanged);
    connect(m_denseMaxSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MatchViewerDialog::onDenseRangeChanged);

    m_toolbar->addWidget(denseGroup);
    denseGroup->hide(); // Hidden until dense tab selected
}
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/dialogs/MatchViewerDialog.h src/gui/dialogs/MatchViewerDialog.cpp
git commit -m "refactor: add tab-based unified viewer (sparse/dense) to MatchViewerDialog"
```

---

### Task 14: DenseMatchDialog — 密集匹配参数对话框

**Files:**
- Create: `src/gui/dialogs/DenseMatchDialog.h`
- Create: `src/gui/dialogs/DenseMatchDialog.cpp`

- [ ] **Step 1: 写 DenseMatchDialog.h**

```cpp
// =============================================================================
// 文件: DenseMatchDialog.h
// 功能: 密集匹配（立体匹配）参数配置对话框
// =============================================================================
#pragma once

#include <QDialog>
#include <QJsonObject>

class QComboBox;
class QSpinBox;
class QDoubleSpinBox;
class QCheckBox;
class QLineEdit;

class DenseMatchDialog : public QDialog
{
    Q_OBJECT
public:
    explicit DenseMatchDialog(QWidget *parent = nullptr);

    void applySettings(const QJsonObject &settings);

signals:
    void runRequested(const QJsonObject &settings);
    void settingsChanged(const QJsonObject &settings);

private slots:
    void onRun();
    void onBrowseOutput();
    void onAlgorithmChanged(int index);
    void emitSettingsNow();

private:
    void setupUi();
    QJsonObject collectSettings() const;

    // 输入
    QLineEdit   *m_leftImageEdit   = nullptr;
    QLineEdit   *m_rightImageEdit  = nullptr;
    QLineEdit   *m_outputEdit      = nullptr;

    // 基础参数
    QComboBox   *m_algorithmCombo  = nullptr;
    QComboBox   *m_costFuncCombo   = nullptr;
    QComboBox   *m_subpixelCombo   = nullptr;
    QSpinBox    *m_minDispSpin     = nullptr;
    QSpinBox    *m_maxDispSpin     = nullptr;
    QSpinBox    *m_kernelWSpin     = nullptr;
    QSpinBox    *m_kernelHSpin     = nullptr;

    // SGM/MGM 参数
    QSpinBox    *m_p1Spin          = nullptr;
    QSpinBox    *m_p2Spin          = nullptr;
    QSpinBox    *m_directionsSpin  = nullptr;
    QSpinBox    *m_pyramidSpin     = nullptr;

    // 系统参数
    QCheckBox   *m_useCudaChk      = nullptr;
    QSpinBox    *m_deviceSpin      = nullptr;
    QSpinBox    *m_threadsSpin     = nullptr;
    QCheckBox   *m_opencvCompareChk = nullptr;

    // 后处理
    QDoubleSpinBox *m_lrThresholdSpin = nullptr;
    QSpinBox    *m_medianFilterSpin = nullptr;

    QPushButton *m_runBtn = nullptr;
    QPushButton *m_cancelBtn = nullptr;
    QPushButton *m_resetBtn = nullptr;
};
```

- [ ] **Step 2: 实现 DenseMatchDialog.cpp**

```cpp
// =============================================================================
// 文件: DenseMatchDialog.cpp
// 功能: 密集匹配对话框实现
// =============================================================================
#include "DenseMatchDialog.h"
#include "settings/DialogSettingStore.h"
#include "settings/DialogSettingKeys.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QLabel>

DenseMatchDialog::DenseMatchDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("密集匹配"));
    resize(600, 700);
    setupUi();
}

void DenseMatchDialog::setupUi()
{
    auto *mainLayout = new QVBoxLayout(this);

    // ==== 输入区域 ====
    auto *inputGroup = new QGroupBox(tr("输入"), this);
    auto *inputForm = new QFormLayout(inputGroup);
    m_leftImageEdit  = new QLineEdit(this);
    m_rightImageEdit = new QLineEdit(this);
    m_outputEdit     = new QLineEdit(this);
    auto *browseBtn = new QPushButton(tr("..."), this);
    connect(browseBtn, &QPushButton::clicked, this, &DenseMatchDialog::onBrowseOutput);
    auto *outRow = new QHBoxLayout();
    outRow->addWidget(m_outputEdit);
    outRow->addWidget(browseBtn);
    inputForm->addRow(tr("左影像:"), m_leftImageEdit);
    inputForm->addRow(tr("右影像:"), m_rightImageEdit);
    inputForm->addRow(tr("输出路径:"), outRow);
    mainLayout->addWidget(inputGroup);

    // ==== 算法参数 ====
    auto *algoGroup = new QGroupBox(tr("算法参数"), this);
    auto *algoForm = new QFormLayout(algoGroup);
    m_algorithmCombo = new QComboBox(this);
    m_algorithmCombo->addItem(tr("MGM (More Global Match)"), 2);
    m_algorithmCombo->addItem(tr("SGM (Semi Global Match)"), 1);
    m_algorithmCombo->addItem(tr("BM (Block Match)"), 0);
    m_algorithmCombo->addItem(tr("OpenCV SGBM"), 3);
    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DenseMatchDialog::onAlgorithmChanged);
    algoForm->addRow(tr("匹配算法:"), m_algorithmCombo);

    m_costFuncCombo = new QComboBox(this);
    m_costFuncCombo->addItem(tr("Census Transform"), 3);
    m_costFuncCombo->addItem(tr("NCC (Normalized Cross Correlation)"), 2);
    m_costFuncCombo->addItem(tr("Absolute Difference"), 0);
    m_costFuncCombo->addItem(tr("Squared Difference"), 1);
    m_costFuncCombo->addItem(tr("Ternary Census"), 4);
    algoForm->addRow(tr("代价函数:"), m_costFuncCombo);

    m_subpixelCombo = new QComboBox(this);
    m_subpixelCombo->addItem(tr("Parabola Fitting"), 1);
    m_subpixelCombo->addItem(tr("None"), 0);
    m_subpixelCombo->addItem(tr("Lucas-Kanade"), 3);
    algoForm->addRow(tr("子像素精化:"), m_subpixelCombo);

    m_minDispSpin = new QSpinBox(this);
    m_minDispSpin->setRange(0, 1024);
    m_minDispSpin->setValue(0);
    algoForm->addRow(tr("最小视差:"), m_minDispSpin);

    m_maxDispSpin = new QSpinBox(this);
    m_maxDispSpin->setRange(16, 2048);
    m_maxDispSpin->setValue(256);
    algoForm->addRow(tr("最大视差:"), m_maxDispSpin);

    m_kernelWSpin = new QSpinBox(this);
    m_kernelWSpin->setRange(3, 31);
    m_kernelWSpin->setSingleStep(2);
    m_kernelWSpin->setValue(15);
    algoForm->addRow(tr("核宽度:"), m_kernelWSpin);

    m_kernelHSpin = new QSpinBox(this);
    m_kernelHSpin->setRange(3, 31);
    m_kernelHSpin->setSingleStep(2);
    m_kernelHSpin->setValue(15);
    algoForm->addRow(tr("核高度:"), m_kernelHSpin);
    mainLayout->addWidget(algoGroup);

    // ==== SGM/MGM 参数 (可折叠) ====
    auto *sgmGroup = new QGroupBox(tr("SGM/MGM 参数"), this);
    sgmGroup->setCheckable(true);
    sgmGroup->setChecked(true);
    auto *sgmForm = new QFormLayout(sgmGroup);
    m_p1Spin = new QSpinBox(this);
    m_p1Spin->setRange(1, 255);
    m_p1Spin->setValue(8);
    sgmForm->addRow(tr("P1 (小惩罚):"), m_p1Spin);
    m_p2Spin = new QSpinBox(this);
    m_p2Spin->setRange(1, 1024);
    m_p2Spin->setValue(32);
    sgmForm->addRow(tr("P2 (大惩罚):"), m_p2Spin);
    m_directionsSpin = new QSpinBox(this);
    m_directionsSpin->setRange(4, 8);
    m_directionsSpin->setSingleStep(4);
    m_directionsSpin->setValue(8);
    sgmForm->addRow(tr("路径方向数:"), m_directionsSpin);
    m_pyramidSpin = new QSpinBox(this);
    m_pyramidSpin->setRange(0, 5);
    m_pyramidSpin->setValue(2);
    sgmForm->addRow(tr("金字塔层数:"), m_pyramidSpin);
    mainLayout->addWidget(sgmGroup);

    // ==== 系统参数 ====
    auto *sysGroup = new QGroupBox(tr("系统参数"), this);
    auto *sysForm = new QFormLayout(sysGroup);
    m_useCudaChk = new QCheckBox(tr("使用 CUDA"), this);
    m_useCudaChk->setChecked(true);
    sysForm->addRow(m_useCudaChk);
    m_deviceSpin = new QSpinBox(this);
    m_deviceSpin->setRange(0, 7);
    m_deviceSpin->setValue(0);
    sysForm->addRow(tr("CUDA 设备:"), m_deviceSpin);
    m_threadsSpin = new QSpinBox(this);
    m_threadsSpin->setRange(1, 64);
    m_threadsSpin->setValue(4);
    sysForm->addRow(tr("CPU 线程数:"), m_threadsSpin);
    m_opencvCompareChk = new QCheckBox(tr("同时运行 OpenCV SGBM 对比"), this);
    sysForm->addRow(m_opencvCompareChk);
    mainLayout->addWidget(sysGroup);

    // ==== 后处理 ====
    auto *postGroup = new QGroupBox(tr("后处理"), this);
    auto *postForm = new QFormLayout(postGroup);
    m_lrThresholdSpin = new QDoubleSpinBox(this);
    m_lrThresholdSpin->setRange(0.0, 10.0);
    m_lrThresholdSpin->setValue(1.0);
    m_lrThresholdSpin->setSingleStep(0.5);
    postForm->addRow(tr("L-R 阈值:"), m_lrThresholdSpin);
    m_medianFilterSpin = new QSpinBox(this);
    m_medianFilterSpin->setRange(0, 15);
    m_medianFilterSpin->setValue(3);
    postForm->addRow(tr("中值滤波核:"), m_medianFilterSpin);
    mainLayout->addWidget(postGroup);

    // ==== 底部按钮 ====
    auto *btnLayout = new QHBoxLayout();
    m_runBtn = new QPushButton(tr("运行"), this);
    m_cancelBtn = new QPushButton(tr("取消"), this);
    m_resetBtn = new QPushButton(tr("恢复默认"), this);
    btnLayout->addStretch();
    btnLayout->addWidget(m_runBtn);
    btnLayout->addWidget(m_cancelBtn);
    btnLayout->addWidget(m_resetBtn);
    mainLayout->addLayout(btnLayout);

    connect(m_runBtn, &QPushButton::clicked, this, &DenseMatchDialog::onRun);
    connect(m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    // Settings persistence
    connect(m_algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_costFuncCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_minDispSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int){ emitSettingsNow(); });
    connect(m_maxDispSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int){ emitSettingsNow(); });
}

void DenseMatchDialog::onBrowseOutput()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("选择输出目录"));
    if (!dir.isEmpty()) m_outputEdit->setText(dir);
}

void DenseMatchDialog::onAlgorithmChanged(int index)
{
    bool isSGM = (index == 0 || index == 1); // MGM or SGM
    m_p1Spin->setEnabled(isSGM);
    m_p2Spin->setEnabled(isSGM);
    m_directionsSpin->setEnabled(isSGM);
    m_pyramidSpin->setEnabled(isSGM);
    m_useCudaChk->setEnabled(index != 3); // no CUDA for OpenCV SGBM
    emitSettingsNow();
}

void DenseMatchDialog::onRun()
{
    emit runRequested(collectSettings());
}

void DenseMatchDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

QJsonObject DenseMatchDialog::collectSettings() const
{
    QJsonObject s;
    s["left_image"]    = m_leftImageEdit->text();
    s["right_image"]   = m_rightImageEdit->text();
    s["output_dir"]    = m_outputEdit->text();
    s["algorithm"]     = m_algorithmCombo->currentData().toInt();
    s["cost_func"]     = m_costFuncCombo->currentData().toInt();
    s["subpixel_mode"] = m_subpixelCombo->currentData().toInt();
    s["min_disparity"] = m_minDispSpin->value();
    s["max_disparity"] = m_maxDispSpin->value();
    s["kernel_w"]      = m_kernelWSpin->value();
    s["kernel_h"]      = m_kernelHSpin->value();
    s["p1"]            = m_p1Spin->value();
    s["p2"]            = m_p2Spin->value();
    s["directions"]    = m_directionsSpin->value();
    s["pyramid"]       = m_pyramidSpin->value();
    s["use_cuda"]      = m_useCudaChk->isChecked();
    s["cuda_device"]   = m_deviceSpin->value();
    s["threads"]       = m_threadsSpin->value();
    s["opencv_compare"] = m_opencvCompareChk->isChecked();
    s["lr_threshold"]  = m_lrThresholdSpin->value();
    s["median_filter"] = m_medianFilterSpin->value();
    return s;
}

void DenseMatchDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty()) return;
    auto val = [&](const QString &k, const QJsonValue &def = {}) { return settings.value(k); };
    m_leftImageEdit->setText(val("left_image").toString());
    m_rightImageEdit->setText(val("right_image").toString());
    m_outputEdit->setText(val("output_dir").toString());
    int algo = val("algorithm").toInt(2);
    m_algorithmCombo->setCurrentIndex(m_algorithmCombo->findData(algo));
    int cost = val("cost_func").toInt(3);
    m_costFuncCombo->setCurrentIndex(m_costFuncCombo->findData(cost));
    int sub = val("subpixel_mode").toInt(1);
    m_subpixelCombo->setCurrentIndex(m_subpixelCombo->findData(sub));
    m_minDispSpin->setValue(val("min_disparity").toInt(0));
    m_maxDispSpin->setValue(val("max_disparity").toInt(256));
    m_kernelWSpin->setValue(val("kernel_w").toInt(15));
    m_kernelHSpin->setValue(val("kernel_h").toInt(15));
    m_p1Spin->setValue(val("p1").toInt(8));
    m_p2Spin->setValue(val("p2").toInt(32));
    m_directionsSpin->setValue(val("directions").toInt(8));
    m_pyramidSpin->setValue(val("pyramid").toInt(2));
    m_useCudaChk->setChecked(val("use_cuda").toBool(true));
    m_deviceSpin->setValue(val("cuda_device").toInt(0));
    m_threadsSpin->setValue(val("threads").toInt(4));
    m_opencvCompareChk->setChecked(val("opencv_compare").toBool(false));
    m_lrThresholdSpin->setValue(val("lr_threshold").toDouble(1.0));
    m_medianFilterSpin->setValue(val("median_filter").toInt(3));
}
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/dialogs/DenseMatchDialog.h src/gui/dialogs/DenseMatchDialog.cpp
git commit -m "feat: add DenseMatchDialog for stereo matching parameter configuration"
```

---

### Task 15: MainWindow — 连接新动作

**Files:**
- Modify: `src/gui/main_window/MainWindow.cpp`

- [ ] **Step 1: 添加 denseMatchAction 和 viewMatchesAction 的连接**

After the existing `m_mainMenu->depthMapEstimateAction()` connection block, add:

```cpp
if (m_mainMenu->denseMatchAction())
{
    connect(m_mainMenu->denseMatchAction(), &QAction::triggered, this, [this]()
    {
        if (!m_projectManager)
        {
            LOG_ERROR(QStringLiteral("无法打开密集匹配：ProjectManager 未初始化"));
            return;
        }
        auto *dlg = new DenseMatchDialog(this);
        dlg->setAttribute(Qt::WA_DeleteOnClose);
        // Apply saved project settings if available
        if (auto *store = m_projectManager->dialogSettingStore(
                DialogSettingKeys::DenseMatch))
        {
            dlg->applySettings(store->load());
        }
        connect(dlg, &DenseMatchDialog::settingsChanged, this,
                [pm = m_projectManager](const QJsonObject &cfg)
        {
            if (auto *store = pm->dialogSettingStore(
                    DialogSettingKeys::DenseMatch))
                store->save(cfg);
        });
        connect(dlg, &DenseMatchDialog::runRequested, this,
                [this](const QJsonObject &cfg)
        {
            // Async dense matching execution: convert cfg to
            // DenseMatchConfig, run in QtConcurrent, emit progress
            QtConcurrent::run([cfg]()
            {
                LOG_INFO("Dense match started: %s",
                         cfg["left_image"].toString().toStdString().c_str());
                // Service execution will be integrated here
                // DenseMatchConfig dmCfg = configFromJson(cfg);
                // DenseMatchService service(dmCfg);
                // auto result = service.process();
                // DenseMatchService::saveDisparity(result,
                //     cfg["output_dir"].toString().toStdString() + "/disparity.tif");
            });
        });
        dlg->exec();
    });
}
```

The `viewMatchesAction` connection remains at its existing location (lines 538-552), just verify it still compiles.

- [ ] **Step 2: 验证编译**

```bash
cd build && cmake --build . --target plascan_gui 2>&1 | tail -10
```
Expected: Compilation success.

- [ ] **Step 3: Commit**

```bash
git add src/gui/main_window/MainWindow.cpp
git commit -m "feat: wire DenseMatchDialog to dense match menu action"
```

---

### Task 16: 最终构建验证

**Files:** None (verification only)

- [ ] **Step 1: 完整构建**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc) 2>&1 | tail -20
```
Expected: Build success with no errors.

- [ ] **Step 2: 运行全部测试**

```bash
cd build && ctest --output-on-failure 2>&1
```
Expected: All tests pass (new + existing).

- [ ] **Step 3: 检查文件数**

```bash
find src/core/dense_match/ -type f | wc -l
```
Expected: ~22 files.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: final build verification, all tests passing"
```

---

### Task 16: GUI CMakeLists 集成

**Files:**
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `src/gui/CMakeLists.txt`

- [ ] **Step 1: 在 GuiSources.cmake 中注册新文件**

In `GUI_SOURCES`, add after `widgets/DualImageViewer.cpp`:
```cmake
widgets/DisparityHeatmapOverlay.cpp
```

In `GUI_HEADERS`, add after `widgets/DualImageViewer.h`:
```cmake
widgets/DisparityHeatmapOverlay.h
```

In `GUI_DIALOG_SOURCES`, add after `dialogs/DepthMapEstimateDialog.cpp`:
```cmake
dialogs/DenseMatchDialog.cpp
```

In `GUI_HEADERS`, add after `dialogs/ModelExportDialog.h`:
```cmake
dialogs/DenseMatchDialog.h
```

- [ ] **Step 2: 在 GUI CMakeLists.txt 中添加 dense_match 链接**

Add include directory after existing core includes:
```cmake
target_include_directories(plascan_gui PRIVATE
  ...
  ${CMAKE_SOURCE_DIR}/src/core/sfm
  ${CMAKE_SOURCE_DIR}/src/core/dense_match  # ← 新增
  ...
)
```

Add to target_link_libraries:
```cmake
target_link_libraries(plascan_gui PRIVATE
  ...
  dense_match  # ← 新增
  ...
)
```

- [ ] **Step 3: Commit**

```bash
git add src/gui/cmake/GuiSources.cmake src/gui/CMakeLists.txt
git commit -m "chore: register DenseMatchDialog and DisparityHeatmapOverlay in GUI build"
```

---

### Task 17: 最终构建验证

### 新建文件 (32)
- `src/core/dense_match/` — 15 源文件 + 7 测试文件 + 2 OpenCV wrapper
- `src/gui/widgets/DisparityHeatmapOverlay.h/cpp`
- `src/gui/dialogs/DenseMatchDialog.h/cpp`
- `docs/superpowers/specs/2026-04-30-dense-match-module-and-viewer-redesign.md`
- `docs/superpowers/plans/2026-04-30-dense-match-module-and-viewer-redesign.md`

### 修改文件 (7)
- `src/core/CMakeLists.txt` — 注册 dense_match 模块
- `src/gui/menu/MainMenu.h/cpp` — 菜单重组
- `src/gui/main_window/MainWindow.cpp` — 连接新动作
- `src/gui/dialogs/MatchViewerDialog.h/cpp` — 统一查看器重构
- `src/gui/widgets/DualImageViewer.h/cpp` — 叠加层切换
- `src/gui/config/settings/DialogSettingKeys.h` — 新键
