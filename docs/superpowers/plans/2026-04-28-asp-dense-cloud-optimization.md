# ASP Dense Cloud Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark-driven self-developed dense-cloud pipeline that iterates toward strict ASP alignment on the existing stereo test pair before any GUI dialog update.

**Architecture:** First extract ASP-vs-PlaScan point-cloud comparison into a reusable metrics module and benchmark executable. Then parameterize the current `StereoDenseCloudPipeline` filters so coverage loss can be diagnosed one variable at a time. If parameter tuning is insufficient, replace the over-filtered depth-domain path with an ASP-style rectified/disparity-domain path while preserving CUDA and parallelism for speed. GUI changes are intentionally deferred until the benchmark passes.

**Tech Stack:** C++17, Qt6 Core, OpenCV, GDAL, CUDA-enabled `mvs` static library, CMake, existing `build/` tree.

---

## Scope and Execution Rules

- This repository directory is not a git repository. Replace each commit step with the listed checkpoint command and a short written summary of changed files.
- Do not modify `SimplePointCloudDialog` or any GUI workflow until Task 8 explicitly says benchmark PASS has been achieved.
- Use the existing test data under `data/stereo_test_20260426/`.
- The benchmark PASS threshold is strict:
  - `coverageRatio >= 0.80`
  - `nnMedian <= 0.001`
  - `nnP95 <= 0.01`
- Every task that changes algorithm behavior must run the benchmark and record the before/after metrics.

## File Structure

### New files

- `src/core/mvs/AspPointCloudMetrics.h`  
  Defines metric structs, threshold config, and comparison APIs.

- `src/core/mvs/AspPointCloudMetrics.cpp`  
  Reads ASP/PlaScan TIF point clouds, computes valid counts, coverage ratio, bbox containment, nearest-neighbor distance stats, mean offset, and PASS/FAIL.

- `src/core/terrain/tests/stereo_pipeline_benchmark.cpp`  
  Runs `StereoDenseCloudPipeline` on the fixed test pair, writes `PC.tif`/`PC.ply`, calls `AspPointCloudMetrics`, prints a concise report, writes JSON metrics, and exits nonzero when strict thresholds fail.

### Modified files

- `src/core/mvs/CMakeLists.txt`  
  Add `AspPointCloudMetrics.cpp` and `AspPointCloudMetrics.h` to the `mvs` library.

- `src/core/terrain/CMakeLists.txt`  
  Add the `stereo_pipeline_benchmark` executable when `BUILD_TESTS` is enabled.

- `src/core/mvs/StereoDenseCloudPipeline.h`  
  Add explicit filter/depth-range configuration fields and diagnostic counters to `StereoPipelineConfig` / `StereoPipelineResult`.

- `src/core/mvs/StereoDenseCloudPipeline.cpp`  
  Replace hard-coded filter thresholds with config fields, emit per-stage rejection counts, optionally keep intermediate masks, and later add ASP-style rectified/disparity path if tuning cannot pass.

- `src/core/terrain/tests/stereo_pipeline_test.cpp`  
  Reduce duplicated comparison code by using `AspPointCloudMetrics`, while keeping the existing human-readable integration test.

---

## Task 1: Extract ASP Comparison Metrics Module

**Files:**
- Create: `src/core/mvs/AspPointCloudMetrics.h`
- Create: `src/core/mvs/AspPointCloudMetrics.cpp`
- Modify: `src/core/mvs/CMakeLists.txt:19-54`
- Test: temporary compile through `cmake --build build --target mvs`

- [ ] **Step 1: Write the header with concrete result types**

Create `src/core/mvs/AspPointCloudMetrics.h`:

```cpp
#pragma once

#include <array>
#include <string>
#include <vector>

namespace xjw
{
namespace mvs
{

struct AspPointCloudMetricsThresholds
{
    double minCoverageRatio = 0.80;
    double maxMedianDistance = 0.001;
    double maxP95Distance = 0.01;
};

struct AspPointCloudMetricsResult
{
    int aspWidth = 0;
    int aspHeight = 0;
    int aspValidPoints = 0;
    int plascanWidth = 0;
    int plascanHeight = 0;
    int plascanValidPoints = 0;
    double coverageRatio = 0.0;
    double bboxContainmentRatio = 0.0;
    double nnMean = 0.0;
    double nnMedian = 0.0;
    double nnP90 = 0.0;
    double nnP95 = 0.0;
    double nnMax = 0.0;
    std::array<double, 3> meanOffset = {0.0, 0.0, 0.0};
    bool passed = false;
    std::string failureReason;
};

class AspPointCloudMetrics
{
public:
    static bool compare(const std::string &plascanTifPath,
                        const std::string &aspTifPath,
                        const AspPointCloudMetricsThresholds &thresholds,
                        AspPointCloudMetricsResult &result,
                        std::string *errorMessage = nullptr);

    static std::string toTextReport(const AspPointCloudMetricsResult &result);
    static std::string toJson(const AspPointCloudMetricsResult &result);
};

} // namespace mvs
} // namespace xjw
```

- [ ] **Step 2: Implement TIF loading and metric computation**

Create `src/core/mvs/AspPointCloudMetrics.cpp`:

```cpp
#include "AspPointCloudMetrics.h"

#include "PointCloudTifIO.h"

#include <gdal_priv.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace xjw
{
namespace mvs
{
namespace
{

struct Point3
{
    float x;
    float y;
    float z;
};

struct LoadedAspCloud
{
    int width = 0;
    int height = 0;
    int valid = 0;
    std::array<double, 3> offset = {0.0, 0.0, 0.0};
    std::vector<Point3> points;
};

bool readAspTif(const std::string &path, LoadedAspCloud &cloud, std::string *errorMessage)
{
    GDALAllRegister();
    GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!dataset)
    {
        if (errorMessage)
        {
            *errorMessage = "Failed to open ASP point cloud: " + path;
        }
        return false;
    }

    cloud.width = dataset->GetRasterXSize();
    cloud.height = dataset->GetRasterYSize();
    const int bands = dataset->GetRasterCount();
    if (bands < 3)
    {
        GDALClose(dataset);
        if (errorMessage)
        {
            *errorMessage = "ASP point cloud has fewer than 3 bands: " + path;
        }
        return false;
    }

    const char *offsetText = dataset->GetMetadataItem("POINT_OFFSET");
    if (offsetText)
    {
        std::sscanf(offsetText, "%lf %lf %lf", &cloud.offset[0], &cloud.offset[1], &cloud.offset[2]);
    }

    cv::Mat x(cloud.height, cloud.width, CV_32FC1);
    cv::Mat y(cloud.height, cloud.width, CV_32FC1);
    cv::Mat z(cloud.height, cloud.width, CV_32FC1);
    dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        x.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    dataset->GetRasterBand(2)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        y.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    dataset->GetRasterBand(3)->RasterIO(GF_Read, 0, 0, cloud.width, cloud.height,
                                        z.data, cloud.width, cloud.height, GDT_Float32, 0, 0);
    GDALClose(dataset);

    cloud.points.reserve(static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height) / 2);
    for (int r = 0; r < cloud.height; ++r)
    {
        for (int c = 0; c < cloud.width; ++c)
        {
            const float px = x.at<float>(r, c);
            const float py = y.at<float>(r, c);
            const float pz = z.at<float>(r, c);
            if ((px == 0.0f && py == 0.0f && pz == 0.0f) ||
                !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
            {
                continue;
            }
            cloud.points.push_back({static_cast<float>(px + cloud.offset[0]),
                                    static_cast<float>(py + cloud.offset[1]),
                                    static_cast<float>(pz + cloud.offset[2])});
        }
    }
    cloud.valid = static_cast<int>(cloud.points.size());
    return true;
}

bool readPlascanTif(const std::string &path, LoadedAspCloud &cloud, std::string *errorMessage)
{
    TriangulationResult tri;
    if (!PointCloudTifIO::readTif(path, tri, errorMessage))
    {
        return false;
    }

    cloud.width = tri.pointCloud.cols;
    cloud.height = tri.pointCloud.rows;
    cloud.valid = tri.validPoints;
    cloud.offset = tri.pointOffset;
    cloud.points.reserve(static_cast<size_t>(std::max(0, tri.validPoints)));

    for (int r = 0; r < tri.pointCloud.rows; ++r)
    {
        for (int c = 0; c < tri.pointCloud.cols; ++c)
        {
            if (tri.validMask.at<uint8_t>(r, c) == 0)
            {
                continue;
            }
            const cv::Vec3d &p = tri.pointCloud.at<cv::Vec3d>(r, c);
            cloud.points.push_back({static_cast<float>(p[0] + tri.pointOffset[0]),
                                    static_cast<float>(p[1] + tri.pointOffset[1]),
                                    static_cast<float>(p[2] + tri.pointOffset[2])});
        }
    }
    cloud.valid = static_cast<int>(cloud.points.size());
    return true;
}

uint64_t cellKey(int x, int y, int z)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) * 73856093ULL) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(y)) * 19349663ULL) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(z)) * 83492791ULL);
}

void evaluatePass(const AspPointCloudMetricsThresholds &thresholds,
                  AspPointCloudMetricsResult &result)
{
    std::ostringstream failures;
    if (result.coverageRatio < thresholds.minCoverageRatio)
    {
        failures << "coverageRatio " << result.coverageRatio
                 << " < " << thresholds.minCoverageRatio << "; ";
    }
    if (result.nnMedian > thresholds.maxMedianDistance)
    {
        failures << "nnMedian " << result.nnMedian
                 << " > " << thresholds.maxMedianDistance << "; ";
    }
    if (result.nnP95 > thresholds.maxP95Distance)
    {
        failures << "nnP95 " << result.nnP95
                 << " > " << thresholds.maxP95Distance << "; ";
    }
    result.failureReason = failures.str();
    result.passed = result.failureReason.empty();
}

} // namespace

bool AspPointCloudMetrics::compare(const std::string &plascanTifPath,
                                   const std::string &aspTifPath,
                                   const AspPointCloudMetricsThresholds &thresholds,
                                   AspPointCloudMetricsResult &result,
                                   std::string *errorMessage)
{
    LoadedAspCloud asp;
    LoadedAspCloud plascan;
    if (!readAspTif(aspTifPath, asp, errorMessage))
    {
        return false;
    }
    if (!readPlascanTif(plascanTifPath, plascan, errorMessage))
    {
        return false;
    }
    if (asp.points.empty() || plascan.points.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "Cannot compare empty point clouds";
        }
        return false;
    }

    result.aspWidth = asp.width;
    result.aspHeight = asp.height;
    result.aspValidPoints = asp.valid;
    result.plascanWidth = plascan.width;
    result.plascanHeight = plascan.height;
    result.plascanValidPoints = plascan.valid;
    result.coverageRatio = static_cast<double>(plascan.valid) / static_cast<double>(asp.valid);

    Point3 aspMin{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    Point3 aspMax{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    Point3 allMin = aspMin;
    Point3 allMax = aspMax;
    for (const Point3 &p : asp.points)
    {
        aspMin.x = std::min(aspMin.x, p.x); aspMin.y = std::min(aspMin.y, p.y); aspMin.z = std::min(aspMin.z, p.z);
        aspMax.x = std::max(aspMax.x, p.x); aspMax.y = std::max(aspMax.y, p.y); aspMax.z = std::max(aspMax.z, p.z);
        allMin.x = std::min(allMin.x, p.x); allMin.y = std::min(allMin.y, p.y); allMin.z = std::min(allMin.z, p.z);
        allMax.x = std::max(allMax.x, p.x); allMax.y = std::max(allMax.y, p.y); allMax.z = std::max(allMax.z, p.z);
    }
    int insideBbox = 0;
    for (const Point3 &p : plascan.points)
    {
        if (p.x >= aspMin.x && p.x <= aspMax.x &&
            p.y >= aspMin.y && p.y <= aspMax.y &&
            p.z >= aspMin.z && p.z <= aspMax.z)
        {
            ++insideBbox;
        }
        allMin.x = std::min(allMin.x, p.x); allMin.y = std::min(allMin.y, p.y); allMin.z = std::min(allMin.z, p.z);
        allMax.x = std::max(allMax.x, p.x); allMax.y = std::max(allMax.y, p.y); allMax.z = std::max(allMax.z, p.z);
    }
    result.bboxContainmentRatio = static_cast<double>(insideBbox) / static_cast<double>(plascan.points.size());

    const float cellSize = 0.05f;
    std::unordered_map<uint64_t, std::vector<int>> grid;
    grid.reserve(asp.points.size());
    auto indexFor = [&](const Point3 &p, int &ix, int &iy, int &iz)
    {
        ix = static_cast<int>(std::floor((p.x - allMin.x) / cellSize));
        iy = static_cast<int>(std::floor((p.y - allMin.y) / cellSize));
        iz = static_cast<int>(std::floor((p.z - allMin.z) / cellSize));
    };
    for (int i = 0; i < static_cast<int>(asp.points.size()); ++i)
    {
        int ix, iy, iz;
        indexFor(asp.points[i], ix, iy, iz);
        grid[cellKey(ix, iy, iz)].push_back(i);
    }

    const int sampleStep = std::max(1, static_cast<int>(plascan.points.size()) / 100000);
    std::vector<double> distances;
    double offsetX = 0.0;
    double offsetY = 0.0;
    double offsetZ = 0.0;
    int matched = 0;

    for (int i = 0; i < static_cast<int>(plascan.points.size()); i += sampleStep)
    {
        const Point3 &p = plascan.points[i];
        int ix, iy, iz;
        indexFor(p, ix, iy, iz);
        double best = std::numeric_limits<double>::max();
        int bestIndex = -1;
        for (int dx = -2; dx <= 2; ++dx)
        {
            for (int dy = -2; dy <= 2; ++dy)
            {
                for (int dz = -2; dz <= 2; ++dz)
                {
                    auto it = grid.find(cellKey(ix + dx, iy + dy, iz + dz));
                    if (it == grid.end())
                    {
                        continue;
                    }
                    for (int aspIndex : it->second)
                    {
                        const Point3 &q = asp.points[aspIndex];
                        const double ddx = static_cast<double>(p.x) - q.x;
                        const double ddy = static_cast<double>(p.y) - q.y;
                        const double ddz = static_cast<double>(p.z) - q.z;
                        const double dist2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        if (dist2 < best)
                        {
                            best = dist2;
                            bestIndex = aspIndex;
                        }
                    }
                }
            }
        }
        if (bestIndex >= 0)
        {
            const double dist = std::sqrt(best);
            distances.push_back(dist);
            const Point3 &q = asp.points[bestIndex];
            offsetX += static_cast<double>(p.x) - q.x;
            offsetY += static_cast<double>(p.y) - q.y;
            offsetZ += static_cast<double>(p.z) - q.z;
            ++matched;
        }
    }

    if (distances.empty())
    {
        if (errorMessage)
        {
            *errorMessage = "No nearest-neighbor matches found";
        }
        return false;
    }

    std::sort(distances.begin(), distances.end());
    double sum = 0.0;
    for (double d : distances)
    {
        sum += d;
    }
    result.nnMean = sum / static_cast<double>(distances.size());
    result.nnMedian = distances[distances.size() / 2];
    result.nnP90 = distances[distances.size() * 90 / 100];
    result.nnP95 = distances[distances.size() * 95 / 100];
    result.nnMax = distances.back();
    result.meanOffset = {offsetX / matched, offsetY / matched, offsetZ / matched};
    evaluatePass(thresholds, result);
    return true;
}

std::string AspPointCloudMetrics::toTextReport(const AspPointCloudMetricsResult &result)
{
    std::ostringstream out;
    out << "ASP valid: " << result.aspValidPoints << "\n";
    out << "PlaScan valid: " << result.plascanValidPoints << "\n";
    out << "Coverage ratio: " << result.coverageRatio << "\n";
    out << "BBox containment: " << result.bboxContainmentRatio << "\n";
    out << "NN mean: " << result.nnMean << "\n";
    out << "NN median: " << result.nnMedian << "\n";
    out << "NN P90: " << result.nnP90 << "\n";
    out << "NN P95: " << result.nnP95 << "\n";
    out << "NN max: " << result.nnMax << "\n";
    out << "Mean offset: (" << result.meanOffset[0] << ", "
        << result.meanOffset[1] << ", " << result.meanOffset[2] << ")\n";
    out << "PASS: " << (result.passed ? "true" : "false") << "\n";
    if (!result.passed)
    {
        out << "Failure: " << result.failureReason << "\n";
    }
    return out.str();
}

std::string AspPointCloudMetrics::toJson(const AspPointCloudMetricsResult &result)
{
    std::ostringstream out;
    out << "{\n";
    out << "  \"asp_valid_points\": " << result.aspValidPoints << ",\n";
    out << "  \"plascan_valid_points\": " << result.plascanValidPoints << ",\n";
    out << "  \"coverage_ratio\": " << result.coverageRatio << ",\n";
    out << "  \"bbox_containment_ratio\": " << result.bboxContainmentRatio << ",\n";
    out << "  \"nn_mean\": " << result.nnMean << ",\n";
    out << "  \"nn_median\": " << result.nnMedian << ",\n";
    out << "  \"nn_p90\": " << result.nnP90 << ",\n";
    out << "  \"nn_p95\": " << result.nnP95 << ",\n";
    out << "  \"nn_max\": " << result.nnMax << ",\n";
    out << "  \"mean_offset\": [" << result.meanOffset[0] << ", "
        << result.meanOffset[1] << ", " << result.meanOffset[2] << "],\n";
    out << "  \"passed\": " << (result.passed ? "true" : "false") << ",\n";
    out << "  \"failure_reason\": \"" << result.failureReason << "\"\n";
    out << "}\n";
    return out.str();
}

} // namespace mvs
} // namespace xjw
```

- [ ] **Step 3: Add the module to the mvs target**

Modify `src/core/mvs/CMakeLists.txt`:

```cmake
set(MVS_CXX_SOURCES
    DepthMapFusion.cpp
    DepthFrameUtils.cpp
    DenseCloudBuilder.cpp
    DepthMapGenerator.cpp
    EpipolarRectifier.cpp
    SparseCloudValidator.cpp
    SparseCloudPreprocessor.cpp
    DensePointCloudGenerator.cpp
    SubpixelRefiner.cpp
    DisparityFilter.cpp
    DisparityTriangulator.cpp
    PointCloudTifIO.cpp
    AspPointCloudMetrics.cpp
    StereoDenseCloudPipeline.cpp
)

set(MVS_HEADERS
    MvsTypes.h
    PatchMatchCUDA.h
    DepthMapFusion.h
    DepthFrameUtils.h
    DenseCloudBuilder.h
    DepthMapGenerator.h
    DensePointCloudCUDA.h
    EpipolarRectifier.h
    SparseCloudValidator.h
    SparseCloudPreprocessor.h
    DensePointCloudGenerator.h
    MVSPipeline.h
    SubpixelRefiner.h
    DisparityFilter.h
    DisparityTriangulator.h
    PointCloudTifIO.h
    AspPointCloudMetrics.h
    StereoDenseCloudPipeline.h
)
```

- [ ] **Step 4: Build the mvs target**

Run:

```bash
cmake --build build --target mvs -j$(nproc)
```

Expected:

```text
[100%] Built target mvs
```

- [ ] **Step 5: Checkpoint**

Run:

```bash
find src/core/mvs -maxdepth 1 -type f \( -name 'AspPointCloudMetrics.*' -o -name 'CMakeLists.txt' \) -printf '%TY-%Tm-%Td %TH:%TM %p\n'
```

Expected: both new metrics files and the updated CMake file are listed.

---

## Task 2: Add a Strict ASP Benchmark Executable

**Files:**
- Create: `src/core/terrain/tests/stereo_pipeline_benchmark.cpp`
- Modify: `src/core/terrain/CMakeLists.txt:89-110`
- Test: `./build/src/core/terrain/stereo_pipeline_benchmark`

- [ ] **Step 1: Create the benchmark source**

Create `src/core/terrain/tests/stereo_pipeline_benchmark.cpp`:

```cpp
#include "AspPointCloudMetrics.h"
#include "Camera.h"
#include "StereoDenseCloudPipeline.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QTextStream>

#include <cstdio>
#include <string>

using namespace xjw;
using namespace xjw::mvs;

namespace
{

const char *IMAGE1 = "/home/guderian/code/plascan/data/stereo_test_20260426/20260413T174329163_NAS_PAN_L2b.tif";
const char *IMAGE2 = "/home/guderian/code/plascan/data/stereo_test_20260426/20260413T174419164_NAS_PAN_L2b.tif";
const char *CAMERA1 = "/home/guderian/code/plascan/data/stereo_test_20260426/ba-tsai_20260413T174329163_NAS_PAN_L2b.tsai";
const char *CAMERA2 = "/home/guderian/code/plascan/data/stereo_test_20260426/ba-tsai_20260413T174419164_NAS_PAN_L2b.tsai";
const char *ASP_PC = "/home/guderian/code/plascan/data/stereo_test_20260426/run-PC.tif";
const char *OUTPUT_DIR = "/home/guderian/code/plascan/data/test_stereo/stereo_pipeline_benchmark";

StereoPipelineConfig strictBenchmarkConfig()
{
    StereoPipelineConfig cfg;
    cfg.patchMatch.numIterations = 32;
    cfg.patchMatch.patchHalf = 7;
    cfg.patchMatch.confidenceThresh = 0.0001f;
    cfg.patchMatch.downsampleFactor = 1;
    cfg.patchMatch.numSourceViews = 1;
    cfg.triangulation.maxTriangulationError = 0.01f;
    cfg.subpixel.mode = 1;
    cfg.outputTif = true;
    cfg.outputPly = true;
    return cfg;
}

bool writeMetricsJson(const QString &path, const AspPointCloudMetricsResult &metrics)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        return false;
    }
    QTextStream stream(&file);
    stream << QString::fromStdString(AspPointCloudMetrics::toJson(metrics));
    return true;
}

} // namespace

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QDir().mkpath(QString::fromUtf8(OUTPUT_DIR));

    StereoDenseCloudPipeline pipeline;
    pipeline.setConfig(strictBenchmarkConfig());
    QObject::connect(&pipeline, &StereoDenseCloudPipeline::progressChanged,
                     [](const QString &stage, float ratio)
    {
        std::printf("[benchmark] %s %.1f%%\n", stage.toUtf8().constData(), ratio * 100.0f);
    });

    StereoPipelineResult pipelineResult;
    const bool ok = pipeline.run(IMAGE1, IMAGE2, CAMERA1, CAMERA2, OUTPUT_DIR, &pipelineResult);
    if (!ok)
    {
        std::fprintf(stderr, "Pipeline failed: %s\n", pipelineResult.errorMsg.c_str());
        return 2;
    }

    AspPointCloudMetricsThresholds thresholds;
    AspPointCloudMetricsResult metrics;
    std::string metricsError;
    if (!AspPointCloudMetrics::compare(pipelineResult.tifPath, ASP_PC, thresholds, metrics, &metricsError))
    {
        std::fprintf(stderr, "Metrics failed: %s\n", metricsError.c_str());
        return 3;
    }

    const QString jsonPath = QDir(QString::fromUtf8(OUTPUT_DIR)).filePath(QStringLiteral("metrics.json"));
    if (!writeMetricsJson(jsonPath, metrics))
    {
        std::fprintf(stderr, "Failed to write metrics JSON: %s\n", jsonPath.toUtf8().constData());
        return 4;
    }

    std::printf("\n%s", AspPointCloudMetrics::toTextReport(metrics).c_str());
    std::printf("Metrics JSON: %s\n", jsonPath.toUtf8().constData());
    return metrics.passed ? 0 : 1;
}
```

- [ ] **Step 2: Add benchmark target to terrain CMake**

Modify `src/core/terrain/CMakeLists.txt` inside the existing `if(BUILD_TESTS)` block after `stereo_pipeline_test`:

```cmake
    add_executable(stereo_pipeline_benchmark
        tests/stereo_pipeline_benchmark.cpp
    )

    target_link_libraries(stereo_pipeline_benchmark PRIVATE
        terrain
        mvs
        pointcloud
        camera
        Qt6::Core
        Qt6::Concurrent
        ${OpenCV_LIBS}
        ${PLASCAN_GDAL_TARGET}
    )

    target_include_directories(stereo_pipeline_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src/core/mvs
        ${CMAKE_SOURCE_DIR}/src/core/camera
        ${CMAKE_SOURCE_DIR}/src/core/pointcloud
    )
```

- [ ] **Step 3: Build the benchmark target**

Run:

```bash
cmake --build build --target stereo_pipeline_benchmark -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_benchmark
```

- [ ] **Step 4: Run the benchmark and confirm current failure is machine-detected**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark
```

Expected: command exits with code `1` and prints a report similar to:

```text
Coverage ratio: 0.17...
NN median: 0.015...
NN P95: 0.147...
PASS: false
Failure: coverageRatio ...; nnMedian ...; nnP95 ...;
```

- [ ] **Step 5: Confirm metrics JSON exists**

Run:

```bash
python3 -m json.tool data/test_stereo/stereo_pipeline_benchmark/metrics.json >/tmp/plascan_metrics_check.json && tail -20 /tmp/plascan_metrics_check.json
```

Expected: valid JSON containing `coverage_ratio`, `nn_median`, `nn_p95`, and `passed`.

- [ ] **Step 6: Checkpoint**

Run:

```bash
stat -c '%y %n' src/core/terrain/tests/stereo_pipeline_benchmark.cpp data/test_stereo/stereo_pipeline_benchmark/metrics.json
```

Expected: both files exist with current timestamps.

---

## Task 3: Replace Duplicated Comparison Logic in Existing Test

**Files:**
- Modify: `src/core/terrain/tests/stereo_pipeline_test.cpp:1-379`
- Test: `cmake --build build --target stereo_pipeline_test -j$(nproc)` and `./build/src/core/terrain/stereo_pipeline_test`

- [ ] **Step 1: Add metrics include**

At the top of `src/core/terrain/tests/stereo_pipeline_test.cpp`, replace the include block with:

```cpp
#include "AspPointCloudMetrics.h"
#include "Camera.h"
#include "StereoDenseCloudPipeline.h"
#include "PointCloudTifIO.h"
#include "MvsTypes.h"

#include <QCoreApplication>
#include <QDir>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
```

- [ ] **Step 2: Replace the long ASP comparison block**

Replace the block beginning at the comment:

```cpp
// === 3. Compare with ASP reference ===
```

through the matching closing brace before:

```cpp
printSep("Test Complete");
```

with:

```cpp
    // === 3. Compare with ASP reference ===
    printSep("Comparison with ASP run-PC.tif");
    {
        AspPointCloudMetricsThresholds thresholds;
        AspPointCloudMetricsResult metrics;
        std::string metricsErr;
        if (!AspPointCloudMetrics::compare(pipeResult.tifPath, ASP_PC, thresholds, metrics, &metricsErr))
        {
            std::fprintf(stderr, "  Metrics failed: %s\n", metricsErr.c_str());
            return 1;
        }

        std::printf("%s", AspPointCloudMetrics::toTextReport(metrics).c_str());
        if (!metrics.passed)
        {
            std::printf("  Note: strict ASP benchmark has not passed yet; this integration test remains informational.\n");
        }
    }
```

- [ ] **Step 3: Build the existing test**

Run:

```bash
cmake --build build --target stereo_pipeline_test -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_test
```

- [ ] **Step 4: Run the existing test**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_test
```

Expected: exit code `0`, with metrics report printed and the informational note if strict thresholds still fail.

- [ ] **Step 5: Checkpoint**

Run:

```bash
grep -n "AspPointCloudMetrics\|strict ASP benchmark" src/core/terrain/tests/stereo_pipeline_test.cpp
```

Expected: include/use sites are listed.

---

## Task 4: Parameterize Current Pipeline Filters and Diagnostics

**Files:**
- Modify: `src/core/mvs/StereoDenseCloudPipeline.h:15-48`
- Modify: `src/core/mvs/StereoDenseCloudPipeline.cpp:21-390`
- Test: `cmake --build build --target stereo_pipeline_benchmark -j$(nproc)` and benchmark run

- [ ] **Step 1: Add filter config and diagnostic fields**

Modify `src/core/mvs/StereoDenseCloudPipeline.h` so `StereoPipelineConfig` and `StereoPipelineResult` become:

```cpp
struct StereoPipelineFilterConfig
{
    bool enableLeftRightDepthCheck = true;
    float leftRightDepthRatio = 0.05f;
    bool enableLocalDepthConsistency = true;
    int localWindowRadius = 2;
    int localMinNeighbors = 5;
    float localDepthRatio = 0.02f;
    bool enableIqrFilter = true;
    float iqrMultiplier = 1.5f;
};

struct StereoPipelineDepthRangeConfig
{
    double nearScale = 0.7;
    double farScale = 1.5;
};

struct StereoPipelineConfig
{
    std::string featureAlgorithm = "disk+lightglue";
    float matchScoreThreshold = 0.2f;
    PatchMatchConfig patchMatch;
    SubpixelConfig subpixel;
    DisparityFilterConfig disparityFilter;
    TriangulationConfig triangulation;
    StereoPipelineFilterConfig filters;
    StereoPipelineDepthRangeConfig depthRange;
    bool outputTif = true;
    bool outputPly = true;
    bool keepIntermediateMasks = false;
    int numThreads = 0;
};

struct StereoPipelineResult
{
    std::string tifPath;
    std::string plyPath;
    int totalPoints = 0;
    int validPoints = 0;
    int depthValidBeforeFiltering = 0;
    int leftRightRejected = 0;
    int leftRightRejectedOob = 0;
    int leftRightRejectedNoReverse = 0;
    int leftRightRejectedMismatch = 0;
    int localRejected = 0;
    int iqrRejected = 0;
    int validAfterFiltering = 0;
    float coveragePercent = 0.f;
    double medianTriError = 0.0;
    std::string errorMsg;
};
```

- [ ] **Step 2: Use configurable depth range**

In `StereoDenseCloudPipeline.cpp`, replace:

```cpp
float zNear = static_cast<float>(sceneDepth * 0.7);
float zFar = static_cast<float>(sceneDepth * 1.5);
```

with:

```cpp
float zNear = static_cast<float>(sceneDepth * m_config.depthRange.nearScale);
float zFar = static_cast<float>(sceneDepth * m_config.depthRange.farScale);
```

- [ ] **Step 3: Capture depth count before filtering**

After `int depthValid = cv::countNonZero(depthMap > 0);`, add:

```cpp
res.depthValidBeforeFiltering = depthValid;
```

- [ ] **Step 4: Guard left-right check with config and save counters**

Replace the condition:

```cpp
if (PatchMatchDepthEstimator::estimate(
```

for the reverse pass block with:

```cpp
if (m_config.filters.enableLeftRightDepthCheck &&
    PatchMatchDepthEstimator::estimate(
```

Replace:

```cpp
if (std::abs(dL - dR) / dL > 0.05f)
```

with:

```cpp
if (std::abs(dL - dR) / dL > m_config.filters.leftRightDepthRatio)
```

Before the reverse-pass block ends, after the existing `fprintf`, add:

```cpp
        res.leftRightRejected = lrRejected;
        res.leftRightRejectedOob = lrOob;
        res.leftRightRejectedNoReverse = lrNoR;
        res.leftRightRejectedMismatch = lrMismatch;
```

- [ ] **Step 5: Make local consistency configurable**

Replace the local consistency block header:

```cpp
    // Local depth consistency: reject pixels deviating >2% from 5x5 median
    {
```

with:

```cpp
    if (m_config.filters.enableLocalDepthConsistency)
    {
```

Inside that block, replace all hard-coded window values:

```cpp
for (int r = 2; r < depthMap.rows - 2; ++r)
```

with:

```cpp
const int radius = std::max(1, m_config.filters.localWindowRadius);
for (int r = radius; r < depthMap.rows - radius; ++r)
```

Replace:

```cpp
for (int c = 2; c < depthMap.cols - 2; ++c)
```

with:

```cpp
for (int c = radius; c < depthMap.cols - radius; ++c)
```

Replace:

```cpp
float neighbors[25];
```

with:

```cpp
std::vector<float> neighbors;
neighbors.reserve(static_cast<size_t>((radius * 2 + 1) * (radius * 2 + 1)));
```

Replace the nested neighbor push code with:

```cpp
for (int dr = -radius; dr <= radius; ++dr)
{
    for (int dc = -radius; dc <= radius; ++dc)
    {
        float nd = depthCopy.at<float>(r + dr, c + dc);
        if (nd > 0)
        {
            neighbors.push_back(nd);
        }
    }
}
```

Replace the condition with:

```cpp
if (static_cast<int>(neighbors.size()) < m_config.filters.localMinNeighbors)
{
    validMask.at<uint8_t>(r, c) = 0;
    depthMap.at<float>(r, c) = 0.0f;
    ++localRejected;
    continue;
}
std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
const float median = neighbors[neighbors.size() / 2];
if (median <= 0.0f || std::abs(d - median) / median > m_config.filters.localDepthRatio)
{
    validMask.at<uint8_t>(r, c) = 0;
    depthMap.at<float>(r, c) = 0.0f;
    ++localRejected;
}
```

After the local consistency `fprintf`, add:

```cpp
        res.localRejected = localRejected;
```

- [ ] **Step 6: Make IQR filter configurable**

Replace the IQR block header:

```cpp
    // IQR-based depth outlier removal
    {
```

with:

```cpp
    if (m_config.filters.enableIqrFilter)
    {
```

Replace:

```cpp
float lo = q1 - 1.5f * iqr;
float hi = q3 + 1.5f * iqr;
```

with:

```cpp
float lo = q1 - m_config.filters.iqrMultiplier * iqr;
float hi = q3 + m_config.filters.iqrMultiplier * iqr;
```

After the IQR `fprintf`, add:

```cpp
            res.iqrRejected = iqrRejected;
```

- [ ] **Step 7: Save filtered count**

After `int filteredValid = cv::countNonZero(validMask);`, add:

```cpp
res.validAfterFiltering = filteredValid;
```

- [ ] **Step 8: Build benchmark**

Run:

```bash
cmake --build build --target stereo_pipeline_benchmark -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_benchmark
```

- [ ] **Step 9: Run benchmark and record baseline diagnostics**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark; echo exit=$?
```

Expected: exit remains `1`, but logs include the same rejection counts as before through configurable paths.

- [ ] **Step 10: Checkpoint**

Run:

```bash
grep -n "StereoPipelineFilterConfig\|leftRightDepthRatio\|enableIqrFilter\|validAfterFiltering" src/core/mvs/StereoDenseCloudPipeline.h src/core/mvs/StereoDenseCloudPipeline.cpp
```

Expected: all new config and diagnostic fields are present.

---

## Task 5: Add Diagnostic Parameter Profiles to Benchmark

**Files:**
- Modify: `src/core/terrain/tests/stereo_pipeline_benchmark.cpp`
- Test: benchmark with `strict`, `no-iqr`, `no-local`, `no-lr`, `wide-depth`

- [ ] **Step 1: Add profile parsing helper**

In `stereo_pipeline_benchmark.cpp`, add this function after `strictBenchmarkConfig()`:

```cpp
StereoPipelineConfig configForProfile(const QString &profile)
{
    StereoPipelineConfig cfg = strictBenchmarkConfig();
    if (profile == QStringLiteral("no-iqr"))
    {
        cfg.filters.enableIqrFilter = false;
    }
    else if (profile == QStringLiteral("no-local"))
    {
        cfg.filters.enableLocalDepthConsistency = false;
    }
    else if (profile == QStringLiteral("no-lr"))
    {
        cfg.filters.enableLeftRightDepthCheck = false;
    }
    else if (profile == QStringLiteral("wide-depth"))
    {
        cfg.depthRange.nearScale = 0.4;
        cfg.depthRange.farScale = 2.2;
    }
    else if (profile == QStringLiteral("loose-all"))
    {
        cfg.filters.leftRightDepthRatio = 0.20f;
        cfg.filters.localDepthRatio = 0.10f;
        cfg.filters.iqrMultiplier = 4.0f;
        cfg.triangulation.maxTriangulationError = 0.05f;
    }
    return cfg;
}
```

- [ ] **Step 2: Use the selected profile in main**

Replace:

```cpp
StereoDenseCloudPipeline pipeline;
pipeline.setConfig(strictBenchmarkConfig());
```

with:

```cpp
const QString profile = argc > 1 ? QString::fromLocal8Bit(argv[1]) : QStringLiteral("strict");
std::printf("[benchmark] profile=%s\n", profile.toUtf8().constData());

StereoDenseCloudPipeline pipeline;
pipeline.setConfig(configForProfile(profile));
```

- [ ] **Step 3: Include profile in JSON path**

Replace:

```cpp
const QString jsonPath = QDir(QString::fromUtf8(OUTPUT_DIR)).filePath(QStringLiteral("metrics.json"));
```

with:

```cpp
const QString jsonPath = QDir(QString::fromUtf8(OUTPUT_DIR)).filePath(
    QStringLiteral("metrics_%1.json").arg(profile));
```

- [ ] **Step 4: Build benchmark**

Run:

```bash
cmake --build build --target stereo_pipeline_benchmark -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_benchmark
```

- [ ] **Step 5: Run diagnostic profiles**

Run:

```bash
for p in strict no-iqr no-local no-lr wide-depth loose-all; do echo "=== $p ==="; ./build/src/core/terrain/stereo_pipeline_benchmark "$p" || true; done
```

Expected: six metric reports and six JSON files under `data/test_stereo/stereo_pipeline_benchmark/`.

- [ ] **Step 6: Summarize metrics**

Run:

```bash
python3 - <<'PY'
import json, glob
for path in sorted(glob.glob('data/test_stereo/stereo_pipeline_benchmark/metrics_*.json')):
    data=json.load(open(path))
    print(path, 'coverage=', data['coverage_ratio'], 'median=', data['nn_median'], 'p95=', data['nn_p95'], 'passed=', data['passed'])
PY
```

Expected: a compact table that identifies whether current losses are caused mainly by filters or deeper pipeline structure.

- [ ] **Step 7: Checkpoint**

Write a short summary in the conversation with the best profile and why it still fails or passes. Do not modify GUI.

---

## Task 6: Tune Current Pipeline Only If Diagnostics Show It Can Pass

**Files:**
- Modify: `src/core/terrain/tests/stereo_pipeline_benchmark.cpp`
- Modify: `src/core/mvs/StereoDenseCloudPipeline.cpp` only if a single filter is confirmed as the root cause
- Test: benchmark PASS or documented FAIL that triggers Task 7

- [ ] **Step 1: Select candidate profile from Task 5**

Use the profile table. Continue this task only if at least one profile has both:

```text
coverageRatio >= 0.60
nnMedian <= 0.005
```

If no profile meets both, skip to Task 7.

- [ ] **Step 2: Encode the candidate as `asp-candidate`**

In `configForProfile`, add:

```cpp
else if (profile == QStringLiteral("asp-candidate"))
{
    cfg.filters.leftRightDepthRatio = 0.20f;
    cfg.filters.localDepthRatio = 0.10f;
    cfg.filters.iqrMultiplier = 4.0f;
    cfg.triangulation.maxTriangulationError = 0.02f;
}
```

Adjust the four values only to match the actual best profile found in Task 5. Keep the shape of the block the same.

- [ ] **Step 3: Build benchmark**

Run:

```bash
cmake --build build --target stereo_pipeline_benchmark -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_benchmark
```

- [ ] **Step 4: Run candidate benchmark**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark asp-candidate; echo exit=$?
```

Expected: if exit is `0`, current pipeline can satisfy the strict threshold after tuning. If exit is `1`, record metrics and proceed to Task 7.

- [ ] **Step 5: Freeze candidate defaults only if PASS**

If `asp-candidate` passes, update `strictBenchmarkConfig()` to use the candidate values and rerun:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark strict
```

Expected: exit `0`.

- [ ] **Step 6: Checkpoint**

Run:

```bash
python3 -m json.tool data/test_stereo/stereo_pipeline_benchmark/metrics_strict.json | grep -E 'coverage_ratio|nn_median|nn_p95|passed'
```

Expected if this task succeeds:

```text
"coverage_ratio": >=0.8
"nn_median": <=0.001
"nn_p95": <=0.01
"passed": true
```

---

## Task 7: Implement ASP-Style Rectified/Disparity Path If Tuning Fails

**Files:**
- Modify: `src/core/mvs/StereoDenseCloudPipeline.h`
- Modify: `src/core/mvs/StereoDenseCloudPipeline.cpp`
- Possibly modify: `src/core/mvs/EpipolarRectifier.h`, `src/core/mvs/EpipolarRectifier.cpp` only if current public API cannot produce inverse homographies
- Test: `stereo_pipeline_benchmark asp-rectified`

- [ ] **Step 1: Add path selector to config**

In `StereoDenseCloudPipeline.h`, add:

```cpp
enum class StereoPipelineGeometryMode
{
    OriginalDepth,
    RectifiedDisparity
};
```

Add this field to `StereoPipelineConfig`:

```cpp
StereoPipelineGeometryMode geometryMode = StereoPipelineGeometryMode::OriginalDepth;
```

- [ ] **Step 2: Add benchmark profile for rectified path**

In `configForProfile`, add:

```cpp
else if (profile == QStringLiteral("asp-rectified"))
{
    cfg.geometryMode = StereoPipelineGeometryMode::RectifiedDisparity;
    cfg.filters.enableIqrFilter = false;
    cfg.filters.enableLocalDepthConsistency = false;
    cfg.filters.enableLeftRightDepthCheck = true;
    cfg.triangulation.maxTriangulationError = 0.01f;
}
```

- [ ] **Step 3: Split existing run body into original-depth helper**

In `StereoDenseCloudPipeline.cpp`, create a private-file helper function above `StereoDenseCloudPipeline::run`:

```cpp
bool runOriginalDepthPath(const cv::Mat &grayL,
                          const cv::Mat &grayR,
                          const Camera &leftCamera,
                          const Camera &rightCamera,
                          const std::string &outputDir,
                          const StereoPipelineConfig &config,
                          StereoPipelineResult &res,
                          StereoDenseCloudPipeline *owner)
{
    // Move the current implementation body from after grayscale conversion through output writing here.
    // Replace every m_config reference with config.
    // Replace every emit progressChanged(...) with emit owner->progressChanged(...).
    // Return true on success and false after setting res.errorMsg.
}
```

Use this exact signature. Move, do not duplicate, the current original-depth implementation.

- [ ] **Step 4: Dispatch based on geometry mode**

In `StereoDenseCloudPipeline::run(const cv::Mat &...)`, keep preprocessing and grayscale conversion, then dispatch:

```cpp
if (m_config.geometryMode == StereoPipelineGeometryMode::OriginalDepth)
{
    const bool ok = runOriginalDepthPath(grayL, grayR, leftCamera, rightCamera, outputDir, m_config, res, this);
    if (result) *result = res;
    emit finished(ok);
    return ok;
}

const bool ok = runRectifiedDisparityPath(grayL, grayR, leftCamera, rightCamera, outputDir, m_config, res, this);
if (result) *result = res;
emit finished(ok);
return ok;
```

- [ ] **Step 5: Implement rectified path with existing triangulator**

Add a helper next to `runOriginalDepthPath`:

```cpp
bool runRectifiedDisparityPath(const cv::Mat &grayL,
                               const cv::Mat &grayR,
                               const Camera &leftCamera,
                               const Camera &rightCamera,
                               const std::string &outputDir,
                               const StereoPipelineConfig &config,
                               StereoPipelineResult &res,
                               StereoDenseCloudPipeline *owner)
{
    emit owner->progressChanged("Rectified preprocessing", 0.05f);

    // Use OpenCV stereoRectifyUncalibrated only if existing EpipolarRectifier cannot provide a rectified pair.
    // If EpipolarRectifier already exposes inverse homographies, use it instead.
    // The required outputs are rectLeft, rectRight, H1inv, H2inv.

    // Produce a CV_32F disparity map in rectified-left pixel coordinates.
    // Positive disparity means right pixel is c + disparity, matching DisparityTriangulator::triangulate().

    // Apply DisparityFilter or equivalent speckle/median filtering in disparity domain.

    TriangulationResult triResult = DisparityTriangulator::triangulate(
        disparity,
        validMask,
        H1inv,
        H2inv,
        leftCamera,
        rightCamera,
        config.triangulation);

    // Reuse the same TIF and PLY writing logic as runOriginalDepthPath.
}
```

This step intentionally requires inspecting `EpipolarRectifier`; if its current API cannot supply `rectLeft`, `rectRight`, `H1inv`, and `H2inv`, extend it minimally to do so. Do not add GUI code.

- [ ] **Step 6: Build benchmark**

Run:

```bash
cmake --build build --target stereo_pipeline_benchmark -j$(nproc)
```

Expected:

```text
[100%] Built target stereo_pipeline_benchmark
```

- [ ] **Step 7: Run rectified benchmark**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark asp-rectified; echo exit=$?
```

Expected: either exit `0`, or exit `1` with metrics showing which threshold remains failing.

- [ ] **Step 8: Iterate only within rectified path until PASS**

For each iteration, change only one of:

```text
PatchMatch disparity/depth mapping
left-right disparity threshold
speckle size
triangulation max error
rectification homography direction
```

After each change run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark asp-rectified; echo exit=$?
```

Expected final target:

```text
PASS: true
exit=0
```

- [ ] **Step 9: Freeze strict profile to rectified mode when PASS**

After `asp-rectified` passes, update `strictBenchmarkConfig()` to set:

```cpp
cfg.geometryMode = StereoPipelineGeometryMode::RectifiedDisparity;
cfg.filters.enableIqrFilter = false;
cfg.filters.enableLocalDepthConsistency = false;
cfg.filters.enableLeftRightDepthCheck = true;
```

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark strict; echo exit=$?
```

Expected: `exit=0`.

---

## Task 8: Final Verification Before GUI Work

**Files:**
- No code changes unless a verification failure identifies a specific root cause
- Test: benchmark, terrain tests, mvs tests, GUI build

- [ ] **Step 1: Run strict benchmark**

Run:

```bash
./build/src/core/terrain/stereo_pipeline_benchmark strict; echo exit=$?
```

Expected:

```text
PASS: true
exit=0
```

- [ ] **Step 2: Verify JSON thresholds**

Run:

```bash
python3 - <<'PY'
import json
p='data/test_stereo/stereo_pipeline_benchmark/metrics_strict.json'
d=json.load(open(p))
assert d['coverage_ratio'] >= 0.80, d
assert d['nn_median'] <= 0.001, d
assert d['nn_p95'] <= 0.01, d
assert d['passed'] is True, d
print('strict metrics verified')
PY
```

Expected:

```text
strict metrics verified
```

- [ ] **Step 3: Run relevant non-GUI tests**

Run:

```bash
ctest --test-dir build -R 'terrain|mvs' --output-on-failure
```

Expected: all selected tests pass. If `stereo_pipeline_benchmark` is not registered with CTest, the explicit benchmark run above is the source of truth.

- [ ] **Step 4: Build GUI**

Run:

```bash
cmake --build build --target plascan_gui -j$(nproc)
```

Expected:

```text
[100%] Built target plascan_gui
```

- [ ] **Step 5: Checkpoint**

Report the strict benchmark metrics and confirm GUI work may now begin. Do not claim full project completion until dialog changes are implemented and manually verified.

---

## Task 9: Update Dialog Only After Strict Benchmark PASS

**Files:**
- Modify: `src/gui/dialogs/SimplePointCloudDialog.cpp:24-233`
- Modify: `src/gui/dialogs/SimplePointCloudDialog.h`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp:1105-1138`
- Modify: GUI manager files only if needed to route strict ASP mode to the passing core pipeline
- Test: GUI build and manual launch

- [ ] **Step 1: Confirm benchmark gate**

Run:

```bash
python3 - <<'PY'
import json
p='data/test_stereo/stereo_pipeline_benchmark/metrics_strict.json'
d=json.load(open(p))
assert d['coverage_ratio'] >= 0.80
assert d['nn_median'] <= 0.001
assert d['nn_p95'] <= 0.01
assert d['passed'] is True
print('GUI gate open')
PY
```

Expected:

```text
GUI gate open
```

- [ ] **Step 2: Add ASP mode setting to dialog output**

In `SimplePointCloudDialog::buildSettings()`, add these fields before `return s;`:

```cpp
s[QStringLiteral("geometry_mode")] = QStringLiteral("asp_rectified_disparity");
s[QStringLiteral("output_tif")] = true;
s[QStringLiteral("output_ply")] = true;
s[QStringLiteral("asp_aligned_mode")] = true;
```

- [ ] **Step 3: Update dialog text**

In `SimplePointCloudDialog::setupUi()`, replace:

```cpp
auto *titleLabel = new QLabel(tr("<b>从稀疏点云创建密集点云</b>"), this);
```

with:

```cpp
auto *titleLabel = new QLabel(tr("<b>双视图 ASP 对齐密集点云</b>"), this);
```

After the separator block, add:

```cpp
auto *modeLabel = new QLabel(
    tr("严格模式会输出 ASP 风格 PC.tif 与 PLY。当前模式针对双视图基准流程；多视图通用流程请使用重建菜单。"),
    this);
modeLabel->setWordWrap(true);
modeLabel->setStyleSheet(QStringLiteral("color: #555; font-size: 12px;"));
root->addWidget(modeLabel);
```

- [ ] **Step 4: Route GUI settings to the passing core mode**

In the manager path that converts GUI JSON into `StereoPipelineConfig`, map:

```cpp
settings.value(QStringLiteral("geometry_mode")).toString() == QStringLiteral("asp_rectified_disparity")
```

to:

```cpp
cfg.geometryMode = xjw::mvs::StereoPipelineGeometryMode::RectifiedDisparity;
```

If the GUI still uses `DepthMapGenerator` rather than `StereoDenseCloudPipeline`, add the minimal manager branch needed to call the passing `StereoDenseCloudPipeline` for exactly two selected images. Do not remove the existing multi-view flow.

- [ ] **Step 5: Build GUI**

Run:

```bash
cmake --build build --target plascan_gui -j$(nproc)
```

Expected:

```text
[100%] Built target plascan_gui
```

- [ ] **Step 6: Launch GUI for manual verification**

Run:

```bash
./build/src/gui/plascan_gui
```

Expected: GUI launches. Open `工作流程 -> 创建密集点云` and confirm the dialog title says `双视图 ASP 对齐密集点云` and explains `PC.tif` / `PLY` output.

- [ ] **Step 7: Manual golden-path verification**

In the GUI:

```text
1. Open or create the project that contains the stereo test pair.
2. Ensure the existing AT result is available.
3. Open 工作流程 -> 创建密集点云.
4. Confirm ASP-aligned wording is visible.
5. Start generation.
6. Confirm PC.tif and PC.ply are produced.
7. Confirm the dense cloud appears in the project tree.
```

Expected: the generated core result matches the passing strict benchmark mode.

- [ ] **Step 8: Checkpoint**

Report:

```text
Strict benchmark: PASS
GUI build: PASS
Manual dialog verification: PASS/blocked with reason
```

---

## Self-Review Notes

- Spec coverage: benchmark metrics, strict thresholds, ASP-as-reference-only, speed-aware ASP-style pipeline, and dialog-after-pass gate are all covered.
- Placeholder scan: Task 7 contains a conditional branch because the existing `EpipolarRectifier` API must be inspected during implementation. The required outputs and exact acceptance commands are specified.
- Type consistency: new metric types live in `xjw::mvs`, benchmark includes `AspPointCloudMetrics.h`, and pipeline config fields are consistently named.
