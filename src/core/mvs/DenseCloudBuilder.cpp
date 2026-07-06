// =============================================================================
// 文件: DenseCloudBuilder.cpp
// 模块: MVS - 稠密点云构建 (CPU)
// =============================================================================

#include "DenseCloudBuilder.h"
#include "io/PathIO.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>
#include "log/Logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mvs
{

namespace
{
static plapoint::PointCloud<float, plamatrix::Device::CPU>
buildPointCloud(const std::vector<DensePoint> &cloud);
} // namespace

// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::unproject(
    const cv::Mat                 &depth,
    const cv::Mat                 &mask,
    const PositiveDepthCameraModel &cam,
    const cv::Mat                 &colorImg,
    const DenseCloudOptions &options)
{
    std::vector<DensePoint> cloud;
    if (depth.empty())
    {
        return cloud;
    }

    const int W = depth.cols, H = depth.rows;
    const bool hasColor = !colorImg.empty() &&
                          colorImg.cols == W && colorImg.rows == H;
    const bool isRGB    = hasColor && colorImg.channels() == 3;
    const bool isGray   = hasColor && colorImg.channels() == 1;

    const int step = std::max(1, options.subsample);
    cloud.reserve(W * H / (step * step));

    for (int v = 0; v < H; v += step)
    {
        for (int u = 0; u < W; u += step)
        {
            // 检查掩码
            if (!mask.empty() && mask.at<uint8_t>(v, u) == 0)
            {
                continue;
            }

            float d = depth.at<float>(v, u);
            if (d < options.minDepth || d > options.maxDepth)
            {
                continue;
            }

            float Xw, Yw, Zw;
            cam.unproject(static_cast<float>(u), static_cast<float>(v),
                          d, Xw, Yw, Zw);

            // AABB 裁剪
            if (options.clipAABB)
            {
                if (Xw < options.minX || Xw > options.maxX)
                {
                    continue;
                }
                if (Yw < options.minY || Yw > options.maxY)
                {
                    continue;
                }
                if (Zw < options.minZ || Zw > options.maxZ)
                {
                    continue;
                }
            }

            DensePoint pt;
            pt.x = Xw;
            pt.y = Yw;
            pt.z = Zw;

            // 颜色（无颜色时保持默认 0，避免未初始化的栈垃圾值）
            if (isRGB)
            {
                const cv::Vec3b &bgr = colorImg.at<cv::Vec3b>(v, u);
                pt.r = bgr[2];
                pt.g = bgr[1];
                pt.b = bgr[0];
            }
            else if (isGray)
            {
                uint8_t g = colorImg.at<uint8_t>(v, u);
                pt.r = pt.g = pt.b = g;
            }
            else
            {
                pt.r = pt.g = pt.b = 128;
            }

            cloud.push_back(pt);
        }
    }

    return cloud;
}

// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::merge(
    const std::vector<std::vector<DensePoint>> &clouds)
{
    std::vector<DensePoint> result;
    size_t total = 0;
    for (const auto &c : clouds)
    {
        total += c.size();
    }
    result.reserve(total);
    for (const auto &c : clouds)
    {
        result.insert(result.end(), c.begin(), c.end());
    }
    return result;
}

// =============================================================================
bool DenseCloudBuilder::savePLY(const std::string &path,
                                 const std::vector<DensePoint> &cloud,
                                 std::string *errorMsg)
{
    try
    {
        plapoint::io::writePly(xjw::common::io::toNativeNarrowPath(path),
                               buildPointCloud(cloud),
                               plapoint::io::PlyFormat::BinaryLE);
        return true;
    }
    catch (const std::exception &e)
    {
        if (errorMsg) *errorMsg = e.what();
        return false;
    }
}

// =============================================================================
// 辅助：在 PlaScan 的 DensePoint 和 plapoint::PointCloud 之间转换，保留颜色属性。
// =============================================================================
namespace
{
using DensePlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

static DensePlaCloud
buildPointCloud(const std::vector<DensePoint> &cloud)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> pts(cloud.size(), 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        pts(row, 0) = cloud[i].x;
        pts(row, 1) = cloud[i].y;
        pts(row, 2) = cloud[i].z;
        colors(row, 0) = cloud[i].r;
        colors(row, 1) = cloud[i].g;
        colors(row, 2) = cloud[i].b;
    }
    DensePlaCloud pointCloud(std::move(pts));
    pointCloud.setColors(std::move(colors));
    return pointCloud;
}

static std::vector<DensePoint> fromPointCloud(const DensePlaCloud &cloud)
{
    std::vector<DensePoint> points;
    points.reserve(cloud.size());
    const auto &matrix = cloud.points();
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        DensePoint point;
        point.x = matrix.getValue(row, 0);
        point.y = matrix.getValue(row, 1);
        point.z = matrix.getValue(row, 2);
        if (cloud.hasColors())
        {
            point.r = cloud.colors()->getValue(row, 0);
            point.g = cloud.colors()->getValue(row, 1);
            point.b = cloud.colors()->getValue(row, 2);
        }
        points.push_back(point);
    }
    return points;
}

} // anonymous namespace

// =============================================================================
// Voxel Downsample
// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::voxelDownsample(
    const std::vector<DensePoint> &cloud,
    float voxelSize,
    plapoint::ProcessingDevice processingDevice)
{
    if (cloud.empty() || voxelSize <= 0.0f)
    {
        return cloud;
    }

    const int N = static_cast<int>(cloud.size());
    auto t0 = std::chrono::steady_clock::now();
    LOG_INFO("[Voxel] 开始 plapoint 体素下采样: %d 点, voxelSize=%.4f", N, voxelSize);

    DensePlaCloud pointCloud = buildPointCloud(cloud);
    plapoint::ProcessingReport report;
    DensePlaCloud filtered = plapoint::voxelDownsample(
        pointCloud, voxelSize, processingDevice, &report);
    std::vector<DensePoint> result = fromPointCloud(filtered);

    auto t1 = std::chrono::steady_clock::now();
    LOG_INFO("[Voxel] plapoint 下采样完成: %d → %d 点 (设备=%d) 耗时 %.3f s",
             N, static_cast<int>(result.size()), static_cast<int>(report.usedDevice),
             std::chrono::duration<double>(t1 - t0).count());
    return result;
}

// =============================================================================
// Statistical Outlier Removal
// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::statisticalOutlierRemoval(
    const std::vector<DensePoint> &cloud,
    int   kNeighbors,
    float stdRatio,
    plapoint::ProcessingDevice processingDevice)
{
    if (cloud.size() < (size_t)kNeighbors + 1)
    {
        return cloud;
    }

    const int N = (int)cloud.size();
    auto t0 = std::chrono::steady_clock::now();

    LOG_INFO("[SOR] 开始 plapoint 统计离群点过滤: %d 点, k=%d, stdRatio=%.2f",
             N, kNeighbors, stdRatio);

    DensePlaCloud pointCloud = buildPointCloud(cloud);
    std::vector<int> removedIndices;
    plapoint::ProcessingReport report;
    DensePlaCloud filtered = plapoint::statisticalOutlierRemoval(
        pointCloud, kNeighbors, stdRatio, processingDevice, &removedIndices, &report);
    std::vector<DensePoint> result = fromPointCloud(filtered);

    auto t3 = std::chrono::steady_clock::now();
    LOG_INFO("[SOR] plapoint 过滤完成: %d → %d 点 (移除 %zu, %.1f%%, 设备=%d) 总耗时 %.3f s",
             N, (int)result.size(), removedIndices.size(), 100.f * removedIndices.size() / N,
             static_cast<int>(report.usedDevice),
             std::chrono::duration<double>(t3 - t0).count());
    return result;
}

// =============================================================================
// Radius Outlier Removal
// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::radiusOutlierRemoval(
    const std::vector<DensePoint> &cloud,
    float radius,
    int   minNeighbors,
    plapoint::ProcessingDevice processingDevice)
{
    if (cloud.empty()) return cloud;

    const int N = (int)cloud.size();
    auto t0 = std::chrono::steady_clock::now();

    LOG_INFO("[RadiusOR] 开始 plapoint 半径离群点过滤: %d 点, radius=%.4f, minNeighbors=%d",
             N, radius, minNeighbors);

    DensePlaCloud pointCloud = buildPointCloud(cloud);
    std::vector<int> removedIndices;
    plapoint::ProcessingReport report;
    DensePlaCloud filtered = plapoint::radiusOutlierRemoval(
        pointCloud, radius, minNeighbors, processingDevice, &removedIndices, &report);
    std::vector<DensePoint> result = fromPointCloud(filtered);

    auto t2 = std::chrono::steady_clock::now();
    LOG_INFO("[RadiusOR] plapoint 过滤完成: %d → %d 点 (移除 %zu, %.1f%%, 设备=%d) 耗时 %.3f s",
             N, (int)result.size(), removedIndices.size(), 100.f * removedIndices.size() / N,
             static_cast<int>(report.usedDevice),
             std::chrono::duration<double>(t2 - t0).count());
    return result;
}

} // namespace mvs
} // namespace xjw
