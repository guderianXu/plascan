// SparseCloudPreprocessor.cpp
#include "SparseCloudPreprocessor.h"
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
#include <plapoint/io/ply_io.h>
#include <plapoint/search/kdtree.h>
#include "log/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <memory>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mvs
{

namespace
{

using SparsePlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

SparsePlaCloud toPlaCloud(const std::vector<std::array<float,3>> &pts)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> matrix(
        static_cast<plamatrix::Index>(pts.size()), 3);
    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        matrix(row, 0) = pts[i][0];
        matrix(row, 1) = pts[i][1];
        matrix(row, 2) = pts[i][2];
    }
    return SparsePlaCloud(std::move(matrix));
}

std::vector<std::array<float,3>> fromPlaCloud(const SparsePlaCloud &cloud)
{
    std::vector<std::array<float,3>> pts;
    pts.reserve(cloud.size());
    const auto &matrix = cloud.points();
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        pts.push_back({matrix.getValue(row, 0),
                       matrix.getValue(row, 1),
                       matrix.getValue(row, 2)});
    }
    return pts;
}

float estimateMedianNearestNeighborDistance(const SparsePlaCloud &cloud)
{
    if (cloud.size() < 2)
    {
        return 0.0f;
    }

    auto cloudPtr = std::shared_ptr<const SparsePlaCloud>(&cloud, [](const SparsePlaCloud*) {});
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(cloudPtr);
    tree.build();

    std::vector<float> distances;
    distances.reserve(cloud.size());
    const auto &matrix = cloud.points();
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        plamatrix::Vec3<float> query{matrix.getValue(row, 0),
                                     matrix.getValue(row, 1),
                                     matrix.getValue(row, 2)};
        const auto neighbors = tree.nearestKSearch(query, 2);
        if (neighbors.size() < 2)
        {
            continue;
        }
        const int nn = neighbors[0] == static_cast<int>(i) ? neighbors[1] : neighbors[0];
        const float dx = matrix.getValue(nn, 0) - query.x;
        const float dy = matrix.getValue(nn, 1) - query.y;
        const float dz = matrix.getValue(nn, 2) - query.z;
        distances.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    if (distances.empty())
    {
        return 0.0f;
    }

    auto mid = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
    std::nth_element(distances.begin(), mid, distances.end());
    return *mid;
}

} // namespace

/**
 * @brief 从文本点云文件加载 XYZ 坐标。
 *
 * 支持简单 `.xyz` 行格式与 ASCII/Binary `.ply`。
 */

// =============================================================================
bool SparseCloudPreprocessor::loadXYZ(const std::string &path,
                                       std::vector<std::array<float,3>> &pts,
                                       std::string *err)
{
    std::ifstream ifs(path);
    if (!ifs)
    {
        if (err)
        {
            *err = "无法打开文件: " + path;
        }
        return false;
    }

    // 检测是否为 PLY
    std::string firstLine;
    std::getline(ifs, firstLine);
    bool isPly = (firstLine.find("ply") != std::string::npos);
    if (isPly)
    {
        try
        {
            auto cloud = plapoint::io::readPly<float>(path);
            if (!cloud)
            {
                if (err)
                {
                    *err = "PLY 读取失败: " + path;
                }
                return false;
            }

            pts.clear();
            pts.reserve(cloud->size());
            const auto &matrix = cloud->points();
            for (std::size_t i = 0; i < cloud->size(); ++i)
            {
                const auto row = static_cast<plamatrix::Index>(i);
                pts.push_back({
                    matrix.getValue(row, 0),
                    matrix.getValue(row, 1),
                    matrix.getValue(row, 2)
                });
            }
            return true;
        }
        catch (const std::exception &ex)
        {
            if (err)
            {
                *err = "PLY 读取失败: " + std::string(ex.what());
            }
            return false;
        }
    }
    else
    {
        ifs.seekg(0);
    }

    pts.clear();
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream ss(line);
        float x, y, z;
        if (!(ss >> x >> y >> z))
        {
            continue;
        }
        pts.push_back({x, y, z});
    }
    return true;
}

// =============================================================================
// PlaPoint-backed outlier filtering (SOR + radius filtering).
// =============================================================================
void SparseCloudPreprocessor::filterOutliers(
    std::vector<std::array<float,3>> &pts,
    float radius,
    int minNeigh,
    plapoint::ProcessingDevice processingDevice)
{
    if (pts.size() < 10 || minNeigh <= 0)
    {
        return;
    }

    const int N = (int)pts.size();
    auto t0 = std::chrono::steady_clock::now();

    LOG_INFO("[SparseFilter] 开始稀疏点云离群点过滤: %d 点", N);

    SparsePlaCloud cloud = toPlaCloud(pts);
    const int kNeighbors = std::clamp(20, 2, std::max(2, N - 1));
    const float medianKnnDist = estimateMedianNearestNeighborDistance(cloud);
    float adaptiveRadius = radius;
    if (adaptiveRadius <= 0.0f || adaptiveRadius < medianKnnDist * 0.5f ||
        adaptiveRadius > medianKnnDist * 15.0f)
    {
        adaptiveRadius = medianKnnDist * 3.0f;
        LOG_WARN("[SparseFilter] 固定半径 %.4f 与点间距 (median=%.4f) 不匹配，使用自适应半径: %.4f",
                 radius, medianKnnDist, adaptiveRadius);
    }

    std::vector<int> sorRemovedIndices;
    plapoint::ProcessingReport sorReport;
    SparsePlaCloud sorCloud = plapoint::statisticalOutlierRemoval(
        cloud, kNeighbors, 1.5f, processingDevice, &sorRemovedIndices, &sorReport);

    std::vector<int> radiusRemovedIndices;
    plapoint::ProcessingReport radiusReport;
    SparsePlaCloud filteredCloud = plapoint::radiusOutlierRemoval(
        sorCloud, adaptiveRadius, minNeigh, processingDevice, &radiusRemovedIndices, &radiusReport);

    if (!filteredCloud.size())
    {
        LOG_WARN("[SparseFilter] plapoint 过滤结果为空，保留输入点云避免 MVS 无输入");
        return;
    }

    pts = fromPlaCloud(filteredCloud);

    auto t2 = std::chrono::steady_clock::now();
    LOG_INFO("[SparseFilter] plapoint 过滤完成: %d → %zu 点 (SOR移除 %zu, 半径移除 %zu, SOR设备=%d, 半径设备=%d) 耗时 %.3f s",
             N, pts.size(), sorRemovedIndices.size(), radiusRemovedIndices.size(),
             static_cast<int>(sorReport.usedDevice), static_cast<int>(radiusReport.usedDevice),
             std::chrono::duration<double>(t2 - t0).count());
}

// =============================================================================
bool SparseCloudPreprocessor::run(const std::string             &cloudPath,
                                   const std::vector<CameraView> &/*views*/,
                                   PreprocessResult              &result,
                                   std::string                   *errorMsg) const
{
    std::vector<std::array<float,3>> pts;
    if (!loadXYZ(cloudPath, pts, errorMsg))
    {
        return false;
    }

    result.rawCount = static_cast<int>(pts.size());
    if (pts.empty())
    {
        if (errorMsg)
        {
            *errorMsg = "点云为空";
        }
        return false;
    }

    // KD-tree 加速离群点过滤
    // 注意 radius 参数会在 filterOutliers 内自适应：当固定值远小于点间距时自动放大
    filterOutliers(pts, -1.0f, 3, _processingDevice);   // radius=-1 强制自适应, minNeigh=3
    // 第二轮: 仅当点数充足时再过滤（小规模点云二次过滤会过于激进）
    if (pts.size() > 200)
    {
        filterOutliers(pts, -1.0f, 4, _processingDevice);  // 第二轮自适应半径, minNeigh=4
    }
    else
    {
        LOG_DEBUG("[SparseCloudPreprocessor] 跳过第二轮过滤 (剩余 %zu 点，阈值 200)", pts.size());
    }

    result.filteredCount = static_cast<int>(pts.size());
    result.tooFewPoints  = (result.filteredCount < 100);

    if (!pts.empty())
    {
        std::array<float,3> minPt = {1e18f, 1e18f, 1e18f};
        std::array<float,3> maxPt = {-1e18f, -1e18f, -1e18f};
        for (const auto &p : pts)
        {
            for (int k = 0; k < 3; ++k)
            {
                if (p[k] < minPt[k])
                {
                    minPt[k] = p[k];
                }
                if (p[k] > maxPt[k])
                {
                    maxPt[k] = p[k];
                }
            }
        }
        result.minPt = minPt;
        result.maxPt = maxPt;
        result.cloud.points = pts;
        result.cloud.minPt  = minPt;
        result.cloud.maxPt  = maxPt;
    }

    LOG_INFO("[SparseCloudPreprocessor] raw=%d filtered=%d AABB=[%.2f %.2f %.2f]-[%.2f %.2f %.2f]",
             result.rawCount, result.filteredCount,
             result.minPt[0], result.minPt[1], result.minPt[2],
             result.maxPt[0], result.maxPt[1], result.maxPt[2]);

    return true;
}

} // namespace mvs
} // namespace xjw
