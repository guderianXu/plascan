// SparseCloudPreprocessor.cpp
#include "SparseCloudPreprocessor.h"
#include <plapoint/search/kdtree.h>
#include <plapoint/core/point_cloud.h>
#include "log/Logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mvs
{

/**
 * @brief 从文本点云文件加载 XYZ 坐标。
 *
 * 支持简单 `.xyz` 行格式与带 header 的 ASCII `.ply`。
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
        std::string line;
        while (std::getline(ifs, line))
        {
            if (line == "end_header")
            {
                break;
            }
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
// KD-tree 加速的离群点过滤 (SOR + 半径过滤，支持任意规模点云)
// =============================================================================
void SparseCloudPreprocessor::filterOutliers(
    std::vector<std::array<float,3>> &pts, float radius, int minNeigh)
{
    if (pts.size() < 10 || radius <= 0)
    {
        return;
    }

    const int N = (int)pts.size();
    auto t0 = std::chrono::steady_clock::now();

    LOG_INFO("[SparseFilter] 开始稀疏点云离群点过滤: %d 点", N);

    // ── 建 KD-tree (plapoint) ───────────────────────────────────────────────
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> ptsMat(N, 3);
    for (int i = 0; i < N; ++i)
    {
        ptsMat(i, 0) = pts[static_cast<size_t>(i)][0];
        ptsMat(i, 1) = pts[static_cast<size_t>(i)][1];
        ptsMat(i, 2) = pts[static_cast<size_t>(i)][2];
    }
    auto pc = std::make_shared<plapoint::PointCloud<float, plamatrix::Device::CPU>>(std::move(ptsMat));
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(pc);
    tree.build();

    auto t1 = std::chrono::steady_clock::now();
    LOG_DEBUG("[SparseFilter] KD-tree 建树完成, 耗时 %.3f s",
              std::chrono::duration<double>(t1 - t0).count());

    // ── 第一步: SOR (Statistical Outlier Removal) ──────────────────────────
    // 使用更多近邻(20)使均值更稳定，并用 IQR 替代 mean+k*std，对重尾分布更鲁棒
    const int kNeighbors = std::min(20, N - 1);  // 12→20，均值更稳定

    std::vector<float> meanDists(N);
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int i = 0; i < N; ++i)
    {
        plamatrix::Vec3<float> query{pts[static_cast<size_t>(i)][0], pts[static_cast<size_t>(i)][1], pts[static_cast<size_t>(i)][2]};
        auto neighbors = tree.nearestKSearch(query, kNeighbors);
        float sum = 0.0f;
        int actualCount = 0;
        for (int nb : neighbors)
        {
            float dx = pts[static_cast<size_t>(i)][0] - pts[static_cast<size_t>(nb)][0];
            float dy = pts[static_cast<size_t>(i)][1] - pts[static_cast<size_t>(nb)][1];
            float dz = pts[static_cast<size_t>(i)][2] - pts[static_cast<size_t>(nb)][2];
            sum += std::sqrt(dx * dx + dy * dy + dz * dz);
            ++actualCount;
        }
        meanDists[i] = (actualCount > 0) ? (sum / static_cast<float>(actualCount)) : 1e9f;
    }

    // ── IQR 鲁棒阈值（替代 mean+k*std，不受极远点拉偏影响）─────────────
    std::vector<float> sortedForIQR = meanDists;
    std::sort(sortedForIQR.begin(), sortedForIQR.end());
    float q1 = sortedForIQR[N / 4];
    float q3 = sortedForIQR[3 * N / 4];
    float iqr = q3 - q1;
    float sorThreshold = q3 + 1.5f * iqr;   // 系数越小越严格；1.5 是 Tukey 标准值

    LOG_DEBUG("[SparseFilter] SOR(IQR): Q1=%.4f Q3=%.4f IQR=%.4f threshold=%.4f",
              q1, q3, iqr, sorThreshold);

    // ── 第二步: 半径过滤 (使用自适应半径) ──────────────────────────────────
    // 传入的固定 radius 可能与点云实际尺度不匹配（例如 BA 结果坐标单位很大），
    // 因此当固定半径远小于点间距中位数时，自动切换为自适应半径。
    float medianKnnDist = sortedForIQR[N / 2];
    float adaptiveRadius = radius;
    if (adaptiveRadius <= 0 || adaptiveRadius < medianKnnDist * 0.5f ||
        adaptiveRadius > sorThreshold * 5.f)
    {
        // 自适应: 3× 中位数 kNN 距离（保证大部分点在该半径内有足够邻居）
        adaptiveRadius = medianKnnDist * 3.0f;
        LOG_WARN("[SparseFilter] 固定半径 %.4f 与点间距 (median=%.4f) 不匹配，使用自适应半径: %.4f",
                 radius, medianKnnDist, adaptiveRadius);
    }

    // ── 联合标记 ───────────────────────────────────────────────────────────
    std::vector<bool> keep(N, false);
    int sorRemoved = 0, radRemoved = 0;
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 64) reduction(+:sorRemoved,radRemoved)
#endif
    for (int i = 0; i < N; ++i)
    {
        // SOR 过滤
        if (meanDists[i] > sorThreshold)
        {
            ++sorRemoved;
            continue;
        }
        // 半径过滤
        plamatrix::Vec3<float> rquery{pts[static_cast<size_t>(i)][0], pts[static_cast<size_t>(i)][1], pts[static_cast<size_t>(i)][2]};
        int cnt = static_cast<int>(tree.radiusSearch(rquery, adaptiveRadius).size());
        if (cnt < minNeigh)
        {
            ++radRemoved;
            continue;
        }
        keep[i] = true;
    }

    // 收集结果
    std::vector<std::array<float,3>> filtered;
    filtered.reserve(N - sorRemoved - radRemoved);
    for (int i = 0; i < N; ++i)
    {
        if (keep[i])
        {
            filtered.push_back(pts[i]);
        }
    }
    pts = std::move(filtered);

    auto t2 = std::chrono::steady_clock::now();
    LOG_INFO("[SparseFilter] 过滤完成: %d → %zu 点 (SOR移除 %d, 半径移除 %d) 耗时 %.3f s",
             N, pts.size(), sorRemoved, radRemoved,
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
    filterOutliers(pts, -1.0f, 3);   // radius=-1 强制自适应, minNeigh=3
    // 第二轮: 仅当点数充足时再过滤（小规模点云二次过滤会过于激进）
    if (pts.size() > 200)
    {
        filterOutliers(pts, -1.0f, 4);  // 第二轮自适应半径, minNeigh=4
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
