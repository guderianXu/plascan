// =============================================================================
// 文件: DenseCloudBuilder.cpp
// 模块: MVS - 稠密点云构建 (CPU)
// =============================================================================

#include "DenseCloudBuilder.h"
#include "spatial/KDTree3D.h"
#include "log/Logger.h"
#include <fstream>
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
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
    {
        if (errorMsg)
        {
            *errorMsg = "无法创建文件: " + path;
        }
        return false;
    }

    // ASCII PLY header
    ofs << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << cloud.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";

#pragma pack(push, 1)
    struct PLYVertex
    {
        float x;
        float y;
        float z;
        uint8_t r;
        uint8_t g;
        uint8_t b;
    };
#pragma pack(pop)
    static_assert(sizeof(PLYVertex) == 15, "PLYVertex must be 15 bytes (no padding)");
    for (const auto &pt : cloud)
    {
        PLYVertex v{pt.x, pt.y, pt.z, pt.r, pt.g, pt.b};
        ofs.write(reinterpret_cast<const char*>(&v), sizeof(PLYVertex));
    }

    return ofs.good();
}

// =============================================================================
// 辅助：将 DensePoint 数组打包为紧凑 float[N*3] 供 common/spatial/KDTree3D 使用
// =============================================================================
namespace
{
static std::vector<float> packCoords(const std::vector<DensePoint> &cloud)
{
    std::vector<float> coords(cloud.size() * 3);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
        coords[i * 3 + 0] = cloud[i].x;
        coords[i * 3 + 1] = cloud[i].y;
        coords[i * 3 + 2] = cloud[i].z;
    }
    return coords;
}
} // anonymous namespace

// =============================================================================
// Statistical Outlier Removal  (KD-tree + OpenMP 并行)
// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::statisticalOutlierRemoval(
    const std::vector<DensePoint> &cloud,
    int   kNeighbors,
    float stdRatio)
{
    if (cloud.size() < (size_t)kNeighbors + 1)
    {
        return cloud;
    }

    const int N = (int)cloud.size();
    auto t0 = std::chrono::steady_clock::now();

#ifdef HAS_OPENMP
    int numThreads = omp_get_max_threads();
#else
    int numThreads = 1;
#endif
    LOG_INFO("[SOR] 开始统计离群点过滤: %d 点, k=%d, stdRatio=%.2f, threads=%d",
             N, kNeighbors, stdRatio, numThreads);

    // ── 建 KD-tree ─────────────────────────────────────────────────────────
    auto coords = packCoords(cloud);
    common::spatial::KDTree3D tree;
    tree.build(coords.data(), N);

    auto t1 = std::chrono::steady_clock::now();
    LOG_DEBUG("[SOR] KD-tree 建树完成, 耗时 %.3f s",
              std::chrono::duration<double>(t1 - t0).count());

    // ── 并行计算每个点的 kNN 平均距离 ──────────────────────────────────────
    std::vector<float> meanDists(N);
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
#endif
    for (int i = 0; i < N; ++i)
    {
        meanDists[i] = tree.knnMeanDist(i, kNeighbors);
    }

    auto t2 = std::chrono::steady_clock::now();
    LOG_DEBUG("[SOR] kNN 查询完成, 耗时 %.3f s",
              std::chrono::duration<double>(t2 - t1).count());

    // ── 计算全局均值和标准差 ────────────────────────────────────────────────
    double sum = 0, sum2 = 0;
    for (float d : meanDists)
    {
        sum += d;
        sum2 += (double)d * d;
    }
    float globalMean = (float)(sum / N);
    float globalStd  = (float)std::sqrt(sum2 / N - (double)globalMean * globalMean);
    float threshold  = globalMean + stdRatio * globalStd;

    LOG_DEBUG("[SOR] 全局 kNN 距离: mean=%.6f std=%.6f threshold=%.6f",
              globalMean, globalStd, threshold);

    // ── 并行标记 + 收集 ────────────────────────────────────────────────────
    std::vector<uint8_t> keep(N, 0);
    int removed = 0;
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:removed)
#endif
    for (int i = 0; i < N; ++i)
    {
        if (meanDists[i] <= threshold)
        {
            keep[i] = 1;
        }
        else
        {
            ++removed;
        }
    }

    std::vector<DensePoint> result;
    result.reserve(N - removed);
    for (int i = 0; i < N; ++i)
    {
        if (keep[i])
        {
            result.push_back(cloud[i]);
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    LOG_INFO("[SOR] 过滤完成: %d → %d 点 (移除 %d, %.1f%%) 总耗时 %.3f s",
             N, (int)result.size(), removed, 100.f * removed / N,
             std::chrono::duration<double>(t3 - t0).count());
    return result;
}

// =============================================================================
// Radius Outlier Removal  (KD-tree + OpenMP 并行)
// =============================================================================
std::vector<DensePoint> DenseCloudBuilder::radiusOutlierRemoval(
    const std::vector<DensePoint> &cloud,
    float radius,
    int   minNeighbors)
{
    if (cloud.empty()) return cloud;

    const int N = (int)cloud.size();
    auto t0 = std::chrono::steady_clock::now();

#ifdef HAS_OPENMP
    int numThreads = omp_get_max_threads();
#else
    int numThreads = 1;
#endif
    LOG_INFO("[RadiusOR] 开始半径离群点过滤: %d 点, radius=%.4f, minNeighbors=%d, threads=%d",
             N, radius, minNeighbors, numThreads);

    // ── 建 KD-tree ─────────────────────────────────────────────────────────
    auto coords = packCoords(cloud);
    common::spatial::KDTree3D tree;
    tree.build(coords.data(), N);

    auto t1 = std::chrono::steady_clock::now();
    LOG_DEBUG("[RadiusOR] KD-tree 建树完成, 耗时 %.3f s",
              std::chrono::duration<double>(t1 - t0).count());

    // ── 并行半径查询 ───────────────────────────────────────────────────────
    std::vector<uint8_t> keep(N, 0);
    int removed = 0;
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic, 256) reduction(+:removed)
#endif
    for (int i = 0; i < N; ++i) {
        int cnt = tree.radiusCount(i, radius, minNeighbors);
        if (cnt >= minNeighbors)
            keep[i] = 1;
        else
            ++removed;
    }

    std::vector<DensePoint> result;
    result.reserve(N - removed);
    for (int i = 0; i < N; ++i)
        if (keep[i]) result.push_back(cloud[i]);

    auto t2 = std::chrono::steady_clock::now();
    LOG_INFO("[RadiusOR] 过滤完成: %d → %d 点 (移除 %d, %.1f%%) 耗时 %.3f s",
             N, (int)result.size(), removed, 100.f * removed / N,
             std::chrono::duration<double>(t2 - t0).count());
    return result;
}

} // namespace mvs
} // namespace xjw