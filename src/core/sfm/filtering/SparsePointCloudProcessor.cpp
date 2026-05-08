#include "SparsePointCloudProcessor.h"

#include "math/Vec3Ops.h"
#include "KDTree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace xjw {
namespace {

using Tree3 = xjw::common::spatial::KDTree<3, double>;
using NeighborList = std::vector<Tree3::Neighbor>;

Tree3 buildTree(const std::vector<SparsePointCloudPoint> &pts)
{
    std::vector<Tree3::CoordinateArray> coords;
    coords.reserve(pts.size());
    for (const auto &p : pts)
        coords.push_back({p.x, p.y, p.z});
    return Tree3(coords);
}

std::array<double, 3> pointToArray(const SparsePointCloudPoint &point)
{
    return {point.x, point.y, point.z};
}

std::vector<NeighborList> buildKnnCache(const Tree3 &tree, int k)
{
    std::vector<NeighborList> knnCache(tree.size());
    for (size_t i = 0; i < tree.size(); ++i)
    {
        knnCache[i] = tree.kNearestByPointIndex(i, static_cast<size_t>(k));
    }
    return knnCache;
}

int filterByMaxReprojError(std::vector<SparsePointCloudPoint> *points,
                           double maxReprojError)
{
    if (!points || maxReprojError <= 0.0)
    {
        return 0;
    }

    const int before = static_cast<int>(points->size());
    points->erase(std::remove_if(points->begin(), points->end(), [maxReprojError](const SparsePointCloudPoint &point)
    {
        return point.rmsReprojPx > maxReprojError;
    }), points->end());
    return before - static_cast<int>(points->size());
}

int filterByMinTrackLength(std::vector<SparsePointCloudPoint> *points,
                           int minTrackLength)
{
    if (!points || minTrackLength <= 0)
    {
        return 0;
    }

    const int before = static_cast<int>(points->size());
    points->erase(std::remove_if(points->begin(), points->end(), [minTrackLength](const SparsePointCloudPoint &point)
    {
        return point.trackLen < minTrackLength;
    }), points->end());
    return before - static_cast<int>(points->size());
}

int filterByMinTriAngle(std::vector<SparsePointCloudPoint> *points,
                        double minTriAngleDeg)
{
    if (!points || minTriAngleDeg <= 0.0)
    {
        return 0;
    }

    const int before = static_cast<int>(points->size());
    points->erase(std::remove_if(points->begin(), points->end(), [minTriAngleDeg](const SparsePointCloudPoint &point)
    {
        return point.minTriAngleDeg < minTriAngleDeg;
    }), points->end());
    return before - static_cast<int>(points->size());
}

int filterByStatisticalOutlier(std::vector<SparsePointCloudPoint> *points,
                               const std::vector<NeighborList> &knnCache,
                               int k,
                               double stdDevMul)
{
    if (!points || points->size() < 3 || k < 2)
    {
        return 0;
    }

    const int n = static_cast<int>(points->size());

    std::vector<double> meanDists(n, 0.0);
    std::vector<bool> valid(n, false);

    for (int i = 0; i < n; ++i)
    {
        const auto &neighbors = knnCache[static_cast<size_t>(i)];
        const size_t usedNeighbors = std::min(neighbors.size(), static_cast<size_t>(k));
        if (usedNeighbors == 0)
        {
            continue;
        }
        double sum = 0.0;
        for (size_t ni = 0; ni < usedNeighbors; ++ni)
        {
            const auto &nb = neighbors[ni];
            sum += std::sqrt(nb.distanceSquared);
        }
        meanDists[i] = sum / static_cast<double>(usedNeighbors);
        valid[i] = true;
    }

    double globalMean = 0.0;
    int validCount = 0;
    for (int i = 0; i < n; ++i)
    {
        if (valid[i]) { globalMean += meanDists[i]; ++validCount; }
    }
    if (validCount < 3) return 0;
    globalMean /= validCount;

    double variance = 0.0;
    for (int i = 0; i < n; ++i)
    {
        if (!valid[i]) continue;
        const double delta = meanDists[i] - globalMean;
        variance += delta * delta;
    }
    variance /= validCount;
    const double threshold = globalMean + stdDevMul * std::sqrt(std::max(variance, 0.0));

    const int before = n;
    std::vector<SparsePointCloudPoint> filtered;
    filtered.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        if (!valid[i] || meanDists[i] <= threshold)
            filtered.push_back((*points)[i]);
    }
    *points = std::move(filtered);
    return before - static_cast<int>(points->size());
}

std::array<double, 3> estimateLocalNormal(const std::vector<SparsePointCloudPoint> &points,
                                          int index,
                                          const NeighborList &neighbors)
{
    if (neighbors.size() < 2)
    {
        return {0.0, 0.0, 0.0};
    }

    const SparsePointCloudPoint &center = points.at(index);
    for (int fi = 0; fi < static_cast<int>(neighbors.size()); ++fi)
    {
        for (int si = fi + 1; si < static_cast<int>(neighbors.size()); ++si)
        {
            const std::array<double, 3> vecA = vec3::subtract(pointToArray(points.at(neighbors.at(fi).index)),
                                                              pointToArray(center));
            const std::array<double, 3> vecB = vec3::subtract(pointToArray(points.at(neighbors.at(si).index)),
                                                              pointToArray(center));
            std::array<double, 3> normal = vec3::cross(vecA, vecB);
            const double length = vec3::norm(normal);
            if (length > 1e-8)
            {
                normal[0] /= length;
                normal[1] /= length;
                normal[2] /= length;
                return normal;
            }
        }
    }

    return {0.0, 0.0, 0.0};
}

int filterByNormalConsistency(std::vector<SparsePointCloudPoint> *points,
                              const std::vector<NeighborList> &knnCache,
                              int k,
                              double minMeanAbsDot)
{
    if (!points || points->size() < 6 || k < 3)
    {
        return 0;
    }

    std::vector<std::array<double, 3>> normals;
    normals.reserve(points->size());
    for (int i = 0; i < static_cast<int>(points->size()); ++i)
    {
        normals.push_back(estimateLocalNormal(*points, i, knnCache[static_cast<size_t>(i)]));
    }

    const int before = static_cast<int>(points->size());
    std::vector<SparsePointCloudPoint> filtered;
    filtered.reserve(points->size());
    for (int i = 0; i < static_cast<int>(points->size()); ++i)
    {
        if (vec3::norm(normals.at(i)) <= 1e-8)
        {
            filtered.push_back(points->at(i));
            continue;
        }

        double sumAbsDot = 0.0;
        int validCount = 0;
        const auto &neighbors = knnCache[static_cast<size_t>(i)];
        const size_t usedNeighbors = std::min(neighbors.size(), static_cast<size_t>(k));
        for (size_t ni = 0; ni < usedNeighbors; ++ni)
        {
            const auto &nb = neighbors[ni];
            if (vec3::norm(normals.at(static_cast<std::size_t>(nb.index))) <= 1e-8)
            {
                continue;
            }
            sumAbsDot += std::abs(vec3::dot(normals.at(i), normals.at(static_cast<std::size_t>(nb.index))));
            ++validCount;
        }

        if (validCount < 3 || (sumAbsDot / validCount) >= minMeanAbsDot)
        {
            filtered.push_back(points->at(i));
        }
    }

    *points = std::move(filtered);
    return before - static_cast<int>(points->size());
}

/**
 * @brief 半径密度过滤：副除李石封0点。
 *
 * 对每个点统计其半径 radius 内的邓居数（自身除外），
 * 少于 minNeighbors 则删除该点。
 */
int filterByDensityRadius(std::vector<SparsePointCloudPoint> *points,
                          const Tree3 &tree,
                          double radius,
                          int minNeighbors)
{
    if (!points || points->size() < 2 || radius <= 0.0 || minNeighbors < 1)
    {
        return 0;
    }

    const int n = static_cast<int>(points->size());

    std::vector<SparsePointCloudPoint> filtered;
    filtered.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        // radiusCountByPointIndex 已排除自身
        const int cnt = tree.radiusCountByPointIndex(
            static_cast<std::size_t>(i), radius, minNeighbors + 1);
        if (cnt >= minNeighbors)
            filtered.push_back((*points)[i]);
    }
    const int removed = n - static_cast<int>(filtered.size());
    *points = std::move(filtered);
    return removed;
}

std::vector<bool> computeStatisticalKeepMask(const std::vector<SparsePointCloudPoint> &points,
                                             const std::vector<NeighborList> &knnCache,
                                             int k,
                                             double stdDevMul,
                                             int *removed)
{
    std::vector<bool> keep(points.size(), true);
    if (removed)
    {
        *removed = 0;
    }
    if (points.size() < 3 || k < 2)
    {
        return keep;
    }

    const int n = static_cast<int>(points.size());
    std::vector<double> meanDists(n, 0.0);
    std::vector<bool> valid(n, false);
    for (int i = 0; i < n; ++i)
    {
        const auto &neighbors = knnCache[static_cast<size_t>(i)];
        const size_t usedNeighbors = std::min(neighbors.size(), static_cast<size_t>(k));
        if (usedNeighbors == 0)
        {
            continue;
        }
        double sum = 0.0;
        for (size_t ni = 0; ni < usedNeighbors; ++ni)
        {
            sum += std::sqrt(neighbors[ni].distanceSquared);
        }
        meanDists[i] = sum / static_cast<double>(usedNeighbors);
        valid[i] = true;
    }

    double globalMean = 0.0;
    int validCount = 0;
    for (int i = 0; i < n; ++i)
    {
        if (valid[i])
        {
            globalMean += meanDists[i];
            ++validCount;
        }
    }
    if (validCount < 3)
    {
        return keep;
    }
    globalMean /= validCount;

    double variance = 0.0;
    for (int i = 0; i < n; ++i)
    {
        if (!valid[i])
        {
            continue;
        }
        const double delta = meanDists[i] - globalMean;
        variance += delta * delta;
    }
    variance /= validCount;
    const double threshold = globalMean + stdDevMul * std::sqrt(std::max(variance, 0.0));

    for (int i = 0; i < n; ++i)
    {
        if (valid[i] && meanDists[i] > threshold)
        {
            keep[static_cast<size_t>(i)] = false;
            if (removed)
            {
                ++(*removed);
            }
        }
    }

    return keep;
}

std::vector<bool> computeNormalConsistencyKeepMask(const std::vector<SparsePointCloudPoint> &points,
                                                   const std::vector<NeighborList> &knnCache,
                                                   int k,
                                                   double minMeanAbsDot,
                                                   int *removed)
{
    std::vector<bool> keep(points.size(), true);
    if (removed)
    {
        *removed = 0;
    }
    if (points.size() < 6 || k < 3)
    {
        return keep;
    }

    std::vector<std::array<double, 3>> normals;
    normals.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        normals.push_back(estimateLocalNormal(points, static_cast<int>(i), knnCache[i]));
    }

    for (size_t i = 0; i < points.size(); ++i)
    {
        if (vec3::norm(normals[i]) <= 1e-8)
        {
            continue;
        }

        const auto &neighbors = knnCache[i];
        const size_t usedNeighbors = std::min(neighbors.size(), static_cast<size_t>(k));
        double sumAbsDot = 0.0;
        int validCount = 0;
        for (size_t ni = 0; ni < usedNeighbors; ++ni)
        {
            const auto &nb = neighbors[ni];
            if (vec3::norm(normals[static_cast<size_t>(nb.index)]) <= 1e-8)
            {
                continue;
            }
            sumAbsDot += std::abs(vec3::dot(normals[i], normals[static_cast<size_t>(nb.index)]));
            ++validCount;
        }

        if (validCount >= 3 && (sumAbsDot / validCount) < minMeanAbsDot)
        {
            keep[i] = false;
            if (removed)
            {
                ++(*removed);
            }
        }
    }

    return keep;
}

std::vector<bool> computeDensityKeepMask(const std::vector<SparsePointCloudPoint> &points,
                                         const Tree3 &tree,
                                         double radius,
                                         int minNeighbors,
                                         int *removed)
{
    std::vector<bool> keep(points.size(), true);
    if (removed)
    {
        *removed = 0;
    }
    if (points.size() < 2 || radius <= 0.0 || minNeighbors < 1)
    {
        return keep;
    }

    for (size_t i = 0; i < points.size(); ++i)
    {
        const int count = tree.radiusCountByPointIndex(i, radius, minNeighbors + 1);
        if (count < minNeighbors)
        {
            keep[i] = false;
            if (removed)
            {
                ++(*removed);
            }
        }
    }

    return keep;
}

} // namespace

SparsePointCloudFilterStats SparsePointCloudProcessor::filter(std::vector<SparsePointCloudPoint> *points,
                                                              const SparsePointCloudFilterOptions &options)
{
    SparsePointCloudFilterStats stats;
    if (!points)
    {
        return stats;
    }

    stats.inputPoints = static_cast<int>(points->size());

    // 辅助宏：若过滤后点集变空，回滚当前步（防止单一阈值误配置清空全部点）
    // 各步骤仍按顺序执行，互相叠加，但不会因为某一步配置过严而清空整个点云。
    auto safeFilter = [&](auto &&filterFn) -> int
    {
        const std::size_t prevSize = points->size();
        if (prevSize == 0)
            return 0;
        std::vector<SparsePointCloudPoint> backup = *points;
        const int removed = filterFn();
        if (points->empty())
        {
            *points = std::move(backup); // 回滚
            return 0;
        }
        return removed;
    };

    if (options.filterByReprojError)
    {
        stats.removedByReprojError = safeFilter([&]{ return filterByMaxReprojError(points, options.maxReprojError); });
    }
    if (options.filterByTrackLen)
    {
        stats.removedByTrackLen = safeFilter([&]{ return filterByMinTrackLength(points, options.minTrackLen); });
    }
    if (options.filterByTriAngle)
    {
        stats.removedByTriAngle = safeFilter([&]{ return filterByMinTriAngle(points, options.minTriAngleDeg); });
    }

    const bool needsSpatialFilter = options.filterByStatistical || options.filterByNormalConsistency ||
                                    options.filterByDensity;
    if (needsSpatialFilter && points->size() >= 2)
    {
        const Tree3 tree = buildTree(*points);
        const int normalK = std::max(6, std::min(options.statK, 20));
        const int cacheK = std::max(options.statK, normalK);
        const std::vector<NeighborList> knnCache = buildKnnCache(tree, cacheK);

        std::vector<bool> keep(points->size(), true);
        if (options.filterByStatistical)
        {
            const auto statKeep = computeStatisticalKeepMask(*points, knnCache, options.statK, options.statStdDevMul,
                                                             &stats.removedByStatistical);
            for (size_t i = 0; i < keep.size(); ++i)
            {
                keep[i] = keep[i] && statKeep[i];
            }
        }
        if (options.filterByNormalConsistency)
        {
            const auto normalKeep = computeNormalConsistencyKeepMask(
                *points, knnCache, normalK, options.normalConsistencyMinMeanAbsDot, &stats.removedByNormalConsistency);
            for (size_t i = 0; i < keep.size(); ++i)
            {
                keep[i] = keep[i] && normalKeep[i];
            }
        }
        if (options.filterByDensity)
        {
            const auto densityKeep = computeDensityKeepMask(*points, tree, options.densityRadius,
                                                            options.densityMinNeighbors, &stats.removedByDensity);
            for (size_t i = 0; i < keep.size(); ++i)
            {
                keep[i] = keep[i] && densityKeep[i];
            }
        }

        std::vector<SparsePointCloudPoint> filtered;
        filtered.reserve(points->size());
        for (size_t i = 0; i < points->size(); ++i)
        {
            if (keep[i])
            {
                filtered.push_back(points->at(i));
            }
        }
        *points = std::move(filtered);
    }

    stats.outputPoints = static_cast<int>(points->size());
    return stats;
}

SparsePointCloudRefineResult SparsePointCloudProcessor::refine(const std::vector<SparsePointCloudPoint> &basePoints,
                                                               const SparsePointCloudRefineOptions &options)
{
    SparsePointCloudFilterOptions filterOptions;
    filterOptions.maxReprojError = options.maxReprojError;
    filterOptions.minTrackLen = options.minTrackLen;
    filterOptions.minTriAngleDeg = options.minTriAngleDeg;
    filterOptions.statK = options.knnNeighbors;
    filterOptions.statStdDevMul = options.stdDevMultiplier;
    filterOptions.filterByNormalConsistency = options.normalConsistency;
    filterOptions.normalConsistencyMinMeanAbsDot = options.normalConsistencyMinMeanAbsDot;

    SparsePointCloudOptimizeOptions optimizeOptions;
    optimizeOptions.filterOptions = filterOptions;
    optimizeOptions.iterative = true;
    optimizeOptions.iterRounds = options.iterRounds;
    optimizeOptions.restartFromInputEachRound = options.retriangulate;
    optimizeOptions.tightenThresholds = true;

    const SparsePointCloudOptimizeResult optimizeResult = optimize(basePoints, optimizeOptions);

    SparsePointCloudRefineResult result;
    result.inputPoints = optimizeResult.inputPoints;
    result.outputPoints = optimizeResult.outputPoints;
    result.removedTotal = optimizeResult.removedTotal;
    result.points = optimizeResult.points;
    result.rounds.reserve(optimizeResult.rounds.size());
    for (const auto &round : optimizeResult.rounds)
    {
        SparsePointCloudRefineRound refineRound;
        refineRound.round = round.round;
        refineRound.inputPoints = round.inputPoints;
        refineRound.maxReprojError = round.filterOptions.maxReprojError;
        refineRound.minTriAngleDeg = round.filterOptions.minTriAngleDeg;
        refineRound.minTrackLen = round.filterOptions.minTrackLen;
        refineRound.statStdDevMul = round.filterOptions.statStdDevMul;
        refineRound.stats = round.filterStats;
        result.rounds.push_back(refineRound);
    }
    return result;
}

SparsePointCloudOptimizeResult SparsePointCloudProcessor::optimize(const std::vector<SparsePointCloudPoint> &basePoints,
                                                                   const SparsePointCloudOptimizeOptions &options)
{
    SparsePointCloudOptimizeResult result;
    result.inputPoints = static_cast<int>(basePoints.size());
    if (basePoints.empty())
    {
        return result;
    }

    std::vector<SparsePointCloudPoint> currentPoints = basePoints;
    std::vector<SparsePointCloudPoint> lastGoodPoints = basePoints; // 保底：某轮全部剔除时回退
    const int rounds = std::max(1, options.iterative ? options.iterRounds : 1);
    for (int roundIndex = 0; roundIndex < rounds; ++roundIndex)
    {
        std::vector<SparsePointCloudPoint> working =
            (options.restartFromInputEachRound && roundIndex > 0) ? basePoints : currentPoints;

        SparsePointCloudOptimizeRound round;
        round.round = roundIndex + 1;
        round.inputPoints = static_cast<int>(working.size());
        round.filterOptions = options.filterOptions;

        if (options.tightenThresholds)
        {
            round.filterOptions.maxReprojError =
                std::max(0.5, round.filterOptions.maxReprojError * (1.0 - 0.08 * roundIndex));
            round.filterOptions.minTriAngleDeg = round.filterOptions.minTriAngleDeg + 0.4 * roundIndex;
            // minTrackLen 是点的固有属性，无需逐轮收紧，保持用户设定值不变
            round.filterOptions.statStdDevMul =
                std::max(0.8, round.filterOptions.statStdDevMul - 0.2 * roundIndex);
        }

        round.filterStats = filter(&working, round.filterOptions);
        if (options.enableSpatialCleanup && !working.empty())
        {
            spatialCleanup(&working, options.spatialCleanupOptions);
        }

        round.outputPoints = static_cast<int>(working.size());
        currentPoints = std::move(working);
        result.rounds.push_back(round);
        if (currentPoints.empty())
        {
            // 本轮过滤结果为空：使用上一轮最佳结果，终止迭代
            currentPoints = lastGoodPoints;
            break;
        }
        lastGoodPoints = currentPoints;
    }

    result.points = std::move(currentPoints);
    result.outputPoints = static_cast<int>(result.points.size());
    result.removedTotal = result.inputPoints - result.outputPoints;
    return result;
}

// ============================================================
// SparsePointCloudProcessor::spatialCleanup
// ============================================================

SparsePointCloudSpatialCleanupResult SparsePointCloudProcessor::spatialCleanup(
    std::vector<SparsePointCloudPoint> *points,
    const SparsePointCloudSpatialCleanupOptions &options)
{
    SparsePointCloudSpatialCleanupResult result;
    if (!points || points->empty())
    {
        return result;
    }
    result.inputPoints = static_cast<int>(points->size());

    const Tree3 tree = buildTree(*points);
    const int n = static_cast<int>(points->size());

    // ── 自动估算体素边长 ─────────────────────────────────────
    double voxelSize = options.voxelSize;
    if (voxelSize <= 0.0)
    {
        std::vector<double> nn1Dists;
        nn1Dists.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            auto nbs = tree.kNearestByPointIndex(static_cast<std::size_t>(i), 1);
            if (!nbs.empty())
                nn1Dists.push_back(std::sqrt(nbs[0].distanceSquared));
        }
        if (!nn1Dists.empty())
        {
            auto mid = nn1Dists.begin() + static_cast<std::ptrdiff_t>(nn1Dists.size() / 2);
            std::nth_element(nn1Dists.begin(), mid, nn1Dists.end());
            voxelSize = *mid * 3.0;
        }
    }

    if (voxelSize <= 0.0)
    {
        result.outputPoints = result.inputPoints;
        return result;
    }

    // ── 体素分组 ───────────────────────────────────────────────
    using VoxelKey = std::tuple<int64_t, int64_t, int64_t>;
    std::map<VoxelKey, std::vector<int>> voxelMap;
    for (int i = 0; i < n; ++i)
    {
        const auto &p = (*points)[i];
        const auto vx = static_cast<int64_t>(std::floor(p.x / voxelSize));
        const auto vy = static_cast<int64_t>(std::floor(p.y / voxelSize));
        const auto vz = static_cast<int64_t>(std::floor(p.z / voxelSize));
        voxelMap[{vx, vy, vz}].push_back(i);
    }

    std::vector<bool> removed(static_cast<std::size_t>(n), false);

    // ── 孤岛体素剔除 + 体素内局部重投影过滤 ────────────────────
    for (auto &kv : voxelMap)
    {
        const auto &indices = kv.second;

        if (static_cast<int>(indices.size()) < options.minVoxelPoints)
        {
            for (int idx : indices)
            {
                removed[static_cast<std::size_t>(idx)] = true;
                ++result.removedByVoxelIsolation;
            }
            continue;
        }

        if (options.localReprojFilter && static_cast<int>(indices.size()) >= 3)
        {
            double sumR = 0.0;
            for (int idx : indices)
                sumR += (*points)[static_cast<std::size_t>(idx)].rmsReprojPx;
            const double meanR = sumR / static_cast<double>(indices.size());

            double varR = 0.0;
            for (int idx : indices)
            {
                const double d = (*points)[static_cast<std::size_t>(idx)].rmsReprojPx - meanR;
                varR += d * d;
            }
            varR /= static_cast<double>(indices.size());
            const double thr = meanR + options.localReprojStdMul * std::sqrt(std::max(varR, 0.0));

            for (int idx : indices)
            {
                if (!removed[static_cast<std::size_t>(idx)] &&
                    (*points)[static_cast<std::size_t>(idx)].rmsReprojPx > thr)
                {
                    removed[static_cast<std::size_t>(idx)] = true;
                    ++result.removedByLocalReproj;
                }
            }
        }
    }

    // ── 几何去重 ───────────────────────────────────────────────
    if (options.deduplicationRadius > 0.0)
    {
        // 构建点的有序下标（按 trackLen 降序，优先保留质量高的点）
        std::vector<int> sortedOrder(static_cast<std::size_t>(n));
        std::iota(sortedOrder.begin(), sortedOrder.end(), 0);
        std::sort(sortedOrder.begin(), sortedOrder.end(), [&](int a, int b) {
            return (*points)[static_cast<std::size_t>(a)].trackLen >
                   (*points)[static_cast<std::size_t>(b)].trackLen;
        });

        // 构建坐标数组（与原始 tree 对应）
        std::vector<Tree3::CoordinateArray> coords;
        coords.reserve(static_cast<std::size_t>(n));
        for (const auto &p : *points)
            coords.push_back({p.x, p.y, p.z});

        for (int si = 0; si < n; ++si)
        {
            const int i = sortedOrder[static_cast<std::size_t>(si)];
            if (removed[static_cast<std::size_t>(i)])
                continue;
            auto nbrs = tree.radiusSearch(coords[static_cast<std::size_t>(i)],
                                          options.deduplicationRadius);
            for (int nbIdx : nbrs)
            {
                if (nbIdx != i && !removed[static_cast<std::size_t>(nbIdx)])
                {
                    removed[static_cast<std::size_t>(nbIdx)] = true;
                    ++result.removedByDeduplication;
                }
            }
        }
    }

    // ── 收集存活点 ─────────────────────────────────────────────
    std::vector<SparsePointCloudPoint> output;
    output.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        if (!removed[static_cast<std::size_t>(i)])
            output.push_back((*points)[static_cast<std::size_t>(i)]);
    }
    *points = std::move(output);
    result.outputPoints = static_cast<int>(points->size());
    return result;
}

SparseCloudLocalOptimResult SparsePointCloudProcessor::localOptim(
    std::vector<SparsePointCloudPoint> *points,
    const SparseCloudLocalOptimOptions &options)
{
    return spatialCleanup(points, options);
}

} // namespace xjw