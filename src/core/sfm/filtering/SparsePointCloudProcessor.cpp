#include "SparsePointCloudProcessor.h"

#include "SparsePointCloudWorkspace.h"

#include <plamatrix/ops/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace xjw {
namespace {

using Neighbor = SparsePointCloudNeighbor;
using NeighborList = SparsePointCloudNeighborList;

std::array<double, 3> pointToArray(const SparsePointCloudPoint &point)
{
    return {point.x, point.y, point.z};
}

double medianValue(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    double median = *mid;
    if (values.size() % 2 == 0)
    {
        const auto lowerMid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2 - 1);
        std::nth_element(values.begin(), lowerMid, values.end());
        median = 0.5 * (median + *lowerMid);
    }
    return median;
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

std::array<double, 3> estimateLocalNormal(const std::vector<SparsePointCloudPoint> &points,
                                          int index,
                                          const NeighborList &neighbors)
{
    if (neighbors.size() < 2)
    {
        return {0.0, 0.0, 0.0};
    }

    const SparsePointCloudPoint &center = points.at(index);
    const plamatrix::Vec3<double> center_vector(pointToArray(center));
    for (int fi = 0; fi < static_cast<int>(neighbors.size()); ++fi)
    {
        for (int si = fi + 1; si < static_cast<int>(neighbors.size()); ++si)
        {
            const plamatrix::Vec3<double> vector_a =
                plamatrix::Vec3<double>(pointToArray(points.at(neighbors.at(fi).index))) - center_vector;
            const plamatrix::Vec3<double> vector_b =
                plamatrix::Vec3<double>(pointToArray(points.at(neighbors.at(si).index))) - center_vector;
            const plamatrix::Vec3<double> normal = plamatrix::cross(vector_a, vector_b);
            const double length = plamatrix::norm(normal);
            if (length > 1e-8)
            {
                return (normal / length).toArray();
            }
        }
    }

    return {0.0, 0.0, 0.0};
}

std::vector<bool> computeStatisticalKeepMask(const SparsePointCloudWorkspace &workspace,
                                             int k,
                                             double stdDevMul,
                                             plapoint::ProcessingDevice processingDevice,
                                             int *removed)
{
    const std::vector<SparsePointCloudPoint> &points = workspace.points();
    std::vector<bool> keep(points.size(), true);
    if (removed)
    {
        *removed = 0;
    }
    if (points.size() < 3 || k < 2)
    {
        return keep;
    }

    (void)processingDevice;

    const int actualK = std::min<int>(k, static_cast<int>(points.size()) - 1);
    const std::vector<NeighborList> knnCache = workspace.knnCache(actualK);
    std::vector<double> meanDistances;
    meanDistances.reserve(points.size());
    for (const NeighborList &neighbors : knnCache)
    {
        if (neighbors.empty())
        {
            meanDistances.push_back(0.0);
            continue;
        }

        double sum = 0.0;
        for (const Neighbor &neighbor : neighbors)
        {
            sum += std::sqrt(neighbor.distanceSquared);
        }
        meanDistances.push_back(sum / static_cast<double>(neighbors.size()));
    }

    const double medianDistance = medianValue(meanDistances);
    std::vector<double> absDeviations;
    absDeviations.reserve(meanDistances.size());
    for (double distance : meanDistances)
    {
        absDeviations.push_back(std::abs(distance - medianDistance));
    }

    double robustScale = 1.4826 * medianValue(absDeviations);
    if (robustScale <= 1e-12)
    {
        std::vector<double> positiveDeviations;
        positiveDeviations.reserve(absDeviations.size());
        for (double deviation : absDeviations)
        {
            if (deviation > 1e-12)
            {
                positiveDeviations.push_back(deviation);
            }
        }
        robustScale = positiveDeviations.empty() ? 0.0 : 1.4826 * medianValue(positiveDeviations);
    }

    const double threshold = medianDistance + std::max(0.0, stdDevMul) * robustScale;
    int removedCount = 0;
    for (std::size_t i = 0; i < meanDistances.size(); ++i)
    {
        if (robustScale > 1e-12 && meanDistances[i] > threshold)
        {
            keep[i] = false;
        }
    }
    for (bool keepPoint : keep)
    {
        if (!keepPoint)
        {
            ++removedCount;
        }
    }
    if (removed)
    {
        *removed = removedCount;
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
        if (plamatrix::norm(plamatrix::Vec3<double>(normals[i])) <= 1e-8)
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
            if (plamatrix::norm(
                    plamatrix::Vec3<double>(normals[static_cast<size_t>(nb.index)])) <= 1e-8)
            {
                continue;
            }
            sumAbsDot += std::abs(plamatrix::dot(
                plamatrix::Vec3<double>(normals[i]),
                plamatrix::Vec3<double>(normals[static_cast<size_t>(nb.index)])));
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

std::vector<bool> computeDensityKeepMask(const SparsePointCloudWorkspace &workspace,
                                         double radius,
                                         int minNeighbors,
                                         plapoint::ProcessingDevice processingDevice,
                                         int *removed)
{
    const std::vector<SparsePointCloudPoint> &points = workspace.points();
    std::vector<bool> keep(points.size(), true);
    if (removed)
    {
        *removed = 0;
    }
    if (points.size() < 2 || radius <= 0.0 || minNeighbors < 1)
    {
        return keep;
    }

    const std::vector<int> removed_indices =
        workspace.radiusOutlierIndices(radius, minNeighbors + 1, processingDevice);
    for (int index : removed_indices)
    {
        if (index >= 0 && static_cast<std::size_t>(index) < keep.size())
        {
            keep[static_cast<std::size_t>(index)] = false;
        }
    }
    if (removed)
    {
        *removed = static_cast<int>(removed_indices.size());
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
        const SparsePointCloudWorkspace workspace =
            SparsePointCloudWorkspace::fromPoints(*points);
        std::vector<NeighborList> knnCache;
        const int normalK = std::max(6, std::min(options.statK, 20));
        if (options.filterByNormalConsistency)
        {
            knnCache = workspace.knnCache(normalK);
        }

        std::vector<bool> keep(points->size(), true);
        if (options.filterByStatistical)
        {
            const auto statKeep = computeStatisticalKeepMask(workspace, options.statK, options.statStdDevMul,
                                                             options.processingDevice,
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
            const auto densityKeep = computeDensityKeepMask(workspace, options.densityRadius,
                                                            options.densityMinNeighbors,
                                                            options.processingDevice,
                                                            &stats.removedByDensity);
            for (size_t i = 0; i < keep.size(); ++i)
            {
                keep[i] = keep[i] && densityKeep[i];
            }
        }

        *points = workspace.filteredPoints(keep);
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
    filterOptions.processingDevice = options.processingDevice;

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

    const SparsePointCloudWorkspace workspace =
        SparsePointCloudWorkspace::fromPoints(*points);
    const int n = static_cast<int>(points->size());

    // ── 自动估算体素边长 ─────────────────────────────────────
    double voxelSize = options.voxelSize;
    if (voxelSize <= 0.0)
    {
        std::vector<double> nn1Dists;
        nn1Dists.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            const auto query = std::array<double, 3>{
                workspace.cloud().points()(i, 0),
                workspace.cloud().points()(i, 1),
                workspace.cloud().points()(i, 2)
            };
            auto indices = workspace.nearestKSearch(query, 2); // k=2: self + nearest neighbor
            if (indices.size() >= 2)
            {
                int nnIdx = (indices[0] == i) ? indices[1] : indices[0];
                double dx = workspace.cloud().points()(nnIdx, 0) - query[0];
                double dy = workspace.cloud().points()(nnIdx, 1) - query[1];
                double dz = workspace.cloud().points()(nnIdx, 2) - query[2];
                nn1Dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
            }
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

        for (int si = 0; si < n; ++si)
        {
            const int i = sortedOrder[static_cast<std::size_t>(si)];
            if (removed[static_cast<std::size_t>(i)])
                continue;
            const auto &p = (*points)[static_cast<std::size_t>(i)];
            const auto query = std::array<double, 3>{p.x, p.y, p.z};
            auto nbrs = workspace.radiusSearch(query, options.deduplicationRadius);
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
