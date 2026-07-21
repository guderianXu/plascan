#include "search/SfmSearchPolicy.h"

#include <algorithm>
#include <cmath>

namespace xjw::aerial_triangulation
{

namespace
{

double networkQualityScore(const SfmCandidateSummary &candidate)
{
    const double angleQuality = std::clamp(candidate.medianTriangulationAngleDeg / 10.0, 0.0, 1.0);
    const double multiViewQuality = std::clamp(1.0 - candidate.twoViewTrackRatio, 0.0, 1.0);
    const double coverageQuality = std::clamp(candidate.observationGridCoverage / 0.25, 0.0, 1.0);
    const double reprojectionQuality = std::isfinite(candidate.meanReprojError)
        ? 1.0 / (1.0 + std::max(0.0, candidate.meanReprojError))
        : 0.0;

    // 三角交会角和多视轨迹决定摄影测量网刚性；空间覆盖和 RMS 用于细化排序。
    return 0.35 * angleQuality +
           0.30 * multiViewQuality +
           0.20 * coverageQuality +
           0.15 * reprojectionQuality;
}

} // namespace

SfmWorkerBudget allocateWorkers(int candidateCount, int totalThreads)
{
    if (candidateCount <= 0)
    {
        return {};
    }

    const int safeThreads = std::max(1, totalThreads);
    const int workerCount = std::min({candidateCount, std::max(1, safeThreads / 8), 4});
    return {workerCount, std::max(1, safeThreads / workerCount)};
}

bool isBetterCandidate(const SfmCandidateSummary &candidate,
                       const SfmCandidateSummary &reference)
{
    if (candidate.success != reference.success)
    {
        return candidate.success;
    }
    if (candidate.registeredImages != reference.registeredImages)
    {
        return candidate.registeredImages > reference.registeredImages;
    }
    if (candidate.hasNetworkQuality != reference.hasNetworkQuality)
    {
        return candidate.hasNetworkQuality;
    }
    if (candidate.hasNetworkQuality)
    {
        const double candidateNetworkScore = networkQualityScore(candidate);
        const double referenceNetworkScore = networkQualityScore(reference);
        if (std::abs(candidateNetworkScore - referenceNetworkScore) > 1e-9)
        {
            return candidateNetworkScore > referenceNetworkScore;
        }
    }
    const bool candidateRmsFinite = std::isfinite(candidate.meanReprojError);
    const bool referenceRmsFinite = std::isfinite(reference.meanReprojError);
    if (candidateRmsFinite != referenceRmsFinite)
    {
        return candidateRmsFinite;
    }
    if (candidateRmsFinite && candidate.meanReprojError != reference.meanReprojError)
    {
        return candidate.meanReprojError < reference.meanReprojError;
    }
    if (candidate.points3D != reference.points3D)
    {
        return candidate.points3D > reference.points3D;
    }
    return candidate.candidateIndex < reference.candidateIndex;
}

std::vector<SfmCandidateSummary> rankCandidates(
    const std::vector<SfmCandidateSummary> &candidates)
{
    std::vector<SfmCandidateSummary> ranked = candidates;
    std::stable_sort(ranked.begin(), ranked.end(), isBetterCandidate);
    return ranked;
}

std::vector<int> replayCandidateIndices(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    int maxReplayCount)
{
    const int count = std::min(
        std::max(0, maxReplayCount),
        static_cast<int>(rankedCandidates.size()));
    std::vector<int> indices;
    indices.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
    {
        indices.push_back(rankedCandidates[static_cast<std::size_t>(i)].candidateIndex);
    }
    return indices;
}

std::vector<double> adaptiveFocalScaleCandidates()
{
    // 覆盖广角、普通视场与转台/长焦影像。Middlebury dino 的真实焦距约为最长边的 5.17 倍。
    return {0.55, 0.70, 0.85, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.2, 6.4};
}

bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud)
{
    return totalImages > 0 &&
           registeredImages >= totalImages &&
           hasProductionSparseCloud;
}

} // namespace xjw::aerial_triangulation
