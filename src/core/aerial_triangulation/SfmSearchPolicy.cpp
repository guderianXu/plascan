#include "SfmSearchPolicy.h"

#include <algorithm>
#include <cmath>

namespace xjw::sfm_search
{

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
    if (candidate.registeredImages != reference.registeredImages)
    {
        return candidate.registeredImages > reference.registeredImages;
    }
    if (candidate.success != reference.success)
    {
        return candidate.success;
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
    // 覆盖普通视场与转台/长焦影像。Middlebury dino 的真实焦距约为最长边的 5.17 倍。
    return {1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 4.0, 5.2, 6.4};
}

bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud)
{
    return totalImages > 0 &&
           registeredImages >= totalImages &&
           hasProductionSparseCloud;
}

} // namespace xjw::sfm_search
