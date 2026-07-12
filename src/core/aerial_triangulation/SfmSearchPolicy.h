#pragma once

#include <cstdint>
#include <vector>

namespace xjw::sfm_search
{

struct SfmWorkerBudget
{
    int workerCount = 0;
    int threadsPerWorker = 0;
};

struct SfmCandidateSummary
{
    int candidateIndex = -1;
    double focalScale = 0.0;
    std::uint32_t initialImageId1 = 0;
    std::uint32_t initialImageId2 = 0;
    int registeredImages = 0;
    int points3D = 0;
    double meanReprojError = 0.0;
    bool success = false;
};

SfmWorkerBudget allocateWorkers(int candidateCount, int totalThreads);

bool isBetterCandidate(const SfmCandidateSummary &candidate,
                       const SfmCandidateSummary &reference);

std::vector<SfmCandidateSummary> rankCandidates(
    const std::vector<SfmCandidateSummary> &candidates);

std::vector<int> replayCandidateIndices(
    const std::vector<SfmCandidateSummary> &rankedCandidates,
    int maxReplayCount);

std::vector<double> adaptiveFocalScaleCandidates();

bool shouldStopAdaptiveFocalReplay(int totalImages,
                                   int registeredImages,
                                   bool hasProductionSparseCloud);

} // namespace xjw::sfm_search
