#pragma once

#include <cstdint>
#include <vector>

namespace xjw::aerial_triangulation
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
    bool hasNetworkQuality = false;
    double medianTriangulationAngleDeg = 0.0;
    double twoViewTrackRatio = 1.0;
    double observationGridCoverage = 0.0;
    bool hasClosedSequenceGeometry = false;
    double sequenceAdjacentDistanceMedian = 0.0;
    double sequenceAdjacentDistanceMaximumRatio = 0.0;
    double sequenceAdjacentDistanceMadRatio = 0.0;
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

} // namespace xjw::aerial_triangulation
