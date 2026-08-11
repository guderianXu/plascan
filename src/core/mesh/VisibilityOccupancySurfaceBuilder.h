#pragma once

#include "BinaryGridMinCutSolver.h"
#include "FramePinholeCamera.h"

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyFrameView
{
    const FramePinholeCamera *camera = nullptr;
    const cv::Mat *depth = nullptr;
    const cv::Mat *confidence = nullptr;
    const cv::Mat *depthValidMask = nullptr;
    const cv::Mat *supportMask = nullptr;
    float frameWeight = 1.0f;
};

struct VisibilityOccupancyOptions
{
    int resolution = 72;
    std::array<int, 3> sampleDimensions{};
    int minimumVisibleViews = 2;
    int minimumSilhouetteViews = 2;
    int minimumDepthFullViewsForSilhouettePrior = 0;
    int allowedSilhouetteViolations = 1;
    float frontTolerancePixelFootprints = 2.5f;
    float behindSurfaceBandPixelFootprints = 5.0f;
    BinaryGridCapacity depthEmptyCapacity = 18;
    BinaryGridCapacity depthFullCapacity = 22;
    BinaryGridCapacity silhouetteEmptyCapacity = 8;
    BinaryGridCapacity silhouetteFullPriorCapacity = 2;
    BinaryGridCapacity pairwiseCapacity = 6;
    BinaryGridCapacity hardBoundaryCapacity = 1000000;
    bool fillInteriorEmptyBubbles = true;
    bool retainLargestFullComponent = true;
    int closingIterations = 0;
    int closingMinimumDepthEmptyViewsToProtect = 16;
    int closingMinimumSilhouetteOutsideViewsToProtect = 2;
    int maximumHandleRepairPasses = 4;
    int maximumHandleRepairAcceptedCandidateCount = 32;
    std::size_t maximumHandleRepairCandidateSampleCount = 512;
    std::size_t maximumHandleRepairSubsetSampleCount = 96;
    int maximumHandleRepairSubsetSeedCount = 0;
    bool repairNonManifoldConfigurations = false;
    int wellComposedRepairMaximumPasses = 8;
    std::size_t wellComposedRepairMaximumFilledSampleCount = 1024;
    bool buildSignedDistanceSamples = true;
    int workerCount = 0;
    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyStatistics
{
    int effectiveWorkerCount = 1;
    std::int64_t projectionElapsedMs = 0;
    double projectionCpuTimeMs = 0.0;
    double projectionCpuDuty = 0.0;
    std::int64_t minCutElapsedMs = 0;
    double minCutCpuTimeMs = 0.0;
    double minCutCpuDuty = 0.0;
    std::int64_t cleanupElapsedMs = 0;
    double cleanupCpuTimeMs = 0.0;
    double cleanupCpuDuty = 0.0;
    std::uint64_t sampleCount = 0;
    std::uint64_t projectedViewCount = 0;
    std::uint64_t silhouetteInsideVoteCount = 0;
    std::uint64_t silhouetteOutsideVoteCount = 0;
    std::uint64_t silhouetteFullPriorCandidateSampleCount = 0;
    std::uint64_t silhouetteFullPriorSampleCount = 0;
    std::uint64_t silhouetteFullPriorRejectedWithoutDepthSupportSampleCount = 0;
    std::uint64_t silhouetteFullPriorCapacityTotal = 0;
    std::uint64_t depthEmptyVoteCount = 0;
    std::uint64_t depthFullVoteCount = 0;
    std::uint64_t fullSampleCountBeforeCleanup = 0;
    std::uint64_t fullSampleCountAfterCleanup = 0;
    std::uint64_t filledInteriorEmptySampleCount = 0;
    std::uint64_t removedFullDustSampleCount = 0;
    std::uint64_t closingChangedSampleCount = 0;
    std::uint64_t closingProposalAddedSampleCount = 0;
    std::uint64_t closingProposalDepthEmptyAtLeastTwoSampleCount = 0;
    std::uint64_t closingProposalDepthEmptyAtLeastThreeSampleCount = 0;
    std::uint64_t closingProposalDepthEmptyAtLeastFourSampleCount = 0;
    std::uint64_t closingProposalDepthFullSampleCount = 0;
    std::uint64_t closingProposalSilhouetteOutsideAtLeastTwoSampleCount = 0;
    std::uint64_t closingProtectedEmptySampleCount = 0;
    std::uint64_t closingDepthEmptyProtectedSampleCount = 0;
    std::uint64_t closingSilhouetteEmptyProtectedSampleCount = 0;
    std::uint64_t closingPreservedOriginalFullSampleCount = 0;
    int effectiveClosingIterations = 0;
    int handleRepairCandidateComponentCount = 0;
    int handleRepairAcceptedCandidateCount = 0;
    int handleRepairAcceptedSubsetCandidateCount = 0;
    int handleRepairAcceptedPlateauSubsetCandidateCount = 0;
    int handleRepairAttemptedSubsetSeedCount = 0;
    int handleRepairRejectedProtectedCandidateCount = 0;
    int handleRepairRejectedOversizedCandidateCount = 0;
    int handleRepairRejectedTopologyCandidateCount = 0;
    int handleRepairRejectedProtectedReachabilityCandidateCount = 0;
    int handleRepairBodyEulerBefore = 0;
    int handleRepairBodyEulerAfter = 0;
    std::uint64_t wellComposedRepairFilledSampleCount = 0;
    int wellComposedRepairAcceptedPassCount = 0;
    int wellComposedRepairBodyEulerBefore = 0;
    int wellComposedRepairBodyEulerAfter = 0;
    std::uint64_t wellComposedRepairRemainingEdgeCheckerboardCount = 0;
    std::uint64_t
        wellComposedRepairRemainingVertexOccupiedComponentDefectCount = 0;
    std::uint64_t
        wellComposedRepairRemainingVertexEmptyComponentDefectCount = 0;
    BinaryGridMinCutStatistics minCut;
};

struct VisibilityOccupancyResult
{
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::array<int, 3> sampleDimensions{};
    std::array<float, 3> boundsMin{};
    std::array<float, 3> boundsMax{};
    std::vector<std::uint8_t> occupied;
    std::vector<float> signedDistanceSamples;
    bool signedDistanceSamplesAreWorldUnits = false;
    VisibilityOccupancyStatistics statistics;

    std::size_t index(int x, int y, int z) const;
};

/**
 * @brief Builds a globally closed occupancy field from depth-ray visibility.
 *
 * Invalid depth and pixels outside a camera field of view stay unknown.
 * Foreground-mask violations vote Empty, camera-to-surface rays vote Empty,
 * and a short band behind each observed surface votes Full. A deterministic
 * regular-grid s-t cut combines these observations with a Potts prior.
 */
class VisibilityOccupancySurfaceBuilder
{
public:
    static VisibilityOccupancyResult build(
        const std::array<float, 3> &boundsMin,
        const std::array<float, 3> &boundsMax,
        const std::vector<VisibilityOccupancyFrameView> &frames,
        const VisibilityOccupancyOptions &options);
};

} // namespace xjw::mesh
