#pragma once

#include "Camera.h"
#include "DepthAnchoredHoleInterpolator.h"

#include <QJsonObject>

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace xjw::mvs
{

struct CrossViewHoleRepairOptions
{
    int minimumDistinctSourceCount = 2;
    float maximumRelativeDepthSpread = 0.025f;
    float maximumProjectionDistancePixels = 1.0f;
    int localDepthRadius = 2;
    float maximumLocalRelativeDepthDifference = 0.05f;
    float repairedConfidence = 0.65f;
    bool enableTwoSourceGrowth = false;
    int maximumGrowthDistancePixels = 3;
    float maximumGrowthInverseDepthSpread = 0.01f;
    float maximumGrowthNormalAngleDegrees = 15.0f;
    float maximumGrowthImageGradient = 80.0f;
    int maximumGrowthComponentArea = 64;
    bool includeValidNativeInterpolationAnchors = false;
    DepthAnchoredHoleInterpolationOptions anchoredInterpolation;
};

struct CrossViewHoleRepairStats
{
    std::uint64_t projectedCandidateCount = 0;
    std::uint64_t consideredHolePixelCount = 0;
    std::uint64_t rejectedInsufficientSourceCount = 0;
    std::uint64_t rejectedDepthSpreadCount = 0;
    std::uint64_t rejectedLocalDepthCount = 0;
    std::uint64_t repairedPixelCount = 0;
    std::uint64_t twoSourceCandidatePixelCount = 0;
    std::uint64_t twoSourceGrownPixelCount = 0;
    std::uint64_t growthRejectedComponentAreaCount = 0;
    std::uint64_t growthRejectedSourceOverlapCount = 0;
    std::uint64_t growthRejectedNormalCount = 0;
    std::uint64_t growthRejectedImageEdgeCount = 0;
    DepthAnchoredHoleInterpolationStats anchoredInterpolation;
};

struct DominantDepthLayerSelectionOptions
{
    int minimumDistinctSourceCount = 2;
    int minimumReplacementSourceCount = 3;
    float maximumRelativeDepthSpread = 0.018f;
    float maximumNativeAgreementRelativeDifference = 0.025f;
    float nativeConsensusBlendWeight = 0.35f;
    float maximumNativeRelativeCorrection = 0.010f;
    float selectedLayerConfidence = 0.72f;
    float ambiguousNativeConfidenceMultiplier = 0.45f;
    bool transferObservedDepthIntoMissingPixels = true;
};

struct DominantDepthLayerSelectionStats
{
    std::uint64_t consideredPixelCount = 0;
    std::uint64_t stableLayerPixelCount = 0;
    std::uint64_t refinedNativePixelCount = 0;
    std::uint64_t switchedNativePixelCount = 0;
    std::uint64_t transferredMissingPixelCount = 0;
    std::uint64_t ambiguousNativePixelCount = 0;
    std::uint64_t unresolvedMissingPixelCount = 0;
};

cv::Mat projectSourceDepthToReference(
    const cv::Mat &sourceDepth,
    const Camera &sourceCamera,
    const Camera &referenceCamera,
    const cv::Size &referenceSize,
    float maximumProjectionDistancePixels,
    std::uint64_t *projectedCandidateCount = nullptr);

/// Selects one occlusion-aware depth layer from source depths projected into
/// the reference view. Stable source clusters may refine a matching native
/// hypothesis, replace a contradicted native layer, or transfer an observed
/// layer into a missing reference pixel. Ambiguous native observations are
/// retained with calibrated confidence instead of being hard-deleted.
DominantDepthLayerSelectionStats selectDominantProjectedDepthLayer(
    cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const cv::Mat &consistentSourceVotes,
    const cv::Mat &contradictedSourceVotes,
    const DominantDepthLayerSelectionOptions &options = {},
    cv::Mat *referenceConfidence = nullptr,
    cv::Mat *selectedLayerMask = nullptr,
    cv::Mat *geometrySourceMask = nullptr,
    cv::Mat *sourceInverseDepthSum = nullptr,
    cv::Mat *sourceInverseDepthSquaredSum = nullptr,
    cv::Mat *selectedSourceVotes = nullptr);

CrossViewHoleRepairStats repairDepthHolesFromProjectedSources(
    cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const CrossViewHoleRepairOptions &options = {},
    cv::Mat *referenceConfidence = nullptr,
    cv::Mat *consistentSourceVotes = nullptr,
    cv::Mat *repairedMask = nullptr,
    cv::Mat *geometrySourceMask = nullptr,
    cv::Mat *sourceInverseDepthSum = nullptr,
    cv::Mat *sourceInverseDepthSquaredSum = nullptr,
    const Camera *referenceCamera = nullptr,
    const cv::Mat *guideGray = nullptr,
    cv::Mat *anchoredInterpolationMask = nullptr);

QJsonObject crossViewHoleRepairStatsToJson(
    const CrossViewHoleRepairStats &stats);

QJsonObject dominantDepthLayerSelectionStatsToJson(
    const DominantDepthLayerSelectionStats &stats);

} // namespace xjw::mvs
