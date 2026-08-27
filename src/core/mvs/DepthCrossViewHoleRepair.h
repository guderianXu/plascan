#pragma once

#include "FramePinholeCamera.h"
#include "DepthAnchoredHoleInterpolator.h"
#include "DepthGeometryHypothesisReranker.h"
#include "DepthLayerReliability.h"

#include <QJsonObject>

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <vector>

namespace xjw::mvs
{

struct CrossViewHoleRepairOptions
{
    int minimumDistinctSourceCount = 2;
    float maximumRelativeDepthSpread = 0.010f;
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
    std::uint64_t nativeInterpolationAnchorCandidateCount = 0;
    std::uint64_t nativeInterpolationAnchorAcceptedCount = 0;
    std::uint64_t nativeInterpolationAnchorRejectedCount = 0;
    DepthAnchoredHoleInterpolationStats anchoredInterpolation;
};

struct DominantDepthLayerSelectionOptions
{
    int minimumDistinctSourceCount = 2;
    int minimumReplacementSourceCount = 3;
    float maximumRelativeDepthSpread = 0.010f;
    float maximumNativeAgreementRelativeDifference = 0.025f;
    float nativeConsensusBlendWeight = 0.35f;
    float maximumNativeRelativeCorrection = 0.010f;
    float selectedLayerConfidence = 0.72f;
    float ambiguousNativeConfidenceMultiplier = 0.45f;
    bool transferObservedDepthIntoMissingPixels = true;
    bool enableReliabilityGuidedCorrection = false;
    /// When true, keep all reliable and unobservable native pixels byte-exact
    /// and evaluate projected layers only for weak native reliability classes.
    /// This is used by non-orbital experiments so enabling the diagnostic
    /// correction cannot activate the legacy orbital hole-transfer behavior.
    bool restrictToReliabilityGuidedCandidates = false;
    int reliabilityGuidedMinimumSourceCount = 3;
    float reliabilityGuidedBlendWeight = 0.85f;
    float reliabilityGuidedMaximumRelativeCorrection = 0.025f;
    DepthGeometryHypothesisRerankOptions geometryRerank;
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
    std::uint64_t reliabilityGuidedCandidatePixelCount = 0;
    std::uint64_t reliabilityGuidedStablePixelCount = 0;
    std::uint64_t reliabilityGuidedRefinedPixelCount = 0;
    std::uint64_t reliabilityGuidedSwitchedPixelCount = 0;
    std::uint64_t reliabilityGuidedInsufficientSourcePixelCount = 0;
    std::uint64_t geometryRerankEvaluatedPixelCount = 0;
    std::uint64_t geometryRerankValidEvidencePixelCount = 0;
    std::uint64_t geometryRerankRefinedPixelCount = 0;
    std::uint64_t geometryRerankSwitchedPixelCount = 0;
    std::uint64_t geometryRerankRejectedSourceCount = 0;
    std::uint64_t geometryRerankRejectedBaselineCount = 0;
    std::uint64_t geometryRerankRejectedWeightCount = 0;
    std::uint64_t geometryRerankRejectedCostCount = 0;
    double geometryRerankNativeCostSum = 0.0;
    double geometryRerankCandidateCostSum = 0.0;
    double geometryRerankCostAdvantageSum = 0.0;
    double geometryRerankCorrectionSum = 0.0;
    double geometryRerankWeakestConfidenceSum = 0.0;
    bool reliabilityGuidedOnlyMode = false;
};

cv::Mat projectSourceDepthToReference(
    const cv::Mat &sourceDepth,
    const FramePinholeCamera &sourceCamera,
    const FramePinholeCamera &referenceCamera,
    const cv::Size &referenceSize,
    float maximumProjectionDistancePixels,
    std::uint64_t *projectedCandidateCount = nullptr,
    int rowWorkerCount = 1,
    const std::atomic<bool> *cancelled = nullptr);

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
    cv::Mat *selectedSourceVotes = nullptr,
    int rowWorkerCount = 1,
    const std::atomic<bool> *cancelled = nullptr,
    /// Optional pre-selection CV_8U reliability classes. When guidance is
    /// enabled, an incompatible map disables only the extra correction path.
    const cv::Mat *nativeReliabilityClassMap = nullptr,
    /// Optional CV_8U output marking only pixels whose native depth was
    /// actually refined or switched by reliability guidance.
    cv::Mat *reliabilityGuidedChangedMask = nullptr,
    /// Optional measured source evidence. When reliability guidance is
    /// enabled, a missing/incompatible vector fails closed to the legacy path.
    const std::vector<ProjectedDepthEvidence> *projectedSourceEvidence = nullptr,
    /// Optional per-pixel audit maps for selected diagnostic snapshots.
    DepthGeometryHypothesisRerankMaps *geometryRerankMaps = nullptr);

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
    const FramePinholeCamera *referenceCamera = nullptr,
    const cv::Mat *guideGray = nullptr,
    cv::Mat *anchoredInterpolationMask = nullptr,
    int rowWorkerCount = 1,
    const std::atomic<bool> *cancelled = nullptr,
    /// Optional CV_8U mask for valid native-depth interpolation anchors.
    /// A non-null incompatible mask fails closed and admits no native anchor.
    const cv::Mat *nativeInterpolationAnchorEligibilityMask = nullptr);

QJsonObject crossViewHoleRepairStatsToJson(
    const CrossViewHoleRepairStats &stats);

QJsonObject dominantDepthLayerSelectionStatsToJson(
    const DominantDepthLayerSelectionStats &stats);

} // namespace xjw::mvs
