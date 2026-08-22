#pragma once

#include "FramePinholeCamera.h"
#include "DepthLayerReliability.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

namespace xjw::mvs
{

/// A source depth z-buffered into the reference grid together with the
/// confidence and sub-pixel rasterization error of the winning measurement.
struct ProjectedDepthEvidence
{
    cv::Mat depth; ///< CV_32F, positive measured depth or zero
    cv::Mat confidence; ///< CV_32F, source confidence in [0, 1]
    cv::Mat reprojectionErrorPixels; ///< CV_32F distance to the reference texel centre
    int baselineSector = -1; ///< Deterministic camera-centre direction sector [0, 5]
};

struct DepthGeometryHypothesisRerankOptions
{
    int minimumDistinctSourceCount = 3;
    int minimumBaselineSectorCount = 2;
    float maximumRelativeDepthSpread = 0.018f;
    float minimumProjectedConfidence = 0.10f;
    float minimumEffectiveSourceWeight = 1.35f;
    float minimumRefinementCostAdvantage = 0.025f;
    float minimumLayerSwitchCostAdvantage = 0.08f;
    float ambiguousMaximumRelativeCorrection = 0.010f;
    float rejectedMaximumRelativeRefinement = 0.025f;
    float refinementBlendWeight = 0.75f;
};

enum class DepthGeometryHypothesisAction : std::uint8_t
{
    None = 0,
    Refine = 1,
    SwitchLayer = 2
};

struct DepthGeometryHypothesisDecision
{
    DepthGeometryHypothesisAction action = DepthGeometryHypothesisAction::None;
    float selectedDepth = 0.0f;
    float evidenceConfidence = 0.0f;
    float nativeCost = 1.0f;
    float candidateCost = 1.0f;
    float costAdvantage = 0.0f;
    float relativeCorrection = 0.0f;
    float effectiveSourceWeight = 0.0f;
    float weakestSourceConfidence = 0.0f;
    int supportingSourceCount = 0;
    int baselineSectorCount = 0;
    std::uint16_t supportingSourceMask = 0;
    float supportingInverseDepthSum = 0.0f;
    float supportingInverseDepthSquaredSum = 0.0f;
    bool validEvidence = false;
    bool rejectedInsufficientSources = false;
    bool rejectedInsufficientBaseline = false;
    bool rejectedInsufficientWeight = false;
    bool rejectedCostAdvantage = false;
};

struct DepthGeometryHypothesisRerankMaps
{
    cv::Mat nativeCost; ///< CV_32F
    cv::Mat candidateCost; ///< CV_32F
    cv::Mat costAdvantage; ///< CV_32F
    cv::Mat effectiveSourceWeight; ///< CV_32F
    cv::Mat relativeCorrection; ///< CV_32F
    cv::Mat weakestSourceConfidence; ///< CV_32F
    cv::Mat supportingSourceCount; ///< CV_8U
    cv::Mat baselineSectorCount; ///< CV_8U
    cv::Mat decisionAction; ///< CV_8U, DepthGeometryHypothesisAction

    void initialize(const cv::Size &size);
    bool compatible(const cv::Size &size) const;
};

ProjectedDepthEvidence projectSourceDepthEvidenceToReference(
    const cv::Mat &sourceDepth,
    const cv::Mat &sourceConfidence,
    const FramePinholeCamera &sourceCamera,
    const FramePinholeCamera &referenceCamera,
    const cv::Size &referenceSize,
    float maximumProjectionDistancePixels,
    int baselineSector,
    std::uint64_t *projectedCandidateCount = nullptr,
    int rowWorkerCount = 1,
    const std::atomic<bool> *cancelled = nullptr);

/// Scores one externally supplied hypothesis (for example a local second-pass
/// PatchMatch candidate) against the same measured source-depth evidence.
DepthGeometryHypothesisDecision scoreMeasuredDepthHypothesis(
    float hypothesisDepth,
    int row,
    int column,
    std::span<const ProjectedDepthEvidence> projectedEvidence,
    const DepthGeometryHypothesisRerankOptions &options = {});

DepthGeometryHypothesisDecision rerankMeasuredDepthHypothesis(
    float nativeDepth,
    DepthLayerReliabilityClass reliabilityClass,
    int row,
    int column,
    std::span<const ProjectedDepthEvidence> projectedEvidence,
    const DepthGeometryHypothesisRerankOptions &options = {});

} // namespace xjw::mvs
