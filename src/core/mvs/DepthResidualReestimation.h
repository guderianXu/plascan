#pragma once

#include "DepthGeometryHypothesisReranker.h"

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace xjw::mvs
{

struct DepthResidualReestimationOptions
{
    int minimumResidualPixelCount = 64;
    float minimumResidualRatio = 0.001f;
    int estimationMaskDilationPixels = 4;
    int minimumLayerSourceCount = 2;
    int minimumLayerSectorCount = 2;
    float maximumLayerInverseDepthRelativeSpread = 0.025f;
    float maximumPriorRadiusRatio = 0.08f;
    float minimumCandidateConfidence = 0.30f;
    float recoveredConfidence = 0.70f;
    int minimumCandidateHypothesisCount = 2;
    float maximumCandidateInverseDepthRelativeSpread = 0.025f;
    float maximumCandidatePriorRelativeDifference = 0.08f;
    int minimumGeometryConfirmationCount = 2;
    int minimumGeometrySectorCount = 2;
    float maximumGeometryRelativeDifference = 0.025f;
    float freeSpaceConflictRelativeDifference = 0.04f;
    bool allowValidDepthReplacement = false;
    float minimumReplacementCostAdvantage = 0.08f;
    float ambiguousMaximumRelativeCorrection = 0.01f;
};

struct DepthResidualReestimationTarget
{
    bool valid = false;
    cv::Mat residualMask;
    cv::Mat estimationMask;
    cv::Mat hintDepth;
    cv::Mat hintRadius;
    cv::Mat layerSourceCount;
    cv::Mat layerSectorCount;
    int supportPixelCount = 0;
    int requestedResidualPixelCount = 0;
    int layerCoveredPixelCount = 0;
    int insufficientSourcePixelCount = 0;
    int insufficientSectorPixelCount = 0;
    int layerSpreadRejectedPixelCount = 0;
    float requestedResidualRatio = 0.0f;
    QString skippedReason;
};

struct DepthResidualReestimationStats
{
    bool attempted = false;
    int supportPixelCount = 0;
    int requestedResidualPixelCount = 0;
    int layerCoveredPixelCount = 0;
    int insufficientSourcePixelCount = 0;
    int insufficientSectorPixelCount = 0;
    int layerSpreadRejectedPixelCount = 0;
    int candidatePixelCount = 0;
    int consensusCandidatePixelCount = 0;
    int rejectedConfidencePixelCount = 0;
    int rejectedInsufficientHypothesisPixelCount = 0;
    int rejectedHypothesisSpreadPixelCount = 0;
    int rejectedPriorPixelCount = 0;
    int rejectedGeometryPixelCount = 0;
    int rejectedGeometrySectorPixelCount = 0;
    int rejectedFreeSpacePixelCount = 0;
    int recoveredPixelCount = 0;
    int replacementCandidatePixelCount = 0;
    int replacedPixelCount = 0;
    int rejectedReplacementReliabilityPixelCount = 0;
    int rejectedReplacementCostPixelCount = 0;
    int rejectedReplacementCorrectionPixelCount = 0;
    int attemptedHypothesisCount = 0;
    int successfulHypothesisCount = 0;
    int failedHypothesisCount = 0;
    int sourceCount = 0;
    float recoveryRatio = 0.0f;
    QString skippedReason;
};

struct DepthResidualReestimationPreflight
{
    bool shouldProjectSources = false;
    cv::Mat normalizedSupport;
    cv::Mat residualMask;
    int supportPixelCount = 0;
    int requestedResidualPixelCount = 0;
    float requestedResidualRatio = 0.0f;
    QString skippedReason;
};

/// Optional authoritative evidence sinks for accepted valid-depth
/// replacements. Source counts exclude the reference view; the caller keeps
/// its existing reference-inclusive convention when deriving support maps.
struct DepthResidualReestimationEvidenceOutputs
{
    cv::Mat *geometrySourceMask = nullptr; ///< CV_16U
    cv::Mat *sourceInverseDepthSum = nullptr; ///< CV_32F
    cv::Mat *sourceInverseDepthSquaredSum = nullptr; ///< CV_32F
    cv::Mat *confirmedSourceCount = nullptr; ///< CV_16U
    DepthGeometryHypothesisRerankMaps *rerankMaps = nullptr;
};

/// Builds two deterministic overlapping local PatchMatch source groups. Each
/// group keeps at least three views when four or more are available; the final
/// merge still requires independent projected-depth evidence, so overlap can
/// improve photometric candidate availability without weakening geometry.
std::vector<std::vector<int>> buildDepthResidualPatchMatchSourceGroups(
    const std::vector<int> &sourceSectorIds);

/**
 * @brief Cheap missing-in-support gate evaluated before source reprojection.
 *
 * This mirrors the initial pixel-count and ratio checks performed by
 * buildDepthResidualReestimationTarget(), but does not allocate projected
 * source maps or alter the reference depth.
 */
DepthResidualReestimationPreflight inspectDepthResidualReestimationNeed(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const DepthResidualReestimationOptions &options = {});

/// Builds the same bounded preflight from an explicit weak-region mask. This
/// is used by the default-off local second-pass treatment and never expands
/// outside the support mask.
DepthResidualReestimationPreflight inspectDepthReestimationMask(
    const cv::Mat &supportMask,
    const cv::Mat &requestedMask,
    const DepthResidualReestimationOptions &options = {});

DepthResidualReestimationTarget buildDepthResidualReestimationTarget(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    const DepthResidualReestimationOptions &options = {});

DepthResidualReestimationTarget buildDepthResidualReestimationTarget(
    const cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    const DepthResidualReestimationOptions &options,
    DepthResidualReestimationPreflight preflight);

DepthResidualReestimationStats mergeDepthResidualReestimationCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const std::vector<cv::Mat> &candidateDepths,
    const std::vector<cv::Mat> &candidateConfidences,
    const DepthResidualReestimationTarget &target,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const std::vector<int> &sourceSectorIds,
    cv::Mat *recoveredMask = nullptr,
    const DepthResidualReestimationOptions &options = {},
    const cv::Mat *nativeReliabilityClassMap = nullptr,
    const std::vector<ProjectedDepthEvidence> *projectedSourceEvidence = nullptr,
    const DepthResidualReestimationEvidenceOutputs *evidenceOutputs = nullptr);

QJsonObject depthResidualReestimationStatsToJson(
    const DepthResidualReestimationStats &stats);

} // namespace xjw::mvs
