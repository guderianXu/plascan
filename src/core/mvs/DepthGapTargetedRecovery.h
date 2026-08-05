#pragma once

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace xjw::mvs
{

struct DepthGapTargetedRecoveryOptions
{
    int minimumGapPixelCount = 64;
    float minimumGapRatio = 0.002f;
    float maximumGapRatio = 0.35f;
    int estimationMaskDilationPixels = 4;
    int maximumPriorDistancePixels = 128;
    float missingPriorRadiusRatio = 0.18f;
    float anchoredPriorRadiusRatio = 0.03f;
    float minimumCandidateConfidence = 0.28f;
    float maximumCandidatePriorRelativeDifference = 0.18f;
    int minimumConsensusHypothesisCount = 2;
    float maximumConsensusInverseDepthRelativeSpread = 0.025f;
    float maximumConsensusPriorRelativeDifference = 0.35f;
    bool enableSurfaceAwarePrior = false;
    int minimumSurfacePriorAnchorCount = 3;
    float maximumSurfaceAnchorInverseDepthRelativeSpread = 0.12f;
    float maximumSurfacePriorFitRelativeResidual = 0.025f;
    float maximumSurfacePriorEnvelopeExpansion = 0.08f;
};

struct DepthGapTarget
{
    bool valid = false;
    cv::Mat gapMask;
    cv::Mat estimationMask;
    cv::Mat hintDepth;
    cv::Mat nearestHintDepth;
    cv::Mat hintRadius;
    cv::Mat surfacePriorMask;
    cv::Mat priorSupportCount;
    cv::Mat priorRelativeResidual;
    int supportPixelCount = 0;
    int requestedGapPixelCount = 0;
    int priorCoveredGapPixelCount = 0;
    float requestedGapRatio = 0.0f;
    int surfacePriorPixelCount = 0;
    int surfacePriorInsufficientAnchorPixelCount = 0;
    int surfacePriorAnchorSpreadRejectedPixelCount = 0;
    int surfacePriorFitRejectedPixelCount = 0;
    int surfacePriorResidualRejectedPixelCount = 0;
    QString skippedReason;
};

struct DepthGapTargetedRecoveryStats
{
    bool attempted = false;
    int supportPixelCount = 0;
    int requestedGapPixelCount = 0;
    int priorCoveredGapPixelCount = 0;
    int candidatePixelCount = 0;
    int recoveredPixelCount = 0;
    int rejectedConfidencePixelCount = 0;
    int rejectedPriorPixelCount = 0;
    int sourceCount = 0;
    int attemptedHypothesisCount = 0;
    int hypothesisCount = 0;
    int failedHypothesisCount = 0;
    int consensusCandidatePixelCount = 0;
    int rejectedInsufficientHypothesisPixelCount = 0;
    int rejectedHypothesisSpreadPixelCount = 0;
    int surfacePriorPixelCount = 0;
    int surfacePriorAcceptedPixelCount = 0;
    int surfacePriorInsufficientAnchorPixelCount = 0;
    int surfacePriorAnchorSpreadRejectedPixelCount = 0;
    int surfacePriorFitRejectedPixelCount = 0;
    int surfacePriorResidualRejectedPixelCount = 0;
    float recoveryRatio = 0.0f;
    QString skippedReason;
};

DepthGapTarget buildDepthGapTarget(
    const cv::Mat &depth,
    const cv::Mat &supportMask,
    const DepthGapTargetedRecoveryOptions &options = {});

DepthGapTargetedRecoveryStats mergeTargetedDepthGapCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const cv::Mat &candidateDepth,
    const cv::Mat &candidateConfidence,
    const DepthGapTarget &target,
    cv::Mat *recoveredMask = nullptr,
    const DepthGapTargetedRecoveryOptions &options = {});

DepthGapTargetedRecoveryStats mergeMultiHypothesisTargetedDepthGapCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const std::vector<cv::Mat> &candidateDepths,
    const std::vector<cv::Mat> &candidateConfidences,
    const DepthGapTarget &target,
    cv::Mat *recoveredMask = nullptr,
    const DepthGapTargetedRecoveryOptions &options = {});

QJsonObject depthGapTargetedRecoveryStatsToJson(
    const DepthGapTargetedRecoveryStats &stats);

} // namespace xjw::mvs
