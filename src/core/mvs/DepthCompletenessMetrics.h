#pragma once

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

namespace xjw::mvs
{

struct DepthCompletenessMetrics
{
    bool validInputs = false;
    int width = 0;
    int height = 0;
    int maskPixelCount = 0;
    int validWithinMaskCount = 0;
    float validWithinMaskRatio = 0.0f;
    int invalidWithinMaskCount = 0;
    int smallInteriorHoleCount = 0;
    int smallInteriorHolePixelCount = 0;
    int largeInteriorOpeningCount = 0;
    int largeInteriorOpeningPixelCount = 0;
    int boundaryConnectedInvalidCount = 0;
    int boundaryConnectedInvalidPixelCount = 0;
    int smallHoleAreaLimit = 0;
};

struct DepthCompletenessDiagnostics
{
    int pyramidValidCount = -1;
    int afterMaskValidCount = -1;
    int afterSparseSupportValidCount = -1;
    int preOutputFilterValidCount = -1;
    int postOutputFilterValidCount = -1;
    int outputFilterRemovedCount = 0;
    float outputFilterRetentionRatio = -1.0f;
    int preConsistencyValidCount = -1;
    int postConsistencyValidCount = -1;          ///< 一致性过滤实际保留数，不含发布回退恢复的原始深度
    float consistencyRetentionRatio = -1.0f;     ///< 实际保留率；质量门只能使用该值
    int publishedPostConsistencyValidCount = -1; ///< 最终发布载体中的有效数，可能包含安全回退
    float publishedConsistencyRetentionRatio = -1.0f;
    bool consistencyPublicationFallbackApplied = false;
    int consistencyConfirmedObservationCount = 0;
    int consistencyOccludedObservationCount = 0;
    int consistencyContradictedObservationCount = 0;
    int consistencyUnverifiableObservationCount = 0;
    int consistencyRejectedPixelCount = 0;
    int crossViewRepairedCount = 0;
    int preFusionPostprocessValidCount = -1;
    int postConfidenceFilterValidCount = -1;
    int postFusionPostprocessValidCount = -1;
    float fusionPostprocessRetentionRatio = -1.0f;
    int restoredFromPrefilterCount = 0;
    int restoredFromParentLevelCount = 0;
    DepthCompletenessMetrics finalMetrics;
};

DepthCompletenessMetrics analyzeDepthCompleteness(
    const cv::Mat &depthMap,
    const cv::Mat &effectiveMask,
    float maximumSmallHoleFraction = 0.002f,
    int minimumSmallHoleAreaLimit = 64);

int restoreSmallInteriorDepthHoles(
    cv::Mat &depthMap,
    const cv::Mat &candidateDepth,
    const cv::Mat &confidenceMap,
    const cv::Mat &effectiveMask,
    float minimumConfidence = 0.75f,
    float maximumRelativeDepthDifference = 0.12f,
    float maximumSmallHoleFraction = 0.002f,
    int minimumSmallHoleAreaLimit = 64);

QJsonObject depthCompletenessMetricsToJson(const DepthCompletenessMetrics &metrics);

QJsonObject depthCompletenessDiagnosticsToJson(
    const DepthCompletenessDiagnostics &diagnostics);

} // namespace xjw::mvs
