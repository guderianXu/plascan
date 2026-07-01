#pragma once

#include <QJsonObject>

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct DepthMapQualityMetrics
{
    int width = 0;
    int height = 0;
    int validPixelCount = 0;
    int sourceViewCount = 0;
    float validCoverage = 0.0f;
    float meanConfidence = 0.0f;
    float p50Confidence = 0.0f;
    float p75Confidence = 0.0f;
    bool lowConfidenceFullCoverage = false;
    int localDepthOutlierCount = 0;
    float localDepthOutlierRatio = 0.0f;
    bool hasLocalDepthOutliers = false;
    float recommendedFusionConfidence = 0.0f;
};

DepthMapQualityMetrics analyzeDepthMapQuality(const cv::Mat &depthMap,
                                              const cv::Mat &confidenceMap,
                                              int sourceViewCount);

QJsonObject depthMapQualityMetricsToJson(const DepthMapQualityMetrics &metrics);

} // namespace mvs
} // namespace xjw
