#pragma once

#include "DepthGeometryHypothesisReranker.h"

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

namespace xjw::mvs
{

struct DepthEvidenceConfidenceOptions
{
    int minimumGeometryObservationCount = 3;
    float maximumInverseDepthRelativeSpread = 0.012f;
    float strongGeometryConfidence = 0.55f;
    float photometricWeight = 0.40f;
};

struct DepthEvidenceConfidenceSummary
{
    bool available = false;
    int validPixelCount = 0;
    int geometryObservedPixelCount = 0;
    int strongGeometryPixelCount = 0;
    int correctedPixelCount = 0;
    float meanPhotometricConfidence = 0.0f;
    float meanGeometricConfidence = 0.0f;
    float meanCombinedConfidence = 0.0f;
    float strongGeometryCoverage = 0.0f;
    float correctedMeanGeometricConfidence = 0.0f;
};

struct DepthEvidenceConfidenceResult
{
    cv::Mat photometric; ///< CV_32F, original PatchMatch confidence
    cv::Mat geometric; ///< CV_32F, independent cross-view evidence
    cv::Mat combined; ///< CV_32F, explicit photometric/geometric combination
    DepthEvidenceConfidenceSummary summary;
};

DepthEvidenceConfidenceResult buildDepthEvidenceConfidence(
    const cv::Mat &depth,
    const cv::Mat &photometricConfidence,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &adaptiveSupportWeight = {},
    const cv::Mat &adaptiveEffectiveViewCount = {},
    const cv::Mat &adaptiveConflictRatio = {},
    const DepthGeometryHypothesisRerankMaps *rerankMaps = nullptr,
    const DepthEvidenceConfidenceOptions &options = {});

QJsonObject depthEvidenceConfidenceSummaryToJson(
    const DepthEvidenceConfidenceSummary &summary);

} // namespace xjw::mvs
