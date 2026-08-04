#pragma once

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace xjw::mvs
{

enum class DepthMissingReason : std::uint8_t
{
    Valid = 0,
    OutsideSupport = 1,
    PatchMatchUnresolved = 2,
    LowConfidence = 3,
    LocalDepthOutlier = 4,
    SmallComponent = 5,
    GeometryContradiction = 6,
    InsufficientGeometrySupport = 7,
    Unclassified = 8
};

struct DepthMissingReasonSummary
{
    bool validInputs = false;
    int supportPixelCount = 0;
    int validPixelCount = 0;
    int missingPixelCount = 0;
    int outsideSupportPixelCount = 0;
    int patchMatchUnresolvedPixelCount = 0;
    int lowConfidencePixelCount = 0;
    int localDepthOutlierPixelCount = 0;
    int smallComponentPixelCount = 0;
    int geometryContradictionPixelCount = 0;
    int insufficientGeometrySupportPixelCount = 0;
    int unclassifiedPixelCount = 0;
    float missingWithinSupportRatio = 0.0f;
};

cv::Mat initializeDepthMissingReasonMap(const cv::Mat &depth,
                                        const cv::Mat &supportMask);

void markDepthLossReason(cv::Mat &reasonMap,
                         const cv::Mat &depthBefore,
                         const cv::Mat &depthAfter,
                         DepthMissingReason reason);

void finalizeDepthMissingReasonMap(cv::Mat &reasonMap,
                                   const cv::Mat &finalDepth,
                                   const cv::Mat &supportMask,
                                   const cv::Mat &geometrySupportCount = {},
                                   const cv::Mat &geometryContradictionCount = {});

DepthMissingReasonSummary summarizeDepthMissingReasons(
    const cv::Mat &reasonMap,
    const cv::Mat &supportMask);

QJsonObject depthMissingReasonSummaryToJson(
    const DepthMissingReasonSummary &summary);

cv::Mat makeDepthMissingReasonPreview(const cv::Mat &reasonMap);

} // namespace xjw::mvs
