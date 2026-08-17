#pragma once

#include <QJsonObject>

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace xjw::mvs
{

enum class DepthProvenance : std::uint8_t
{
    Invalid = 0,
    NativePatchMatch = 1,
    TargetedPatchMatch = 2,
    CrossViewMeasured = 3,
    AnchoredInterpolation = 4,
    ResidualPatchMatch = 5,
    LearnedGeometryGated = 6
};

struct DepthProvenanceSummary
{
    bool available = false;
    int validPixelCount = 0;
    int nativePatchMatchPixelCount = 0;
    int targetedPatchMatchPixelCount = 0;
    int crossViewMeasuredPixelCount = 0;
    int anchoredInterpolationPixelCount = 0;
    int residualPatchMatchPixelCount = 0;
    int learnedGeometryGatedPixelCount = 0;
    int unclassifiedValidPixelCount = 0;
};

cv::Mat initializeDepthProvenance(
    const cv::Mat &depth,
    const cv::Mat &targetedPatchMatchMask = cv::Mat());

void updateDepthProvenance(
    cv::Mat &provenance,
    const cv::Mat &depth,
    const cv::Mat &targetedPatchMatchMask = cv::Mat(),
    const cv::Mat &crossViewMeasuredMask = cv::Mat(),
    const cv::Mat &anchoredInterpolationMask = cv::Mat(),
    const cv::Mat &residualPatchMatchMask = cv::Mat(),
    const cv::Mat &learnedGeometryGatedMask = cv::Mat());

DepthProvenanceSummary summarizeDepthProvenance(
    const cv::Mat &provenance,
    const cv::Mat &depth);

QJsonObject depthProvenanceSummaryToJson(
    const DepthProvenanceSummary &summary);

[[nodiscard]] bool isInterpolatedDepthProvenance(std::uint8_t value);

} // namespace xjw::mvs
