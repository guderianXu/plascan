#include "DepthProvenance.h"

#include <opencv2/core.hpp>

#include <algorithm>

namespace xjw::mvs
{
namespace
{

bool compatibleMask(const cv::Mat &mask, const cv::Size &size)
{
    return !mask.empty() && mask.type() == CV_8UC1 && mask.size() == size;
}

void assignMasked(cv::Mat &provenance,
                  const cv::Mat &mask,
                  DepthProvenance value)
{
    if (compatibleMask(mask, provenance.size()))
    {
        provenance.setTo(
            cv::Scalar(static_cast<std::uint8_t>(value)), mask != 0);
    }
}

} // namespace

cv::Mat initializeDepthProvenance(
    const cv::Mat &depth,
    const cv::Mat &targetedPatchMatchMask)
{
    if (depth.empty() || depth.type() != CV_32FC1)
    {
        return {};
    }
    cv::Mat provenance(depth.size(), CV_8UC1, cv::Scalar(0));
    provenance.setTo(
        cv::Scalar(static_cast<std::uint8_t>(
            DepthProvenance::NativePatchMatch)),
        depth > 0.0f);
    assignMasked(
        provenance,
        targetedPatchMatchMask,
        DepthProvenance::TargetedPatchMatch);
    provenance.setTo(cv::Scalar(0), depth <= 0.0f);
    return provenance;
}

void updateDepthProvenance(
    cv::Mat &provenance,
    const cv::Mat &depth,
    const cv::Mat &targetedPatchMatchMask,
    const cv::Mat &crossViewMeasuredMask,
    const cv::Mat &anchoredInterpolationMask)
{
    if (depth.empty() || depth.type() != CV_32FC1)
    {
        provenance.release();
        return;
    }
    if (provenance.empty() || provenance.type() != CV_8UC1 ||
        provenance.size() != depth.size())
    {
        provenance = initializeDepthProvenance(depth, targetedPatchMatchMask);
    }
    provenance.setTo(
        cv::Scalar(static_cast<std::uint8_t>(
            DepthProvenance::NativePatchMatch)),
        (depth > 0.0f) & (provenance == 0));
    assignMasked(
        provenance,
        targetedPatchMatchMask,
        DepthProvenance::TargetedPatchMatch);
    if (compatibleMask(crossViewMeasuredMask, provenance.size()))
    {
        cv::Mat measured_mask;
        cv::bitwise_and(
            crossViewMeasuredMask,
            provenance != static_cast<std::uint8_t>(
                DepthProvenance::AnchoredInterpolation),
            measured_mask);
        assignMasked(
            provenance,
            measured_mask,
            DepthProvenance::CrossViewMeasured);
    }
    assignMasked(
        provenance,
        anchoredInterpolationMask,
        DepthProvenance::AnchoredInterpolation);
    provenance.setTo(cv::Scalar(0), depth <= 0.0f);
}

DepthProvenanceSummary summarizeDepthProvenance(
    const cv::Mat &provenance,
    const cv::Mat &depth)
{
    DepthProvenanceSummary summary;
    if (depth.empty() || depth.type() != CV_32FC1 ||
        provenance.empty() || provenance.type() != CV_8UC1 ||
        provenance.size() != depth.size())
    {
        return summary;
    }
    summary.available = true;
    const cv::Mat valid = depth > 0.0f;
    summary.validPixelCount = cv::countNonZero(valid);
    summary.nativePatchMatchPixelCount = cv::countNonZero(
        valid & (provenance == static_cast<std::uint8_t>(
            DepthProvenance::NativePatchMatch)));
    summary.targetedPatchMatchPixelCount = cv::countNonZero(
        valid & (provenance == static_cast<std::uint8_t>(
            DepthProvenance::TargetedPatchMatch)));
    summary.crossViewMeasuredPixelCount = cv::countNonZero(
        valid & (provenance == static_cast<std::uint8_t>(
            DepthProvenance::CrossViewMeasured)));
    summary.anchoredInterpolationPixelCount = cv::countNonZero(
        valid & (provenance == static_cast<std::uint8_t>(
            DepthProvenance::AnchoredInterpolation)));
    const int classified = summary.nativePatchMatchPixelCount +
        summary.targetedPatchMatchPixelCount +
        summary.crossViewMeasuredPixelCount +
        summary.anchoredInterpolationPixelCount;
    summary.unclassifiedValidPixelCount =
        std::max(0, summary.validPixelCount - classified);
    return summary;
}

QJsonObject depthProvenanceSummaryToJson(
    const DepthProvenanceSummary &summary)
{
    return QJsonObject{
        {QStringLiteral("available"), summary.available},
        {QStringLiteral("valid_pixel_count"), summary.validPixelCount},
        {QStringLiteral("native_patchmatch_pixel_count"),
         summary.nativePatchMatchPixelCount},
        {QStringLiteral("targeted_patchmatch_pixel_count"),
         summary.targetedPatchMatchPixelCount},
        {QStringLiteral("cross_view_measured_pixel_count"),
         summary.crossViewMeasuredPixelCount},
        {QStringLiteral("anchored_interpolation_pixel_count"),
         summary.anchoredInterpolationPixelCount},
        {QStringLiteral("unclassified_valid_pixel_count"),
         summary.unclassifiedValidPixelCount},
        {QStringLiteral("schema_version"), 1}};
}

bool isInterpolatedDepthProvenance(std::uint8_t value)
{
    return value == static_cast<std::uint8_t>(
        DepthProvenance::AnchoredInterpolation);
}

} // namespace xjw::mvs
