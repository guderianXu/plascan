#include "DepthMissingReason.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace xjw::mvs
{
namespace
{

bool compatibleDepthAndMask(const cv::Mat &depth, const cv::Mat &mask)
{
    return !depth.empty() && depth.type() == CV_32FC1 &&
        !mask.empty() && mask.type() == CV_8UC1 && depth.size() == mask.size();
}

int reasonCount(const cv::Mat &reasonMap, DepthMissingReason reason)
{
    return cv::countNonZero(
        reasonMap == static_cast<std::uint8_t>(reason));
}

} // namespace

cv::Mat initializeDepthMissingReasonMap(const cv::Mat &depth,
                                        const cv::Mat &supportMask)
{
    if (!compatibleDepthAndMask(depth, supportMask))
    {
        return {};
    }

    cv::Mat result(depth.size(), CV_8UC1,
                   cv::Scalar(static_cast<std::uint8_t>(
                       DepthMissingReason::PatchMatchUnresolved)));
    result.setTo(static_cast<std::uint8_t>(DepthMissingReason::Valid),
                 depth > 0.0f);
    result.setTo(static_cast<std::uint8_t>(DepthMissingReason::OutsideSupport),
                 supportMask == 0);
    return result;
}

void markDepthLossReason(cv::Mat &reasonMap,
                         const cv::Mat &depthBefore,
                         const cv::Mat &depthAfter,
                         DepthMissingReason reason)
{
    if (reasonMap.empty() || reasonMap.type() != CV_8UC1 ||
        depthBefore.empty() || depthBefore.type() != CV_32FC1 ||
        depthAfter.empty() || depthAfter.type() != CV_32FC1 ||
        reasonMap.size() != depthBefore.size() ||
        reasonMap.size() != depthAfter.size())
    {
        return;
    }

    cv::Mat lost;
    cv::bitwise_and(depthBefore > 0.0f, depthAfter <= 0.0f, lost);
    reasonMap.setTo(static_cast<std::uint8_t>(reason), lost);
}

void finalizeDepthMissingReasonMap(cv::Mat &reasonMap,
                                   const cv::Mat &finalDepth,
                                   const cv::Mat &supportMask,
                                   const cv::Mat &geometrySupportCount,
                                   const cv::Mat &geometryContradictionCount)
{
    if (!compatibleDepthAndMask(finalDepth, supportMask))
    {
        reasonMap.release();
        return;
    }
    if (reasonMap.empty() || reasonMap.type() != CV_8UC1 ||
        reasonMap.size() != finalDepth.size())
    {
        reasonMap = initializeDepthMissingReasonMap(finalDepth, supportMask);
    }

    reasonMap.setTo(static_cast<std::uint8_t>(DepthMissingReason::Valid),
                    finalDepth > 0.0f);
    reasonMap.setTo(static_cast<std::uint8_t>(DepthMissingReason::OutsideSupport),
                    supportMask == 0);

    const bool has_support = !geometrySupportCount.empty() &&
        geometrySupportCount.type() == CV_16UC1 &&
        geometrySupportCount.size() == finalDepth.size();
    const bool has_contradiction = !geometryContradictionCount.empty() &&
        geometryContradictionCount.type() == CV_16UC1 &&
        geometryContradictionCount.size() == finalDepth.size();
    const cv::Mat unresolved = (finalDepth <= 0.0f) & (supportMask != 0);
    if (has_contradiction)
    {
        cv::Mat contradicted;
        cv::bitwise_and(unresolved, geometryContradictionCount > 0, contradicted);
        reasonMap.setTo(
            static_cast<std::uint8_t>(DepthMissingReason::GeometryContradiction),
            contradicted);
    }
    if (has_support)
    {
        cv::Mat insufficient;
        cv::bitwise_and(unresolved, geometrySupportCount == 0, insufficient);
        cv::bitwise_and(
            insufficient,
            reasonMap == static_cast<std::uint8_t>(
                DepthMissingReason::PatchMatchUnresolved),
            insufficient);
        reasonMap.setTo(
            static_cast<std::uint8_t>(
                DepthMissingReason::InsufficientGeometrySupport),
            insufficient);
    }

    cv::Mat unclassified;
    cv::bitwise_and(
        unresolved,
        reasonMap == static_cast<std::uint8_t>(DepthMissingReason::Valid),
        unclassified);
    reasonMap.setTo(
        static_cast<std::uint8_t>(DepthMissingReason::Unclassified),
        unclassified);
}

DepthMissingReasonSummary summarizeDepthMissingReasons(
    const cv::Mat &reasonMap,
    const cv::Mat &supportMask)
{
    DepthMissingReasonSummary summary;
    if (reasonMap.empty() || reasonMap.type() != CV_8UC1 ||
        supportMask.empty() || supportMask.type() != CV_8UC1 ||
        reasonMap.size() != supportMask.size())
    {
        return summary;
    }

    summary.validInputs = true;
    summary.supportPixelCount = cv::countNonZero(supportMask);
    summary.outsideSupportPixelCount = reasonCount(
        reasonMap, DepthMissingReason::OutsideSupport);
    summary.validPixelCount = reasonCount(reasonMap, DepthMissingReason::Valid);
    summary.patchMatchUnresolvedPixelCount = reasonCount(
        reasonMap, DepthMissingReason::PatchMatchUnresolved);
    summary.lowConfidencePixelCount = reasonCount(
        reasonMap, DepthMissingReason::LowConfidence);
    summary.localDepthOutlierPixelCount = reasonCount(
        reasonMap, DepthMissingReason::LocalDepthOutlier);
    summary.smallComponentPixelCount = reasonCount(
        reasonMap, DepthMissingReason::SmallComponent);
    summary.geometryContradictionPixelCount = reasonCount(
        reasonMap, DepthMissingReason::GeometryContradiction);
    summary.insufficientGeometrySupportPixelCount = reasonCount(
        reasonMap, DepthMissingReason::InsufficientGeometrySupport);
    summary.unclassifiedPixelCount = reasonCount(
        reasonMap, DepthMissingReason::Unclassified);
    summary.missingPixelCount = std::max(
        0, summary.supportPixelCount - summary.validPixelCount);
    if (summary.supportPixelCount > 0)
    {
        summary.missingWithinSupportRatio =
            static_cast<float>(summary.missingPixelCount) /
            static_cast<float>(summary.supportPixelCount);
    }
    return summary;
}

QJsonObject depthMissingReasonSummaryToJson(
    const DepthMissingReasonSummary &summary)
{
    QJsonObject object;
    object.insert(QStringLiteral("available"), summary.validInputs);
    if (!summary.validInputs)
    {
        return object;
    }
    object.insert(QStringLiteral("support_pixel_count"),
                  summary.supportPixelCount);
    object.insert(QStringLiteral("valid_pixel_count"), summary.validPixelCount);
    object.insert(QStringLiteral("missing_pixel_count"),
                  summary.missingPixelCount);
    object.insert(QStringLiteral("missing_within_support_ratio"),
                  summary.missingWithinSupportRatio);
    object.insert(QStringLiteral("outside_support_pixel_count"),
                  summary.outsideSupportPixelCount);
    object.insert(QStringLiteral("patchmatch_unresolved_pixel_count"),
                  summary.patchMatchUnresolvedPixelCount);
    object.insert(QStringLiteral("low_confidence_pixel_count"),
                  summary.lowConfidencePixelCount);
    object.insert(QStringLiteral("local_depth_outlier_pixel_count"),
                  summary.localDepthOutlierPixelCount);
    object.insert(QStringLiteral("small_component_pixel_count"),
                  summary.smallComponentPixelCount);
    object.insert(QStringLiteral("geometry_contradiction_pixel_count"),
                  summary.geometryContradictionPixelCount);
    object.insert(QStringLiteral("insufficient_geometry_support_pixel_count"),
                  summary.insufficientGeometrySupportPixelCount);
    object.insert(QStringLiteral("unclassified_pixel_count"),
                  summary.unclassifiedPixelCount);
    object.insert(QStringLiteral("schema_version"), 1);
    return object;
}

cv::Mat makeDepthMissingReasonPreview(const cv::Mat &reasonMap)
{
    if (reasonMap.empty() || reasonMap.type() != CV_8UC1)
    {
        return {};
    }

    cv::Mat preview(reasonMap.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));
    const auto paint = [&preview, &reasonMap](DepthMissingReason reason,
                                              const cv::Scalar &bgra)
    {
        preview.setTo(bgra, reasonMap == static_cast<std::uint8_t>(reason));
    };
    paint(DepthMissingReason::PatchMatchUnresolved,
          cv::Scalar(255, 0, 255, 220));
    paint(DepthMissingReason::LowConfidence, cv::Scalar(0, 255, 255, 220));
    paint(DepthMissingReason::LocalDepthOutlier, cv::Scalar(0, 128, 255, 220));
    paint(DepthMissingReason::SmallComponent, cv::Scalar(42, 42, 165, 220));
    paint(DepthMissingReason::GeometryContradiction,
          cv::Scalar(0, 0, 255, 235));
    paint(DepthMissingReason::InsufficientGeometrySupport,
          cv::Scalar(255, 200, 0, 220));
    paint(DepthMissingReason::Unclassified, cv::Scalar(160, 160, 160, 220));
    return preview;
}

} // namespace xjw::mvs
