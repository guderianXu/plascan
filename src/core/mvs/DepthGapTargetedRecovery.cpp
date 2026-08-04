#include "DepthGapTargetedRecovery.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw::mvs
{
namespace
{

bool compatibleDepthAndMask(const cv::Mat &depth, const cv::Mat &mask)
{
    return !depth.empty() && depth.type() == CV_32FC1 &&
        !mask.empty() && mask.type() == CV_8UC1 && depth.size() == mask.size();
}

} // namespace

DepthGapTarget buildDepthGapTarget(
    const cv::Mat &depth,
    const cv::Mat &supportMask,
    const DepthGapTargetedRecoveryOptions &options)
{
    DepthGapTarget target;
    if (!compatibleDepthAndMask(depth, supportMask))
    {
        target.skippedReason = QStringLiteral("invalid_inputs");
        return target;
    }

    cv::Mat normalized_support;
    cv::compare(supportMask, 0, normalized_support, cv::CMP_GT);
    target.supportPixelCount = cv::countNonZero(normalized_support);
    if (target.supportPixelCount <= 0)
    {
        target.skippedReason = QStringLiteral("empty_support_mask");
        return target;
    }

    cv::bitwise_and(depth <= 0.0f, normalized_support, target.gapMask);
    target.requestedGapPixelCount = cv::countNonZero(target.gapMask);
    target.requestedGapRatio = static_cast<float>(target.requestedGapPixelCount) /
        static_cast<float>(target.supportPixelCount);
    if (target.requestedGapPixelCount < std::max(1, options.minimumGapPixelCount) ||
        target.requestedGapRatio < std::max(0.0f, options.minimumGapRatio))
    {
        target.skippedReason = QStringLiteral("gap_below_trigger");
        return target;
    }
    if (target.requestedGapRatio >
        std::clamp(options.maximumGapRatio, 0.0f, 1.0f))
    {
        target.skippedReason = QStringLiteral("gap_above_safety_limit");
        return target;
    }

    const cv::Mat valid_depth = (depth > 0.0f) & normalized_support;
    if (cv::countNonZero(valid_depth) <= 0)
    {
        target.skippedReason = QStringLiteral("no_depth_anchors");
        return target;
    }

    cv::Mat distance_input(depth.size(), CV_8UC1, cv::Scalar(255));
    distance_input.setTo(cv::Scalar(0), valid_depth);
    cv::Mat distance;
    cv::Mat labels;
    cv::distanceTransform(
        distance_input, distance, labels, cv::DIST_L2, 5, cv::DIST_LABEL_PIXEL);
    double maximum_label_value = 0.0;
    cv::minMaxLoc(labels, nullptr, &maximum_label_value);
    std::vector<float> depth_by_label(
        static_cast<std::size_t>(std::max(0.0, maximum_label_value)) + 1U,
        0.0f);
    for (int row = 0; row < depth.rows; ++row)
    {
        const float *depth_row = depth.ptr<float>(row);
        const std::uint8_t *valid_row = valid_depth.ptr<std::uint8_t>(row);
        const int *label_row = labels.ptr<int>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (valid_row[column] == 0)
            {
                continue;
            }
            const int label = label_row[column];
            if (label > 0 &&
                label < static_cast<int>(depth_by_label.size()))
            {
                depth_by_label[static_cast<std::size_t>(label)] =
                    depth_row[column];
            }
        }
    }

    target.hintDepth = depth.clone();
    target.hintRadius = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    const int maximum_distance = std::max(1, options.maximumPriorDistancePixels);
    for (int row = 0; row < depth.rows; ++row)
    {
        const std::uint8_t *gap_row = target.gapMask.ptr<std::uint8_t>(row);
        const float *distance_row = distance.ptr<float>(row);
        const int *label_row = labels.ptr<int>(row);
        float *hint_row = target.hintDepth.ptr<float>(row);
        float *radius_row = target.hintRadius.ptr<float>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (gap_row[column] == 0)
            {
                if (hint_row[column] > 0.0f)
                {
                    radius_row[column] = hint_row[column] *
                        std::max(0.0f, options.anchoredPriorRadiusRatio);
                }
                continue;
            }
            const int label = label_row[column];
            const float prior = label > 0 &&
                    label < static_cast<int>(depth_by_label.size())
                ? depth_by_label[static_cast<std::size_t>(label)]
                : 0.0f;
            if (distance_row[column] > static_cast<float>(maximum_distance) ||
                !std::isfinite(prior) || prior <= 0.0f)
            {
                hint_row[column] = 0.0f;
                continue;
            }
            hint_row[column] = prior;
            radius_row[column] = prior *
                std::max(0.0f, options.missingPriorRadiusRatio);
            ++target.priorCoveredGapPixelCount;
        }
    }

    cv::Mat prior_covered_gap;
    cv::bitwise_and(target.gapMask, target.hintDepth > 0.0f,
                    prior_covered_gap);
    target.gapMask = prior_covered_gap;
    target.priorCoveredGapPixelCount = cv::countNonZero(target.gapMask);
    if (target.priorCoveredGapPixelCount <= 0)
    {
        target.skippedReason = QStringLiteral("gap_has_no_bounded_prior");
        return target;
    }

    const int dilation = std::max(0, options.estimationMaskDilationPixels);
    if (dilation > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(dilation * 2 + 1, dilation * 2 + 1));
        cv::dilate(target.gapMask, target.estimationMask, kernel);
    }
    else
    {
        target.estimationMask = target.gapMask.clone();
    }
    cv::bitwise_and(target.estimationMask,
                    normalized_support,
                    target.estimationMask);
    target.hintDepth.setTo(0.0f, target.estimationMask == 0);
    target.hintRadius.setTo(0.0f, target.estimationMask == 0);
    target.valid = true;
    return target;
}

DepthGapTargetedRecoveryStats mergeTargetedDepthGapCandidates(
    cv::Mat &depth,
    cv::Mat &confidence,
    const cv::Mat &candidateDepth,
    const cv::Mat &candidateConfidence,
    const DepthGapTarget &target,
    cv::Mat *recoveredMask,
    const DepthGapTargetedRecoveryOptions &options)
{
    DepthGapTargetedRecoveryStats stats;
    stats.supportPixelCount = target.supportPixelCount;
    stats.requestedGapPixelCount = target.requestedGapPixelCount;
    stats.priorCoveredGapPixelCount = target.priorCoveredGapPixelCount;
    stats.skippedReason = target.skippedReason;
    if (!target.valid || depth.empty() || depth.type() != CV_32FC1 ||
        candidateDepth.empty() || candidateDepth.type() != CV_32FC1 ||
        candidateConfidence.empty() || candidateConfidence.type() != CV_32FC1 ||
        depth.size() != target.gapMask.size() ||
        depth.size() != candidateDepth.size() ||
        depth.size() != candidateConfidence.size())
    {
        if (stats.skippedReason.isEmpty())
        {
            stats.skippedReason = QStringLiteral("invalid_candidates");
        }
        return stats;
    }

    stats.attempted = true;
    if (confidence.empty() || confidence.type() != CV_32FC1 ||
        confidence.size() != depth.size())
    {
        confidence = cv::Mat(depth.size(), CV_32FC1, cv::Scalar(0.0f));
    }
    cv::Mat recovered(depth.size(), CV_8UC1, cv::Scalar(0));
    const float minimum_confidence = std::clamp(
        options.minimumCandidateConfidence, 0.0f, 1.0f);
    const float maximum_relative_difference = std::max(
        0.0f, options.maximumCandidatePriorRelativeDifference);
    for (int row = 0; row < depth.rows; ++row)
    {
        float *depth_row = depth.ptr<float>(row);
        float *confidence_row = confidence.ptr<float>(row);
        const float *candidate_depth_row = candidateDepth.ptr<float>(row);
        const float *candidate_confidence_row =
            candidateConfidence.ptr<float>(row);
        const float *prior_row = target.hintDepth.ptr<float>(row);
        const std::uint8_t *gap_row = target.gapMask.ptr<std::uint8_t>(row);
        std::uint8_t *recovered_row = recovered.ptr<std::uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column)
        {
            if (gap_row[column] == 0 || depth_row[column] > 0.0f)
            {
                continue;
            }
            const float candidate_depth = candidate_depth_row[column];
            if (!std::isfinite(candidate_depth) || candidate_depth <= 0.0f)
            {
                continue;
            }
            ++stats.candidatePixelCount;
            const float candidate_confidence =
                candidate_confidence_row[column];
            if (!std::isfinite(candidate_confidence) ||
                candidate_confidence < minimum_confidence)
            {
                ++stats.rejectedConfidencePixelCount;
                continue;
            }
            const float prior = prior_row[column];
            const float relative_difference = std::fabs(candidate_depth - prior) /
                std::max(prior, 1.0e-6f);
            if (!std::isfinite(prior) || prior <= 0.0f ||
                relative_difference > maximum_relative_difference)
            {
                ++stats.rejectedPriorPixelCount;
                continue;
            }
            depth_row[column] = candidate_depth;
            confidence_row[column] = candidate_confidence;
            recovered_row[column] = 255;
            ++stats.recoveredPixelCount;
        }
    }
    if (stats.priorCoveredGapPixelCount > 0)
    {
        stats.recoveryRatio = static_cast<float>(stats.recoveredPixelCount) /
            static_cast<float>(stats.priorCoveredGapPixelCount);
    }
    if (recoveredMask)
    {
        *recoveredMask = std::move(recovered);
    }
    return stats;
}

QJsonObject depthGapTargetedRecoveryStatsToJson(
    const DepthGapTargetedRecoveryStats &stats)
{
    return QJsonObject{
        {QStringLiteral("attempted"), stats.attempted},
        {QStringLiteral("support_pixel_count"), stats.supportPixelCount},
        {QStringLiteral("requested_gap_pixel_count"),
         stats.requestedGapPixelCount},
        {QStringLiteral("prior_covered_gap_pixel_count"),
         stats.priorCoveredGapPixelCount},
        {QStringLiteral("candidate_pixel_count"), stats.candidatePixelCount},
        {QStringLiteral("recovered_pixel_count"), stats.recoveredPixelCount},
        {QStringLiteral("rejected_confidence_pixel_count"),
         stats.rejectedConfidencePixelCount},
        {QStringLiteral("rejected_prior_pixel_count"),
         stats.rejectedPriorPixelCount},
        {QStringLiteral("recovery_ratio"), stats.recoveryRatio},
        {QStringLiteral("skipped_reason"), stats.skippedReason},
        {QStringLiteral("schema_version"), 1}};
}

} // namespace xjw::mvs
