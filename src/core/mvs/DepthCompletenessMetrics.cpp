#include "DepthCompletenessMetrics.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw::mvs
{
namespace
{

bool isCompatible(const cv::Mat &depthMap, const cv::Mat &effectiveMask)
{
    return !depthMap.empty()
        && depthMap.type() == CV_32FC1
        && !effectiveMask.empty()
        && effectiveMask.type() == CV_8UC1
        && depthMap.size() == effectiveMask.size();
}

bool touchesMaskBoundary(const cv::Mat &labels,
                         const cv::Mat &mask,
                         int label,
                         const cv::Rect &bounds)
{
    constexpr int neighbors[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},            {0, 1},
        {1, -1},  {1, 0},   {1, 1}
    };
    for (int row = bounds.y; row < bounds.y + bounds.height; ++row)
    {
        const int *label_row = labels.ptr<int>(row);
        for (int column = bounds.x; column < bounds.x + bounds.width; ++column)
        {
            if (label_row[column] != label)
            {
                continue;
            }
            for (const auto &offset : neighbors)
            {
                const int neighbor_row = row + offset[0];
                const int neighbor_column = column + offset[1];
                if (neighbor_row < 0 || neighbor_row >= mask.rows
                    || neighbor_column < 0 || neighbor_column >= mask.cols)
                {
                    return true;
                }
                if (mask.at<std::uint8_t>(neighbor_row, neighbor_column) == 0)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

} // namespace

DepthCompletenessMetrics analyzeDepthCompleteness(const cv::Mat &depthMap,
                                                  const cv::Mat &effectiveMask,
                                                  float maximumSmallHoleFraction,
                                                  int minimumSmallHoleAreaLimit)
{
    DepthCompletenessMetrics metrics;
    if (!isCompatible(depthMap, effectiveMask))
    {
        return metrics;
    }

    metrics.validInputs = true;
    metrics.width = depthMap.cols;
    metrics.height = depthMap.rows;

    cv::Mat normalized_mask;
    cv::compare(effectiveMask, 0, normalized_mask, cv::CMP_GT);
    metrics.maskPixelCount = cv::countNonZero(normalized_mask);
    const float bounded_fraction = std::max(0.0f, maximumSmallHoleFraction);
    metrics.smallHoleAreaLimit = std::max(
        std::max(1, minimumSmallHoleAreaLimit),
        static_cast<int>(std::lround(
            static_cast<double>(metrics.maskPixelCount) * bounded_fraction)));
    if (metrics.maskPixelCount == 0)
    {
        return metrics;
    }

    cv::Mat positive_depth;
    cv::compare(depthMap, 0.0f, positive_depth, cv::CMP_GT);
    cv::Mat valid_within_mask;
    cv::bitwise_and(positive_depth, normalized_mask, valid_within_mask);
    metrics.validWithinMaskCount = cv::countNonZero(valid_within_mask);
    metrics.invalidWithinMaskCount = metrics.maskPixelCount - metrics.validWithinMaskCount;
    metrics.validWithinMaskRatio = static_cast<float>(metrics.validWithinMaskCount)
        / static_cast<float>(metrics.maskPixelCount);
    if (metrics.invalidWithinMaskCount <= 0)
    {
        return metrics;
    }

    cv::Mat invalid_within_mask;
    cv::bitwise_xor(normalized_mask, valid_within_mask, invalid_within_mask);
    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        invalid_within_mask, labels, statistics, centroids, 8, CV_32S);
    for (int label = 1; label < component_count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect bounds(statistics.at<int>(label, cv::CC_STAT_LEFT),
                              statistics.at<int>(label, cv::CC_STAT_TOP),
                              statistics.at<int>(label, cv::CC_STAT_WIDTH),
                              statistics.at<int>(label, cv::CC_STAT_HEIGHT));
        if (touchesMaskBoundary(labels, normalized_mask, label, bounds))
        {
            ++metrics.boundaryConnectedInvalidCount;
            metrics.boundaryConnectedInvalidPixelCount += area;
        }
        else if (area <= metrics.smallHoleAreaLimit)
        {
            ++metrics.smallInteriorHoleCount;
            metrics.smallInteriorHolePixelCount += area;
        }
        else
        {
            ++metrics.largeInteriorOpeningCount;
            metrics.largeInteriorOpeningPixelCount += area;
        }
    }
    return metrics;
}

int restoreSmallInteriorDepthHoles(cv::Mat &depthMap,
                                   const cv::Mat &candidateDepth,
                                   const cv::Mat &confidenceMap,
                                   const cv::Mat &effectiveMask,
                                   float minimumConfidence,
                                   float maximumRelativeDepthDifference,
                                   float maximumSmallHoleFraction,
                                   int minimumSmallHoleAreaLimit)
{
    if (!isCompatible(depthMap, effectiveMask) ||
        candidateDepth.type() != CV_32FC1 || candidateDepth.size() != depthMap.size() ||
        confidenceMap.type() != CV_32FC1 || confidenceMap.size() != depthMap.size())
    {
        return 0;
    }

    cv::Mat normalized_mask;
    cv::compare(effectiveMask, 0, normalized_mask, cv::CMP_GT);
    const int mask_pixel_count = cv::countNonZero(normalized_mask);
    if (mask_pixel_count <= 0)
    {
        return 0;
    }
    const int area_limit = std::max(
        std::max(1, minimumSmallHoleAreaLimit),
        static_cast<int>(std::lround(
            static_cast<double>(mask_pixel_count) *
            std::max(0.0f, maximumSmallHoleFraction))));

    cv::Mat positive_depth;
    cv::compare(depthMap, 0.0f, positive_depth, cv::CMP_GT);
    cv::Mat valid_within_mask;
    cv::bitwise_and(positive_depth, normalized_mask, valid_within_mask);
    cv::Mat invalid_within_mask;
    cv::bitwise_xor(normalized_mask, valid_within_mask, invalid_within_mask);

    cv::Mat labels;
    cv::Mat statistics;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        invalid_within_mask, labels, statistics, centroids, 8, CV_32S);
    const float confidence_threshold = std::clamp(minimumConfidence, 0.0f, 1.0f);
    const float relative_threshold = std::max(0.0f, maximumRelativeDepthDifference);
    int restored_count = 0;
    for (int label = 1; label < component_count; ++label)
    {
        const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
        const cv::Rect bounds(statistics.at<int>(label, cv::CC_STAT_LEFT),
                              statistics.at<int>(label, cv::CC_STAT_TOP),
                              statistics.at<int>(label, cv::CC_STAT_WIDTH),
                              statistics.at<int>(label, cv::CC_STAT_HEIGHT));
        if (area > area_limit || touchesMaskBoundary(labels, normalized_mask, label, bounds))
        {
            continue;
        }

        std::vector<float> boundary_depths;
        boundary_depths.reserve(static_cast<std::size_t>(std::max(8, area * 2)));
        const int expanded_left = std::max(0, bounds.x - 1);
        const int expanded_top = std::max(0, bounds.y - 1);
        const int expanded_right = std::min(depthMap.cols, bounds.x + bounds.width + 1);
        const int expanded_bottom = std::min(depthMap.rows, bounds.y + bounds.height + 1);
        const cv::Rect expanded(expanded_left,
                                expanded_top,
                                expanded_right - expanded_left,
                                expanded_bottom - expanded_top);
        for (int row = expanded.y; row < expanded.y + expanded.height; ++row)
        {
            for (int column = expanded.x; column < expanded.x + expanded.width; ++column)
            {
                if (labels.at<int>(row, column) == label)
                {
                    continue;
                }
                const float depth = depthMap.at<float>(row, column);
                if (normalized_mask.at<std::uint8_t>(row, column) != 0 &&
                    std::isfinite(depth) && depth > 0.0f)
                {
                    boundary_depths.push_back(depth);
                }
            }
        }
        if (boundary_depths.size() < 3)
        {
            continue;
        }
        const auto middle = boundary_depths.begin() +
            static_cast<std::ptrdiff_t>(boundary_depths.size() / 2);
        std::nth_element(boundary_depths.begin(), middle, boundary_depths.end());
        const float boundary_median = *middle;
        if (!std::isfinite(boundary_median) || boundary_median <= 0.0f)
        {
            continue;
        }

        std::vector<cv::Point> candidates;
        candidates.reserve(static_cast<std::size_t>(area));
        for (int row = bounds.y; row < bounds.y + bounds.height; ++row)
        {
            for (int column = bounds.x; column < bounds.x + bounds.width; ++column)
            {
                if (labels.at<int>(row, column) != label)
                {
                    continue;
                }
                const float candidate_depth = candidateDepth.at<float>(row, column);
                const float confidence = confidenceMap.at<float>(row, column);
                const float relative_difference = std::fabs(candidate_depth - boundary_median) /
                    boundary_median;
                if (std::isfinite(candidate_depth) && candidate_depth > 0.0f &&
                    std::isfinite(confidence) && confidence >= confidence_threshold &&
                    relative_difference <= relative_threshold)
                {
                    candidates.emplace_back(column, row);
                }
            }
        }

        if (candidates.size() * 4 < static_cast<std::size_t>(area) * 3)
        {
            continue;
        }
        for (const cv::Point &point : candidates)
        {
            depthMap.at<float>(point.y, point.x) =
                candidateDepth.at<float>(point.y, point.x);
        }
        restored_count += static_cast<int>(candidates.size());
    }
    return restored_count;
}

QJsonObject depthCompletenessMetricsToJson(const DepthCompletenessMetrics &metrics)
{
    QJsonObject object;
    if (!metrics.validInputs)
    {
        object.insert(QStringLiteral("available"), false);
        return object;
    }
    object.insert(QStringLiteral("available"), true);
    object.insert(QStringLiteral("mask_pixel_count"), metrics.maskPixelCount);
    object.insert(QStringLiteral("valid_within_mask_count"), metrics.validWithinMaskCount);
    object.insert(QStringLiteral("valid_within_mask_ratio"), metrics.validWithinMaskRatio);
    object.insert(QStringLiteral("invalid_within_mask_count"), metrics.invalidWithinMaskCount);
    object.insert(QStringLiteral("small_internal_hole_count"), metrics.smallInteriorHoleCount);
    object.insert(QStringLiteral("small_internal_hole_pixel_count"),
                  metrics.smallInteriorHolePixelCount);
    object.insert(QStringLiteral("large_internal_opening_count"),
                  metrics.largeInteriorOpeningCount);
    object.insert(QStringLiteral("large_internal_opening_pixel_count"),
                  metrics.largeInteriorOpeningPixelCount);
    object.insert(QStringLiteral("boundary_connected_invalid_count"),
                  metrics.boundaryConnectedInvalidCount);
    object.insert(QStringLiteral("boundary_connected_invalid_pixel_count"),
                  metrics.boundaryConnectedInvalidPixelCount);
    object.insert(QStringLiteral("small_hole_area_limit"), metrics.smallHoleAreaLimit);
    return object;
}

QJsonObject depthCompletenessDiagnosticsToJson(
    const DepthCompletenessDiagnostics &diagnostics)
{
    QJsonObject object = depthCompletenessMetricsToJson(diagnostics.finalMetrics);
    const auto insert_count = [&object](const QString &key, int value)
    {
        if (value >= 0)
        {
            object.insert(key, value);
        }
    };
    insert_count(QStringLiteral("pyramid_valid_count"), diagnostics.pyramidValidCount);
    insert_count(QStringLiteral("after_mask_valid_count"), diagnostics.afterMaskValidCount);
    insert_count(QStringLiteral("after_sparse_support_valid_count"),
                 diagnostics.afterSparseSupportValidCount);
    insert_count(QStringLiteral("pre_output_filter_valid_count"),
                 diagnostics.preOutputFilterValidCount);
    insert_count(QStringLiteral("post_output_filter_valid_count"),
                 diagnostics.postOutputFilterValidCount);
    object.insert(QStringLiteral("output_filter_removed_count"),
                  diagnostics.outputFilterRemovedCount);
    if (diagnostics.outputFilterRetentionRatio >= 0.0f)
    {
        object.insert(QStringLiteral("output_filter_retention_ratio"),
                      diagnostics.outputFilterRetentionRatio);
    }
    insert_count(QStringLiteral("pre_consistency_valid_count"), diagnostics.preConsistencyValidCount);
    insert_count(QStringLiteral("post_consistency_valid_count"), diagnostics.postConsistencyValidCount);
    if (diagnostics.consistencyRetentionRatio >= 0.0f)
    {
        object.insert(QStringLiteral("consistency_retention_ratio"), diagnostics.consistencyRetentionRatio);
    }
    insert_count(QStringLiteral("published_post_consistency_valid_count"),
                 diagnostics.publishedPostConsistencyValidCount);
    if (diagnostics.publishedConsistencyRetentionRatio >= 0.0f)
    {
        object.insert(QStringLiteral("published_consistency_retention_ratio"),
                      diagnostics.publishedConsistencyRetentionRatio);
    }
    object.insert(QStringLiteral("consistency_publication_fallback_applied"),
                  diagnostics.consistencyPublicationFallbackApplied);
    object.insert(QStringLiteral("consistency_confirmed_observation_count"),
                  diagnostics.consistencyConfirmedObservationCount);
    object.insert(QStringLiteral("consistency_occluded_observation_count"),
                  diagnostics.consistencyOccludedObservationCount);
    object.insert(QStringLiteral("consistency_contradicted_observation_count"),
                  diagnostics.consistencyContradictedObservationCount);
    object.insert(QStringLiteral("consistency_unverifiable_observation_count"),
                  diagnostics.consistencyUnverifiableObservationCount);
    object.insert(QStringLiteral("consistency_rejected_pixel_count"), diagnostics.consistencyRejectedPixelCount);
    object.insert(QStringLiteral("cross_view_repaired_count"), diagnostics.crossViewRepairedCount);
    insert_count(QStringLiteral("pre_fusion_postprocess_valid_count"), diagnostics.preFusionPostprocessValidCount);
    insert_count(QStringLiteral("post_confidence_filter_valid_count"),
                 diagnostics.postConfidenceFilterValidCount);
    insert_count(QStringLiteral("post_fusion_postprocess_valid_count"),
                 diagnostics.postFusionPostprocessValidCount);
    if (diagnostics.fusionPostprocessRetentionRatio >= 0.0f)
    {
        object.insert(QStringLiteral("fusion_postprocess_retention_ratio"),
                      diagnostics.fusionPostprocessRetentionRatio);
    }
    object.insert(QStringLiteral("restored_from_prefilter_count"),
                  diagnostics.restoredFromPrefilterCount);
    object.insert(QStringLiteral("restored_from_parent_level_count"),
                  diagnostics.restoredFromParentLevelCount);
    return object;
}

} // namespace xjw::mvs
