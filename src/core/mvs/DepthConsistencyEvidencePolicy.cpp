#include "DepthConsistencyEvidencePolicy.h"

#include <algorithm>
#include <cmath>

namespace xjw::mvs
{

WeakNativeDepthRetentionStats retainWeaklyVerifiedNativeDepth(
    const cv::Mat &originalDepth,
    const cv::Mat &originalConfidence,
    const cv::Mat &supportRegionMask,
    const cv::Mat &consistentVotes,
    const cv::Mat &contradictedVotes,
    const WeakNativeDepthRetentionOptions &options,
    cv::Mat *filteredDepth,
    cv::Mat *filteredConfidence)
{
    WeakNativeDepthRetentionStats stats;
    if (!filteredDepth ||
        originalDepth.type() != CV_32FC1 ||
        filteredDepth->type() != CV_32FC1 ||
        consistentVotes.type() != CV_16UC1 ||
        contradictedVotes.type() != CV_16UC1 ||
        originalDepth.size() != filteredDepth->size() ||
        originalDepth.size() != consistentVotes.size() ||
        originalDepth.size() != contradictedVotes.size())
    {
        return stats;
    }

    const bool use_support_mask =
        supportRegionMask.type() == CV_8UC1 &&
        supportRegionMask.size() == originalDepth.size();
    const bool use_original_confidence =
        originalConfidence.type() == CV_32FC1 &&
        originalConfidence.size() == originalDepth.size();
    const bool update_filtered_confidence =
        filteredConfidence &&
        filteredConfidence->type() == CV_32FC1 &&
        filteredConfidence->size() == originalDepth.size();
    const int minimum_confirmation_count =
        std::max(1, options.minimumConfirmationCount);
    const float confidence_multiplier =
        std::clamp(options.confidenceMultiplier, 0.0f, 1.0f);
    const float minimum_retained_confidence =
        std::clamp(options.minimumRetainedConfidence, 0.0f, 1.0f);

    for (int row = 0; row < originalDepth.rows; ++row)
    {
        const float *original_depth_row = originalDepth.ptr<float>(row);
        float *filtered_depth_row = filteredDepth->ptr<float>(row);
        const float *original_confidence_row =
            use_original_confidence ? originalConfidence.ptr<float>(row) : nullptr;
        float *filtered_confidence_row =
            update_filtered_confidence ? filteredConfidence->ptr<float>(row) : nullptr;
        const std::uint8_t *support_mask_row =
            use_support_mask ? supportRegionMask.ptr<std::uint8_t>(row) : nullptr;
        const std::uint16_t *consistent_votes_row =
            consistentVotes.ptr<std::uint16_t>(row);
        const std::uint16_t *contradicted_votes_row =
            contradictedVotes.ptr<std::uint16_t>(row);

        for (int column = 0; column < originalDepth.cols; ++column)
        {
            const float original_depth = original_depth_row[column];
            if (!std::isfinite(original_depth) ||
                original_depth <= 0.0f ||
                filtered_depth_row[column] > 0.0f ||
                (support_mask_row && support_mask_row[column] == 0))
            {
                continue;
            }

            ++stats.consideredPixelCount;
            if (contradicted_votes_row[column] > 0)
            {
                ++stats.rejectedContradictionPixelCount;
                continue;
            }
            const bool confirmed =
                consistent_votes_row[column] >= minimum_confirmation_count;
            if (!confirmed && !options.retainUnconfirmedWithoutContradiction)
            {
                ++stats.rejectedNoConfirmationPixelCount;
                continue;
            }

            filtered_depth_row[column] = original_depth;
            if (filtered_confidence_row)
            {
                const float original_confidence = original_confidence_row
                    ? original_confidence_row[column]
                    : filtered_confidence_row[column];
                filtered_confidence_row[column] =
                    std::isfinite(original_confidence)
                    ? std::max(
                          std::clamp(
                              original_confidence * confidence_multiplier,
                              0.0f,
                              1.0f),
                          std::min(
                              original_confidence,
                              minimum_retained_confidence))
                    : 0.0f;
            }
            ++stats.retainedPixelCount;
            stats.retainedUnconfirmedPixelCount += !confirmed;
        }
    }

    return stats;
}

} // namespace xjw::mvs
