#include "DepthGeometryConsistency.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xjw
{
namespace mvs
{

ProjectedDepthConsistencyResult evaluateProjectedDepthConsistency(
    const Camera &referenceCamera,
    const cv::Point2f &referencePixel,
    float referenceDepth,
    const Camera &sourceCamera,
    const cv::Mat &sourceDepth,
    float relativeThreshold,
    int searchRadius,
    float maximumRoundTripErrorPixels)
{
    ProjectedDepthConsistencyResult result;
    if (!referenceCamera.isValid() || !sourceCamera.isValid() ||
        sourceDepth.empty() || sourceDepth.type() != CV_32F ||
        !std::isfinite(referenceDepth) || referenceDepth <= 0.0f)
    {
        return result;
    }

    const double reference_pixel[2] = {
        static_cast<double>(referencePixel.x), static_cast<double>(referencePixel.y)};
    double reference_world[3] = {0.0, 0.0, 0.0};
    if (!referenceCamera.unprojectPixel(reference_pixel, referenceDepth, reference_world))
    {
        return result;
    }

    double projected_source_pixel[2] = {0.0, 0.0};
    double expected_source_depth = 0.0;
    if (!sourceCamera.projectWorldPointWithDepth(
            reference_world, projected_source_pixel, expected_source_depth) ||
        !std::isfinite(expected_source_depth) || expected_source_depth <= 0.0)
    {
        return result;
    }

    const int center_column = static_cast<int>(std::lround(projected_source_pixel[0]));
    const int center_row = static_cast<int>(std::lround(projected_source_pixel[1]));
    if (center_column < 0 || center_column >= sourceDepth.cols ||
        center_row < 0 || center_row >= sourceDepth.rows)
    {
        return result;
    }

    const float center_depth = sourceDepth.at<float>(center_row, center_column);
    result.evidence = classifyDepthConsistencyEvidence(
        static_cast<float>(expected_source_depth), center_depth, relativeThreshold);
    result.sourcePixel = cv::Point(center_column, center_row);
    if (center_depth > 0.0f && std::isfinite(center_depth))
    {
        result.relativeDepthError = std::fabs(
            center_depth - static_cast<float>(expected_source_depth)) /
            static_cast<float>(expected_source_depth);
    }

    const int radius = std::clamp(searchRadius, 0, 2);
    const float maximum_round_trip_error = std::max(0.0f, maximumRoundTripErrorPixels);
    float best_score = std::numeric_limits<float>::max();
    bool found_consistent = false;
    for (int delta_row = -radius; delta_row <= radius; ++delta_row)
    {
        for (int delta_column = -radius; delta_column <= radius; ++delta_column)
        {
            const int source_row = center_row + delta_row;
            const int source_column = center_column + delta_column;
            if (source_column < 0 || source_column >= sourceDepth.cols ||
                source_row < 0 || source_row >= sourceDepth.rows)
            {
                continue;
            }

            const float measured_depth = sourceDepth.at<float>(source_row, source_column);
            if (classifyDepthConsistencyEvidence(
                    static_cast<float>(expected_source_depth),
                    measured_depth,
                    relativeThreshold) != DepthConsistencyEvidence::Consistent)
            {
                continue;
            }

            const double source_pixel[2] = {
                static_cast<double>(source_column), static_cast<double>(source_row)};
            double measured_world[3] = {0.0, 0.0, 0.0};
            if (!sourceCamera.unprojectPixel(source_pixel, measured_depth, measured_world))
            {
                continue;
            }

            double round_trip_pixel[2] = {0.0, 0.0};
            double round_trip_depth = 0.0;
            if (!referenceCamera.projectWorldPointWithDepth(
                    measured_world, round_trip_pixel, round_trip_depth) ||
                round_trip_depth <= 0.0)
            {
                continue;
            }

            const float delta_x = static_cast<float>(round_trip_pixel[0]) - referencePixel.x;
            const float delta_y = static_cast<float>(round_trip_pixel[1]) - referencePixel.y;
            const float round_trip_error = std::sqrt(delta_x * delta_x + delta_y * delta_y);

            float depth_tolerance_pixel_error = 0.0f;
            const float threshold = std::max(0.0f, relativeThreshold);
            for (const float depth_scale : {1.0f - threshold, 1.0f + threshold})
            {
                const float tolerance_depth =
                    static_cast<float>(expected_source_depth) * std::max(0.01f, depth_scale);
                double tolerance_world[3] = {0.0, 0.0, 0.0};
                double tolerance_pixel[2] = {0.0, 0.0};
                double tolerance_reference_depth = 0.0;
                if (!sourceCamera.unprojectPixel(
                        source_pixel, tolerance_depth, tolerance_world) ||
                    !referenceCamera.projectWorldPointWithDepth(
                        tolerance_world,
                        tolerance_pixel,
                        tolerance_reference_depth) ||
                    tolerance_reference_depth <= 0.0)
                {
                    continue;
                }
                const float tolerance_delta_x =
                    static_cast<float>(tolerance_pixel[0]) - referencePixel.x;
                const float tolerance_delta_y =
                    static_cast<float>(tolerance_pixel[1]) - referencePixel.y;
                depth_tolerance_pixel_error = std::max(
                    depth_tolerance_pixel_error,
                    std::sqrt(tolerance_delta_x * tolerance_delta_x +
                              tolerance_delta_y * tolerance_delta_y));
            }
            if (round_trip_error >
                depth_tolerance_pixel_error + maximum_round_trip_error)
            {
                continue;
            }

            const float relative_error = std::fabs(
                measured_depth - static_cast<float>(expected_source_depth)) /
                static_cast<float>(expected_source_depth);
            const float projected_delta_x =
                static_cast<float>(source_column - projected_source_pixel[0]);
            const float projected_delta_y =
                static_cast<float>(source_row - projected_source_pixel[1]);
            const float source_pixel_error = std::sqrt(
                projected_delta_x * projected_delta_x + projected_delta_y * projected_delta_y);
            const float score = round_trip_error + 0.25f * source_pixel_error + relative_error;
            if (score >= best_score)
            {
                continue;
            }

            best_score = score;
            found_consistent = true;
            result.evidence = DepthConsistencyEvidence::Consistent;
            result.sourcePixel = cv::Point(source_column, source_row);
            result.relativeDepthError = relative_error;
            result.roundTripErrorPixels = round_trip_error;
            result.consistentReferenceDepth = static_cast<float>(round_trip_depth);
        }
    }

    if (!found_consistent && result.evidence == DepthConsistencyEvidence::Consistent)
    {
        result.evidence = DepthConsistencyEvidence::Unverifiable;
    }
    return result;
}

bool shouldRetainDepthFromConsistencyVotes(int sourceViewCount,
                                           int consistentVotes,
                                           int,
                                           int contradictedVotes,
                                           int minimumSourceConfirmations)
{
    if (sourceViewCount <= 1)
    {
        return contradictedVotes <= 0;
    }
    return consistentVotes >= std::clamp(
        minimumSourceConfirmations, 1, sourceViewCount);
}

cv::Mat makeGeometrySupportCount(const cv::Mat &retainedDepth,
                                 const cv::Mat &consistentVotes)
{
    if (retainedDepth.empty() || retainedDepth.type() != CV_32FC1 ||
        consistentVotes.empty() || consistentVotes.type() != CV_16UC1 ||
        retainedDepth.size() != consistentVotes.size())
    {
        return {};
    }

    cv::Mat support(retainedDepth.size(), CV_16UC1, cv::Scalar(0));
    for (int row = 0; row < retainedDepth.rows; ++row)
    {
        const float *depth_row = retainedDepth.ptr<float>(row);
        const std::uint16_t *votes_row = consistentVotes.ptr<std::uint16_t>(row);
        std::uint16_t *support_row = support.ptr<std::uint16_t>(row);
        for (int column = 0; column < retainedDepth.cols; ++column)
        {
            if (!std::isfinite(depth_row[column]) || depth_row[column] <= 0.0f)
            {
                continue;
            }
            support_row[column] = static_cast<std::uint16_t>(std::min<int>(
                std::numeric_limits<std::uint16_t>::max(),
                static_cast<int>(votes_row[column]) + 1));
        }
    }
    return support;
}

GeometryEvidenceMaps makeGeometryEvidenceMaps(
    const cv::Mat &retained_depth,
    const cv::Mat &consistent_votes,
    const cv::Mat &source_mask,
    const cv::Mat &source_inverse_depth_sum,
    const cv::Mat &source_inverse_depth_squared_sum)
{
    GeometryEvidenceMaps result;
    if (retained_depth.empty() || retained_depth.type() != CV_32FC1 ||
        consistent_votes.empty() || consistent_votes.type() != CV_16UC1 ||
        source_mask.empty() || source_mask.type() != CV_16UC1 ||
        source_inverse_depth_sum.empty() || source_inverse_depth_sum.type() != CV_32FC1 ||
        source_inverse_depth_squared_sum.empty() ||
        source_inverse_depth_squared_sum.type() != CV_32FC1 ||
        retained_depth.size() != consistent_votes.size() ||
        retained_depth.size() != source_mask.size() ||
        retained_depth.size() != source_inverse_depth_sum.size() ||
        retained_depth.size() != source_inverse_depth_squared_sum.size())
    {
        return result;
    }

    result.supportCount = cv::Mat(retained_depth.size(), CV_16UC1, cv::Scalar(0));
    result.sourceMask = cv::Mat(retained_depth.size(), CV_16UC1, cv::Scalar(0));
    result.inverseDepthMean = cv::Mat(retained_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    result.inverseDepthRelativeSpread = cv::Mat(
        retained_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    for (int row = 0; row < retained_depth.rows; ++row)
    {
        const float *depth_row = retained_depth.ptr<float>(row);
        const std::uint16_t *votes_row = consistent_votes.ptr<std::uint16_t>(row);
        const std::uint16_t *mask_row = source_mask.ptr<std::uint16_t>(row);
        const float *sum_row = source_inverse_depth_sum.ptr<float>(row);
        const float *squared_sum_row = source_inverse_depth_squared_sum.ptr<float>(row);
        std::uint16_t *support_row = result.supportCount.ptr<std::uint16_t>(row);
        std::uint16_t *output_mask_row = result.sourceMask.ptr<std::uint16_t>(row);
        float *mean_row = result.inverseDepthMean.ptr<float>(row);
        float *spread_row = result.inverseDepthRelativeSpread.ptr<float>(row);
        for (int column = 0; column < retained_depth.cols; ++column)
        {
            const float depth = depth_row[column];
            if (!std::isfinite(depth) || depth <= 0.0f)
            {
                continue;
            }
            const int source_count = static_cast<int>(votes_row[column]);
            const int observation_count = source_count + 1;
            const float reference_inverse_depth = 1.0f / depth;
            const float total_sum = sum_row[column] + reference_inverse_depth;
            const float total_squared_sum = squared_sum_row[column] +
                reference_inverse_depth * reference_inverse_depth;
            const float mean = total_sum / static_cast<float>(observation_count);
            const float variance = std::max(
                0.0f,
                total_squared_sum / static_cast<float>(observation_count) - mean * mean);
            support_row[column] = static_cast<std::uint16_t>(std::min<int>(
                observation_count,
                static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
            output_mask_row[column] = mask_row[column];
            mean_row[column] = mean;
            spread_row[column] = mean > 1.0e-12f
                ? std::sqrt(variance) / mean : 0.0f;
        }
    }
    return result;
}

} // namespace mvs
} // namespace xjw
