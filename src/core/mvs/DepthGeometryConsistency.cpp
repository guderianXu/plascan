#include "DepthGeometryConsistency.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace xjw
{
namespace mvs
{

namespace
{

float worldPixelFootprint(const Camera &camera, double positive_depth)
{
    const double focal_product =
        std::fabs(camera.focalX() * camera.focalY());
    if (!std::isfinite(positive_depth) || positive_depth <= 0.0 ||
        !std::isfinite(focal_product) ||
        focal_product <= std::numeric_limits<double>::epsilon())
    {
        return 0.0f;
    }
    return static_cast<float>(
        positive_depth / std::sqrt(focal_product));
}

float worldDistance(const double first[3], const double second[3])
{
    const double delta_x = first[0] - second[0];
    const double delta_y = first[1] - second[1];
    const double delta_z = first[2] - second[2];
    return static_cast<float>(std::sqrt(
        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z));
}

void assignContinuousMetrics(
    const Camera &reference_camera,
    const cv::Point2f &reference_pixel,
    float reference_depth,
    const double reference_world[3],
    const Camera &source_camera,
    const cv::Point &source_pixel,
    float source_depth,
    ProjectedDepthConsistencyResult *result)
{
    if (!result || !std::isfinite(source_depth) || source_depth <= 0.0f)
    {
        return;
    }
    const double source_pixel_array[2] = {
        static_cast<double>(source_pixel.x),
        static_cast<double>(source_pixel.y)};
    double measured_world[3] = {0.0, 0.0, 0.0};
    if (!source_camera.unprojectPixel(
            source_pixel_array, source_depth, measured_world))
    {
        return;
    }
    double round_trip_pixel[2] = {0.0, 0.0};
    double round_trip_depth = 0.0;
    if (!reference_camera.projectWorldPointWithDepth(
            measured_world, round_trip_pixel, round_trip_depth) ||
        !std::isfinite(round_trip_depth) || round_trip_depth <= 0.0)
    {
        return;
    }
    const float reference_footprint =
        worldPixelFootprint(reference_camera, reference_depth);
    const float source_footprint =
        worldPixelFootprint(source_camera, source_depth);
    const float joint_footprint =
        0.5f * (reference_footprint + source_footprint);
    if (!std::isfinite(joint_footprint) ||
        joint_footprint <= std::numeric_limits<float>::epsilon())
    {
        return;
    }
    const float delta_x =
        static_cast<float>(round_trip_pixel[0]) - reference_pixel.x;
    const float delta_y =
        static_cast<float>(round_trip_pixel[1]) - reference_pixel.y;
    result->roundTripErrorPixels =
        std::sqrt(delta_x * delta_x + delta_y * delta_y);
    result->worldSurfaceResidual =
        worldDistance(reference_world, measured_world);
    result->jointWorldPixelFootprint = joint_footprint;
    result->continuousGeometryValid =
        std::isfinite(result->roundTripErrorPixels) &&
        std::isfinite(result->worldSurfaceResidual);
}

} // namespace

ProjectedDepthConsistencyResult evaluateProjectedDepthConsistency(
    const Camera &referenceCamera,
    const cv::Point2f &referencePixel,
    float referenceDepth,
    const Camera &sourceCamera,
    const cv::Mat &sourceDepth,
    float relativeThreshold,
    int searchRadius,
    float maximumRoundTripErrorPixels,
    bool computeContinuousMetrics)
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
        if (computeContinuousMetrics)
        {
            assignContinuousMetrics(
                referenceCamera,
                referencePixel,
                referenceDepth,
                reference_world,
                sourceCamera,
                result.sourcePixel,
                center_depth,
                &result);
        }
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
            result.worldSurfaceResidual =
                worldDistance(reference_world, measured_world);
            const float reference_footprint =
                worldPixelFootprint(referenceCamera, referenceDepth);
            const float source_footprint =
                worldPixelFootprint(sourceCamera, measured_depth);
            result.jointWorldPixelFootprint =
                0.5f * (reference_footprint + source_footprint);
            result.continuousGeometryValid =
                std::isfinite(result.worldSurfaceResidual) &&
                std::isfinite(result.jointWorldPixelFootprint) &&
                result.jointWorldPixelFootprint >
                    std::numeric_limits<float>::epsilon();
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

AdaptiveGeometryEvidenceAccumulatorMaps makeAdaptiveGeometryEvidenceAccumulatorMaps(
    cv::Size size)
{
    AdaptiveGeometryEvidenceAccumulatorMaps result;
    if (size.width <= 0 || size.height <= 0)
    {
        return result;
    }
    result.positiveSupport = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    result.squaredPositiveSupport = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    result.conflict = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    result.observable = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    return result;
}

AdaptiveGeometryEvidenceMaps makeAdaptiveGeometryEvidenceMaps(
    const cv::Mat &retained_depth,
    const AdaptiveGeometryEvidenceAccumulatorMaps &accumulator_maps,
    const AdaptiveGeometryEvidenceOptions &options)
{
    AdaptiveGeometryEvidenceMaps result;
    const cv::Size size = retained_depth.size();
    if (retained_depth.empty() || retained_depth.type() != CV_32FC1 ||
        accumulator_maps.positiveSupport.empty() ||
        accumulator_maps.positiveSupport.type() != CV_32FC1 ||
        accumulator_maps.squaredPositiveSupport.empty() ||
        accumulator_maps.squaredPositiveSupport.type() != CV_32FC1 ||
        accumulator_maps.conflict.empty() ||
        accumulator_maps.conflict.type() != CV_32FC1 ||
        accumulator_maps.observable.empty() ||
        accumulator_maps.observable.type() != CV_32FC1 ||
        accumulator_maps.positiveSupport.size() != size ||
        accumulator_maps.squaredPositiveSupport.size() != size ||
        accumulator_maps.conflict.size() != size ||
        accumulator_maps.observable.size() != size)
    {
        return result;
    }

    result.supportWeight = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    result.effectiveViewCount = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    result.conflictWeight = cv::Mat(size, CV_32FC1, cv::Scalar(0.0f));
    for (int row = 0; row < retained_depth.rows; ++row)
    {
        const float *depth_row = retained_depth.ptr<float>(row);
        const float *positive_row = accumulator_maps.positiveSupport.ptr<float>(row);
        const float *squared_row =
            accumulator_maps.squaredPositiveSupport.ptr<float>(row);
        const float *conflict_row = accumulator_maps.conflict.ptr<float>(row);
        const float *observable_row = accumulator_maps.observable.ptr<float>(row);
        float *support_row = result.supportWeight.ptr<float>(row);
        float *effective_view_row = result.effectiveViewCount.ptr<float>(row);
        float *output_conflict_row = result.conflictWeight.ptr<float>(row);
        for (int column = 0; column < retained_depth.cols; ++column)
        {
            if (!std::isfinite(depth_row[column]) || depth_row[column] <= 0.0f)
            {
                continue;
            }
            AdaptiveGeometryEvidenceAccumulator accumulator;
            accumulator.positiveSupport = positive_row[column];
            accumulator.squaredPositiveSupport = squared_row[column];
            accumulator.conflict = conflict_row[column];
            accumulator.observable = observable_row[column];
            const AdaptiveGeometryEvidenceResult evidence =
                finalizeAdaptiveGeometryEvidence(accumulator, options);
            support_row[column] = evidence.supportWeight;
            effective_view_row[column] = evidence.effectiveViewCount;
            output_conflict_row[column] = evidence.conflictWeight;
        }
    }
    return result;
}

QJsonObject geometryEvidenceDiagnosticsToJson(
    const cv::Mat &depth,
    const cv::Mat &geometry_support_count,
    const cv::Mat &inverse_depth_relative_spread,
    const cv::Mat &cross_view_repaired_mask,
    const cv::Mat &support_region_mask)
{
    QJsonObject object;
    if (depth.empty() || depth.type() != CV_32FC1 ||
        geometry_support_count.empty() ||
        geometry_support_count.type() != CV_16UC1 ||
        geometry_support_count.size() != depth.size())
    {
        object.insert(QStringLiteral("valid_inputs"), false);
        return object;
    }
    const bool has_spread =
        !inverse_depth_relative_spread.empty() &&
        inverse_depth_relative_spread.type() == CV_32FC1 &&
        inverse_depth_relative_spread.size() == depth.size();
    const bool has_repaired =
        !cross_view_repaired_mask.empty() &&
        cross_view_repaired_mask.type() == CV_8UC1 &&
        cross_view_repaired_mask.size() == depth.size();
    const bool has_support_region =
        !support_region_mask.empty() &&
        support_region_mask.type() == CV_8UC1 &&
        support_region_mask.size() == depth.size();

    std::array<std::uint64_t, 6> support_histogram{};
    std::uint64_t mask_pixel_count = 0;
    std::uint64_t valid_pixel_count = 0;
    std::uint64_t native_valid_pixel_count = 0;
    std::uint64_t repaired_valid_pixel_count = 0;
    std::vector<float> spreads;
    spreads.reserve(static_cast<std::size_t>(depth.total() / 2));
    for (int row = 0; row < depth.rows; ++row)
    {
        const float *depth_row = depth.ptr<float>(row);
        const std::uint16_t *support_row =
            geometry_support_count.ptr<std::uint16_t>(row);
        const float *spread_row = has_spread
            ? inverse_depth_relative_spread.ptr<float>(row) : nullptr;
        const std::uint8_t *repaired_row = has_repaired
            ? cross_view_repaired_mask.ptr<std::uint8_t>(row) : nullptr;
        const std::uint8_t *region_row = has_support_region
            ? support_region_mask.ptr<std::uint8_t>(row) : nullptr;
        for (int column = 0; column < depth.cols; ++column)
        {
            if (region_row && region_row[column] == 0)
            {
                continue;
            }
            ++mask_pixel_count;
            const float depth_value = depth_row[column];
            if (!std::isfinite(depth_value) || depth_value <= 0.0f)
            {
                ++support_histogram[0];
                continue;
            }
            ++valid_pixel_count;
            const bool repaired = repaired_row && repaired_row[column] != 0;
            if (repaired)
            {
                ++repaired_valid_pixel_count;
            }
            else
            {
                ++native_valid_pixel_count;
            }
            const std::size_t support_bucket = std::min<std::size_t>(
                5, static_cast<std::size_t>(support_row[column]));
            ++support_histogram[support_bucket];
            if (spread_row && std::isfinite(spread_row[column]) &&
                spread_row[column] >= 0.0f)
            {
                spreads.push_back(spread_row[column]);
            }
        }
    }

    std::sort(spreads.begin(), spreads.end());
    const auto percentile = [&spreads](double quantile)
    {
        if (spreads.empty())
        {
            return 0.0;
        }
        const double position = std::clamp(quantile, 0.0, 1.0) *
            static_cast<double>(spreads.size() - 1);
        const std::size_t lower = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = std::min(lower + 1, spreads.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return static_cast<double>(spreads[lower]) * (1.0 - fraction) +
            static_cast<double>(spreads[upper]) * fraction;
    };
    QJsonObject histogram;
    histogram.insert(
        QStringLiteral("support_0"), static_cast<double>(support_histogram[0]));
    histogram.insert(
        QStringLiteral("support_1"), static_cast<double>(support_histogram[1]));
    histogram.insert(
        QStringLiteral("support_2"), static_cast<double>(support_histogram[2]));
    histogram.insert(
        QStringLiteral("support_3"), static_cast<double>(support_histogram[3]));
    histogram.insert(
        QStringLiteral("support_4"), static_cast<double>(support_histogram[4]));
    histogram.insert(
        QStringLiteral("support_5_plus"),
        static_cast<double>(support_histogram[5]));

    object.insert(QStringLiteral("valid_inputs"), true);
    object.insert(
        QStringLiteral("mask_pixel_count"),
        static_cast<double>(mask_pixel_count));
    object.insert(
        QStringLiteral("valid_pixel_count"),
        static_cast<double>(valid_pixel_count));
    object.insert(
        QStringLiteral("native_valid_pixel_count"),
        static_cast<double>(native_valid_pixel_count));
    object.insert(
        QStringLiteral("repaired_valid_pixel_count"),
        static_cast<double>(repaired_valid_pixel_count));
    object.insert(
        QStringLiteral("native_valid_ratio"),
        mask_pixel_count > 0
            ? static_cast<double>(native_valid_pixel_count) /
                  static_cast<double>(mask_pixel_count)
            : 0.0);
    object.insert(
        QStringLiteral("repaired_valid_ratio"),
        mask_pixel_count > 0
            ? static_cast<double>(repaired_valid_pixel_count) /
                  static_cast<double>(mask_pixel_count)
            : 0.0);
    object.insert(QStringLiteral("geometry_support_histogram"), histogram);
    object.insert(
        QStringLiteral("inverse_depth_spread_p50"), percentile(0.50));
    object.insert(
        QStringLiteral("inverse_depth_spread_p90"), percentile(0.90));
    object.insert(
        QStringLiteral("inverse_depth_spread_p95"), percentile(0.95));
    return object;
}

} // namespace mvs
} // namespace xjw
