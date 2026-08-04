#include "DepthCrossViewHoleRepair.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>

namespace xjw::mvs
{
namespace
{

struct SourceDepthCandidate
{
    float depth = 0.0f;
    int sourceOrdinal = -1;
};

bool validDepth(float depth)
{
    return std::isfinite(depth) && depth > 0.0f;
}

float relativeDifference(float first, float second)
{
    return std::fabs(first - second) /
        std::max(1.0e-6f, std::min(first, second));
}

int bitCount(std::uint16_t mask)
{
    int count = 0;
    while (mask != 0)
    {
        mask = static_cast<std::uint16_t>(mask & (mask - 1));
        ++count;
    }
    return count;
}

bool surfaceNormalAt(const cv::Mat &depth,
                     int row,
                     int column,
                     const Camera &camera,
                     cv::Vec3f *normal)
{
    if (!normal || depth.type() != CV_32FC1 || !camera.isValid() ||
        row <= 0 || row + 1 >= depth.rows ||
        column <= 0 || column + 1 >= depth.cols)
    {
        return false;
    }
    const float left_depth = depth.at<float>(row, column - 1);
    const float right_depth = depth.at<float>(row, column + 1);
    const float upper_depth = depth.at<float>(row - 1, column);
    const float lower_depth = depth.at<float>(row + 1, column);
    if (!validDepth(left_depth) || !validDepth(right_depth) ||
        !validDepth(upper_depth) || !validDepth(lower_depth))
    {
        return false;
    }
    auto unproject = [&](double x, double y, float value, cv::Vec3f *point)
    {
        const double pixel[2] = {x, y};
        double world[3] = {};
        if (!camera.unprojectPixel(pixel, value, world))
        {
            return false;
        }
        *point = cv::Vec3f(static_cast<float>(world[0]),
                          static_cast<float>(world[1]),
                          static_cast<float>(world[2]));
        return true;
    };
    cv::Vec3f left;
    cv::Vec3f right;
    cv::Vec3f upper;
    cv::Vec3f lower;
    if (!unproject(column - 1, row, left_depth, &left) ||
        !unproject(column + 1, row, right_depth, &right) ||
        !unproject(column, row - 1, upper_depth, &upper) ||
        !unproject(column, row + 1, lower_depth, &lower))
    {
        return false;
    }
    cv::Vec3f value = (right - left).cross(lower - upper);
    const float length = std::sqrt(value.dot(value));
    if (!std::isfinite(length) || length <= 1.0e-8f)
    {
        return false;
    }
    *normal = value / length;
    return true;
}

bool normalsAgree(const cv::Mat &surface,
                  int first_row,
                  int first_column,
                  int second_row,
                  int second_column,
                  const Camera &camera,
                  float maximum_angle_degrees)
{
    cv::Vec3f first;
    cv::Vec3f second;
    if (!surfaceNormalAt(surface, first_row, first_column, camera, &first) ||
        !surfaceNormalAt(surface, second_row, second_column, camera, &second))
    {
        return false;
    }
    const float cosine = std::clamp(std::fabs(first.dot(second)), 0.0f, 1.0f);
    const float angle = std::acos(cosine) * 180.0f /
        static_cast<float>(CV_PI);
    return angle <= maximum_angle_degrees;
}

cv::Mat guideGradient(const cv::Mat *guide_gray, const cv::Size &size)
{
    if (!guide_gray || guide_gray->empty())
    {
        return {};
    }
    cv::Mat gray;
    if (guide_gray->channels() == 1)
    {
        gray = *guide_gray;
    }
    else
    {
        cv::cvtColor(*guide_gray, gray, cv::COLOR_BGR2GRAY);
    }
    if (gray.size() != size)
    {
        cv::resize(gray, gray, size, 0.0, 0.0, cv::INTER_AREA);
    }
    cv::Mat gradient_x;
    cv::Mat gradient_y;
    cv::Sobel(gray, gradient_x, CV_32FC1, 1, 0, 3);
    cv::Sobel(gray, gradient_y, CV_32FC1, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradient_x, gradient_y, magnitude);
    return magnitude;
}

bool agreesWithLocalReference(const cv::Mat &reference_depth,
                              int row,
                              int column,
                              float candidate,
                              int radius,
                              float relative_threshold)
{
    int valid_neighbor_count = 0;
    int agreeing_neighbor_count = 0;
    for (int delta_row = -radius; delta_row <= radius; ++delta_row)
    {
        for (int delta_column = -radius; delta_column <= radius; ++delta_column)
        {
            if (delta_row == 0 && delta_column == 0)
            {
                continue;
            }
            const int neighbor_row = row + delta_row;
            const int neighbor_column = column + delta_column;
            if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                neighbor_column < 0 || neighbor_column >= reference_depth.cols)
            {
                continue;
            }
            const float neighbor = reference_depth.at<float>(neighbor_row, neighbor_column);
            if (!validDepth(neighbor))
            {
                continue;
            }
            ++valid_neighbor_count;
            if (relativeDifference(candidate, neighbor) <= relative_threshold)
            {
                ++agreeing_neighbor_count;
            }
        }
    }
    return valid_neighbor_count < 3 || agreeing_neighbor_count > 0;
}

} // namespace

cv::Mat projectSourceDepthToReference(
    const cv::Mat &source_depth,
    const Camera &source_camera,
    const Camera &reference_camera,
    const cv::Size &reference_size,
    float maximum_projection_distance_pixels,
    std::uint64_t *projected_candidate_count)
{
    if (projected_candidate_count)
    {
        *projected_candidate_count = 0;
    }
    if (source_depth.empty() || source_depth.type() != CV_32FC1 ||
        !source_camera.isValid() || !reference_camera.isValid() ||
        reference_size.width <= 0 || reference_size.height <= 0)
    {
        return {};
    }

    cv::Mat projected(reference_size, CV_32FC1, cv::Scalar(0.0f));
    const float maximum_distance = std::clamp(
        maximum_projection_distance_pixels, 0.25f, 1.5f);
    std::uint64_t candidate_count = 0;
    for (int source_row = 0; source_row < source_depth.rows; ++source_row)
    {
        const float *source_values = source_depth.ptr<float>(source_row);
        for (int source_column = 0; source_column < source_depth.cols; ++source_column)
        {
            const float source_value = source_values[source_column];
            if (!validDepth(source_value))
            {
                continue;
            }
            const double source_pixel[2] = {
                static_cast<double>(source_column), static_cast<double>(source_row)};
            double world[3] = {};
            if (!source_camera.unprojectPixel(source_pixel, source_value, world))
            {
                continue;
            }
            double reference_pixel[2] = {};
            double reference_value = 0.0;
            if (!reference_camera.projectWorldPointWithDepth(
                    world, reference_pixel, reference_value) ||
                !std::isfinite(reference_value) || reference_value <= 0.0)
            {
                continue;
            }

            const int first_column = static_cast<int>(std::floor(reference_pixel[0]));
            const int first_row = static_cast<int>(std::floor(reference_pixel[1]));
            for (int delta_row = 0; delta_row <= 1; ++delta_row)
            {
                for (int delta_column = 0; delta_column <= 1; ++delta_column)
                {
                    const int column = first_column + delta_column;
                    const int row = first_row + delta_row;
                    if (column < 0 || column >= projected.cols ||
                        row < 0 || row >= projected.rows)
                    {
                        continue;
                    }
                    const double offset_x = reference_pixel[0] - column;
                    const double offset_y = reference_pixel[1] - row;
                    if (std::sqrt(offset_x * offset_x + offset_y * offset_y) >
                        maximum_distance)
                    {
                        continue;
                    }
                    float &stored = projected.at<float>(row, column);
                    const float candidate = static_cast<float>(reference_value);
                    if (!validDepth(stored) || candidate < stored)
                    {
                        stored = candidate;
                    }
                    ++candidate_count;
                }
            }
        }
    }
    if (projected_candidate_count)
    {
        *projected_candidate_count = candidate_count;
    }
    return projected;
}

DominantDepthLayerSelectionStats selectDominantProjectedDepthLayer(
    cv::Mat &reference_depth,
    const cv::Mat &support_mask,
    const std::vector<cv::Mat> &projected_source_depths,
    const cv::Mat &consistent_source_votes,
    const cv::Mat &contradicted_source_votes,
    const DominantDepthLayerSelectionOptions &options,
    cv::Mat *reference_confidence,
    cv::Mat *selected_layer_mask,
    cv::Mat *geometry_source_mask,
    cv::Mat *source_inverse_depth_sum,
    cv::Mat *source_inverse_depth_squared_sum,
    cv::Mat *selected_source_votes)
{
    DominantDepthLayerSelectionStats stats;
    if (reference_depth.type() != CV_32FC1 ||
        projected_source_depths.empty())
    {
        return stats;
    }

    const cv::Size size = reference_depth.size();
    const bool has_support = support_mask.type() == CV_8UC1 &&
        support_mask.size() == size;
    const bool has_confidence = reference_confidence &&
        reference_confidence->type() == CV_32FC1 &&
        reference_confidence->size() == size;
    const bool has_consistent_votes = consistent_source_votes.type() == CV_16UC1 &&
        consistent_source_votes.size() == size;
    const bool has_contradicted_votes = contradicted_source_votes.type() == CV_16UC1 &&
        contradicted_source_votes.size() == size;
    const bool update_selection_mask = selected_layer_mask != nullptr;
    if (update_selection_mask &&
        (selected_layer_mask->type() != CV_8UC1 ||
         selected_layer_mask->size() != size))
    {
        *selected_layer_mask = cv::Mat(size, CV_8UC1, cv::Scalar(0));
    }
    const bool update_geometry = geometry_source_mask &&
        source_inverse_depth_sum && source_inverse_depth_squared_sum &&
        geometry_source_mask->type() == CV_16UC1 &&
        source_inverse_depth_sum->type() == CV_32FC1 &&
        source_inverse_depth_squared_sum->type() == CV_32FC1 &&
        geometry_source_mask->size() == size &&
        source_inverse_depth_sum->size() == size &&
        source_inverse_depth_squared_sum->size() == size;
    const bool update_votes = selected_source_votes &&
        selected_source_votes->type() == CV_16UC1 &&
        selected_source_votes->size() == size;

    const int minimum_sources = std::max(2, options.minimumDistinctSourceCount);
    const int minimum_replacement_sources = std::max(
        minimum_sources, options.minimumReplacementSourceCount);
    const float maximum_spread = std::clamp(
        options.maximumRelativeDepthSpread, 0.001f, 0.10f);
    const float native_agreement = std::clamp(
        options.maximumNativeAgreementRelativeDifference,
        maximum_spread,
        0.20f);
    const float blend_weight = std::clamp(
        options.nativeConsensusBlendWeight, 0.0f, 1.0f);
    const float maximum_correction = std::clamp(
        options.maximumNativeRelativeCorrection, 0.0f, 0.05f);
    const float selected_confidence = std::clamp(
        options.selectedLayerConfidence, 0.0f, 1.0f);
    const float ambiguous_multiplier = std::clamp(
        options.ambiguousNativeConfidenceMultiplier, 0.0f, 1.0f);

    std::vector<SourceDepthCandidate> candidates;
    candidates.reserve(projected_source_depths.size());
    for (int row = 0; row < reference_depth.rows; ++row)
    {
        float *depth_row = reference_depth.ptr<float>(row);
        float *confidence_row = has_confidence
            ? reference_confidence->ptr<float>(row) : nullptr;
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if (has_support && support_mask.at<std::uint8_t>(row, column) == 0)
            {
                depth_row[column] = 0.0f;
                if (confidence_row)
                {
                    confidence_row[column] = 0.0f;
                }
                continue;
            }

            ++stats.consideredPixelCount;
            const float native_depth = depth_row[column];
            const bool native_valid = validDepth(native_depth);
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 || projected.size() != size)
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });

            int best_begin = -1;
            int best_count = 0;
            float best_spread = std::numeric_limits<float>::infinity();
            float best_native_distance = std::numeric_limits<float>::infinity();
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth,
                                          candidates[end].depth) <= maximum_spread)
                {
                    ++end;
                }
                const int count = end - begin;
                const float median = candidates[
                    static_cast<std::size_t>(begin + (count - 1) / 2)].depth;
                const float spread = count > 1
                    ? relativeDifference(candidates[begin].depth,
                                         candidates[end - 1].depth)
                    : 0.0f;
                const float native_distance = native_valid
                    ? relativeDifference(native_depth, median) : 0.0f;
                if (count > best_count ||
                    (count == best_count && spread < best_spread) ||
                    (count == best_count && spread == best_spread &&
                     native_distance < best_native_distance))
                {
                    best_begin = begin;
                    best_count = count;
                    best_spread = spread;
                    best_native_distance = native_distance;
                }
            }

            if (best_count < minimum_sources)
            {
                if (native_valid)
                {
                    ++stats.ambiguousNativePixelCount;
                    if (confidence_row && has_contradicted_votes &&
                        contradicted_source_votes.at<std::uint16_t>(row, column) > 0)
                    {
                        confidence_row[column] = std::clamp(
                            confidence_row[column] * ambiguous_multiplier,
                            0.0f,
                            1.0f);
                    }
                }
                else
                {
                    ++stats.unresolvedMissingPixelCount;
                }
                continue;
            }

            ++stats.stableLayerPixelCount;
            const int median_index = best_begin + (best_count - 1) / 2;
            const float selected_depth = candidates[
                static_cast<std::size_t>(median_index)].depth;
            bool selected_from_sources = false;
            if (!native_valid)
            {
                if (!options.transferObservedDepthIntoMissingPixels)
                {
                    ++stats.unresolvedMissingPixelCount;
                    continue;
                }
                depth_row[column] = selected_depth;
                selected_from_sources = true;
                ++stats.transferredMissingPixelCount;
            }
            else if (relativeDifference(native_depth, selected_depth) <= native_agreement)
            {
                const float native_inverse = 1.0f / native_depth;
                const float selected_inverse = 1.0f / selected_depth;
                float refined_depth = 1.0f /
                    ((1.0f - blend_weight) * native_inverse +
                     blend_weight * selected_inverse);
                const float maximum_delta = native_depth * maximum_correction;
                refined_depth = std::clamp(
                    refined_depth,
                    native_depth - maximum_delta,
                    native_depth + maximum_delta);
                if (std::fabs(refined_depth - native_depth) > 1.0e-7f)
                {
                    depth_row[column] = refined_depth;
                    ++stats.refinedNativePixelCount;
                }
            }
            else if (best_count >= minimum_replacement_sources &&
                     (!has_consistent_votes || !has_contradicted_votes ||
                      contradicted_source_votes.at<std::uint16_t>(row, column) >
                          consistent_source_votes.at<std::uint16_t>(row, column)))
            {
                depth_row[column] = selected_depth;
                selected_from_sources = true;
                ++stats.switchedNativePixelCount;
            }
            else
            {
                ++stats.ambiguousNativePixelCount;
                if (confidence_row)
                {
                    confidence_row[column] = std::clamp(
                        confidence_row[column] * ambiguous_multiplier,
                        0.0f,
                        1.0f);
                }
                continue;
            }

            if (selected_from_sources && update_selection_mask)
            {
                selected_layer_mask->at<std::uint8_t>(row, column) = 255;
            }
            if (confidence_row && selected_from_sources)
            {
                confidence_row[column] = std::min(
                    selected_confidence,
                    std::max(confidence_row[column], selected_confidence * 0.75f));
            }

            std::uint16_t source_bits = 0;
            float inverse_sum = 0.0f;
            float inverse_squared_sum = 0.0f;
            for (int candidate_index = best_begin;
                 candidate_index < best_begin + best_count;
                 ++candidate_index)
            {
                const SourceDepthCandidate &candidate = candidates[
                    static_cast<std::size_t>(candidate_index)];
                if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                {
                    source_bits = static_cast<std::uint16_t>(
                        source_bits |
                        (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                }
                const float inverse_depth = 1.0f / candidate.depth;
                inverse_sum += inverse_depth;
                inverse_squared_sum += inverse_depth * inverse_depth;
            }
            if (update_geometry)
            {
                geometry_source_mask->at<std::uint16_t>(row, column) = source_bits;
                source_inverse_depth_sum->at<float>(row, column) = inverse_sum;
                source_inverse_depth_squared_sum->at<float>(row, column) =
                    inverse_squared_sum;
            }
            if (update_votes)
            {
                selected_source_votes->at<std::uint16_t>(row, column) =
                    static_cast<std::uint16_t>(std::min(
                        best_count,
                        static_cast<int>(
                            std::numeric_limits<std::uint16_t>::max())));
            }
        }
    }
    return stats;
}

CrossViewHoleRepairStats repairDepthHolesFromProjectedSources(
    cv::Mat &reference_depth,
    const cv::Mat &support_mask,
    const std::vector<cv::Mat> &projected_source_depths,
    const CrossViewHoleRepairOptions &options,
    cv::Mat *reference_confidence,
    cv::Mat *consistent_source_votes,
    cv::Mat *repaired_mask,
    cv::Mat *geometry_source_mask,
    cv::Mat *source_inverse_depth_sum,
    cv::Mat *source_inverse_depth_squared_sum,
    const Camera *reference_camera,
    const cv::Mat *guide_gray,
    cv::Mat *anchored_interpolation_mask)
{
    CrossViewHoleRepairStats stats;
    if (reference_depth.empty() || reference_depth.type() != CV_32FC1 ||
        projected_source_depths.empty())
    {
        return stats;
    }
    const bool has_support = support_mask.type() == CV_8UC1 &&
        support_mask.size() == reference_depth.size();
    const bool has_confidence = reference_confidence &&
        reference_confidence->type() == CV_32FC1 &&
        reference_confidence->size() == reference_depth.size();
    const bool has_votes = consistent_source_votes &&
        consistent_source_votes->type() == CV_16UC1 &&
        consistent_source_votes->size() == reference_depth.size();
    const bool has_geometry_evidence = geometry_source_mask &&
        geometry_source_mask->type() == CV_16UC1 &&
        geometry_source_mask->size() == reference_depth.size() &&
        source_inverse_depth_sum && source_inverse_depth_sum->type() == CV_32FC1 &&
        source_inverse_depth_sum->size() == reference_depth.size() &&
        source_inverse_depth_squared_sum &&
        source_inverse_depth_squared_sum->type() == CV_32FC1 &&
        source_inverse_depth_squared_sum->size() == reference_depth.size();
    cv::Mat strong_repaired_mask = repaired_mask &&
            repaired_mask->type() == CV_8UC1 &&
            repaired_mask->size() == reference_depth.size()
        ? repaired_mask->clone()
        : cv::Mat(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    if (repaired_mask)
    {
        if (repaired_mask->type() != CV_8UC1 ||
            repaired_mask->size() != reference_depth.size())
        {
            *repaired_mask = cv::Mat(
                reference_depth.size(), CV_8UC1, cv::Scalar(0));
        }
    }
    if (anchored_interpolation_mask)
    {
        *anchored_interpolation_mask = cv::Mat(
            reference_depth.size(), CV_8UC1, cv::Scalar(0));
    }
    auto interpolate_anchored_components = [&]()
    {
        cv::Mat interpolation_anchor_mask = strong_repaired_mask.clone();
        if (options.includeValidNativeInterpolationAnchors)
        {
            for (int row = 0; row < reference_depth.rows; ++row)
            {
                const float *depth_row = reference_depth.ptr<float>(row);
                std::uint8_t *anchor_row =
                    interpolation_anchor_mask.ptr<std::uint8_t>(row);
                for (int column = 0; column < reference_depth.cols; ++column)
                {
                    if (validDepth(depth_row[column]))
                    {
                        anchor_row[column] = 255;
                    }
                }
            }
        }
        cv::Mat *interpolation_output = anchored_interpolation_mask
            ? anchored_interpolation_mask : repaired_mask;
        stats.anchoredInterpolation = interpolateAnchoredInternalDepthHoles(
            reference_depth,
            has_support
                ? support_mask
                : cv::Mat(reference_depth.size(), CV_8UC1, cv::Scalar(255)),
            interpolation_anchor_mask,
            guide_gray,
            options.anchoredInterpolation,
            has_confidence ? reference_confidence : nullptr,
            interpolation_output);
        if (anchored_interpolation_mask && repaired_mask)
        {
            cv::bitwise_or(
                *repaired_mask, *anchored_interpolation_mask, *repaired_mask);
        }
        stats.repairedPixelCount +=
            stats.anchoredInterpolation.interpolatedPixelCount;
    };
    const int minimum_sources = std::max(2, options.minimumDistinctSourceCount);
    const float maximum_spread = std::clamp(
        options.maximumRelativeDepthSpread, 0.001f, 0.10f);
    const int local_radius = std::clamp(options.localDepthRadius, 0, 3);
    const float local_threshold = std::clamp(
        options.maximumLocalRelativeDepthDifference, maximum_spread, 0.20f);
    const cv::Mat original_depth = reference_depth.clone();

    std::vector<SourceDepthCandidate> candidates;
    candidates.reserve(projected_source_depths.size());
    for (int row = 0; row < reference_depth.rows; ++row)
    {
        float *depth_row = reference_depth.ptr<float>(row);
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if (validDepth(depth_row[column]) ||
                (has_support && support_mask.at<std::uint8_t>(row, column) == 0))
            {
                continue;
            }
            ++stats.consideredHolePixelCount;
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 ||
                    projected.size() != reference_depth.size())
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                    ++stats.projectedCandidateCount;
                }
            }
            if (static_cast<int>(candidates.size()) < minimum_sources)
            {
                ++stats.rejectedInsufficientSourceCount;
                continue;
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });
            int best_begin = -1;
            int best_count = 0;
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth, candidates[end].depth) <=
                           maximum_spread)
                {
                    ++end;
                }
                const int count = end - begin;
                if (count > best_count)
                {
                    best_begin = begin;
                    best_count = count;
                }
            }
            if (best_count < minimum_sources)
            {
                ++stats.rejectedDepthSpreadCount;
                continue;
            }
            const int reference_candidate_index = best_begin + (best_count - 1) / 2;
            const float repaired_depth = candidates[
                static_cast<std::size_t>(reference_candidate_index)].depth;
            if (!agreesWithLocalReference(original_depth,
                                          row,
                                          column,
                                          repaired_depth,
                                          local_radius,
                                          local_threshold))
            {
                ++stats.rejectedLocalDepthCount;
                continue;
            }

            depth_row[column] = repaired_depth;
            if (has_confidence)
            {
                reference_confidence->at<float>(row, column) = std::clamp(
                    options.repairedConfidence + 0.05f * (best_count - minimum_sources),
                    0.0f,
                    0.85f);
            }
            if (has_votes)
            {
                consistent_source_votes->at<std::uint16_t>(row, column) =
                    static_cast<std::uint16_t>(std::min(
                        best_count - 1,
                        static_cast<int>(std::numeric_limits<std::uint16_t>::max())));
            }
            if (has_geometry_evidence)
            {
                std::uint16_t source_bits = 0;
                float inverse_sum = 0.0f;
                float inverse_squared_sum = 0.0f;
                for (int candidate_index = best_begin;
                     candidate_index < best_begin + best_count;
                     ++candidate_index)
                {
                    const SourceDepthCandidate &candidate = candidates[
                        static_cast<std::size_t>(candidate_index)];
                    if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                    {
                        source_bits = static_cast<std::uint16_t>(
                            source_bits |
                            (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                    }
                    if (candidate_index == reference_candidate_index)
                    {
                        continue;
                    }
                    const float inverse_depth = 1.0f / candidate.depth;
                    inverse_sum += inverse_depth;
                    inverse_squared_sum += inverse_depth * inverse_depth;
                }
                geometry_source_mask->at<std::uint16_t>(row, column) = source_bits;
                source_inverse_depth_sum->at<float>(row, column) = inverse_sum;
                source_inverse_depth_squared_sum->at<float>(row, column) =
                    inverse_squared_sum;
            }
            if (repaired_mask)
            {
                repaired_mask->at<std::uint8_t>(row, column) = 255;
            }
            strong_repaired_mask.at<std::uint8_t>(row, column) = 255;
            ++stats.repairedPixelCount;
        }
    }
    if (!options.enableTwoSourceGrowth || !has_votes || !has_geometry_evidence ||
        !reference_camera || !reference_camera->isValid())
    {
        interpolate_anchored_components();
        return stats;
    }

    const int maximum_growth_distance = std::clamp(
        options.maximumGrowthDistancePixels, 1, 8);
    const float maximum_growth_spread = std::clamp(
        options.maximumGrowthInverseDepthSpread, 0.001f, 0.05f);
    const float maximum_normal_angle = std::clamp(
        options.maximumGrowthNormalAngleDegrees, 5.0f, 45.0f);
    const int maximum_component_area = std::clamp(
        options.maximumGrowthComponentArea, 1, 512);
    const cv::Mat image_gradient = guideGradient(guide_gray, reference_depth.size());
    if (image_gradient.empty())
    {
        interpolate_anchored_components();
        return stats;
    }

    cv::Mat weak_mask(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat weak_depth(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_source_mask(reference_depth.size(), CV_16UC1, cv::Scalar(0));
    cv::Mat weak_inverse_sum(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_inverse_squared_sum(
        reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::Mat weak_spread(reference_depth.size(), CV_32FC1, cv::Scalar(0.0f));
    for (int row = 0; row < reference_depth.rows; ++row)
    {
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if (validDepth(reference_depth.at<float>(row, column)) ||
                (has_support && support_mask.at<std::uint8_t>(row, column) == 0))
            {
                continue;
            }
            candidates.clear();
            for (int source_ordinal = 0;
                 source_ordinal < static_cast<int>(projected_source_depths.size());
                 ++source_ordinal)
            {
                const cv::Mat &projected = projected_source_depths[
                    static_cast<std::size_t>(source_ordinal)];
                if (projected.type() != CV_32FC1 ||
                    projected.size() != reference_depth.size())
                {
                    continue;
                }
                const float candidate = projected.at<float>(row, column);
                if (validDepth(candidate))
                {
                    candidates.push_back({candidate, source_ordinal});
                }
            }
            if (candidates.size() < 2)
            {
                continue;
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const SourceDepthCandidate &left,
                         const SourceDepthCandidate &right)
                      {
                          return left.depth < right.depth;
                      });
            int best_begin = -1;
            int best_count = 0;
            for (int begin = 0; begin < static_cast<int>(candidates.size()); ++begin)
            {
                int end = begin + 1;
                while (end < static_cast<int>(candidates.size()) &&
                       relativeDifference(candidates[begin].depth,
                                          candidates[end].depth) <= maximum_spread)
                {
                    ++end;
                }
                if (end - begin > best_count)
                {
                    best_begin = begin;
                    best_count = end - begin;
                }
            }
            if (best_count != 2)
            {
                continue;
            }
            const int reference_candidate_index = best_begin;
            const float candidate_depth = candidates[
                static_cast<std::size_t>(reference_candidate_index)].depth;
            float inverse_sum = 0.0f;
            float inverse_squared_sum = 0.0f;
            float all_inverse_sum = 0.0f;
            float all_inverse_squared_sum = 0.0f;
            std::uint16_t source_bits = 0;
            for (int index = best_begin; index < best_begin + best_count; ++index)
            {
                const SourceDepthCandidate &candidate = candidates[
                    static_cast<std::size_t>(index)];
                if (candidate.sourceOrdinal >= 0 && candidate.sourceOrdinal < 16)
                {
                    source_bits = static_cast<std::uint16_t>(
                        source_bits |
                        (static_cast<std::uint16_t>(1U) << candidate.sourceOrdinal));
                }
                const float inverse_depth = 1.0f / candidate.depth;
                all_inverse_sum += inverse_depth;
                all_inverse_squared_sum += inverse_depth * inverse_depth;
                if (index != reference_candidate_index)
                {
                    inverse_sum += inverse_depth;
                    inverse_squared_sum += inverse_depth * inverse_depth;
                }
            }
            if (bitCount(source_bits) < 2)
            {
                continue;
            }
            const float inverse_mean = all_inverse_sum / best_count;
            const float inverse_variance = std::max(
                0.0f, all_inverse_squared_sum / best_count - inverse_mean * inverse_mean);
            const float relative_spread = inverse_mean > 1.0e-12f
                ? std::sqrt(inverse_variance) / inverse_mean : 1.0f;
            if (relative_spread > maximum_growth_spread)
            {
                continue;
            }
            weak_mask.at<std::uint8_t>(row, column) = 255;
            weak_depth.at<float>(row, column) = candidate_depth;
            weak_source_mask.at<std::uint16_t>(row, column) = source_bits;
            weak_inverse_sum.at<float>(row, column) = inverse_sum;
            weak_inverse_squared_sum.at<float>(row, column) = inverse_squared_sum;
            weak_spread.at<float>(row, column) = relative_spread;
            ++stats.twoSourceCandidatePixelCount;
        }
    }
    if (cv::countNonZero(weak_mask) == 0)
    {
        interpolate_anchored_components();
        return stats;
    }

    cv::Mat candidate_surface = reference_depth.clone();
    weak_depth.copyTo(candidate_surface, weak_mask);
    cv::Mat strong_mask(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < reference_depth.rows; ++row)
    {
        for (int column = 0; column < reference_depth.cols; ++column)
        {
            if (!validDepth(reference_depth.at<float>(row, column)))
            {
                continue;
            }
            if (strong_repaired_mask.at<std::uint8_t>(row, column) != 0 ||
                consistent_source_votes->at<std::uint16_t>(row, column) >= 3)
            {
                strong_mask.at<std::uint8_t>(row, column) = 255;
            }
        }
    }

    cv::Mat labels;
    cv::Mat component_stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        weak_mask, labels, component_stats, centroids, 8, CV_32S);
    struct GrowthNode
    {
        float cost = 0.0f;
        int distance = 0;
        int row = 0;
        int column = 0;
        int parentRow = 0;
        int parentColumn = 0;
    };
    struct GrowthNodeGreater
    {
        bool operator()(const GrowthNode &left, const GrowthNode &right) const
        {
            return left.cost > right.cost;
        }
    };
    constexpr int neighbor_offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    cv::Mat accepted(reference_depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat accepted_distance(
        reference_depth.size(), CV_16UC1, cv::Scalar(0xffff));
    for (int label = 1; label < component_count; ++label)
    {
        const int area = component_stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > maximum_component_area)
        {
            ++stats.growthRejectedComponentAreaCount;
            continue;
        }
        std::priority_queue<GrowthNode,
                            std::vector<GrowthNode>,
                            GrowthNodeGreater> queue;
        const int top = component_stats.at<int>(label, cv::CC_STAT_TOP);
        const int left = component_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int height = component_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int width = component_stats.at<int>(label, cv::CC_STAT_WIDTH);
        for (int row = top; row < top + height; ++row)
        {
            for (int column = left; column < left + width; ++column)
            {
                if (labels.at<int>(row, column) != label)
                {
                    continue;
                }
                for (const auto &offset : neighbor_offsets)
                {
                    const int neighbor_row = row + offset[0];
                    const int neighbor_column = column + offset[1];
                    if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                        neighbor_column < 0 || neighbor_column >= reference_depth.cols ||
                        strong_mask.at<std::uint8_t>(neighbor_row, neighbor_column) == 0)
                    {
                        continue;
                    }
                    const std::uint16_t candidate_sources =
                        weak_source_mask.at<std::uint16_t>(row, column);
                    const std::uint16_t seed_sources =
                        geometry_source_mask->at<std::uint16_t>(neighbor_row,
                                                                neighbor_column);
                    if ((candidate_sources & seed_sources) == 0)
                    {
                        ++stats.growthRejectedSourceOverlapCount;
                        continue;
                    }
                    const float gradient = image_gradient.at<float>(row, column);
                    const float cost = weak_spread.at<float>(row, column) * 1000.0f +
                        gradient * 0.01f;
                    queue.push({cost,
                                1,
                                row,
                                column,
                                neighbor_row,
                                neighbor_column});
                }
            }
        }

        while (!queue.empty())
        {
            const GrowthNode node = queue.top();
            queue.pop();
            if (accepted.at<std::uint8_t>(node.row, node.column) != 0 ||
                node.distance > maximum_growth_distance)
            {
                continue;
            }
            const float gradient = image_gradient.at<float>(node.row, node.column);
            if (gradient > options.maximumGrowthImageGradient)
            {
                ++stats.growthRejectedImageEdgeCount;
                continue;
            }
            if (!normalsAgree(candidate_surface,
                              node.row,
                              node.column,
                              node.parentRow,
                              node.parentColumn,
                              *reference_camera,
                              maximum_normal_angle))
            {
                ++stats.growthRejectedNormalCount;
                continue;
            }
            const std::uint16_t candidate_sources =
                weak_source_mask.at<std::uint16_t>(node.row, node.column);
            const std::uint16_t parent_sources =
                geometry_source_mask->at<std::uint16_t>(node.parentRow,
                                                        node.parentColumn);
            if ((candidate_sources & parent_sources) == 0)
            {
                ++stats.growthRejectedSourceOverlapCount;
                continue;
            }

            accepted.at<std::uint8_t>(node.row, node.column) = 255;
            accepted_distance.at<std::uint16_t>(node.row, node.column) =
                static_cast<std::uint16_t>(node.distance);
            reference_depth.at<float>(node.row, node.column) =
                weak_depth.at<float>(node.row, node.column);
            consistent_source_votes->at<std::uint16_t>(node.row, node.column) = 1;
            geometry_source_mask->at<std::uint16_t>(node.row, node.column) =
                candidate_sources;
            source_inverse_depth_sum->at<float>(node.row, node.column) =
                weak_inverse_sum.at<float>(node.row, node.column);
            source_inverse_depth_squared_sum->at<float>(node.row, node.column) =
                weak_inverse_squared_sum.at<float>(node.row, node.column);
            if (has_confidence)
            {
                reference_confidence->at<float>(node.row, node.column) =
                    std::min(options.repairedConfidence, 0.60f);
            }
            if (repaired_mask)
            {
                repaired_mask->at<std::uint8_t>(node.row, node.column) = 255;
            }
            ++stats.twoSourceGrownPixelCount;
            ++stats.repairedPixelCount;

            for (const auto &offset : neighbor_offsets)
            {
                const int neighbor_row = node.row + offset[0];
                const int neighbor_column = node.column + offset[1];
                if (neighbor_row < 0 || neighbor_row >= reference_depth.rows ||
                    neighbor_column < 0 || neighbor_column >= reference_depth.cols ||
                    labels.at<int>(neighbor_row, neighbor_column) != label ||
                    accepted.at<std::uint8_t>(neighbor_row, neighbor_column) != 0)
                {
                    continue;
                }
                const float cost = weak_spread.at<float>(neighbor_row, neighbor_column) *
                    1000.0f + image_gradient.at<float>(neighbor_row, neighbor_column) * 0.01f;
                queue.push({cost,
                            node.distance + 1,
                            neighbor_row,
                            neighbor_column,
                            node.row,
                            node.column});
            }
        }
    }
    interpolate_anchored_components();
    return stats;
}

QJsonObject crossViewHoleRepairStatsToJson(
    const CrossViewHoleRepairStats &stats)
{
    QJsonObject object;
    object.insert(
        QStringLiteral("projected_candidate_count"),
        static_cast<double>(stats.projectedCandidateCount));
    object.insert(
        QStringLiteral("considered_hole_pixel_count"),
        static_cast<double>(stats.consideredHolePixelCount));
    object.insert(
        QStringLiteral("rejected_insufficient_source_count"),
        static_cast<double>(stats.rejectedInsufficientSourceCount));
    object.insert(
        QStringLiteral("rejected_depth_spread_count"),
        static_cast<double>(stats.rejectedDepthSpreadCount));
    object.insert(
        QStringLiteral("rejected_local_depth_count"),
        static_cast<double>(stats.rejectedLocalDepthCount));
    object.insert(
        QStringLiteral("repaired_pixel_count"),
        static_cast<double>(stats.repairedPixelCount));
    object.insert(
        QStringLiteral("two_source_candidate_pixel_count"),
        static_cast<double>(stats.twoSourceCandidatePixelCount));
    object.insert(
        QStringLiteral("two_source_grown_pixel_count"),
        static_cast<double>(stats.twoSourceGrownPixelCount));
    object.insert(
        QStringLiteral("growth_rejected_component_area_count"),
        static_cast<double>(stats.growthRejectedComponentAreaCount));
    object.insert(
        QStringLiteral("growth_rejected_source_overlap_count"),
        static_cast<double>(stats.growthRejectedSourceOverlapCount));
    object.insert(
        QStringLiteral("growth_rejected_normal_count"),
        static_cast<double>(stats.growthRejectedNormalCount));
    object.insert(
        QStringLiteral("growth_rejected_image_edge_count"),
        static_cast<double>(stats.growthRejectedImageEdgeCount));
    object.insert(
        QStringLiteral("anchored_interpolation"),
        depthAnchoredHoleInterpolationStatsToJson(stats.anchoredInterpolation));
    return object;
}

QJsonObject dominantDepthLayerSelectionStatsToJson(
    const DominantDepthLayerSelectionStats &stats)
{
    return {
        {QStringLiteral("considered_pixel_count"),
         static_cast<double>(stats.consideredPixelCount)},
        {QStringLiteral("stable_layer_pixel_count"),
         static_cast<double>(stats.stableLayerPixelCount)},
        {QStringLiteral("refined_native_pixel_count"),
         static_cast<double>(stats.refinedNativePixelCount)},
        {QStringLiteral("switched_native_pixel_count"),
         static_cast<double>(stats.switchedNativePixelCount)},
        {QStringLiteral("transferred_missing_pixel_count"),
         static_cast<double>(stats.transferredMissingPixelCount)},
        {QStringLiteral("ambiguous_native_pixel_count"),
         static_cast<double>(stats.ambiguousNativePixelCount)},
        {QStringLiteral("unresolved_missing_pixel_count"),
         static_cast<double>(stats.unresolvedMissingPixelCount)}
    };
}

} // namespace xjw::mvs
