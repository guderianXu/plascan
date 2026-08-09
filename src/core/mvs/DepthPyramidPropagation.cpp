#include "DepthPyramidPropagation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{
namespace
{

cv::Mat makeGrayGuide(const cv::Mat &guide_image, cv::Size target_size)
{
    cv::Mat resized;
    if (guide_image.size() == target_size)
    {
        resized = guide_image;
    }
    else
    {
        cv::resize(guide_image, resized, target_size, 0.0, 0.0, cv::INTER_AREA);
    }

    cv::Mat gray;
    if (resized.channels() == 3)
    {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    }
    else if (resized.channels() == 4)
    {
        cv::cvtColor(resized, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        gray = resized;
    }
    if (gray.type() != CV_32F)
    {
        gray.convertTo(gray, CV_32F);
    }
    return gray;
}

cv::Mat makeParentValidMask(const DepthLevelResult &parent)
{
    cv::Mat valid_mask = parent.depth > 0.0f;
    cv::Mat finite_mask;
    cv::compare(parent.depth,
                std::numeric_limits<float>::max(),
                finite_mask,
                cv::CMP_LE);
    cv::bitwise_and(valid_mask, finite_mask, valid_mask);
    if (parent.validMask.empty())
    {
        return valid_mask;
    }

    cv::Mat configured_mask;
    cv::compare(parent.validMask, 0, configured_mask, cv::CMP_GT);
    cv::bitwise_and(valid_mask, configured_mask, valid_mask);
    return valid_mask;
}

float parentConfidence(const DepthLevelResult &parent, int row, int column)
{
    if (parent.confidence.empty())
    {
        return 1.0f;
    }
    return std::clamp(parent.confidence.at<float>(row, column), 0.0f, 1.0f);
}

float parentUncertainty(const DepthLevelResult &parent, int row, int column, float depth)
{
    if (!parent.uncertainty.empty())
    {
        const float uncertainty = parent.uncertainty.at<float>(row, column);
        if (std::isfinite(uncertainty) && uncertainty > 0.0f)
        {
            return uncertainty;
        }
    }
    return std::max(0.01f, depth * 0.02f);
}

std::vector<int> makeNearestNeighborIndices(int native_extent, int logical_extent)
{
    std::vector<int> indices(static_cast<std::size_t>(logical_extent));
    const double scale = static_cast<double>(native_extent) / logical_extent;
    for (int logical_index = 0; logical_index < logical_extent; ++logical_index)
    {
        indices[static_cast<std::size_t>(logical_index)] = std::min(
            static_cast<int>(logical_index * scale),
            native_extent - 1);
    }
    return indices;
}

bool isLogicalParentValid(const cv::Mat &native_valid_mask,
                          int logical_row,
                          int logical_column,
                          const std::vector<int> &native_rows,
                          const std::vector<int> &native_columns)
{
    const int native_row = native_rows[static_cast<std::size_t>(logical_row)];
    const int native_column = native_columns[static_cast<std::size_t>(logical_column)];
    return native_valid_mask.at<std::uint8_t>(native_row, native_column) != 0;
}

float localDepthGradient(const DepthLevelResult &parent,
                         const cv::Mat &native_valid_mask,
                         int logical_row,
                         int logical_column,
                         float center_depth,
                         const std::vector<int> &native_rows,
                         const std::vector<int> &native_columns,
                         cv::Size logical_size)
{
    constexpr int kOffsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    float result = 0.0f;
    for (const auto &offset : kOffsets)
    {
        const int sample_row = logical_row + offset[0];
        const int sample_column = logical_column + offset[1];
        if (sample_row < 0 || sample_row >= logical_size.height ||
            sample_column < 0 || sample_column >= logical_size.width ||
            !isLogicalParentValid(native_valid_mask,
                                  sample_row,
                                  sample_column,
                                  native_rows,
                                  native_columns))
        {
            continue;
        }

        const int native_row = native_rows[static_cast<std::size_t>(sample_row)];
        const int native_column = native_columns[static_cast<std::size_t>(sample_column)];
        result = std::max(
            result,
            std::abs(parent.depth.at<float>(native_row, native_column) - center_depth));
    }
    return result;
}

struct TargetAxisMapping
{
    int baseIndex = 0;
    std::array<float, 4> spatialWeights{};
};

constexpr float kSpatialSigma = 1.2f;
constexpr float kSpatialWeightDenominator = 2.0f * kSpatialSigma * kSpatialSigma;

std::vector<TargetAxisMapping> makeTargetAxisMappings(int logical_extent, int target_extent)
{
    std::vector<TargetAxisMapping> mappings(static_cast<std::size_t>(target_extent));
    const float scale = static_cast<float>(logical_extent) / target_extent;
    for (int target_index = 0; target_index < target_extent; ++target_index)
    {
        const float parent_coordinate = (target_index + 0.5f) * scale - 0.5f;
        TargetAxisMapping &mapping = mappings[static_cast<std::size_t>(target_index)];
        mapping.baseIndex = static_cast<int>(std::floor(parent_coordinate));
        for (int delta = -1; delta <= 2; ++delta)
        {
            const float offset = mapping.baseIndex + delta - parent_coordinate;
            mapping.spatialWeights[static_cast<std::size_t>(delta + 1)] =
                std::exp(-(offset * offset) / kSpatialWeightDenominator);
        }
    }
    return mappings;
}

std::vector<int> makeParentGuideIndices(int logical_extent,
                                        int target_extent,
                                        float parent_to_target_scale)
{
    std::vector<int> indices(static_cast<std::size_t>(logical_extent));
    for (int parent_index = 0; parent_index < logical_extent; ++parent_index)
    {
        indices[static_cast<std::size_t>(parent_index)] = std::clamp(
            static_cast<int>(std::lround(
                (parent_index + 0.5f) / parent_to_target_scale - 0.5f)),
            0,
            target_extent - 1);
    }
    return indices;
}

} // namespace

DepthSearchPrior propagateDepthPrior(const DepthLevelResult &parent,
                                    const cv::Mat &guide_image,
                                    cv::Size target_size,
                                    cv::Size parent_logical_size)
{
    DepthSearchPrior result;
    if (parent.depth.empty() || parent.depth.type() != CV_32F ||
        target_size.width <= 0 || target_size.height <= 0 || guide_image.empty())
    {
        return result;
    }

    if (parent_logical_size.width <= 0 || parent_logical_size.height <= 0)
    {
        parent_logical_size = parent.depth.size();
    }

    const cv::Mat guide = makeGrayGuide(guide_image, target_size);
    const cv::Mat parent_valid_mask = makeParentValidMask(parent);
    result.center = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
    result.radius = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
    result.validMask = cv::Mat(target_size, CV_8U, cv::Scalar(0));
    if (!parent.normalMap.empty() && parent.normalMap.type() == CV_32FC3)
    {
        result.normalMap = cv::Mat(target_size, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    }

    const float scale_x = static_cast<float>(parent_logical_size.width) / target_size.width;
    const float scale_y = static_cast<float>(parent_logical_size.height) / target_size.height;
    const std::vector<int> native_columns = makeNearestNeighborIndices(
        parent.depth.cols, parent_logical_size.width);
    const std::vector<int> native_rows = makeNearestNeighborIndices(
        parent.depth.rows, parent_logical_size.height);
    const std::vector<TargetAxisMapping> column_mappings = makeTargetAxisMappings(
        parent_logical_size.width, target_size.width);
    const std::vector<TargetAxisMapping> row_mappings = makeTargetAxisMappings(
        parent_logical_size.height, target_size.height);
    const std::vector<int> parent_guide_columns = makeParentGuideIndices(
        parent_logical_size.width, target_size.width, scale_x);
    const std::vector<int> parent_guide_rows = makeParentGuideIndices(
        parent_logical_size.height, target_size.height, scale_y);
    constexpr float kColorSigma = 24.0f;

    cv::parallel_for_(cv::Range(0, target_size.height), [&](const cv::Range &rows)
    {
        for (int row = rows.start; row < rows.end; ++row)
        {
            for (int column = 0; column < target_size.width; ++column)
            {
                const TargetAxisMapping &column_mapping =
                    column_mappings[static_cast<std::size_t>(column)];
                const TargetAxisMapping &row_mapping =
                    row_mappings[static_cast<std::size_t>(row)];
                const float target_guide = guide.at<float>(row, column);

                float weight_sum = 0.0f;
                float confidence_sum = 0.0f;
                float uncertainty_sum = 0.0f;
                float gradient_sum = 0.0f;
                cv::Vec3f normal_sum(0.0f, 0.0f, 0.0f);
                std::array<std::pair<float, float>, 16> weighted_depths;
                std::size_t weighted_depth_count = 0;

                for (int delta_row = -1; delta_row <= 2; ++delta_row)
                {
                    for (int delta_column = -1; delta_column <= 2; ++delta_column)
                    {
                        const int logical_row = row_mapping.baseIndex + delta_row;
                        const int logical_column = column_mapping.baseIndex + delta_column;
                        if (logical_row < 0 || logical_row >= parent_logical_size.height ||
                            logical_column < 0 || logical_column >= parent_logical_size.width ||
                            !isLogicalParentValid(parent_valid_mask,
                                                  logical_row,
                                                  logical_column,
                                                  native_rows,
                                                  native_columns))
                        {
                            continue;
                        }

                        const int guide_column =
                            parent_guide_columns[static_cast<std::size_t>(logical_column)];
                        const int guide_row =
                            parent_guide_rows[static_cast<std::size_t>(logical_row)];
                        const float spatial_weight =
                            column_mapping.spatialWeights[
                                static_cast<std::size_t>(delta_column + 1)] *
                            row_mapping.spatialWeights[
                                static_cast<std::size_t>(delta_row + 1)];
                        const float color_difference =
                            guide.at<float>(guide_row, guide_column) - target_guide;
                        const float weight = spatial_weight *
                            std::exp(-(color_difference * color_difference) /
                                     (2.0f * kColorSigma * kColorSigma));

                        const int native_row = native_rows[static_cast<std::size_t>(logical_row)];
                        const int native_column =
                            native_columns[static_cast<std::size_t>(logical_column)];
                        const float depth = parent.depth.at<float>(native_row, native_column);
                        const float confidence = parentConfidence(parent, native_row, native_column);
                        weight_sum += weight;
                        weighted_depths[weighted_depth_count++] = {depth, weight};
                        confidence_sum += weight * confidence;
                        uncertainty_sum +=
                            weight * parentUncertainty(parent, native_row, native_column, depth);
                        gradient_sum += weight * localDepthGradient(parent,
                                                                    parent_valid_mask,
                                                                    logical_row,
                                                                    logical_column,
                                                                    depth,
                                                                    native_rows,
                                                                    native_columns,
                                                                    parent_logical_size);
                        if (!result.normalMap.empty())
                        {
                            normal_sum += weight *
                                parent.normalMap.at<cv::Vec3f>(native_row, native_column);
                        }
                    }
                }

                if (weight_sum <= 1e-6f)
                {
                    continue;
                }

                std::sort(weighted_depths.begin(),
                          weighted_depths.begin() + weighted_depth_count,
                          [](const auto &left, const auto &right)
                          {
                              return left.first < right.first;
                          });
                const float median_weight = 0.5f * weight_sum;
                float accumulated_weight = 0.0f;
                float center = weighted_depths[weighted_depth_count - 1].first;
                for (std::size_t index = 0; index < weighted_depth_count; ++index)
                {
                    const auto &[depth, weight] = weighted_depths[index];
                    accumulated_weight += weight;
                    if (accumulated_weight >= median_weight)
                    {
                        center = depth;
                        break;
                    }
                }
                const float confidence = std::max(0.1f, confidence_sum / weight_sum);
                const float uncertainty = uncertainty_sum / weight_sum;
                const float gradient_radius = 0.5f * gradient_sum / weight_sum;
                result.center.at<float>(row, column) = center;
                result.radius.at<float>(row, column) =
                    std::max(uncertainty, gradient_radius) / confidence;
                result.validMask.at<uint8_t>(row, column) = 255;

                if (!result.normalMap.empty())
                {
                    cv::Vec3f normal = normal_sum * (1.0f / weight_sum);
                    const float length = std::sqrt(normal.dot(normal));
                    if (length > 1e-6f)
                    {
                        normal *= 1.0f / length;
                    }
                    result.normalMap.at<cv::Vec3f>(row, column) = normal;
                }
            }
        }
    });
    return result;
}

} // namespace mvs
} // namespace xjw
