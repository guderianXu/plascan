#include "PatchMatchHostUtils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

namespace xjw
{
namespace mvs
{
namespace
{

bool validDepth(float depth) noexcept
{
    return std::isfinite(depth) && depth > 0.0f;
}

int normalizedMedianKernelSize(int requested)
{
    if (requested <= 1)
    {
        return 1;
    }
    return (requested % 2) == 0 ? requested + 1 : requested;
}

void maskedMedianFilter(const cv::Mat &source,
                        const cv::Mat &validMask,
                        int requestedKernelSize,
                        cv::Mat *filtered)
{
    const int kernel_size = normalizedMedianKernelSize(requestedKernelSize);
    const int radius = kernel_size / 2;
    *filtered = cv::Mat(source.size(), CV_32FC1, cv::Scalar(0.0f));

    cv::parallel_for_(cv::Range(0, source.rows), [&](const cv::Range &range)
    {
        std::vector<float> samples;
        samples.reserve(static_cast<std::size_t>(kernel_size * kernel_size));
        for (int row = range.start; row < range.end; ++row)
        {
            const std::uint8_t *valid_row = validMask.ptr<std::uint8_t>(row);
            float *output_row = filtered->ptr<float>(row);
            for (int column = 0; column < source.cols; ++column)
            {
                if (valid_row[column] == 0)
                {
                    continue;
                }

                samples.clear();
                const int first_row = std::max(0, row - radius);
                const int last_row = std::min(source.rows - 1, row + radius);
                const int first_column = std::max(0, column - radius);
                const int last_column = std::min(source.cols - 1, column + radius);
                for (int sample_row = first_row; sample_row <= last_row; ++sample_row)
                {
                    const float *source_row = source.ptr<float>(sample_row);
                    const std::uint8_t *sample_valid_row =
                        validMask.ptr<std::uint8_t>(sample_row);
                    for (int sample_column = first_column;
                         sample_column <= last_column;
                         ++sample_column)
                    {
                        if (sample_valid_row[sample_column] != 0)
                        {
                            samples.push_back(source_row[sample_column]);
                        }
                    }
                }

                const auto median = samples.begin() +
                    static_cast<std::ptrdiff_t>(samples.size() / 2);
                std::nth_element(samples.begin(), median, samples.end());
                output_row[column] = *median;
            }
        }
    });
}

void maskedRelativeBilateralFilter(const cv::Mat &source,
                                   const cv::Mat &validMask,
                                   const cv::Mat &guidance,
                                   const PatchMatchConfig &config,
                                   cv::Mat *filtered)
{
    const float sigma_space = std::max(config.bilateralSigmaSpace, 1.0e-3f);
    // Keep the legacy numeric control while giving it a dimensionless meaning:
    // 50 -> 5% log-depth sigma, 25 -> 2.5%. This preserves useful defaults and
    // removes dependence on metres, millimetres, or arbitrary SfM scale.
    const float sigma_log_depth = std::clamp(
        config.bilateralSigmaColor * 0.001f, 1.0e-4f, 1.0f);
    const int radius = config.bilateralD > 0
        ? std::max(1, config.bilateralD / 2)
        : std::max(1, static_cast<int>(std::ceil(1.5f * sigma_space)));
    const float inverse_two_sigma_space_squared =
        0.5f / (sigma_space * sigma_space);
    const float inverse_two_sigma_depth_squared =
        0.5f / (sigma_log_depth * sigma_log_depth);
    const bool has_guidance = !guidance.empty() &&
        guidance.type() == CV_32FC1 && guidance.size() == source.size();
    const float sigma_guidance = std::max(
        config.bilateralSigmaGuidance, 1.0e-3f);
    const float inverse_two_sigma_guidance_squared =
        0.5f / (sigma_guidance * sigma_guidance);

    *filtered = cv::Mat(source.size(), CV_32FC1, cv::Scalar(0.0f));
    cv::parallel_for_(cv::Range(0, source.rows), [&](const cv::Range &range)
    {
        for (int row = range.start; row < range.end; ++row)
        {
            const float *center_row = source.ptr<float>(row);
            const std::uint8_t *valid_row = validMask.ptr<std::uint8_t>(row);
            float *output_row = filtered->ptr<float>(row);
            for (int column = 0; column < source.cols; ++column)
            {
                if (valid_row[column] == 0)
                {
                    continue;
                }

                const float center_depth = center_row[column];
                const float center_guidance = has_guidance
                    ? guidance.at<float>(row, column)
                    : 0.0f;
                double weighted_depth_sum = 0.0;
                double weight_sum = 0.0;
                const int first_row = std::max(0, row - radius);
                const int last_row = std::min(source.rows - 1, row + radius);
                const int first_column = std::max(0, column - radius);
                const int last_column = std::min(source.cols - 1, column + radius);
                for (int sample_row = first_row; sample_row <= last_row; ++sample_row)
                {
                    const float *source_row = source.ptr<float>(sample_row);
                    const std::uint8_t *sample_valid_row =
                        validMask.ptr<std::uint8_t>(sample_row);
                    const int delta_row = sample_row - row;
                    for (int sample_column = first_column;
                         sample_column <= last_column;
                         ++sample_column)
                    {
                        if (sample_valid_row[sample_column] == 0)
                        {
                            continue;
                        }

                        const float sample_depth = source_row[sample_column];
                        const float log_depth_delta = std::log(
                            sample_depth / center_depth);
                        const int delta_column = sample_column - column;
                        const float spatial_distance_squared = static_cast<float>(
                            delta_row * delta_row + delta_column * delta_column);
                        const float guidance_delta = has_guidance
                            ? guidance.at<float>(sample_row, sample_column) -
                                center_guidance
                            : 0.0f;
                        const float exponent =
                            -spatial_distance_squared * inverse_two_sigma_space_squared
                            -log_depth_delta * log_depth_delta *
                                inverse_two_sigma_depth_squared
                            -(has_guidance
                                  ? guidance_delta * guidance_delta *
                                      inverse_two_sigma_guidance_squared
                                  : 0.0f);
                        const double weight = std::exp(static_cast<double>(exponent));
                        weighted_depth_sum += weight * static_cast<double>(sample_depth);
                        weight_sum += weight;
                    }
                }

                output_row[column] = weight_sum > std::numeric_limits<double>::epsilon()
                    ? static_cast<float>(weighted_depth_sum / weight_sum)
                    : center_depth;
            }
        }
    });
}

cv::Mat normalizedFilterGuidance(const cv::Mat &referenceGuide,
                                 const cv::Size &targetSize)
{
    if (referenceGuide.empty() || targetSize.empty())
    {
        return {};
    }

    cv::Mat gray;
    if (referenceGuide.channels() == 1)
    {
        gray = referenceGuide;
    }
    else if (referenceGuide.channels() == 3)
    {
        cv::cvtColor(referenceGuide, gray, cv::COLOR_BGR2GRAY);
    }
    else if (referenceGuide.channels() == 4)
    {
        cv::cvtColor(referenceGuide, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        return {};
    }

    cv::Mat resized;
    if (gray.size() == targetSize)
    {
        resized = gray;
    }
    else
    {
        cv::resize(gray, resized, targetSize, 0.0, 0.0, cv::INTER_AREA);
    }

    cv::Mat normalized;
    if (resized.depth() == CV_8U)
    {
        resized.convertTo(normalized, CV_32F, 1.0 / 255.0);
    }
    else if (resized.depth() == CV_16U)
    {
        resized.convertTo(normalized, CV_32F, 1.0 / 65535.0);
    }
    else
    {
        resized.convertTo(normalized, CV_32F);
        double maximum = 0.0;
        cv::minMaxLoc(normalized, nullptr, &maximum);
        if (maximum > 1.0)
        {
            normalized *= static_cast<float>(1.0 / maximum);
        }
    }
    return normalized;
}

} // namespace

std::array<float, 16> buildPatchMatchSourceCameraData(
    const FramePinholeCamera &reference,
    const FramePinholeCamera &source,
    int downsampleFactor)
{
    const float scale = 1.0f /
        static_cast<float>(std::max(1, downsampleFactor));
    const FramePinholeCamera::Intrinsics intrinsics = source.intrinsics();
    const std::array<double, 9> reference_rotation =
        reference.worldToCameraRotation();
    const std::array<double, 9> source_rotation =
        source.worldToCameraRotation();
    const std::array<double, 3> reference_center = reference.cameraCenter();
    const std::array<double, 3> source_center = source.cameraCenter();

    std::array<float, 16> result{};
    result[0] = static_cast<float>(intrinsics.focalX) * scale;
    result[1] = static_cast<float>(intrinsics.principalX) * scale;
    result[2] = static_cast<float>(intrinsics.focalY) * scale;
    result[3] = static_cast<float>(intrinsics.principalY) * scale;

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            double value = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                value += source_rotation[row * 3 + k] *
                    reference_rotation[column * 3 + k];
            }
            result[4 + row * 3 + column] = static_cast<float>(value);
        }
    }

    const std::array<double, 3> center_delta{{
        reference_center[0] - source_center[0],
        reference_center[1] - source_center[1],
        reference_center[2] - source_center[2]}};
    for (int row = 0; row < 3; ++row)
    {
        double value = 0.0;
        for (int column = 0; column < 3; ++column)
        {
            value += source_rotation[row * 3 + column] * center_delta[column];
        }
        result[13 + row] = static_cast<float>(value);
    }
    return result;
}

void postprocessPatchMatchDepth(cv::Mat &depth,
                                const PatchMatchConfig &config,
                                const cv::Mat &referenceGuide)
{
    if (depth.empty() || depth.type() != CV_32FC1)
    {
        return;
    }

    cv::Mat valid_mask(depth.size(), CV_8UC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, depth.rows), [&](const cv::Range &range)
    {
        for (int row = range.start; row < range.end; ++row)
        {
            float *depth_row = depth.ptr<float>(row);
            std::uint8_t *valid_row = valid_mask.ptr<std::uint8_t>(row);
            for (int column = 0; column < depth.cols; ++column)
            {
                if (validDepth(depth_row[column]))
                {
                    valid_row[column] = 255;
                }
                else
                {
                    depth_row[column] = 0.0f;
                }
            }
        }
    });

    if (config.doMedianBlur && config.medianKernelSize > 1)
    {
        cv::Mat filtered;
        maskedMedianFilter(depth, valid_mask, config.medianKernelSize, &filtered);
        depth = std::move(filtered);
    }

    if (config.doBilateralFilter)
    {
        const cv::Mat guidance = config.enableReferenceGuidedFilter
            ? normalizedFilterGuidance(referenceGuide, depth.size())
            : cv::Mat();
        cv::Mat filtered;
        maskedRelativeBilateralFilter(
            depth, valid_mask, guidance, config, &filtered);
        depth = std::move(filtered);
    }
}

} // namespace mvs
} // namespace xjw
