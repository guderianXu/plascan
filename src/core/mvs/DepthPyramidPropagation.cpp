#include "DepthPyramidPropagation.h"

#include <algorithm>
#include <cmath>

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

bool isParentValid(const DepthLevelResult &parent, int row, int column)
{
    const float depth = parent.depth.at<float>(row, column);
    if (!std::isfinite(depth) || depth <= 0.0f)
    {
        return false;
    }
    return parent.validMask.empty() || parent.validMask.at<uint8_t>(row, column) != 0;
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

float localDepthGradient(const DepthLevelResult &parent, int row, int column, float center_depth)
{
    constexpr int kOffsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    float result = 0.0f;
    for (const auto &offset : kOffsets)
    {
        const int sample_row = row + offset[0];
        const int sample_column = column + offset[1];
        if (sample_row < 0 || sample_row >= parent.depth.rows ||
            sample_column < 0 || sample_column >= parent.depth.cols ||
            !isParentValid(parent, sample_row, sample_column))
        {
            continue;
        }
        result = std::max(result,
                          std::abs(parent.depth.at<float>(sample_row, sample_column) - center_depth));
    }
    return result;
}

} // namespace

DepthSearchPrior propagateDepthPrior(const DepthLevelResult &parent,
                                    const cv::Mat &guide_image,
                                    cv::Size target_size)
{
    DepthSearchPrior result;
    if (parent.depth.empty() || parent.depth.type() != CV_32F ||
        target_size.width <= 0 || target_size.height <= 0 || guide_image.empty())
    {
        return result;
    }

    const cv::Mat guide = makeGrayGuide(guide_image, target_size);
    result.center = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
    result.radius = cv::Mat(target_size, CV_32F, cv::Scalar(0.0f));
    result.validMask = cv::Mat(target_size, CV_8U, cv::Scalar(0));
    if (!parent.normalMap.empty() && parent.normalMap.type() == CV_32FC3)
    {
        result.normalMap = cv::Mat(target_size, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    }

    const float scale_x = static_cast<float>(parent.depth.cols) / target_size.width;
    const float scale_y = static_cast<float>(parent.depth.rows) / target_size.height;
    constexpr float kSpatialSigma = 1.2f;
    constexpr float kColorSigma = 24.0f;

    for (int row = 0; row < target_size.height; ++row)
    {
        for (int column = 0; column < target_size.width; ++column)
        {
            const float parent_x = (column + 0.5f) * scale_x - 0.5f;
            const float parent_y = (row + 0.5f) * scale_y - 0.5f;
            const int base_column = static_cast<int>(std::floor(parent_x));
            const int base_row = static_cast<int>(std::floor(parent_y));
            const float target_guide = guide.at<float>(row, column);

            float weight_sum = 0.0f;
            float depth_sum = 0.0f;
            float confidence_sum = 0.0f;
            float uncertainty_sum = 0.0f;
            float gradient_sum = 0.0f;
            cv::Vec3f normal_sum(0.0f, 0.0f, 0.0f);

            for (int delta_row = -1; delta_row <= 2; ++delta_row)
            {
                for (int delta_column = -1; delta_column <= 2; ++delta_column)
                {
                    const int sample_row = base_row + delta_row;
                    const int sample_column = base_column + delta_column;
                    if (sample_row < 0 || sample_row >= parent.depth.rows ||
                        sample_column < 0 || sample_column >= parent.depth.cols ||
                        !isParentValid(parent, sample_row, sample_column))
                    {
                        continue;
                    }

                    const int guide_column = std::clamp(
                        static_cast<int>(std::lround((sample_column + 0.5f) / scale_x - 0.5f)),
                        0,
                        target_size.width - 1);
                    const int guide_row = std::clamp(
                        static_cast<int>(std::lround((sample_row + 0.5f) / scale_y - 0.5f)),
                        0,
                        target_size.height - 1);
                    const float spatial_distance =
                        (sample_column - parent_x) * (sample_column - parent_x) +
                        (sample_row - parent_y) * (sample_row - parent_y);
                    const float color_difference = guide.at<float>(guide_row, guide_column) - target_guide;
                    const float weight = std::exp(-spatial_distance / (2.0f * kSpatialSigma * kSpatialSigma)) *
                                         std::exp(-(color_difference * color_difference) /
                                                  (2.0f * kColorSigma * kColorSigma));

                    const float depth = parent.depth.at<float>(sample_row, sample_column);
                    const float confidence = parentConfidence(parent, sample_row, sample_column);
                    weight_sum += weight;
                    depth_sum += weight * depth;
                    confidence_sum += weight * confidence;
                    uncertainty_sum += weight * parentUncertainty(parent, sample_row, sample_column, depth);
                    gradient_sum += weight * localDepthGradient(parent, sample_row, sample_column, depth);
                    if (!result.normalMap.empty())
                    {
                        normal_sum += weight * parent.normalMap.at<cv::Vec3f>(sample_row, sample_column);
                    }
                }
            }

            if (weight_sum <= 1e-6f)
            {
                continue;
            }

            const float center = depth_sum / weight_sum;
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
    return result;
}

} // namespace mvs
} // namespace xjw
