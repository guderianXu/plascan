#include "TextureNaturalBlender.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace xjw::mesh::texture_v4
{
    namespace
    {

        cv::Mat srgbToLinear(const cv::Mat& encoded)
        {
            cv::Mat lookup(1, 256, CV_32FC1);
            for (int value = 0; value < 256; ++value)
            {
                const float normalized = value / 255.0f;
                lookup.at<float>(0, value) =
                    normalized <= 0.04045f ? normalized / 12.92f : std::pow((normalized + 0.055f) / 1.055f, 2.4f);
            }
            cv::Mat linear;
            cv::LUT(encoded, lookup, linear);
            return linear;
        }

        cv::Mat linearToSrgb(const cv::Mat& linear)
        {
            std::vector<cv::Mat> channels;
            cv::split(linear, channels);
            for (cv::Mat& channel : channels)
            {
                cv::max(channel, 0.0, channel);
                cv::min(channel, 1.0, channel);
                cv::Mat low_mask = channel <= 0.0031308f;
                cv::Mat power;
                cv::pow(channel, 1.0 / 2.4, power);
                cv::Mat encoded = 1.055f * power - 0.055f;
                cv::Mat low = channel * 12.92f;
                low.copyTo(encoded, low_mask);
                channel = encoded;
            }
            cv::Mat encoded_float;
            cv::merge(channels, encoded_float);
            cv::Mat encoded;
            encoded_float.convertTo(encoded, CV_8UC3, 255.0);
            return encoded;
        }

        cv::Mat normalizedLowFrequency(const cv::Mat& linear, const cv::Mat& mask, int levels)
        {
            cv::Mat weight;
            mask.convertTo(weight, CV_32FC1, 1.0 / 255.0);
            std::array<cv::Mat, 3> weights{{weight, weight, weight}};
            cv::Mat weight_three;
            cv::merge(weights.data(), static_cast<int>(weights.size()), weight_three);
            cv::Mat weighted;
            cv::multiply(linear, weight_three, weighted);

            std::vector<cv::Size> sizes{linear.size()};
            for (int level = 0; level < levels; ++level)
            {
                const cv::Size next_size(std::max(1, (weighted.cols + 1) / 2), std::max(1, (weighted.rows + 1) / 2));
                cv::Mat next_weighted;
                cv::Mat next_weight;
                cv::pyrDown(weighted, next_weighted, next_size);
                cv::pyrDown(weight, next_weight, next_size);
                weighted = std::move(next_weighted);
                weight = std::move(next_weight);
                sizes.push_back(next_size);
            }

            cv::Mat denominator;
            cv::max(weight, 1.0e-6, denominator);
            std::vector<cv::Mat> channels;
            cv::split(weighted, channels);
            for (cv::Mat& channel : channels)
            {
                cv::divide(channel, denominator, channel);
            }
            cv::Mat low_frequency;
            cv::merge(channels, low_frequency);
            for (int level = levels; level > 0; --level)
            {
                cv::Mat expanded;
                cv::pyrUp(low_frequency, expanded, sizes[static_cast<std::size_t>(level - 1)]);
                low_frequency = std::move(expanded);
            }
            return low_frequency;
        }

    } // namespace

    TextureNaturalBlendStats applyTextureNaturalBlend(cv::Mat* atlas,
                                                      const cv::Mat& primaryAtlas,
                                                      const cv::Mat& filledMask,
                                                      const cv::Mat& primaryMask,
                                                      const cv::Rect& region,
                                                      int requestedLevels,
                                                      float lowFrequencyStrength)
    {
        TextureNaturalBlendStats stats;
        if (!atlas || atlas->type() != CV_8UC3 || primaryAtlas.type() != CV_8UC3 || filledMask.type() != CV_8UC1 ||
            primaryMask.type() != CV_8UC1 || atlas->size() != primaryAtlas.size() ||
            atlas->size() != filledMask.size() || atlas->size() != primaryMask.size())
        {
            return stats;
        }
        cv::Rect roi = region & cv::Rect(0, 0, atlas->cols, atlas->rows);
        if (roi.empty() || lowFrequencyStrength <= 0.0f)
        {
            return stats;
        }
        cv::Mat correction_mask;
        cv::bitwise_and(filledMask(roi), primaryMask(roi), correction_mask);
        stats.correctedPixelCount = cv::countNonZero(correction_mask);
        if (stats.correctedPixelCount == 0)
        {
            return stats;
        }

        int levels = std::clamp(requestedLevels, 1, 5);
        int minimum_dimension = std::min(roi.width, roi.height);
        while (levels > 1 && minimum_dimension < (1 << levels))
        {
            --levels;
        }
        stats.pyramidLevels = levels;

        const cv::Mat robust_linear = srgbToLinear((*atlas)(roi));
        const cv::Mat primary_linear = srgbToLinear(primaryAtlas(roi));
        const cv::Mat robust_low = normalizedLowFrequency(robust_linear, filledMask(roi), levels);
        const cv::Mat primary_low = normalizedLowFrequency(primary_linear, primaryMask(roi), levels);
        cv::Mat correction = (robust_low - primary_low) * std::clamp(lowFrequencyStrength, 0.0f, 1.0f);
        const cv::Scalar mean_correction = cv::mean(cv::abs(correction), correction_mask);
        stats.meanAbsoluteLinearCorrection = (mean_correction[0] + mean_correction[1] + mean_correction[2]) / 3.0;

        cv::Mat composed = primary_linear + correction;
        cv::max(composed, 0.0, composed);
        cv::min(composed, 1.0, composed);
        const cv::Mat encoded = linearToSrgb(composed);
        encoded.copyTo((*atlas)(roi), correction_mask);
        return stats;
    }

} // namespace xjw::mesh::texture_v4
