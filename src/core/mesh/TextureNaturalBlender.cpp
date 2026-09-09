#include "TextureNaturalBlender.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>

namespace xjw::mesh::texture_v4
{
    namespace
    {
        template <typename Pixel> Pixel bilinear(const cv::Mat& image, double x, double y)
        {
            x = std::clamp(x, 0.0, static_cast<double>(image.cols - 1));
            y = std::clamp(y, 0.0, static_cast<double>(image.rows - 1));
            const int left = static_cast<int>(x);
            const int top = static_cast<int>(y);
            const int right = std::min(left + 1, image.cols - 1);
            const int bottom = std::min(top + 1, image.rows - 1);
            const float dx = static_cast<float>(x - left);
            const float dy = static_cast<float>(y - top);
            return (image.at<Pixel>(top, left) * (1.0f - dx) + image.at<Pixel>(top, right) * dx) * (1.0f - dy) +
                   (image.at<Pixel>(bottom, left) * (1.0f - dx) + image.at<Pixel>(bottom, right) * dx) * dy;
        }

        cv::Mat downFour(const cv::Mat& image)
        {
            cv::Mat half;
            cv::Mat quarter;
            cv::pyrDown(image, half);
            cv::pyrDown(half, quarter);
            return quarter;
        }

        cv::Mat normalizedColor(const cv::Mat& numerator, const cv::Mat& support)
        {
            std::vector<cv::Mat> channels;
            cv::split(numerator, channels);
            cv::Mat denominator;
            cv::max(support, 1.0e-8, denominator);
            for (cv::Mat& channel : channels)
            {
                cv::divide(channel, denominator, channel);
            }
            cv::Mat result;
            cv::merge(channels, result);
            return result;
        }

        cv::Mat seedWeights(const cv::Mat& support, const cv::Mat& winner)
        {
            cv::Mat seeds;
            cv::bitwise_and(support, winner, seeds);
            if (cv::countNonZero(seeds) == 0)
            {
                return cv::Mat(support.size(), CV_32FC1, cv::Scalar(0));
            }
            cv::Mat inside;
            cv::Mat outside;
            cv::distanceTransform(seeds, inside, cv::DIST_L2, cv::DIST_MASK_5);
            cv::distanceTransform(255 - seeds, outside, cv::DIST_L2, cv::DIST_MASK_5);
            // Five-pixel signed distance transition, analogous to the reference
            // winner/non-winner distance weights; explicit support stays masked.
            cv::Mat weights = 0.5f + (inside - outside) * 0.1f;
            cv::max(weights, 0.0, weights);
            cv::min(weights, 1.0, weights);
            weights.setTo(0.0f, support == 0);
            return weights;
        }
    } // namespace

    cv::Vec3f textureSrgbToLinear(const cv::Vec3f& encoded)
    {
        cv::Vec3f linear;
        for (int channel = 0; channel < 3; ++channel)
        {
            const float value = std::clamp(encoded[channel] / 255.0f, 0.0f, 1.0f);
            linear[channel] = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
        }
        return linear;
    }

    cv::Vec3b textureLinearToSrgb(const cv::Vec3f& linear)
    {
        cv::Vec3b encoded;
        for (int channel = 0; channel < 3; ++channel)
        {
            const float value = std::clamp(linear[channel], 0.0f, 1.0f);
            encoded[channel] = cv::saturate_cast<std::uint8_t>(
                255.0f * (value <= 0.0031308f ? 12.92f * value : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f));
        }
        return encoded;
    }

    bool sampleSupportedBilinear(const cv::Mat& image,
                                 const cv::Mat& support,
                                 double x,
                                 double y,
                                 cv::Vec3f* color)
    {
        if (!color || image.type() != CV_8UC3 || support.type() != CV_8UC1 ||
            image.size() != support.size() || image.empty())
        {
            return false;
        }
        x = std::clamp(x, 0.0, static_cast<double>(image.cols - 1));
        y = std::clamp(y, 0.0, static_cast<double>(image.rows - 1));
        const int left = static_cast<int>(std::floor(x));
        const int top = static_cast<int>(std::floor(y));
        const int right = std::min(left + 1, image.cols - 1);
        const int bottom = std::min(top + 1, image.rows - 1);
        const float dx = static_cast<float>(x - left);
        const float dy = static_cast<float>(y - top);
        const std::array<std::tuple<int, int, float>, 4> taps{{
            {left, top, (1.0f - dx) * (1.0f - dy)},
            {right, top, dx * (1.0f - dy)},
            {left, bottom, (1.0f - dx) * dy},
            {right, bottom, dx * dy}}};
        cv::Vec3f sum{};
        float normalization = 0.0f;
        for (const auto& [column, row, weight] : taps)
        {
            if (support.at<std::uint8_t>(row, column) == 0)
            {
                continue;
            }
            sum += cv::Vec3f(image.at<cv::Vec3b>(row, column)) * weight;
            normalization += weight;
        }
        if (normalization <= 1.0e-8f)
        {
            return false;
        }
        *color = sum / normalization;
        return true;
    }

    TextureSourcePyramid buildTextureSourcePyramid(const cv::Mat& image,
                                                   const cv::Mat& support,
                                                   const cv::Mat& winner,
                                                   float exposure_gain,
                                                   const std::function<bool()>& is_cancelled)
    {
        if (image.empty() || image.type() != CV_8UC3 || support.type() != CV_8UC1 || winner.type() != CV_8UC1 ||
            image.size() != support.size() || image.size() != winner.size() || !std::isfinite(exposure_gain) ||
            exposure_gain <= 0.0f)
        {
            throw std::invalid_argument("invalid texture source pyramid image, masks or exposure gain");
        }
        TextureSourcePyramid pyramid;
        if (is_cancelled && is_cancelled())
        {
            return pyramid;
        }
        cv::Mat valid;
        cv::compare(support, 0, valid, cv::CMP_GT);
        pyramid.weight[0] = seedWeights(valid, winner);
        cv::Mat numerator(image.size(), CV_32FC3, cv::Scalar(0, 0, 0));
        for (int row = 0; row < image.rows; ++row)
        {
            if (row % 64 == 0 && is_cancelled && is_cancelled())
            {
                return {};
            }
            for (int column = 0; column < image.cols; ++column)
            {
                if (valid.at<std::uint8_t>(row, column) != 0)
                {
                    cv::Vec3f color = textureSrgbToLinear(cv::Vec3f(image.at<cv::Vec3b>(row, column))) * exposure_gain;
                    for (float& value : color.val)
                    {
                        value = std::min(value, 1.0f);
                    }
                    numerator.at<cv::Vec3f>(row, column) = color;
                }
            }
        }
        cv::Mat coverage;
        valid.convertTo(coverage, CV_32FC1, 1.0 / 255.0);
        cv::Mat weights = pyramid.weight[0];
        for (int level = 1; level < kTextureBlendLevels; ++level)
        {
            if (is_cancelled && is_cancelled())
            {
                return {};
            }
            numerator = downFour(numerator);
            coverage = downFour(coverage);
            weights = downFour(weights);
            pyramid.lowFrequency[level] = normalizedColor(numerator, coverage);
            cv::Mat denominator;
            cv::max(coverage, 1.0e-8, denominator);
            cv::divide(weights, denominator, pyramid.weight[level]);
            // A small low-frequency floor lets overlapping non-winner cameras
            // contribute illumination, without duplicating their fine detail.
            pyramid.weight[level] += 0.10f * static_cast<float>(level) / (kTextureBlendLevels - 1);
            pyramid.weight[level].setTo(0.0f, coverage <= 1.0e-8f);
        }
        return pyramid;
    }

    cv::Vec3f blendTexturePyramidSamples(std::span<const TexturePyramidSample> samples,
                                         bool filter_ghosts,
                                         float ghost_threshold,
                                         std::uint64_t* rejected_count)
    {
        if (samples.empty())
        {
            return {};
        }
        std::size_t reference_index = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            double cost = 0.0;
            for (const auto& other : samples)
            {
                cost += cv::norm(samples[index].encodedColor - other.encodedColor) * other.confidence;
            }
            if (cost < best_cost)
            {
                best_cost = cost;
                reference_index = index;
            }
        }
        std::array<cv::Vec3f, kTextureBlendLevels> sum{};
        std::array<float, kTextureBlendLevels> total{};
        std::array<cv::Vec3f, kTextureBlendLevels> fallback{};
        bool has_fallback = false;
        for (const auto& sample : samples)
        {
            const float distance =
                static_cast<float>(cv::norm(sample.encodedColor - samples[reference_index].encodedColor));
            if (filter_ghosts && samples.size() >= 3 && distance > ghost_threshold)
            {
                if (rejected_count)
                {
                    ++*rejected_count;
                }
                continue;
            }
            const float robust_distance = distance / std::max(ghost_threshold, 1.0f);
            const float confidence = sample.confidence / (1.0f + robust_distance * robust_distance);
            cv::Vec3f fine = textureSrgbToLinear(sample.encodedColor);
            double divisor = 1.0;
            for (int level = 0; level < kTextureBlendLevels; ++level)
            {
                cv::Vec3f coarse{};
                float weight = sample.primary ? 1.0f : 0.0f;
                if (sample.pyramid && !sample.pyramid->weight[level].empty())
                {
                    weight = bilinear<float>(
                        sample.pyramid->weight[level], sample.pixel.x / divisor, sample.pixel.y / divisor);
                    if (level + 1 < kTextureBlendLevels)
                    {
                        coarse = bilinear<cv::Vec3f>(sample.pyramid->lowFrequency[level + 1],
                                                     sample.pixel.x / (divisor * 4.0),
                                                     sample.pixel.y / (divisor * 4.0));
                    }
                }
                const cv::Vec3f band = fine - coarse;
                if (!has_fallback || sample.primary)
                {
                    fallback[level] = band;
                }
                sum[level] += band * (weight * confidence);
                total[level] += weight * confidence;
                fine = coarse;
                divisor *= 4.0;
            }
            has_fallback = true;
        }
        cv::Vec3f composed{};
        for (int level = 0; level < kTextureBlendLevels; ++level)
        {
            composed += total[level] > 1.0e-8f ? sum[level] / total[level] : fallback[level];
        }
        return composed;
    }
} // namespace xjw::mesh::texture_v4
