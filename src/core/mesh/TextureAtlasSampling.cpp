#include "TextureAtlasSampling.h"

#include "TextureOverlapExposure.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace xjw::mesh::texture_v4
{
namespace
{

cv::Vec3f bilinearColor(const cv::Mat &image, double x, double y)
{
    const int left = std::clamp(
        static_cast<int>(std::floor(x)), 0, image.cols - 1);
    const int top = std::clamp(
        static_cast<int>(std::floor(y)), 0, image.rows - 1);
    const int right = std::min(left + 1, image.cols - 1);
    const int bottom = std::min(top + 1, image.rows - 1);
    const float fx = static_cast<float>(x - left);
    const float fy = static_cast<float>(y - top);
    const cv::Vec3f first = image.at<cv::Vec3b>(top, left);
    const cv::Vec3f second = image.at<cv::Vec3b>(top, right);
    const cv::Vec3f third = image.at<cv::Vec3b>(bottom, left);
    const cv::Vec3f fourth = image.at<cv::Vec3b>(bottom, right);
    return (first * (1.0f - fx) + second * fx) * (1.0f - fy) +
           (third * (1.0f - fx) + fourth * fx) * fy;
}

float colorDistance(const cv::Vec3f &left, const cv::Vec3f &right)
{
    const cv::Vec3f difference = left - right;
    return std::sqrt(difference.dot(difference));
}

bool bilinearFootprintSupported(const PreparedView &view,
                                double x,
                                double y,
                                float *minimumDistance)
{
    if (view.supportDistance.type() != CV_32FC1 ||
        view.supportDistance.size() != view.colorBgr.size())
    {
        return false;
    }
    const int left = static_cast<int>(std::floor(x));
    const int top = static_cast<int>(std::floor(y));
    const int right = std::min(left + 1, view.supportDistance.cols - 1);
    const int bottom = std::min(top + 1, view.supportDistance.rows - 1);
    const float fx = static_cast<float>(x - left);
    const float fy = static_cast<float>(y - top);
    const std::array<std::pair<cv::Point, float>, 4> footprint{{
        {{left, top}, (1.0f - fx) * (1.0f - fy)},
        {{right, top}, fx * (1.0f - fy)},
        {{left, bottom}, (1.0f - fx) * fy},
        {{right, bottom}, fx * fy}
    }};
    float minimum_distance = std::numeric_limits<float>::infinity();
    for (const auto &[point, weight] : footprint)
    {
        if (weight <= 1.0e-6f)
        {
            continue;
        }
        const float distance =
            view.supportDistance.at<float>(point.y, point.x);
        if (!std::isfinite(distance) || distance <= 0.0f)
        {
            return false;
        }
        minimum_distance = std::min(minimum_distance, distance);
    }
    *minimumDistance = minimum_distance;
    return true;
}

TextureSampleStatus localDepthEvidence(
    const PreparedView &view,
    const std::array<double, 3> &world,
    const TextureMappingConfig &config,
    double medianEdgeLength,
    bool strict,
    float *evidenceWeight)
{
    if (!view.depth || !view.confidence || !view.depthValidMask ||
        !view.supportMask)
    {
        return TextureSampleStatus::MissingDepthEvidence;
    }
    double pixel[2]{};
    double camera_depth = 0.0;
    if (!view.evidenceCamera.projectWorldPointWithDepth(
            world.data(), pixel, camera_depth) ||
        !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        !std::isfinite(camera_depth) || camera_depth <= 0.0 ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > view.supportMask->cols - 1.0 ||
        pixel[1] > view.supportMask->rows - 1.0)
    {
        return TextureSampleStatus::Rejected;
    }
    const int column = static_cast<int>(std::lround(pixel[0]));
    const int row = static_cast<int>(std::lround(pixel[1]));
    if (row < 0 || column < 0 ||
        row >= view.supportMask->rows || column >= view.supportMask->cols ||
        view.supportMask->at<std::uint8_t>(row, column) == 0)
    {
        return TextureSampleStatus::Rejected;
    }
    if (view.depthValidMask->at<std::uint8_t>(row, column) == 0)
    {
        return TextureSampleStatus::MissingDepthEvidence;
    }

    const float observed_depth = view.depth->at<float>(row, column);
    const float confidence = view.confidence->at<float>(row, column);
    if (!std::isfinite(observed_depth) || observed_depth <= 0.0f ||
        !std::isfinite(confidence) || confidence < config.minimumConfidence)
    {
        return TextureSampleStatus::MissingDepthEvidence;
    }
    const float tolerance = std::max(
        static_cast<float>(config.edgeLengthDepthTolerance *
                           medianEdgeLength),
        config.relativeDepthTolerance *
            std::fabs(static_cast<float>(camera_depth)));
    const float allowed = strict ? tolerance : tolerance * 2.0f;
    const float residual =
        std::fabs(observed_depth - static_cast<float>(camera_depth));
    if (residual > allowed)
    {
        return TextureSampleStatus::Rejected;
    }
    const float ratio = residual / std::max(allowed, 1.0e-8f);
    *evidenceWeight = confidence * std::exp(-0.5f * ratio * ratio);
    return TextureSampleStatus::Sampled;
}

std::size_t medoidIndex(const std::vector<WeightedColor> &samples)
{
    std::size_t best_index = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    float best_weight = -1.0f;
    for (std::size_t candidate = 0; candidate < samples.size(); ++candidate)
    {
        double cost = 0.0;
        for (std::size_t other = 0; other < samples.size(); ++other)
        {
            cost += colorDistance(
                        samples[candidate].color, samples[other].color) *
                std::max(samples[other].weight, 1.0e-8f);
        }
        if (cost < best_cost - 1.0e-6 ||
            (std::fabs(cost - best_cost) <= 1.0e-6 &&
             samples[candidate].weight > best_weight))
        {
            best_index = candidate;
            best_cost = cost;
            best_weight = samples[candidate].weight;
        }
    }
    return best_index;
}

cv::Vec3b saturateColor(const cv::Vec3f &color)
{
    return cv::Vec3b(
        cv::saturate_cast<std::uint8_t>(color[0]),
        cv::saturate_cast<std::uint8_t>(color[1]),
        cv::saturate_cast<std::uint8_t>(color[2]));
}

} // namespace

TextureSampleStatus sampleTextureView(
    const PreparedView &view,
    const std::array<double, 3> &world,
    const FaceCandidate &candidate,
    const TextureMappingConfig &config,
    double medianEdgeLength,
    int padding,
    WeightedColor *sample,
    int faceIndex)
{
    if (!sample || candidate.score <= 0.0f || view.colorBgr.empty())
    {
        return TextureSampleStatus::Rejected;
    }
    double pixel[2]{};
    double color_depth = 0.0;
    if (!view.colorCamera.projectWorldPointWithDepth(
            world.data(), pixel, color_depth) ||
        !std::isfinite(pixel[0]) || !std::isfinite(pixel[1]) ||
        pixel[0] < 0.0 || pixel[1] < 0.0 ||
        pixel[0] > view.colorBgr.cols - 1.0 ||
        pixel[1] > view.colorBgr.rows - 1.0)
    {
        return TextureSampleStatus::Rejected;
    }

    float support_distance = 0.0f;
    if (!bilinearFootprintSupported(
            view, pixel[0], pixel[1], &support_distance))
    {
        return TextureSampleStatus::Rejected;
    }
    if (candidate.finalMeshVisibilityRequired && faceIndex >= 0 &&
        !isFinalMeshFaceVisible(view, faceIndex, world))
    {
        return TextureSampleStatus::Rejected;
    }
    sample->color = applyLinearSrgbExposureGain(
        bilinearColor(view.colorBgr, pixel[0], pixel[1]),
        view.exposureGain);
    const float border_weight = std::clamp(
        support_distance / std::max(static_cast<float>(padding), 1.0f),
        0.10f,
        1.0f);
    const float base_weight = std::max(candidate.score, 1.0e-8f) *
        border_weight;

    float evidence_weight = 0.0f;
    const TextureSampleStatus evidence_status = localDepthEvidence(
        view,
        world,
        config,
        medianEdgeLength,
        candidate.strict,
        &evidence_weight);
    if (evidence_status == TextureSampleStatus::MissingDepthEvidence)
    {
        sample->weight = base_weight * 0.10f;
        return evidence_status;
    }
    if (evidence_status == TextureSampleStatus::Rejected)
    {
        return evidence_status;
    }
    sample->weight = base_weight * evidence_weight;
    return TextureSampleStatus::Sampled;
}

cv::Vec3b blendTextureSamples(std::vector<WeightedColor> samples,
                              const TextureMappingConfig &config,
                              TextureMappingResult *result)
{
    if (samples.empty())
    {
        return {};
    }
    if (config.blendMode == TextureBlendMode::BestView)
    {
        return saturateColor(samples.front().color);
    }

    const cv::Vec3f reference = samples[medoidIndex(samples)].color;
    if (config.enableGhostFilter && samples.size() >= 3)
    {
        samples.erase(
            std::remove_if(samples.begin(), samples.end(), [&](const auto &sample)
            {
                if (colorDistance(sample.color, reference) <=
                    config.ghostColorThreshold)
                {
                    return false;
                }
                if (result)
                {
                    ++result->rejectedColorOutlierCount;
                }
                return true;
            }),
            samples.end());
    }

    cv::Vec3f color{};
    float total_weight = 0.0f;
    const float robust_scale = std::max(config.ghostColorThreshold, 1.0f);
    for (const WeightedColor &sample : samples)
    {
        float weight = sample.weight;
        if (config.blendMode == TextureBlendMode::Natural)
        {
            const float normalized =
                colorDistance(sample.color, reference) / robust_scale;
            weight /= 1.0f + normalized * normalized;
        }
        color += sample.color * weight;
        total_weight += weight;
    }
    if (total_weight <= 1.0e-8f)
    {
        return saturateColor(reference);
    }
    return saturateColor(color * (1.0f / total_weight));
}

float atlasCoordinateToNormalizedUv(double coordinate, int atlasSize)
{
    if (atlasSize <= 0 || !std::isfinite(coordinate))
    {
        return 0.0f;
    }
    return std::clamp(
        static_cast<float>(coordinate / static_cast<double>(atlasSize)),
        0.0f,
        1.0f);
}

} // namespace xjw::mesh::texture_v4
