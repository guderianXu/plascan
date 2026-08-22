#include "TextureSeamLeveling.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace xjw::mesh::texture_v4
{
namespace
{

float srgbToLinear(float value)
{
    const float normalized = std::clamp(value / 255.0f, 0.0f, 1.0f);
    return normalized <= 0.04045f
        ? normalized / 12.92f
        : std::pow((normalized + 0.055f) / 1.055f, 2.4f);
}

float linearToSrgb(float value)
{
    const float normalized = std::clamp(value, 0.0f, 1.0f);
    const float encoded = normalized <= 0.0031308f
        ? 12.92f * normalized
        : 1.055f * std::pow(normalized, 1.0f / 2.4f) - 0.055f;
    return 255.0f * encoded;
}

struct NeighborConstraint
{
    int neighbor = -1;
    cv::Vec3f targetDifference{};
    float weight = 0.0f;
};

} // namespace

TextureSeamLevelingStats applyTextureSeamLeveling(
    cv::Mat *atlasBgr,
    const cv::Mat &chartIndexMap,
    const std::vector<TextureSeamConstraint> &constraints,
    int chartCount,
    int borderBlendRadiusPixels,
    float maximumAbsoluteLinearCorrection,
    float globalCorrectionStrength)
{
    TextureSeamLevelingStats stats;
    if (!atlasBgr || atlasBgr->type() != CV_8UC3 ||
        chartIndexMap.type() != CV_32SC1 ||
        chartIndexMap.size() != atlasBgr->size() || chartCount <= 0)
    {
        return stats;
    }

    std::vector<std::vector<NeighborConstraint>> graph(
        static_cast<std::size_t>(chartCount));
    for (const TextureSeamConstraint &constraint : constraints)
    {
        if (constraint.firstChart < 0 || constraint.firstChart >= chartCount ||
            constraint.secondChart < 0 || constraint.secondChart >= chartCount ||
            constraint.firstChart == constraint.secondChart ||
            !std::isfinite(constraint.weight) || constraint.weight <= 0.0f)
        {
            continue;
        }
        const cv::Vec3f difference = constraint.differenceLinearBgr;
        if (!std::isfinite(difference[0]) ||
            !std::isfinite(difference[1]) ||
            !std::isfinite(difference[2]))
        {
            continue;
        }
        graph[static_cast<std::size_t>(constraint.firstChart)].push_back(
            {constraint.secondChart, difference, constraint.weight});
        graph[static_cast<std::size_t>(constraint.secondChart)].push_back(
            {constraint.firstChart, -difference, constraint.weight});
        ++stats.constraintCount;
    }
    if (stats.constraintCount == 0)
    {
        return stats;
    }

    std::vector<cv::Vec3f> offsets(
        static_cast<std::size_t>(chartCount), cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::vector<cv::Vec3f> next = offsets;
    for (int iteration = 0; iteration < 96; ++iteration)
    {
        for (int chart = 0; chart < chartCount; ++chart)
        {
            const auto &neighbors = graph[static_cast<std::size_t>(chart)];
            if (neighbors.empty())
            {
                next[static_cast<std::size_t>(chart)] = cv::Vec3f{};
                continue;
            }
            cv::Vec3f sum{};
            float weight_sum = 0.0f;
            for (const NeighborConstraint &neighbor : neighbors)
            {
                sum += (offsets[static_cast<std::size_t>(neighbor.neighbor)] +
                        neighbor.targetDifference) * neighbor.weight;
                weight_sum += neighbor.weight;
            }
            const cv::Vec3f target = sum / weight_sum;
            next[static_cast<std::size_t>(chart)] =
                0.5f * offsets[static_cast<std::size_t>(chart)] +
                0.5f * target;
        }
        offsets.swap(next);
    }

    std::vector<int> component(static_cast<std::size_t>(chartCount), -1);
    int component_count = 0;
    for (int seed = 0; seed < chartCount; ++seed)
    {
        if (graph[static_cast<std::size_t>(seed)].empty() ||
            component[static_cast<std::size_t>(seed)] >= 0)
        {
            continue;
        }
        std::queue<int> pending;
        pending.push(seed);
        component[static_cast<std::size_t>(seed)] = component_count;
        while (!pending.empty())
        {
            const int chart = pending.front();
            pending.pop();
            ++stats.connectedChartCount;
            for (const NeighborConstraint &neighbor :
                 graph[static_cast<std::size_t>(chart)])
            {
                if (component[static_cast<std::size_t>(neighbor.neighbor)] < 0)
                {
                    component[static_cast<std::size_t>(neighbor.neighbor)] =
                        component_count;
                    pending.push(neighbor.neighbor);
                }
            }
        }
        ++component_count;
    }

    std::vector<cv::Vec3f> component_mean(
        static_cast<std::size_t>(component_count), cv::Vec3f{});
    std::vector<int> component_size(
        static_cast<std::size_t>(component_count), 0);
    for (int chart = 0; chart < chartCount; ++chart)
    {
        const int id = component[static_cast<std::size_t>(chart)];
        if (id >= 0)
        {
            component_mean[static_cast<std::size_t>(id)] +=
                offsets[static_cast<std::size_t>(chart)];
            ++component_size[static_cast<std::size_t>(id)];
        }
    }
    const float correction_limit = std::clamp(
        maximumAbsoluteLinearCorrection, 0.0f, 0.25f);
    for (int chart = 0; chart < chartCount; ++chart)
    {
        const int id = component[static_cast<std::size_t>(chart)];
        if (id < 0)
        {
            offsets[static_cast<std::size_t>(chart)] = cv::Vec3f{};
            continue;
        }
        offsets[static_cast<std::size_t>(chart)] -=
            component_mean[static_cast<std::size_t>(id)] /
            static_cast<float>(component_size[static_cast<std::size_t>(id)]);
        cv::Vec3f &offset = offsets[static_cast<std::size_t>(chart)];
        for (int channel = 0; channel < 3; ++channel)
        {
            offset[channel] = std::clamp(
                offset[channel], -correction_limit, correction_limit);
            stats.maximumAbsoluteLinearCorrection = std::max(
                stats.maximumAbsoluteLinearCorrection,
                std::fabs(offset[channel]));
        }
        if (cv::norm(offset) > 1.0e-6f)
        {
            ++stats.adjustedChartCount;
        }
    }

    const int radius = std::clamp(borderBlendRadiusPixels, 1, 64);
    const float global_strength = std::clamp(
        globalCorrectionStrength, 0.0f, 1.0f);
    std::vector<cv::Rect> bounds(
        static_cast<std::size_t>(chartCount),
        cv::Rect(atlasBgr->cols, atlasBgr->rows, 0, 0));
    for (int row = 0; row < chartIndexMap.rows; ++row)
    {
        const int *chart_row = chartIndexMap.ptr<int>(row);
        for (int column = 0; column < chartIndexMap.cols; ++column)
        {
            const int chart = chart_row[column];
            if (chart < 0 || chart >= chartCount)
            {
                continue;
            }
            cv::Rect &bound = bounds[static_cast<std::size_t>(chart)];
            if (bound.width == 0)
            {
                bound = cv::Rect(column, row, 1, 1);
            }
            else
            {
                bound |= cv::Rect(column, row, 1, 1);
            }
        }
    }

    for (int chart = 0; chart < chartCount; ++chart)
    {
        cv::Rect bound = bounds[static_cast<std::size_t>(chart)];
        if (bound.empty() ||
            cv::norm(offsets[static_cast<std::size_t>(chart)]) <= 1.0e-6f)
        {
            continue;
        }
        bound &= cv::Rect(0, 0, atlasBgr->cols, atlasBgr->rows);
        cv::Mat chart_mask(
            bound.height + 2,
            bound.width + 2,
            CV_8UC1,
            cv::Scalar(0));
        for (int row = 0; row < bound.height; ++row)
        {
            const int *chart_row = chartIndexMap.ptr<int>(bound.y + row);
            std::uint8_t *mask_row =
                chart_mask.ptr<std::uint8_t>(row + 1);
            for (int column = 0; column < bound.width; ++column)
            {
                if (chart_row[bound.x + column] == chart)
                {
                    mask_row[column + 1] = 255;
                }
            }
        }
        cv::Mat distance;
        cv::distanceTransform(chart_mask, distance, cv::DIST_L2, cv::DIST_MASK_3);
        for (int row = 0; row < bound.height; ++row)
        {
            const std::uint8_t *mask_row =
                chart_mask.ptr<std::uint8_t>(row + 1);
            const float *distance_row = distance.ptr<float>(row + 1);
            cv::Vec3b *atlas_row = atlasBgr->ptr<cv::Vec3b>(bound.y + row);
            for (int column = 0; column < bound.width; ++column)
            {
                if (mask_row[column + 1] == 0)
                {
                    continue;
                }
                const float border_alpha = std::clamp(
                    (static_cast<float>(radius) + 1.0f -
                     distance_row[column + 1]) /
                        static_cast<float>(radius),
                    0.0f,
                    1.0f);
                const float alpha = global_strength +
                    (1.0f - global_strength) * border_alpha;
                if (alpha <= 0.0f)
                {
                    continue;
                }
                cv::Vec3b &pixel = atlas_row[bound.x + column];
                for (int channel = 0; channel < 3; ++channel)
                {
                    const float linear = srgbToLinear(pixel[channel]);
                    pixel[channel] = cv::saturate_cast<std::uint8_t>(
                        linearToSrgb(
                            linear + alpha *
                                offsets[static_cast<std::size_t>(chart)][channel]));
                }
                ++stats.adjustedPixelCount;
            }
        }
    }
    return stats;
}

} // namespace xjw::mesh::texture_v4
