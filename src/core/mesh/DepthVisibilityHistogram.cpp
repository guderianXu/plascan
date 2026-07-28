#include "DepthVisibilityHistogram.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>

namespace xjw::mesh
{
namespace
{

constexpr float kQuantizationScale = 16.0f;

float binCenter(std::size_t index)
{
    return -1.0f +
        (static_cast<float>(index) + 0.5f) *
            (2.0f / static_cast<float>(kDepthVisibilityHistogramBinCount));
}

} // namespace

float DepthVisibilityHistogramSummary::totalWeight() const
{
    return std::accumulate(bins.cbegin(), bins.cend(), 0.0f);
}

float DepthVisibilityHistogramSummary::weightedMedian() const
{
    const float total = totalWeight();
    if (total <= 0.0f)
    {
        return 1.0f;
    }
    const float half_weight = total * 0.5f;
    float accumulated = 0.0f;
    for (std::size_t index = 0; index < bins.size(); ++index)
    {
        accumulated += bins[index];
        if (accumulated >= half_weight)
        {
            if (std::fabs(accumulated - half_weight) <=
                std::numeric_limits<float>::epsilon() * total)
            {
                const auto next = std::find_if(
                    bins.cbegin() + static_cast<std::ptrdiff_t>(index + 1),
                    bins.cend(),
                    [](float weight)
                    {
                        return weight > 0.0f;
                    });
                if (next != bins.cend())
                {
                    const std::size_t next_index =
                        static_cast<std::size_t>(
                            std::distance(bins.cbegin(), next));
                    return 0.5f *
                        (binCenter(index) + binCenter(next_index));
                }
            }
            return binCenter(index);
        }
    }
    return binCenter(bins.size() - 1);
}

float DepthVisibilityHistogramSummary::dominantBinRatio() const
{
    const float total = totalWeight();
    if (total <= 0.0f)
    {
        return 0.0f;
    }
    return *std::max_element(bins.cbegin(), bins.cend()) / total;
}

float DepthVisibilityHistogramSummary::conflictingSignRatio() const
{
    const std::size_t middle = bins.size() / 2;
    const float negative = std::accumulate(
        bins.cbegin(), bins.cbegin() + middle, 0.0f);
    const float positive = std::accumulate(
        bins.cbegin() + middle + 1, bins.cend(), 0.0f);
    const float total = negative + positive;
    return total > 0.0f ? std::min(negative, positive) / total : 0.0f;
}

void DepthVisibilityHistogram::add(float normalizedSignedDistance,
                                   float weight)
{
    if (!std::isfinite(normalizedSignedDistance) ||
        !std::isfinite(weight) ||
        weight <= 0.0f)
    {
        return;
    }
    const float normalized = std::clamp(
        normalizedSignedDistance, -1.0f, 1.0f);
    const float scaled =
        (normalized + 1.0f) * 0.5f *
        static_cast<float>(kDepthVisibilityHistogramBinCount);
    const std::size_t index = std::min(
        static_cast<std::size_t>(std::max(0.0f, std::floor(scaled))),
        kDepthVisibilityHistogramBinCount - 1);
    const int quantized = std::clamp(
        static_cast<int>(std::lround(weight * kQuantizationScale)),
        1,
        255);
    _bins[index] = static_cast<std::uint8_t>(
        std::min(255, static_cast<int>(_bins[index]) + quantized));
}

bool DepthVisibilityHistogram::empty() const
{
    return quantizedWeight() == 0;
}

std::uint32_t DepthVisibilityHistogram::quantizedWeight() const
{
    return std::accumulate(
        _bins.cbegin(), _bins.cend(), std::uint32_t{0});
}

DepthVisibilityHistogramSummary DepthVisibilityHistogram::summary() const
{
    DepthVisibilityHistogramSummary result;
    accumulate(&result);
    return result;
}

void DepthVisibilityHistogram::accumulate(
    DepthVisibilityHistogramSummary *target) const
{
    if (!target)
    {
        return;
    }
    for (std::size_t index = 0; index < _bins.size(); ++index)
    {
        target->bins[index] +=
            static_cast<float>(_bins[index]) / kQuantizationScale;
    }
}

} // namespace xjw::mesh
