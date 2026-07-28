#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace xjw::mesh
{

// Keep an odd number of bins so that zero signed distance has an exact bin
// centre. An even count biases near-surface observations to either side of
// the zero isosurface.
constexpr std::size_t kDepthVisibilityHistogramBinCount = 9;
static_assert(kDepthVisibilityHistogramBinCount % 2 == 1);

struct DepthVisibilityHistogramSummary
{
    std::array<float, kDepthVisibilityHistogramBinCount> bins{};

    float totalWeight() const;
    float weightedMedian() const;
    float dominantBinRatio() const;
    float conflictingSignRatio() const;
};

class DepthVisibilityHistogram
{
public:
    void add(float normalizedSignedDistance, float weight);

    bool empty() const;
    std::uint32_t quantizedWeight() const;
    DepthVisibilityHistogramSummary summary() const;
    void accumulate(DepthVisibilityHistogramSummary *target) const;

private:
    std::array<std::uint8_t, kDepthVisibilityHistogramBinCount> _bins{};
};

static_assert(sizeof(DepthVisibilityHistogram) ==
              kDepthVisibilityHistogramBinCount);

} // namespace xjw::mesh
