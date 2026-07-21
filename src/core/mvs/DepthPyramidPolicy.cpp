#include "DepthPyramidPolicy.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <vector>

namespace xjw
{
namespace mvs
{
namespace
{

constexpr int kMinimumLevelShortSide = 160;
constexpr int kMinimumFinalShortSide = 320;

DepthPyramidLevelConfig makeLevel(const PatchMatchConfig &base_config,
                                  int level,
                                  int downsample_factor)
{
    DepthPyramidLevelConfig result;
    result.level = level;
    result.patchMatch = base_config;
    result.patchMatch.downsampleFactor = downsample_factor;

    if (level == 3)
    {
        result.patchMatch.numIterations = std::max(4, base_config.numIterations / 2);
        result.minSupportViews = 2;
        result.radiusScale = 2.0f;
    }
    else if (level == 2)
    {
        result.patchMatch.numIterations = std::max(4, base_config.numIterations * 3 / 4);
        result.minSupportViews = 2;
        result.radiusScale = 1.5f;
    }
    else
    {
        result.minSupportViews = 3;
        result.radiusScale = 1.0f;
    }
    return result;
}

} // namespace

cv::Size depthPyramidWorkingSize(int image_width,
                                 int image_height,
                                 int downsample_factor)
{
    const int factor = std::max(1, downsample_factor);
    return cv::Size(std::max(1, image_width / factor),
                    std::max(1, image_height / factor));
}

int depthPyramidMinimumLevelShortSide()
{
    return kMinimumLevelShortSide;
}

DepthPyramidConfig makeDepthPyramidConfig(const PatchMatchConfig &base_config,
                                          int image_width,
                                          int image_height)
{
    DepthPyramidConfig result;
    const int short_side = std::max(1, std::min(image_width, image_height));
    const int requested_final_factor = std::max(1, base_config.downsampleFactor);
    const int maximum_final_factor = std::max(1, short_side / kMinimumFinalShortSide);
    const int final_factor = std::min(requested_final_factor, maximum_final_factor);
    const int maximum_factor = std::max(1, short_side / kMinimumLevelShortSide);
    const std::array<int, 3> desired_factors = {
        final_factor * 4,
        final_factor * 2,
        final_factor
    };
    const std::array<int, 3> desired_levels = {3, 2, 1};

    std::vector<DepthPyramidLevelConfig> active_levels;
    active_levels.reserve(3);
    for (size_t index = 0; index < desired_factors.size(); ++index)
    {
        const int factor = desired_factors[index];
        const bool is_final = index + 1 == desired_factors.size();
        if (!is_final && factor > maximum_factor)
        {
            continue;
        }
        if (!active_levels.empty() &&
            active_levels.back().patchMatch.downsampleFactor <= factor)
        {
            continue;
        }
        active_levels.push_back(makeLevel(base_config, desired_levels[index], factor));
    }

    if (active_levels.empty())
    {
        active_levels.push_back(makeLevel(base_config, 1, final_factor));
    }

    result.activeLevelCount = static_cast<int>(active_levels.size());
    for (size_t index = 0; index < result.levels.size(); ++index)
    {
        result.levels[index] = index < active_levels.size()
            ? active_levels[index]
            : active_levels.back();
    }

    if (result.activeLevelCount < 3)
    {
        std::ostringstream message;
        message << "image short side " << short_side
                << " cannot keep three pyramid levels above "
                << kMinimumLevelShortSide << " pixels";
        result.degradedReason = message.str();
    }
    return result;
}

} // namespace mvs
} // namespace xjw
