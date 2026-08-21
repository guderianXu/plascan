#include "DepthPyramidPolicy.h"

#include <algorithm>
#include <array>
#include <cmath>
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
    // The high-quality profile normally requests half resolution. For small
    // 1K-class images that saves little memory but discards the fine surface
    // structure needed by orbital-object silhouettes. Preserve the requested
    // downsample for larger inputs while restoring a native-resolution final
    // level for manageable high-quality images.
    constexpr int kMaximumNativeHighQualityShortSide = 1280;
    const int final_factor =
        requested_final_factor == 2 &&
        short_side <= kMaximumNativeHighQualityShortSide
        ? 1
        : std::min(requested_final_factor, maximum_final_factor);
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

bool shouldPreserveNativeFinalDepthGrid(bool requested,
                                        MvsSceneProfile scene_profile,
                                        bool epipolar_rectified) noexcept
{
    return requested &&
           scene_profile == MvsSceneProfile::Custom &&
           !epipolar_rectified;
}

FramePinholeCamera cameraForDepthGrid(const FramePinholeCamera &raster_camera,
                                      const cv::Size &raster_size,
                                      const cv::Size &depth_grid_size)
{
    if (raster_size.width <= 0 || raster_size.height <= 0 ||
        depth_grid_size.width <= 0 || depth_grid_size.height <= 0 ||
        depth_grid_size == raster_size)
    {
        return raster_camera;
    }

    return raster_camera.scaledIntrinsics(
        static_cast<double>(depth_grid_size.width) / raster_size.width,
        static_cast<double>(depth_grid_size.height) / raster_size.height);
}

DepthPixelDomainScale depthPixelDomainScale(const cv::Size &raster_size,
                                             const cv::Size &depth_grid_size) noexcept
{
    DepthPixelDomainScale result;
    result.rasterSize = raster_size;
    result.gridSize = depth_grid_size;
    if (raster_size.width <= 0 || raster_size.height <= 0 ||
        depth_grid_size.width <= 0 || depth_grid_size.height <= 0)
    {
        return result;
    }

    result.scaleX = static_cast<double>(depth_grid_size.width) /
                    static_cast<double>(raster_size.width);
    result.scaleY = static_cast<double>(depth_grid_size.height) /
                    static_cast<double>(raster_size.height);
    result.areaScale = result.scaleX * result.scaleY;
    result.linearScale = std::sqrt(result.areaScale);
    return result;
}

int scaleDepthPixelRadius(int raster_radius,
                          const DepthPixelDomainScale &scale) noexcept
{
    return std::max(0, static_cast<int>(std::lround(
        static_cast<double>(std::max(0, raster_radius)) * scale.linearScale)));
}

float scaleDepthPixelDistance(float raster_distance,
                              const DepthPixelDomainScale &scale) noexcept
{
    if (!std::isfinite(raster_distance) || raster_distance <= 0.0f)
    {
        return 0.0f;
    }
    return static_cast<float>(
        static_cast<double>(raster_distance) * scale.linearScale);
}

int scaleDepthPixelArea(int raster_area,
                        const DepthPixelDomainScale &scale) noexcept
{
    if (raster_area <= 0)
    {
        return 0;
    }
    return std::max(1, static_cast<int>(std::lround(
        static_cast<double>(raster_area) * scale.areaScale)));
}

int scaleDepthLocalOutlierKernel(
    int raster_kernel_size,
    const DepthPixelDomainScale &scale) noexcept
{
    if (raster_kernel_size < 3)
    {
        return 1;
    }
    const int normalized_kernel = std::clamp(raster_kernel_size | 1, 3, 5);
    const int raster_radius = (normalized_kernel - 1) / 2;
    return scaleDepthPixelRadius(raster_radius, scale) * 2 + 1;
}

} // namespace mvs
} // namespace xjw
