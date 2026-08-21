#pragma once

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

struct DepthPixelDomainScale
{
    cv::Size rasterSize;
    cv::Size gridSize;
    double scaleX = 1.0;
    double scaleY = 1.0;
    double linearScale = 1.0;
    double areaScale = 1.0;

    bool usesReducedGrid() const noexcept
    {
        return gridSize.width > 0 && gridSize.height > 0 &&
               rasterSize.width > 0 && rasterSize.height > 0 &&
               gridSize != rasterSize;
    }
};

DepthPyramidConfig makeDepthPyramidConfig(const PatchMatchConfig &baseConfig,
                                          int imageWidth,
                                          int imageHeight);

bool shouldPreserveNativeFinalDepthGrid(bool requested,
                                        MvsSceneProfile sceneProfile,
                                        bool epipolarRectified) noexcept;

FramePinholeCamera cameraForDepthGrid(const FramePinholeCamera &rasterCamera,
                                      const cv::Size &rasterSize,
                                      const cv::Size &depthGridSize);

/// Pixel-domain configuration is expressed in prepared full-raster pixels.
/// These helpers quantize it onto the actual depth grid. Scalar distances use
/// the geometric mean of the exact x/y camera scales; both exact scales remain
/// available for diagnostics.
DepthPixelDomainScale depthPixelDomainScale(const cv::Size &rasterSize,
                                             const cv::Size &depthGridSize) noexcept;
int scaleDepthPixelRadius(int rasterRadius,
                          const DepthPixelDomainScale &scale) noexcept;
float scaleDepthPixelDistance(float rasterDistance,
                              const DepthPixelDomainScale &scale) noexcept;
int scaleDepthPixelArea(int rasterArea,
                        const DepthPixelDomainScale &scale) noexcept;
int scaleDepthLocalOutlierKernel(int rasterKernelSize,
                                 const DepthPixelDomainScale &scale) noexcept;

cv::Size depthPyramidWorkingSize(int imageWidth,
                                 int imageHeight,
                                 int downsampleFactor);

int depthPyramidMinimumLevelShortSide();

} // namespace mvs
} // namespace xjw
