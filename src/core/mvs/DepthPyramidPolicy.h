#pragma once

#include "MvsTypes.h"

namespace xjw
{
namespace mvs
{

DepthPyramidConfig makeDepthPyramidConfig(const PatchMatchConfig &baseConfig,
                                          int imageWidth,
                                          int imageHeight);

cv::Size depthPyramidWorkingSize(int imageWidth,
                                 int imageHeight,
                                 int downsampleFactor);

int depthPyramidMinimumLevelShortSide();

} // namespace mvs
} // namespace xjw
