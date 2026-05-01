#pragma once

#include "StereoDenseCloudPipeline.h"

namespace xjw
{
namespace mvs
{

bool runOriginalDepthPath(const cv::Mat &grayL,
                          const cv::Mat &grayR,
                          const Camera &leftCamera,
                          const Camera &rightCamera,
                          const std::string &outputDir,
                          const StereoPipelineConfig &config,
                          StereoPipelineResult &res,
                          StereoDenseCloudPipeline *owner);

bool runRectifiedDisparityPath(const cv::Mat &grayL,
                               const cv::Mat &grayR,
                               const Camera &leftCamera,
                               const Camera &rightCamera,
                               const std::string &outputDir,
                               const StereoPipelineConfig &config,
                               StereoPipelineResult &res,
                               StereoDenseCloudPipeline *owner);

} // namespace mvs
} // namespace xjw
