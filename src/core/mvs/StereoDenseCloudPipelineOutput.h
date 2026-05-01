#pragma once

#include "DisparityTriangulator.h"

#include <opencv2/core.hpp>

#include <string>

namespace xjw
{
namespace mvs
{

bool writeStereoPipelinePly(const std::string &path,
                            const TriangulationResult &triResult,
                            const cv::Mat &grayImage,
                            std::string &errorMessage);

} // namespace mvs
} // namespace xjw
