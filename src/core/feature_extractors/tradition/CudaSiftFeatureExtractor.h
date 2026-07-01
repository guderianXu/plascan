#pragma once

#include "SuperPoint.h"

#include <opencv2/opencv.hpp>

namespace xjw::feature_extractors
{

bool isCudaSiftAvailable();

FeatureOutput detectCudaSift(const cv::Mat &grayImage,
                             const SuperPointConfig &config,
                             int cudaDevice);

} // namespace xjw::feature_extractors
