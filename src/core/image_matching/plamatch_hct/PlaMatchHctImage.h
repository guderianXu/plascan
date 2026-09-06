#pragma once

#include "metalign/image.hpp"

#include <opencv2/core.hpp>

namespace xjw::image_matching
{

    metalign::Image makePlaMatchHctImage(const cv::Mat& grayImage, const cv::Mat& colorImage);
    metalign::Image makePlaMatchHctMask(const cv::Mat& validMask);

} // namespace xjw::image_matching
