#pragma once

#include <opencv2/core.hpp>

namespace xjw::point_cloud_dom_internal
{

void fillSmallGaps(cv::Mat *image, cv::Mat *mask, int iterations);

} // namespace xjw::point_cloud_dom_internal
