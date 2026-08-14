#pragma once

#include <opencv2/core/mat.hpp>

namespace xjw::mesh
{

struct StudioForegroundMask
{
    cv::Mat mask;
    float coverage = 0.0f;
    float borderCoverage = 1.0f;
    float borderLuminance = 255.0f;

    bool isUsable() const;
    bool isUsableForColorSampling() const;
};

StudioForegroundMask buildStudioForegroundMask(const cv::Mat &color_image);

} // namespace xjw::mesh
