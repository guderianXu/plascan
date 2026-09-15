#pragma once

#include "MvsTypes.h"
#include "DepthPyramidTypes.h"

#include <opencv2/core.hpp>

namespace xjw
{
    namespace mvs
    {

        struct DepthSearchPrior
        {
            cv::Mat center;
            cv::Mat radius;
            cv::Mat normalMap;
            cv::Mat validMask;
        };

        DepthSearchPrior propagateDepthPrior(const DepthLevelResult& parent,
                                             const cv::Mat& guideImage,
                                             cv::Size targetSize,
                                             cv::Size parentLogicalSize = cv::Size());

    } // namespace mvs
} // namespace xjw
