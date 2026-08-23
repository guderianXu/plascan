#pragma once

#include "MvsTypes.h"

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct DepthLevelResult
{
    int level = 1;
    int downsampleFactor = 1;
    cv::Mat depth;
    cv::Mat normalMap;
    cv::Mat confidence;
    cv::Mat supportCount; ///< PatchMatch 候选来源数诊断图，不代表逐像素几何确认数
    cv::Mat photometricSourceMask; ///< CV_32SC1；bit 位对应当前帧 source ordinal
    cv::Mat uncertainty;
    cv::Mat validMask;
};

struct DepthSearchPrior
{
    cv::Mat center;
    cv::Mat radius;
    cv::Mat normalMap;
    cv::Mat validMask;
};

DepthSearchPrior propagateDepthPrior(const DepthLevelResult &parent,
                                    const cv::Mat &guideImage,
                                    cv::Size targetSize,
                                    cv::Size parentLogicalSize = cv::Size());

} // namespace mvs
} // namespace xjw
