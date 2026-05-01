#pragma once

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct SubpixelConfig
{
    int mode = 1;           // 0=off, 1=parabola
    int kernelSize = 21;    // NCC window size for cost recomputation
};

class SubpixelRefiner
{
public:
    static void refine(cv::Mat &disparity,
                       const cv::Mat &leftImg,
                       const cv::Mat &rightImg,
                       const SubpixelConfig &cfg = {});
};

} // namespace mvs
} // namespace xjw
