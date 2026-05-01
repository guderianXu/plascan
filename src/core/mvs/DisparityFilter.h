#pragma once

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct DisparityFilterConfig
{
    int medianFilterSize = 3;
    int speckleSize = 60;
    float speckleRange = 3.0f;
    bool leftRightCheck = true;
    float lrThreshold = 1.0f;
};

class DisparityFilter
{
public:
    static void filter(cv::Mat &disparity,
                       cv::Mat &validMask,
                       const DisparityFilterConfig &cfg = {});

    static void filterWithLR(cv::Mat &dispLeft,
                             const cv::Mat &dispRight,
                             cv::Mat &validMask,
                             const DisparityFilterConfig &cfg = {});
};

} // namespace mvs
} // namespace xjw
