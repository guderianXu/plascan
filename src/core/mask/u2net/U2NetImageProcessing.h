#pragma once

#include <opencv2/core.hpp>

namespace xjw::mask
{

    cv::Mat makeU2NetBlob(const cv::Mat& image, int inputSize);
    cv::Mat u2netProbabilityFromOutput(const cv::Mat& output);
    cv::Mat makeU2NetMask(const cv::Mat& probability,
                          const cv::Size& outputSize,
                          float foregroundThreshold,
                          int morphologyRadius,
                          int minComponentArea,
                          bool keepLargestComponent);

} // namespace xjw::mask
