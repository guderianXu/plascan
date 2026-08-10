#pragma once

#include <opencv2/core.hpp>

namespace xjw::mask
{

struct BiRefNetLetterbox
{
    cv::Size sourceSize;
    cv::Size resizedSize;
    int inputSize = 0;
    int left = 0;
    int top = 0;

    bool isValid() const;
};

cv::Mat makeBiRefNetBlob(const cv::Mat& image, int inputSize, BiRefNetLetterbox* letterbox);
cv::Mat biRefNetProbabilityFromOutput(const cv::Mat& output);
cv::Mat makeBiRefNetMask(const cv::Mat& probability,
                         const BiRefNetLetterbox& letterbox,
                         float foregroundThreshold,
                         int morphologyRadius,
                         int minComponentArea,
                         bool keepLargestComponent);

} // namespace xjw::mask
