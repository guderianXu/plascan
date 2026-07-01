#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace xjw::feature_match::tradition
{

class CudaSiftMatcher
{
public:
    static bool isAvailable();

    static std::vector<std::vector<cv::DMatch>> knnMatchL2(const cv::Mat &queryDescriptors,
                                                           const cv::Mat &trainDescriptors,
                                                           int k,
                                                           int cudaDevice);
};

} // namespace xjw::feature_match::tradition
