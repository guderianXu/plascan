#pragma once
#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace xjw::dense_match
{

class DisparityValidator
{
public:
    explicit DisparityValidator(const DenseMatchConfig &cfg);

    DisparityResult validate(const cv::Mat &disparity,
                             const cv::Mat &confidence);

    cv::Mat checkLRConsistency(const cv::Mat &dispLR,
                               const cv::Mat &dispRL);

    cv::Mat medianFilter(const cv::Mat &disp, int kernelSize);

    cv::Mat speckleFilter(const cv::Mat &disp, const cv::Mat &valid,
                          int maxSpeckleSize = 100);

    void applyImageSupportMask(DisparityResult &result,
                               const cv::Mat &left,
                               const cv::Mat &right) const;

private:
    DenseMatchConfig _config;
};

} // namespace xjw::dense_match
