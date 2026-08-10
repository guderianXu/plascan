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
                             const cv::Mat &confidence,
                             const cv::Mat &validMask = cv::Mat());

    cv::Mat checkLRConsistency(const cv::Mat &dispLR,
                               const cv::Mat &dispRL,
                               const cv::Mat &validLR = cv::Mat(),
                               const cv::Mat &validRL = cv::Mat());

    cv::Mat medianFilter(const cv::Mat &disp, int kernelSize);

    cv::Mat medianFilter(const cv::Mat &disp,
                         const cv::Mat &valid,
                         int kernelSize);

    cv::Mat speckleFilter(const cv::Mat &disp, const cv::Mat &valid,
                          int maxSpeckleSize = 100);

    void applyImageSupportMask(DisparityResult &result,
                               const cv::Mat &left,
                               const cv::Mat &right) const;

private:
    DenseMatchConfig _config;
};

} // namespace xjw::dense_match
