#pragma once
#include "../DenseMatchConfig.h"
#include "../DenseMatchTypes.h"
#include <opencv2/core.hpp>

namespace xjw::dense_match
{

class OpenCVSgbmWrapper
{
public:
    explicit OpenCVSgbmWrapper(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

private:
    DenseMatchConfig _config;
};

} // namespace xjw::dense_match
