#pragma once
#include "DenseMatchConfig.h"
#include "DenseMatchTypes.h"
#include <opencv2/core.hpp>
#include <string>

namespace xjw::dense_match
{

class DenseMatchService
{
public:
    explicit DenseMatchService(const DenseMatchConfig &cfg);

    DisparityResult process();
    DisparityResult process(const cv::Mat &left, const cv::Mat &right);

    static bool saveDisparity(const DisparityResult &result,
                              const std::string &filepath);

private:
    DisparityResult computeRawDisparity(const cv::Mat &left, const cv::Mat &right) const;

    DenseMatchConfig _config;
    cv::Mat _left, _right;
};

} // namespace xjw::dense_match
