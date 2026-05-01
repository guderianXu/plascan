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
    DenseMatchConfig m_cfg;
    cv::Mat m_left, m_right;
};

} // namespace xjw::dense_match
