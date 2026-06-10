#include "OpenCVSgbmWrapper.h"
#include <opencv2/calib3d.hpp>
#include <algorithm>

namespace xjw::dense_match
{

OpenCVSgbmWrapper::OpenCVSgbmWrapper(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult OpenCVSgbmWrapper::compute(const cv::Mat &left, const cv::Mat &right)
{
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;
    int numDisp16 = std::max(16, ((numDisp + 15) / 16) * 16);
    int blockSize = std::max(3, m_cfg.corrKernelW | 1);
    int p1 = std::max(m_cfg.p1, 8 * blockSize * blockSize);
    int p2 = std::max(m_cfg.p2, 32 * blockSize * blockSize);

    auto sgbm = cv::StereoSGBM::create(
        m_cfg.minDisparity,
        numDisp16,
        blockSize,
        p1,
        p2,
        1,
        31,
        5,
        50,
        2,
        cv::StereoSGBM::MODE_SGBM
    );

    cv::Mat disp16;
    sgbm->compute(left, right, disp16);

    DisparityResult result;
    disp16.convertTo(result.disparity, CV_32FC1, 1.0 / 16.0);
    result.confidence = cv::Mat(left.rows, left.cols, CV_32FC1, cv::Scalar(1.0));
    result.validMask  = cv::Mat(left.rows, left.cols, CV_8UC1, cv::Scalar(1));
    return result;
}

} // namespace xjw::dense_match
