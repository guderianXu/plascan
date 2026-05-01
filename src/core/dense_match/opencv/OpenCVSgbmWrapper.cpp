#include "OpenCVSgbmWrapper.h"
#include <opencv2/calib3d.hpp>

namespace xjw::dense_match
{

OpenCVSgbmWrapper::OpenCVSgbmWrapper(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

DisparityResult OpenCVSgbmWrapper::compute(const cv::Mat &left, const cv::Mat &right)
{
    int numDisp = m_cfg.maxDisparity - m_cfg.minDisparity;
    int numDisp16 = ((numDisp + 15) / 16) * 16;

    auto sgbm = cv::StereoSGBM::create(
        m_cfg.minDisparity,
        numDisp16,
        std::max(3, m_cfg.corrKernelW | 1),
        8 * m_cfg.corrKernelW * m_cfg.corrKernelH,
        32 * m_cfg.corrKernelW * m_cfg.corrKernelH,
        0, 0, 100, 0, 0,
        cv::StereoSGBM::MODE_SGBM_3WAY
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
