#include "OpenCVSgbmWrapper.h"

#include "OpenCvCompat.h"

#include <algorithm>

namespace xjw::dense_match
{

OpenCVSgbmWrapper::OpenCVSgbmWrapper(const DenseMatchConfig &cfg) : _config(cfg) {}

DisparityResult OpenCVSgbmWrapper::compute(const cv::Mat &left, const cv::Mat &right)
{
    int numDisp = _config.maxDisparity - _config.minDisparity;
    int numDisp16 = std::max(16, ((numDisp + 15) / 16) * 16);
    int blockSize = std::max(3, _config.corrKernelW | 1);
    int p1 = std::max(_config.p1, 8 * blockSize * blockSize);
    int p2 = std::max(_config.p2, 32 * blockSize * blockSize);

    auto sgbm = cv::StereoSGBM::create(
        _config.minDisparity,
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
