#include "OpenCVSgbmWrapper.h"

#include <opencv2/stereo.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace xjw::dense_match
{

OpenCVSgbmWrapper::OpenCVSgbmWrapper(const DenseMatchConfig &cfg) : _config(cfg) {}

DisparityResult OpenCVSgbmWrapper::compute(const cv::Mat &left, const cv::Mat &right)
{
    const std::int64_t disparityCount = static_cast<std::int64_t>(_config.maxDisparity)
        - static_cast<std::int64_t>(_config.minDisparity);
    if (disparityCount <= 0
        || disparityCount > std::numeric_limits<int>::max()
        || disparityCount % 16 != 0)
    {
        CV_Error(
            cv::Error::StsBadArg,
            "OpenCV SGBM requires maxDisparity - minDisparity to be an int-representable "
            "positive multiple of 16; "
            "the requested half-open disparity range cannot be expanded without changing its semantics");
    }
    const int numDisp = static_cast<int>(disparityCount);
    if (_config.corrKernelW <= 0 || _config.corrKernelW % 2 == 0)
    {
        CV_Error(cv::Error::StsBadArg, "OpenCV SGBM correlation kernel width must be positive and odd");
    }
    const std::int64_t blockSize64 = std::max<std::int64_t>(3, _config.corrKernelW);
    const std::int64_t blockArea = blockSize64 * blockSize64;
    if (blockArea > std::numeric_limits<int>::max() / 32)
    {
        CV_Error(
            cv::Error::StsOutOfRange,
            "OpenCV SGBM correlation kernel is too large for int P1/P2 parameters");
    }
    const int blockSize = static_cast<int>(blockSize64);
    const int p1 = std::max(_config.p1, static_cast<int>(8 * blockArea));
    const int p2 = std::max(_config.p2, static_cast<int>(32 * blockArea));

    auto sgbm = cv::StereoSGBM::create(
        _config.minDisparity,
        numDisp,
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
    result.confidence = cv::Mat(left.rows, left.cols, CV_32FC1, cv::Scalar(0.0f));
    result.validMask  = cv::Mat(left.rows, left.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < result.disparity.rows; ++y)
    {
        const float *disparityRow = result.disparity.ptr<float>(y);
        float *confidenceRow = result.confidence.ptr<float>(y);
        uchar *validRow = result.validMask.ptr<uchar>(y);
        for (int x = 0; x < result.disparity.cols; ++x)
        {
            const float disparity = disparityRow[x];
            const int rightX = cvRound(static_cast<float>(x) - disparity);
            if (std::isfinite(disparity)
                && disparity >= static_cast<float>(_config.minDisparity)
                && disparity < static_cast<float>(_config.maxDisparity)
                && rightX >= 0
                && rightX < right.cols)
            {
                confidenceRow[x] = 1.0f;
                validRow[x] = 1;
            }
        }
    }
    return result;
}

} // namespace xjw::dense_match
