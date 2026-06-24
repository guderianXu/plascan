#include "DisparityValidator.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>

namespace xjw::dense_match
{

DisparityValidator::DisparityValidator(const DenseMatchConfig &cfg) : _config(cfg) {}

DisparityResult DisparityValidator::validate(const cv::Mat &disparity,
                                             const cv::Mat &confidence)
{
    DisparityResult result;
    result.disparity = disparity.clone();
    result.confidence = confidence.clone();
    if (result.disparity.empty())
    {
        return result;
    }

    if (_config.medianFilterSize > 0)
        result.disparity = medianFilter(result.disparity, _config.medianFilterSize);

    result.validMask = cv::Mat(disparity.rows, disparity.cols, CV_8UC1, cv::Scalar(1));
    return result;
}

cv::Mat DisparityValidator::checkLRConsistency(const cv::Mat &dispLR,
                                               const cv::Mat &dispRL)
{
    CV_Assert(dispLR.size() == dispRL.size());
    cv::Mat valid(dispLR.rows, dispLR.cols, CV_8UC1);

    for (int y = 0; y < dispLR.rows; ++y)
    {
        for (int x = 0; x < dispLR.cols; ++x)
        {
            float dLR = dispLR.at<float>(y, x);
            int xR = static_cast<int>(x - dLR + 0.5f);
            if (xR >= 0 && xR < dispRL.cols)
            {
                float dRL = dispRL.at<float>(y, xR);
                valid.at<uchar>(y, x) =
                    (std::abs(dLR - dRL) <= _config.lrCheckThreshold) ? 1 : 0;
            }
            else
            {
                valid.at<uchar>(y, x) = 0;
            }
        }
    }
    return valid;
}

cv::Mat DisparityValidator::medianFilter(const cv::Mat &disp, int kernelSize)
{
    if (kernelSize % 2 == 0) kernelSize += 1;
    cv::Mat result;
    cv::medianBlur(disp, result, kernelSize);
    return result;
}

cv::Mat DisparityValidator::speckleFilter(const cv::Mat &disp,
                                          const cv::Mat &valid,
                                          int maxSpeckleSize)
{
    // Use connected components to remove small valid regions (speckles)
    cv::Mat binaryValid;
    valid.convertTo(binaryValid, CV_8UC1);
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(
        binaryValid, labels, stats, centroids, 4, CV_32S);
    cv::Mat filtered = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int i = 1; i < nLabels; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= maxSpeckleSize)
        {
            cv::Mat mask = (labels == i);
            filtered.setTo(1, mask);
        }
    }
    return filtered;
}

void DisparityValidator::applyImageSupportMask(DisparityResult &result,
                                               const cv::Mat &left,
                                               const cv::Mat &right) const
{
    if (result.disparity.empty() || left.empty() || right.empty())
        return;
    CV_Assert(result.disparity.size() == left.size());
    CV_Assert(result.disparity.size() == right.size());
    CV_Assert(result.disparity.type() == CV_32FC1);
    CV_Assert(left.type() == CV_8UC1);
    CV_Assert(right.type() == CV_8UC1);

    if (result.validMask.empty())
        result.validMask = cv::Mat(result.disparity.size(), CV_8UC1, cv::Scalar(1));

    const int threshold = std::max(0, _config.supportIntensityThreshold);
    int removed = 0;
    int kept = 0;

    for (int y = 0; y < result.disparity.rows; ++y)
    {
        float *dispRow = result.disparity.ptr<float>(y);
        uchar *maskRow = result.validMask.ptr<uchar>(y);
        float *confRow = result.confidence.empty() ? nullptr : result.confidence.ptr<float>(y);
        const uchar *leftRow = left.ptr<uchar>(y);
        const uchar *rightRow = right.ptr<uchar>(y);

        for (int x = 0; x < result.disparity.cols; ++x)
        {
            if (maskRow[x] == 0)
            {
                dispRow[x] = 0.0f;
                if (confRow) confRow[x] = 0.0f;
                continue;
            }

            const float disparity = dispRow[x];
            const int rightX = cvRound(static_cast<float>(x) - disparity);
            const bool supported =
                std::isfinite(disparity)
                && disparity > 0.0f
                && rightX >= 0
                && rightX < result.disparity.cols
                && leftRow[x] > threshold
                && rightRow[rightX] > threshold;

            if (!supported)
            {
                maskRow[x] = 0;
                dispRow[x] = 0.0f;
                if (confRow) confRow[x] = 0.0f;
                ++removed;
            }
            else
            {
                ++kept;
            }
        }
    }

    fprintf(stderr, "[DisparityValidator] image support mask: kept=%d removed=%d threshold=%d\n",
            kept, removed, threshold);
}

} // namespace xjw::dense_match
