#include "DisparityValidator.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xjw::dense_match
{

namespace
{

int normalizedMedianKernelSize(int kernelSize)
{
    if (kernelSize <= 1)
    {
        return kernelSize;
    }
    std::int64_t normalizedSize = kernelSize;
    if (normalizedSize % 2 == 0)
    {
        ++normalizedSize;
    }
    if (normalizedSize > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("Median filter kernel cannot be represented by int");
    }
    return static_cast<int>(normalizedSize);
}

std::size_t checkedMedianSampleCapacity(
    int kernelSize,
    int imageWidth,
    int imageHeight)
{
    const std::size_t sampleWidth = static_cast<std::size_t>(
        std::min(kernelSize, imageWidth));
    const std::size_t sampleHeight = static_cast<std::size_t>(
        std::min(kernelSize, imageHeight));
    if (sampleWidth != 0
        && sampleHeight > std::numeric_limits<std::size_t>::max() / sampleWidth)
    {
        throw std::overflow_error("Median filter sample capacity exceeds size_t");
    }
    return sampleWidth * sampleHeight;
}

} // namespace

DisparityValidator::DisparityValidator(const DenseMatchConfig &cfg) : _config(cfg) {}

DisparityResult DisparityValidator::validate(const cv::Mat &disparity,
                                             const cv::Mat &confidence,
                                             const cv::Mat &validMask)
{
    DisparityResult result;
    result.disparity = disparity.clone();
    result.confidence = confidence.clone();
    if (result.disparity.empty())
    {
        return result;
    }

    if (validMask.empty())
    {
        result.validMask = cv::Mat(disparity.rows, disparity.cols, CV_8UC1, cv::Scalar(1));
    }
    else
    {
        CV_Assert(validMask.type() == CV_8UC1 && validMask.size() == disparity.size());
        cv::compare(validMask, 0, result.validMask, cv::CMP_NE);
        result.validMask /= 255;
    }

    if (_config.medianFilterSize > 0)
    {
        result.disparity = medianFilter(
            result.disparity,
            result.validMask,
            _config.medianFilterSize);
    }

    for (int y = 0; y < result.disparity.rows; ++y)
    {
        float *disparityRow = result.disparity.ptr<float>(y);
        float *confidenceRow = result.confidence.empty()
            ? nullptr
            : result.confidence.ptr<float>(y);
        const uchar *validRow = result.validMask.ptr<uchar>(y);
        for (int x = 0; x < result.disparity.cols; ++x)
        {
            if (validRow[x] != 0)
            {
                continue;
            }
            disparityRow[x] = 0.0f;
            if (confidenceRow != nullptr)
            {
                confidenceRow[x] = 0.0f;
            }
        }
    }
    return result;
}

cv::Mat DisparityValidator::checkLRConsistency(const cv::Mat &dispLR,
                                               const cv::Mat &dispRL,
                                               const cv::Mat &validLR,
                                               const cv::Mat &validRL)
{
    CV_Assert(dispLR.size() == dispRL.size());
    CV_Assert(dispLR.type() == CV_32FC1 && dispRL.type() == CV_32FC1);
    CV_Assert(validLR.empty()
              || (validLR.type() == CV_8UC1 && validLR.size() == dispLR.size()));
    CV_Assert(validRL.empty()
              || (validRL.type() == CV_8UC1 && validRL.size() == dispRL.size()));
    cv::Mat valid(dispLR.rows, dispLR.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < dispLR.rows; ++y)
    {
        for (int x = 0; x < dispLR.cols; ++x)
        {
            if (!validLR.empty() && validLR.at<uchar>(y, x) == 0)
            {
                continue;
            }
            const float dLR = dispLR.at<float>(y, x);
            if (!std::isfinite(dLR))
            {
                continue;
            }
            const int xR = cvRound(static_cast<float>(x) - dLR);
            if (xR < 0 || xR >= dispRL.cols
                || (!validRL.empty() && validRL.at<uchar>(y, xR) == 0))
            {
                continue;
            }

            const float dRL = dispRL.at<float>(y, xR);
            valid.at<uchar>(y, x) =
                std::isfinite(dRL)
                && std::abs(dLR + dRL) <= _config.lrCheckThreshold
                ? 1
                : 0;
        }
    }
    return valid;
}

cv::Mat DisparityValidator::medianFilter(const cv::Mat &disp, int kernelSize)
{
    kernelSize = normalizedMedianKernelSize(kernelSize);
    if (kernelSize <= 1)
    {
        return disp.clone();
    }
    if (kernelSize > 255)
    {
        CV_Error(
            cv::Error::StsOutOfRange,
            "OpenCV median filter kernel must be no larger than 255");
    }
    cv::Mat result;
    cv::medianBlur(disp, result, kernelSize);
    return result;
}

cv::Mat DisparityValidator::medianFilter(const cv::Mat &disp,
                                         const cv::Mat &valid,
                                         int kernelSize)
{
    CV_Assert(disp.type() == CV_32FC1);
    CV_Assert(valid.type() == CV_8UC1 && valid.size() == disp.size());

    if (kernelSize <= 1)
    {
        return disp.clone();
    }
    kernelSize = normalizedMedianKernelSize(kernelSize);

    const int radius = kernelSize / 2;
    cv::Mat result = disp.clone();
    std::vector<float> samples;
    const std::size_t sampleCapacity = checkedMedianSampleCapacity(
        kernelSize,
        disp.cols,
        disp.rows);
    if (sampleCapacity > samples.max_size())
    {
        throw std::length_error("Median filter sample capacity exceeds the host vector limit");
    }
    samples.reserve(sampleCapacity);

    for (int y = 0; y < disp.rows; ++y)
    {
        const uchar *centerValidRow = valid.ptr<uchar>(y);
        float *resultRow = result.ptr<float>(y);
        for (int x = 0; x < disp.cols; ++x)
        {
            // The raw matcher mask owns center-pixel validity.  Filtering may
            // only change the value of a center that was already valid.
            if (centerValidRow[x] == 0)
            {
                continue;
            }

            samples.clear();
            const int firstY = static_cast<int>(std::max<std::int64_t>(
                0,
                static_cast<std::int64_t>(y) - radius));
            const int lastY = static_cast<int>(std::min<std::int64_t>(
                disp.rows - 1,
                static_cast<std::int64_t>(y) + radius));
            const int firstX = static_cast<int>(std::max<std::int64_t>(
                0,
                static_cast<std::int64_t>(x) - radius));
            const int lastX = static_cast<int>(std::min<std::int64_t>(
                disp.cols - 1,
                static_cast<std::int64_t>(x) + radius));
            for (int sampleY = firstY; sampleY <= lastY; ++sampleY)
            {
                const float *disparityRow = disp.ptr<float>(sampleY);
                const uchar *validRow = valid.ptr<uchar>(sampleY);
                for (int sampleX = firstX; sampleX <= lastX; ++sampleX)
                {
                    const float sample = disparityRow[sampleX];
                    if (validRow[sampleX] != 0 && std::isfinite(sample))
                    {
                        samples.push_back(sample);
                    }
                }
            }

            if (samples.empty())
            {
                continue;
            }

            const std::size_t middle = samples.size() / 2;
            auto middleIterator = samples.begin() + static_cast<std::ptrdiff_t>(middle);
            std::nth_element(samples.begin(), middleIterator, samples.end());
            if (samples.size() % 2 != 0)
            {
                resultRow[x] = *middleIterator;
                continue;
            }

            const float lower = *std::max_element(samples.begin(), middleIterator);
            resultRow[x] = lower + (*middleIterator - lower) * 0.5f;
        }
    }

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
