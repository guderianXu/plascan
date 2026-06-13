#include "DisparityFilter.h"

#include "OpenCvCompat.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstdio>

namespace xjw
{
namespace mvs
{

void DisparityFilter::filter(cv::Mat &disparity,
                             cv::Mat &validMask,
                             const DisparityFilterConfig &cfg)
{
    if (disparity.empty()) return;

    const int rows = disparity.rows;
    const int cols = disparity.cols;

    // Build initial valid mask
    validMask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float d = disparity.at<float>(r, c);
            if (std::isfinite(d) && d != 0.0f)
                validMask.at<uint8_t>(r, c) = 255;
        }
    }

    // Median filter
    if (cfg.medianFilterSize >= 3)
    {
        int kSize = cfg.medianFilterSize | 1;
        cv::Mat filtered;
        cv::medianBlur(disparity, filtered, kSize);
        filtered.copyTo(disparity, validMask);
    }

    // Speckle removal
    if (cfg.speckleSize > 0)
    {
        cv::Mat disp16;
        disparity.convertTo(disp16, CV_16S, 16.0);
        cv::filterSpeckles(disp16, 0, cfg.speckleSize,
                           static_cast<int>(cfg.speckleRange * 16));
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                if (disp16.at<int16_t>(r, c) == 0)
                {
                    validMask.at<uint8_t>(r, c) = 0;
                    disparity.at<float>(r, c) = 0.0f;
                }
            }
        }
    }

    int validCount = cv::countNonZero(validMask);
    fprintf(stderr, "[DisparityFilter] valid pixels after filter: %d / %d (%.1f%%)\n",
            validCount, rows * cols,
            100.0f * validCount / (rows * cols));
}

void DisparityFilter::filterWithLR(cv::Mat &dispLeft,
                                   const cv::Mat &dispRight,
                                   cv::Mat &validMask,
                                   const DisparityFilterConfig &cfg)
{
    filter(dispLeft, validMask, cfg);

    if (!cfg.leftRightCheck || dispRight.empty()) return;

    const int rows = dispLeft.rows;
    const int cols = dispLeft.cols;
    int rejected = 0;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            if (validMask.at<uint8_t>(r, c) == 0) continue;

            float dL = dispLeft.at<float>(r, c);
            int cR = static_cast<int>(std::round(c - dL));
            if (cR < 0 || cR >= cols)
            {
                validMask.at<uint8_t>(r, c) = 0;
                dispLeft.at<float>(r, c) = 0.0f;
                ++rejected;
                continue;
            }

            float dR = dispRight.at<float>(r, cR);
            if (!std::isfinite(dR) || dR == 0.0f ||
                std::abs(dL - dR) > cfg.lrThreshold)
            {
                validMask.at<uint8_t>(r, c) = 0;
                dispLeft.at<float>(r, c) = 0.0f;
                ++rejected;
            }
        }
    }

    int validCount = cv::countNonZero(validMask);
    fprintf(stderr, "[DisparityFilter] LR check rejected %d, remaining %d (%.1f%%)\n",
            rejected, validCount,
            100.0f * validCount / (rows * cols));
}

} // namespace mvs
} // namespace xjw
