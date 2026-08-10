// =============================================================================
// 文件: SubpixelRefiner.cpp
// 功能: 亚像素视差细化实现
// =============================================================================
#include "SubpixelRefiner.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

SubpixelRefiner::SubpixelRefiner(const DenseMatchConfig &cfg)
    : _config(cfg)
{
}

cv::Mat SubpixelRefiner::refine(const cv::Mat &disparityInt,
                                const CostVolume &costVolume,
                                int minDisp, int maxDisp,
                                const cv::Mat &validMask)
{
    switch (_config.subpixel)
    {
    case SubpixelMode::None:
        return disparityInt.clone();
    case SubpixelMode::Parabola:
        return refineParabola(disparityInt, costVolume, minDisp, maxDisp, validMask);
    default:
        return disparityInt.clone();
    }
}

cv::Mat SubpixelRefiner::refineParabola(const cv::Mat &disp,
                                        const CostVolume &costVol,
                                        int minDisp, int maxDisp,
                                        const cv::Mat &validMask)
{
    const int numDisp = maxDisp - minDisp;
    cv::Mat result = disp.clone();
    if (disp.empty() || numDisp <= 0
        || costVol.size() != static_cast<std::size_t>(numDisp)
        || costVol.minDisparity() != minDisp
        || costVol.maxDisparity() != maxDisp)
    {
        return result;
    }
    if (!validMask.empty())
    {
        CV_Assert(validMask.type() == CV_8UC1 && validMask.size() == disp.size());
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int y = 0; y < disp.rows; ++y)
    {
        for (int x = 0; x < disp.cols; ++x)
        {
            if (!validMask.empty() && validMask.at<uchar>(y, x) == 0)
            {
                continue;
            }

            const float integerDisparity = disp.at<float>(y, x);
            if (!std::isfinite(integerDisparity))
            {
                continue;
            }
            const int dIdx = cvRound(integerDisparity) - minDisp;
            if (dIdx <= 0 || dIdx >= numDisp - 1)
            {
                continue;
            }
            if (!costVol.isValid(static_cast<std::size_t>(dIdx - 1), y, x)
                || !costVol.isValid(static_cast<std::size_t>(dIdx), y, x)
                || !costVol.isValid(static_cast<std::size_t>(dIdx + 1), y, x))
            {
                continue;
            }

            const float c0 = costVol[static_cast<std::size_t>(dIdx - 1)].at<float>(y, x);
            const float c1 = costVol[static_cast<std::size_t>(dIdx)].at<float>(y, x);
            const float c2 = costVol[static_cast<std::size_t>(dIdx + 1)].at<float>(y, x);
            if (!std::isfinite(c0) || !std::isfinite(c1) || !std::isfinite(c2)
                || c0 >= kInvalidCost || c1 >= kInvalidCost || c2 >= kInvalidCost)
            {
                continue;
            }

            const float denominator = 2.0f * (c0 + c2 - 2.0f * c1);
            if (denominator > 1.0e-10f)
            {
                const float delta = std::clamp((c0 - c2) / denominator, -1.0f, 1.0f);
                result.at<float>(y, x) = static_cast<float>(minDisp + dIdx) + delta;
            }
        }
    }
    return result;
}

} // namespace xjw::dense_match
