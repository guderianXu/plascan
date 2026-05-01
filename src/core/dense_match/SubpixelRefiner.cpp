// =============================================================================
// 文件: SubpixelRefiner.cpp
// 功能: 亚像素视差细化实现
// =============================================================================
#include "SubpixelRefiner.h"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw::dense_match
{

SubpixelRefiner::SubpixelRefiner(const DenseMatchConfig &cfg) : m_cfg(cfg) {}

cv::Mat SubpixelRefiner::refine(const cv::Mat &disparityInt,
                                const CostVolume &costVolume,
                                int minDisp, int maxDisp)
{
    switch (m_cfg.subpixel)
    {
    case SubpixelMode::None:
        return disparityInt.clone();
    case SubpixelMode::Parabola:
        return refineParabola(disparityInt, costVolume, minDisp, maxDisp);
    default:
        return disparityInt.clone();
    }
}

cv::Mat SubpixelRefiner::refineParabola(const cv::Mat &disp,
                                        const CostVolume &costVol,
                                        int minDisp, int maxDisp)
{
    int numDisp = maxDisp - minDisp;
    cv::Mat result = disp.clone();

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int y = 0; y < disp.rows; ++y)
    {
        for (int x = 0; x < disp.cols; ++x)
        {
            int dIdx = static_cast<int>(disp.at<float>(y, x)) - minDisp;
            if (dIdx <= 0 || dIdx >= numDisp - 1)
            {
                continue;
            }

            float c0 = costVol[dIdx - 1].at<float>(y, x);
            float c1 = costVol[dIdx].at<float>(y, x);
            float c2 = costVol[dIdx + 1].at<float>(y, x);

            float denom = 2.0f * (c0 + c2 - 2.0f * c1);
            if (std::abs(denom) > 1e-10f)
            {
                float delta = (c0 - c2) / denom;
                result.at<float>(y, x) = static_cast<float>(minDisp + dIdx) + delta;
            }
        }
    }
    return result;
}

} // namespace xjw::dense_match
