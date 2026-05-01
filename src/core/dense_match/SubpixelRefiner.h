// =============================================================================
// 文件: SubpixelRefiner.h
// 功能: 亚像素视差细化声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include "CostFunctions.h"
#include <opencv2/core.hpp>

namespace xjw::dense_match
{

class SubpixelRefiner
{
public:
    explicit SubpixelRefiner(const DenseMatchConfig &cfg);

    cv::Mat refine(const cv::Mat &disparityInt,
                   const CostVolume &costVolume,
                   int minDisp, int maxDisp);

private:
    cv::Mat refineParabola(const cv::Mat &disp,
                           const CostVolume &costVol,
                           int minDisp, int maxDisp);

    DenseMatchConfig m_cfg;
};

} // namespace xjw::dense_match
