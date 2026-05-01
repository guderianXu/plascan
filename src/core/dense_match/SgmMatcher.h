// =============================================================================
// 文件: SgmMatcher.h
// 功能: 半全局/多全局立体匹配器声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include "CostFunctions.h"
#include <opencv2/core.hpp>

namespace xjw::dense_match
{

class SgmMatcher
{
public:
    explicit SgmMatcher(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

private:
    void aggregatePath(CostVolume &L, const CostVolume &C,
                       int imgW, int imgH, int numDisp,
                       int dirX, int dirY) const;

    DenseMatchConfig m_cfg;
};

} // namespace xjw::dense_match
