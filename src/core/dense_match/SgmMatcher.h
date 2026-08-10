// =============================================================================
// 文件: SgmMatcher.h
// 功能: 半全局/多全局立体匹配器声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include "CostFunctions.h"
#include <opencv2/core.hpp>
#include <vector>

namespace xjw::dense_match
{

struct SgmDirection
{
    int x = 0;
    int y = 0;
};

// Testable SGM reference aggregation.  Each direction owns only two disparity
// scan buffers; completed path costs are accumulated into one full volume.
CostVolume aggregateSgmCostVolume(const CostVolume &costVolume,
                                  int p1,
                                  int p2,
                                  const std::vector<SgmDirection> &directions);

class SgmMatcher
{
public:
    explicit SgmMatcher(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

private:
    DenseMatchConfig _config;
};

} // namespace xjw::dense_match
