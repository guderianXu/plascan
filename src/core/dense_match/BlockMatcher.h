// =============================================================================
// 文件: BlockMatcher.h
// 功能: WTA块匹配器声明
// =============================================================================
#pragma once

#include "DenseMatchConfig.h"
#include <opencv2/core.hpp>

namespace xjw::dense_match
{

class BlockMatcher
{
public:
    explicit BlockMatcher(const DenseMatchConfig &cfg);
    DisparityResult compute(const cv::Mat &left, const cv::Mat &right);

private:
    DenseMatchConfig m_cfg;
};

} // namespace xjw::dense_match
