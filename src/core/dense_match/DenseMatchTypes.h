// =============================================================================
// 文件: DenseMatchTypes.h
// 功能: 密集匹配模块公共类型定义
// =============================================================================
#pragma once

#include <opencv2/core.hpp>

namespace xjw::dense_match
{

enum class CostFunction
{
    AbsoluteDifference      = 0,
    SquaredDifference       = 1,
    NormalizedCrossCorr     = 2,
    CensusTransform         = 3,
    TernaryCensusTransform  = 4
};

enum class StereoAlgorithm
{
    BlockMatch      = 0,
    SemiGlobalMatch = 1,
    MoreGlobalMatch = 2,
    OpenCV_SGBM     = 3
};

enum class SubpixelMode
{
    None        = 0,
    Parabola    = 1,
    AffineBayes = 2,
    LucasKanade = 3
};

struct DisparityResult
{
    cv::Mat disparity;
    cv::Mat confidence;
    cv::Mat validMask;
};

} // namespace xjw::dense_match
