// =============================================================================
// 文件: DenseMatchConfig.h
// 功能: 密集匹配参数配置结构体
// =============================================================================
#pragma once

#include "DenseMatchTypes.h"
#include <string>

namespace xjw::dense_match
{

struct DenseMatchConfig
{
    StereoAlgorithm algorithm  = StereoAlgorithm::MoreGlobalMatch;
    CostFunction    costFunc   = CostFunction::CensusTransform;
    SubpixelMode    subpixel   = SubpixelMode::Parabola;

    int minDisparity = 0;
    int maxDisparity = 256;

    int corrKernelW = 15;
    int corrKernelH = 15;

    int p1            = 8;
    int p2            = 32;
    int sgmDirections = 8;
    int sgmCollarSize = 512;

    int pyramidLevels = 2;

    int subpixelKernelW = 21;
    int subpixelKernelH = 21;

    float lrCheckThreshold = 1.0f;
    int   medianFilterSize = 3;
    int   supportIntensityThreshold = 5;
    bool  enableLRCheck = false;

    bool useCuda     = true;
    int  cudaDevice  = 0;
    int  numThreads  = 4;

    std::string leftImagePath;
    std::string rightImagePath;
    std::string outputDisparityPath;
};

} // namespace xjw::dense_match
