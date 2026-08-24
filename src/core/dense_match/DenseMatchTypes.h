// =============================================================================
// 文件: DenseMatchTypes.h
// 功能: 密集匹配模块公共类型定义
// =============================================================================
#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace xjw::dense_match
{

    enum class DenseMatchComputeBackend
    {
        Automatic = 0,
        Cpu = 1,
        Cuda = 2,
        OpenCl = 3
    };

    struct DenseMatchExecutionReport
    {
        DenseMatchComputeBackend requestedBackend = DenseMatchComputeBackend::Automatic;
        DenseMatchComputeBackend actualBackend = DenseMatchComputeBackend::Cpu;
        int deviceIndex = -1;
        bool workSubmitted = false;
        bool fallbackUsed = false;
        std::string fallbackReason;
    };

    enum class CostFunction
    {
        AbsoluteDifference = 0,
        SquaredDifference = 1,
        NormalizedCrossCorr = 2,
        CensusTransform = 3,
        TernaryCensusTransform = 4
    };

    enum class StereoAlgorithm
    {
        BlockMatch = 0,
        SemiGlobalMatch = 1,
        MoreGlobalMatch = 2,
        OpenCV_SGBM = 3
    };

    enum class SubpixelMode
    {
        None = 0,
        Parabola = 1,
        AffineBayes = 2,
        LucasKanade = 3
    };

    struct DisparityResult
    {
        // Left-reference disparity: d = x_left - x_right.
        cv::Mat disparity;
        cv::Mat confidence;
        cv::Mat validMask;
    };

} // namespace xjw::dense_match
