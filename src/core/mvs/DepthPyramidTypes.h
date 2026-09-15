#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace xjw::mvs
{

    struct DepthLevelResult
    {
        int level = 1;
        int downsampleFactor = 1;
        cv::Mat depth;
        cv::Mat normalMap;
        cv::Mat confidence;
        cv::Mat supportCount;          ///< PatchMatch 候选来源数诊断图，不代表逐像素几何确认数
        cv::Mat photometricSourceMask; ///< CV_32SC1；bit 位对应当前帧 source ordinal
        cv::Mat uncertainty;
        cv::Mat validMask;
    };

    struct DepthLevelSummary
    {
        int level = 1;
        int downsampleFactor = 1;
        int validPixelCount = 0;
        float validCoverage = 0.0f;
        float meanConfidence = 0.0f;
        float meanSupportViews = 0.0f;
        float depthDiscontinuityRatio = 0.0f;
        double elapsedMs = 0.0;
        bool success = false;
        std::string errorMessage;
    };

} // namespace xjw::mvs
