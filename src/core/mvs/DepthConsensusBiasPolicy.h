#pragma once

#include <opencv2/core.hpp>

namespace xjw
{
namespace mvs
{

struct ReferenceAnchoredDepthConsensusOptions
{
    int minimumGeometrySupport = 3;
    float maximumInverseDepthSpread = 0.008f;
    float minimumConfidence = 0.50f;
    int contourExclusionPixels = 8;
    int minimumBiasSampleCount = 256;
    float maximumCandidateRelativeDifference = 0.03f;
    float maximumAppliedRelativeCorrection = 0.005f;
    float blendWeight = 0.50f;
};

struct ReferenceAnchoredDepthConsensusCalibration
{
    bool valid = false;
    float additiveDepthBias = 0.0f;
    int sampleCount = 0;
};

struct ReferenceAnchoredDepthConsensusResult
{
    cv::Mat depth;
    cv::Mat eligibleMask;
    cv::Mat appliedMask;
    ReferenceAnchoredDepthConsensusCalibration calibration;
    int appliedPixelCount = 0;
};

ReferenceAnchoredDepthConsensusResult makeReferenceAnchoredDepthConsensus(
    const cv::Mat &rawDepth,
    const cv::Mat &inverseDepthMean,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthRelativeSpread,
    const cv::Mat &confidence,
    const cv::Mat &crossViewRepairedMask,
    const cv::Mat &supportMask,
    const cv::Mat &depthValidMask,
    const ReferenceAnchoredDepthConsensusOptions &options = {});

} // namespace mvs
} // namespace xjw
