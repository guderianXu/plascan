#pragma once

#include <opencv2/core/mat.hpp>

namespace xjw::mvs
{

struct LearnedDepthCandidateGateOptions
{
    float minimumCandidateConfidence = 0.50f;
    int minimumGeometryObservationCount = 3;
    float maximumInverseDepthRelativeSpread = 0.015f;
    float maximumRelativeDepthDifference = 0.03f;
    float replacementConfidenceMargin = 0.10f;
};

struct LearnedDepthCandidateGateStats
{
    bool validInputs = false;
    int candidatePixelCount = 0;
    int geometrySupportedPixelCount = 0;
    int acceptedPixelCount = 0;
    int filledPixelCount = 0;
    int replacedPixelCount = 0;
    int rejectedConfidenceCount = 0;
    int rejectedGeometryCount = 0;
    int rejectedDepthDifferenceCount = 0;
};

/// Merge learned depth only after independent camera-geometry evidence exists.
/// Candidate depth never contributes to supportCount, inverseDepthMean, or
/// spread, so it cannot validate itself.
LearnedDepthCandidateGateStats gateLearnedDepthCandidate(
    cv::Mat &depth,
    cv::Mat &confidence,
    const cv::Mat &candidateDepth,
    const cv::Mat &candidateConfidence,
    const cv::Mat &geometrySupportCount,
    const cv::Mat &inverseDepthMean,
    const cv::Mat &inverseDepthRelativeSpread,
    cv::Mat *acceptedMask,
    const LearnedDepthCandidateGateOptions &options = {});

} // namespace xjw::mvs
