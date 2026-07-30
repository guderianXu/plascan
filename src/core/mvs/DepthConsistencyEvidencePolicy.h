#pragma once

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace xjw::mvs
{

struct WeakNativeDepthRetentionOptions
{
    int minimumConfirmationCount = 1;
    float confidenceMultiplier = 0.55f;
    float minimumRetainedConfidence = 0.80f;
    bool retainUnconfirmedWithoutContradiction = false;
};

struct WeakNativeDepthRetentionStats
{
    std::uint64_t consideredPixelCount = 0;
    std::uint64_t retainedPixelCount = 0;
    std::uint64_t retainedUnconfirmedPixelCount = 0;
    std::uint64_t rejectedContradictionPixelCount = 0;
    std::uint64_t rejectedNoConfirmationPixelCount = 0;
};

WeakNativeDepthRetentionStats retainWeaklyVerifiedNativeDepth(
    const cv::Mat &originalDepth,
    const cv::Mat &originalConfidence,
    const cv::Mat &supportRegionMask,
    const cv::Mat &consistentVotes,
    const cv::Mat &contradictedVotes,
    const WeakNativeDepthRetentionOptions &options,
    cv::Mat *filteredDepth,
    cv::Mat *filteredConfidence);

} // namespace xjw::mvs
