#pragma once

/**
 * @file SiftLowTextureRecovery.h
 * @brief 为原始 SIFT 覆盖不足的有效网格规划局部对比度增强补点。
 */

#include "ImageMatchingAlgorithm.h"
#include "SiftComputeBackend.h"

#include <opencv2/core.hpp>

#include <optional>
#include <vector>

namespace xjw::image_matching
{

    struct SiftLowTextureRecoveryPlan
    {
        cv::Mat enhancedImage;
        cv::Mat recoveryMask;
        int maximumFeatures = 0;
        int targetFeatures = 0;
        float thresholdScale = 0.5f;

        bool isValid() const;
    };

    std::optional<SiftLowTextureRecoveryPlan> planSiftLowTextureRecovery(
        const ImageFeatureInput& input,
        const ImageMatchingRuntimeConfig& runtime,
        const std::vector<cv::KeyPoint>& baseKeypoints,
        int targetFeatureCount);

    SiftRawFeatures filterRecoveredSiftFeatures(
        const SiftRawFeatures& base,
        const SiftRawFeatures& recovered,
        const cv::Mat& recoveryMask);

} // namespace xjw::image_matching
