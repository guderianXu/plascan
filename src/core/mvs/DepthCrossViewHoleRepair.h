#pragma once

#include "Camera.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace xjw::mvs
{

struct CrossViewHoleRepairOptions
{
    int minimumDistinctSourceCount = 2;
    float maximumRelativeDepthSpread = 0.025f;
    float maximumProjectionDistancePixels = 1.0f;
    int localDepthRadius = 2;
    float maximumLocalRelativeDepthDifference = 0.05f;
    float repairedConfidence = 0.65f;
    bool enableTwoSourceGrowth = false;
    int maximumGrowthDistancePixels = 3;
    float maximumGrowthInverseDepthSpread = 0.01f;
    float maximumGrowthNormalAngleDegrees = 15.0f;
    float maximumGrowthImageGradient = 80.0f;
    int maximumGrowthComponentArea = 64;
};

struct CrossViewHoleRepairStats
{
    std::uint64_t projectedCandidateCount = 0;
    std::uint64_t consideredHolePixelCount = 0;
    std::uint64_t rejectedInsufficientSourceCount = 0;
    std::uint64_t rejectedDepthSpreadCount = 0;
    std::uint64_t rejectedLocalDepthCount = 0;
    std::uint64_t repairedPixelCount = 0;
    std::uint64_t twoSourceCandidatePixelCount = 0;
    std::uint64_t twoSourceGrownPixelCount = 0;
    std::uint64_t growthRejectedComponentAreaCount = 0;
    std::uint64_t growthRejectedSourceOverlapCount = 0;
    std::uint64_t growthRejectedNormalCount = 0;
    std::uint64_t growthRejectedImageEdgeCount = 0;
};

cv::Mat projectSourceDepthToReference(
    const cv::Mat &sourceDepth,
    const Camera &sourceCamera,
    const Camera &referenceCamera,
    const cv::Size &referenceSize,
    float maximumProjectionDistancePixels,
    std::uint64_t *projectedCandidateCount = nullptr);

CrossViewHoleRepairStats repairDepthHolesFromProjectedSources(
    cv::Mat &referenceDepth,
    const cv::Mat &supportMask,
    const std::vector<cv::Mat> &projectedSourceDepths,
    const CrossViewHoleRepairOptions &options = {},
    cv::Mat *referenceConfidence = nullptr,
    cv::Mat *consistentSourceVotes = nullptr,
    cv::Mat *repairedMask = nullptr,
    cv::Mat *geometrySourceMask = nullptr,
    cv::Mat *sourceInverseDepthSum = nullptr,
    cv::Mat *sourceInverseDepthSquaredSum = nullptr,
    const Camera *referenceCamera = nullptr,
    const cv::Mat *guideGray = nullptr);

} // namespace xjw::mvs
