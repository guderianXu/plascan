#pragma once

#include <QJsonObject>

#include <opencv2/core.hpp>

#include <cstdint>

namespace xjw::mvs
{

struct DepthAnchoredHoleInterpolationOptions
{
    bool enabled = false;
    int maximumComponentArea = 12000;
    float maximumComponentAreaRatio = 0.12f;
    int minimumBoundarySampleCount = 12;
    int minimumAnchorContactPixelCount = 6;
    int anchorSearchRadius = 2;
    float maximumBoundaryInverseDepthSpread = 0.18f;
    float guideColorSigma = 24.0f;
    int maximumIterations = 160;
    float convergenceTolerance = 1.0e-5f;
    float interpolatedConfidence = 0.45f;
    bool allowSilhouetteConnectedInterior = false;
    int silhouetteProtectionRadiusPixels = 4;
};

struct DepthAnchoredHoleInterpolationStats
{
    std::uint64_t candidateComponentCount = 0;
    std::uint64_t acceptedComponentCount = 0;
    std::uint64_t interpolatedPixelCount = 0;
    std::uint64_t rejectedSilhouetteComponentCount = 0;
    std::uint64_t rejectedAreaComponentCount = 0;
    std::uint64_t rejectedAnchorComponentCount = 0;
    std::uint64_t rejectedBoundarySpreadComponentCount = 0;
    std::uint64_t protectedSilhouetteComponentCount = 0;
    std::uint64_t protectedSilhouettePixelCount = 0;
};

DepthAnchoredHoleInterpolationStats interpolateAnchoredInternalDepthHoles(
    cv::Mat &depth,
    const cv::Mat &supportMask,
    const cv::Mat &strongAnchorMask,
    const cv::Mat *guideGray,
    const DepthAnchoredHoleInterpolationOptions &options,
    cv::Mat *confidence = nullptr,
    cv::Mat *repairedMask = nullptr);

QJsonObject depthAnchoredHoleInterpolationStatsToJson(
    const DepthAnchoredHoleInterpolationStats &stats);

} // namespace xjw::mvs
