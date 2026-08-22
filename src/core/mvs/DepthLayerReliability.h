#pragma once

#include <QJsonObject>

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace xjw::mvs
{

enum class DepthLayerReliabilityClass : std::uint8_t
{
    Unobservable = 0,
    Reliable = 1,
    AmbiguousLowTexture = 2,
    RejectedLayer = 3
};

struct DepthLayerReliabilityOptions
{
    int textureRadiusPixels = 3;
    float maximumLowTextureStandardDeviation = 0.03f;
    float maximumWeakEffectiveViewCount = 1.50f;
    float minimumWeakConflictRatio = 0.50f;
    float minimumWeakInverseDepthSpread = 0.0025f;
    float minimumBoundaryEffectiveViewCount = 1.50f;
    float maximumBoundaryConflictRatio = 0.50f;
    float maximumBoundaryInverseDepthSpread = 0.0025f;
    int minimumComponentArea = 16;
    int boundaryRingRadiusPixels = 6;
    int minimumBoundaryAnchorCount = 24;
    float minimumRejectedLayerRelativeResidual = 0.008f;
    float maximumBoundarySurfaceFitP90 = 0.004f;
    int maximumReportedComponents = 32;
};

struct DepthLayerReliabilityComponent
{
    cv::Rect bounds;
    int pixelCount = 0;
    int boundaryAnchorCount = 0;
    float boundarySurfaceFitP90 = -1.0f;
    float signedRelativeResidualMedian = 0.0f;
    float absoluteRelativeResidualMedian = 0.0f;
    DepthLayerReliabilityClass reliabilityClass =
        DepthLayerReliabilityClass::AmbiguousLowTexture;
};

struct DepthLayerReliabilityResult
{
    bool validInputs = false;
    std::string errorMessage;
    cv::Mat classMap; ///< CV_8UC1, values from DepthLayerReliabilityClass.
    int validPixelCount = 0;
    int lowTexturePixelCount = 0;
    int weakGeometryPixelCount = 0;
    int candidatePixelCount = 0;
    int reliablePixelCount = 0;
    int ambiguousPixelCount = 0;
    int rejectedLayerPixelCount = 0;
    int rejectedLayerComponentCount = 0;
    std::vector<DepthLayerReliabilityComponent> components;
};

/// Observe coherent weak-evidence depth layers without modifying depth,
/// confidence, validity, or any production admission decision.
DepthLayerReliabilityResult analyzeDepthLayerReliability(
    const cv::Mat &depth,
    const cv::Mat &guideGray,
    const cv::Mat &effectiveViewCount,
    const cv::Mat &conflictRatio,
    const cv::Mat &inverseDepthRelativeSpread,
    const DepthLayerReliabilityOptions &options = {});

QJsonObject depthLayerReliabilityDiagnosticsToJson(
    const DepthLayerReliabilityResult &result,
    const DepthLayerReliabilityOptions &options = {});

} // namespace xjw::mvs
