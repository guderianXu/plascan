#pragma once

#include "MvsTypes.h"

#include <string>
#include <vector>

namespace xjw
{
namespace mvs
{

enum class DepthFrameAcceptance
{
    Accepted,
    ValidationOnly,
    Rejected
};

struct DepthConfidenceComponents
{
    float photometric = 0.0f;
    float support = 0.0f;
    float uniqueness = 0.0f;
    float geometry = 0.0f;
    float texture = 0.0f;
};

struct DepthFilterSettings
{
    int minComponentArea = 24;
    float localDepthOutlierRelThreshold = 0.25f;
    int minConsistentViews = 3;
};

struct DepthFrameQualityInput
{
    MvsSceneProfile sceneProfile = MvsSceneProfile::OrbitalObject;
    DepthFilterMode filterMode = DepthFilterMode::Moderate;
    int sourceViewCount = 0;
    float validCoverage = 0.0f;
    float largestComponentRatio = 0.0f;
    float meanConfidence = 0.0f;
    float multiViewConsistency = 0.0f;
    float depthAtSearchBoundaryRatio = 0.0f;
    float sparseDepthMedianRelativeError = 0.0f;
};

struct DepthFrameQualityDecision
{
    DepthFrameAcceptance acceptance = DepthFrameAcceptance::Rejected;
    DepthFilterSettings filterSettings;
    float calibratedConfidence = 0.0f;
    std::vector<std::string> reasons;
};

float calibrateDepthConfidence(const DepthConfidenceComponents &components);

DepthFilterSettings depthFilterSettings(DepthFilterMode mode, int availableSourceViews);

DepthFrameQualityDecision evaluateDepthFrame(const DepthFrameQualityInput &input);

const char *depthFrameAcceptanceId(DepthFrameAcceptance acceptance);

} // namespace mvs
} // namespace xjw
