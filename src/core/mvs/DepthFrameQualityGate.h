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

struct DepthConfidenceThresholds
{
    float patchMatch = 0.60f;
    float fusion = 0.65f;
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
    bool hasConstrainedSupportMask = false;
    float validWithinMaskRatio = -1.0f;
    float outputFilterRetentionRatio = -1.0f;
    float consistencyRetentionRatio = -1.0f;
    float fusionPostprocessRetentionRatio = -1.0f;
};

struct DepthFrameQualityDecision
{
    DepthFrameAcceptance acceptance = DepthFrameAcceptance::Rejected;
    DepthFilterSettings filterSettings;
    float calibratedConfidence = 0.0f;
    std::vector<std::string> reasons;
};

enum class DepthConsistencyEvidence
{
    Unverifiable,
    Consistent,
    Occluded,
    Contradicted
};

float calibrateDepthConfidence(const DepthConfidenceComponents &components);

DepthFilterSettings depthFilterSettings(DepthFilterMode mode, int availableSourceViews);

DepthConfidenceThresholds depthConfidenceThresholds(
    MvsSceneProfile sceneProfile,
    DepthFilterMode filterMode,
    int availableSourceViews,
    float configuredPatchMatch,
    float configuredFusion);

int minimumDepthConsistencySourceConfirmations(DepthFilterMode mode,
                                               int availableSourceViews);
int minimumDepthConsistencySourceConfirmations(MvsSceneProfile sceneProfile,
                                               DepthFilterMode mode,
                                               int availableSourceViews);

float depthConsistencyRelativeThreshold(
    MvsSceneProfile sceneProfile,
    int viewCount,
    DepthFilterMode filterMode = DepthFilterMode::Moderate);

bool useContradictionOnlyDepthConsistency(int sourceViewCount);

DepthConsistencyEvidence classifyDepthConsistencyEvidence(float expectedDepth,
                                                           float measuredDepth,
                                                           float relativeThreshold);

DepthFrameQualityDecision evaluateDepthFrame(const DepthFrameQualityInput &input);

const char *depthFrameAcceptanceId(DepthFrameAcceptance acceptance);

} // namespace mvs
} // namespace xjw
