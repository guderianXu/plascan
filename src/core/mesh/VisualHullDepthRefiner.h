#pragma once

#include "MeshTypes.h"

#include <QVector>

#include <cstdint>

namespace xjw::mesh
{

struct DepthTsdfFrame;

struct VisualHullDepthRefineOptions
{
    float maximumEvidenceDistance = 0.0f;
    float maximumDisplacement = 0.0f;
    int minimumViewCount = 2;
    int minimumNativeViewCount = 1;
    float minimumDepthConfidence = 0.25f;
    float repairedObservationWeight = 0.35f;
    float maximumViewMedianAbsoluteDeviation = 0.0f;
    float minimumAnchorWeight = 0.05f;
    int regularizationIterations = 30;
    float regularizationWeight = 4.0f;
    float propagationDecay = 0.90f;
    float regularizationMaximumNormalAngleDegrees = 50.0f;
};

struct VisualHullDepthRefineStatistics
{
    bool applied = false;
    std::uint64_t projectedObservationCount = 0;
    std::uint64_t acceptedObservationCount = 0;
    std::uint64_t anchoredVertexCount = 0;
    std::uint64_t displacedVertexCount = 0;
    float medianSupportingViewCount = 0.0f;
    float p90SupportingViewCount = 0.0f;
    float maximumAppliedDisplacement = 0.0f;
    float medianAppliedDisplacement = 0.0f;
    float p90AppliedDisplacement = 0.0f;
};

class VisualHullDepthRefiner
{
public:
    static VisualHullDepthRefineStatistics refine(
        TriMesh *mesh,
        const QVector<DepthTsdfFrame> &frames,
        const VisualHullDepthRefineOptions &options);
};

} // namespace xjw::mesh
