#pragma once

#include "VisualHullDepthRefiner.h"

#include <QVector>

#include <cstdint>
#include <functional>

namespace xjw::mesh
{

struct DepthTsdfFrame;

struct DepthConstrainedSurfaceRefineOptions
{
    VisualHullDepthRefineOptions depthRefine;
    int passes = 2;
    double minimumAreaRatio = 0.85;
    double maximumAreaRatio = 1.15;
    double minimumVolumeRatio = 0.85;
    double maximumVolumeRatio = 1.15;
    double minimumFaceNormalDot = 0.0;
    double minimumFaceAreaRatio = 0.10;
    bool removeMedianNormalBias = false;
};

struct DepthConstrainedSurfaceRefineStatistics
{
    bool attempted = false;
    int attemptedPassCount = 0;
    int appliedPassCount = 0;
    int revertedPassCount = 0;
    bool applied = false;
    bool reverted = false;
    float acceptedBlend = 0.0f;
    double areaBefore = 0.0;
    double areaAfter = 0.0;
    double absoluteVolumeBefore = 0.0;
    double absoluteVolumeAfter = 0.0;
    std::uint64_t flippedFaceCount = 0;
    std::uint64_t degenerateFaceCount = 0;
    int locallyProjectedCandidateCount = 0;
    int localSafetyProjectionIterationCount = 0;
    // Per-candidate maxima; repeated blend attempts do not double-count vertices.
    std::uint64_t locallyRejectedFaceCount = 0;
    std::uint64_t locallyFrozenVertexCount = 0;
    // Maxima among passes that were ultimately accepted.
    std::uint64_t acceptedLocallyRejectedFaceCount = 0;
    std::uint64_t acceptedLocallyFrozenVertexCount = 0;
    double removedMedianNormalBias = 0.0;
    VisualHullDepthRefineStatistics refiner;
};

class DepthConstrainedSurfaceRefiner
{
public:
    static DepthConstrainedSurfaceRefineStatistics refine(
        TriMesh *mesh,
        const QVector<DepthTsdfFrame> &frames,
        const DepthConstrainedSurfaceRefineOptions &options,
        const std::function<bool()> &isCancelled = {});
};

} // namespace xjw::mesh
