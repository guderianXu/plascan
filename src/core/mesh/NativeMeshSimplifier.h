#pragma once

#include "MeshTypes.h"

#include <functional>
#include <string>

namespace xjw::mesh
{

struct NativeMeshSimplifyOptions
{
    int targetFaceCount = 0;
    int maximumPasses = 12;
    int workerCount = 0;
    float minimumFaceArea = 5.0e-9f;
    float maximumResultFaceAspectRatio = 0.0f;
    float featureAngleDegrees = 35.0f;
    float maximumNormalDeviationDegrees = 45.0f;
    float maximumNormalFlippingDegrees = 80.0f;
    int smoothingIterations = 0;
    float smoothingMaximumDisplacement = 0.0f;
    float smoothingFeatureAngleDegrees = 60.0f;
    std::function<bool()> isCancelled;
    std::function<void(int, int)> progress;
};

struct NativeMeshSimplifyStatistics
{
    bool initialized = false;
    bool succeeded = false;
    bool cancelled = false;
    bool reachedTarget = false;
    bool smoothingApplied = false;
    int inputVertexCount = 0;
    int inputFaceCount = 0;
    int outputVertexCount = 0;
    int outputFaceCount = 0;
    int collapsedEdgeCount = 0;
    int rejectedBoundaryEdgeCount = 0;
    int rejectedFeatureEdgeCount = 0;
    int rejectedTopologyEdgeCount = 0;
    int rejectedFlipEdgeCount = 0;
    int rejectedTriangleQualityEdgeCount = 0;
    int passCount = 0;
    int inconsistentSharedEdgeCountBefore = 0;
    int reorientedInputFaceCount = 0;
    int removedContradictoryFaceCount = 0;
    int orientationConflictCount = 0;
    std::string error;
};

NativeMeshSimplifyStatistics simplifyMeshTopologySafe(
    TriMesh *mesh,
    const NativeMeshSimplifyOptions &options = {});

} // namespace xjw::mesh
