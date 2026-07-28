#pragma once

#include "MeshTypes.h"

#include <functional>
#include <string>

namespace xjw::mesh
{

struct OpenMeshSimplifyOptions
{
    int targetFaceCount = 0;
    float maximumNormalDeviationDegrees = 35.0f;
    float maximumNormalFlippingDegrees = 80.0f;
    int smoothingIterations = 0;
    float smoothingMaximumDisplacement = 0.0f;
    float smoothingFeatureAngleDegrees = 60.0f;
    int notificationInterval = 4096;
    std::function<bool()> isCancelled;
    std::function<void(int, int)> progress;
};

struct OpenMeshSimplifyStatistics
{
    bool available = false;
    bool initialized = false;
    bool succeeded = false;
    bool cancelled = false;
    bool reachedTarget = false;
    int inputVertexCount = 0;
    int inputFaceCount = 0;
    int outputVertexCount = 0;
    int outputFaceCount = 0;
    int collapsedVertexCount = 0;
    int rejectedInputFaceCount = 0;
    int inconsistentSharedEdgeCountBefore = 0;
    int reorientedInputFaceCount = 0;
    int removedContradictoryFaceCount = 0;
    int orientationConflictCount = 0;
    bool smoothingApplied = false;
    std::string error;
};

bool openMeshSimplifierAvailable();

OpenMeshSimplifyStatistics simplifyMeshWithOpenMesh(
    TriMesh *mesh,
    const OpenMeshSimplifyOptions &options = {});

} // namespace xjw::mesh
