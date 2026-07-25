#pragma once

#include "MeshTypes.h"

#include <functional>

namespace xjw::mesh
{

struct QuadricSimplifyOptions
{
    int targetFaceCount = 0;
    int maximumPasses = 12;
    int workerCount = 0;
    float minimumFaceReductionRatio = 0.001f;
    int maximumStagnantPasses = 2;
    float featureAngleDegrees = 35.0f;
    float maximumNormalDeviationDegrees = 45.0f;
    bool preserveOpenBoundaries = true;
    int minimumSharpEdgeEndpointDegree = 1;
    bool simplifySimpleOpenBoundaries = false;
    std::function<bool()> isCancelled;
    std::function<void(int, int)> progress;
};

struct QuadricSimplifyStatistics
{
    int inputVertexCount = 0;
    int inputFaceCount = 0;
    int outputVertexCount = 0;
    int outputFaceCount = 0;
    int collapsedEdgeCount = 0;
    int rejectedBoundaryEdgeCount = 0;
    int rejectedFeatureEdgeCount = 0;
    int rejectedTopologyEdgeCount = 0;
    int rejectedFlipEdgeCount = 0;
    int passCount = 0;
    bool reachedTarget = false;
    bool stoppedByStagnation = false;
    bool cancelled = false;
};

QuadricSimplifyStatistics simplifyMeshQuadric(
    TriMesh *mesh,
    const QuadricSimplifyOptions &options);

} // namespace xjw::mesh
