#pragma once

#include "MeshTypes.h"

namespace xjw::mesh
{

struct QuadricSimplifyOptions
{
    int targetFaceCount = 0;
    int maximumPasses = 12;
    float featureAngleDegrees = 35.0f;
    float maximumNormalDeviationDegrees = 45.0f;
    bool preserveOpenBoundaries = true;
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
};

QuadricSimplifyStatistics simplifyMeshQuadric(
    TriMesh *mesh,
    const QuadricSimplifyOptions &options);

} // namespace xjw::mesh
