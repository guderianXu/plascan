#pragma once

#include "MeshTypes.h"

#include <functional>

namespace xjw::mesh
{

struct MeshIsotropicRemeshOptions
{
    int maximumPasses = 2;
    double shortEdgeRatio = 0.25;
    double longEdgeRatio = 2.0;
    double minimumAffectedAspectRatio = 10.0;
    double minimumWorstAspectImprovementRatio = 0.01;
    double maximumFeatureAngleDegrees = 45.0;
    double maximumNormalDeviationDegrees = 35.0;
    double maximumFaceGrowthRatio = 0.08;
    std::function<bool()> isCancelled;
};

struct MeshIsotropicRemeshStatistics
{
    int passCount = 0;
    int collapsedEdgeCount = 0;
    int splitEdgeCount = 0;
    int rejectedTopologyCount = 0;
    int rejectedFeatureCount = 0;
    int rejectedNormalCount = 0;
    int rejectedQualityCount = 0;
    bool cancelled = false;
};

MeshIsotropicRemeshStatistics remeshInteriorHighAspectTriangles(
    TriMesh *mesh,
    const MeshIsotropicRemeshOptions &options = {});

} // namespace xjw::mesh
