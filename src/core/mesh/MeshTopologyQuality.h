#pragma once

#include "MeshTypes.h"

#include <functional>

namespace xjw::mesh
{

struct MeshTopologyQualityThresholds
{
    double skinnyTriangleQuality = 0.05;
    double highAspectRatio = 10.0;
    double extremeAspectRatio = 20.0;
    double maximumBoundaryEdgeRatio = 0.01;
    double maximumHighAspectFaceRatio = 0.02;
    double maximumExtremeAspectFaceRatio = 0.005;
    int maximumNonManifoldEdgeCount = 0;
    int maximumComponentCount = 1;
    double minimumLargestComponentFaceRatio = 0.995;
};

struct MeshTopologyQualityStatistics
{
    int validFaceCount = 0;
    int uniqueEdgeCount = 0;
    int boundaryEdgeCount = 0;
    int nonManifoldEdgeCount = 0;
    int componentCount = 0;
    int largestComponentFaceCount = 0;
    int skinnyFaceCount = 0;
    int highAspectFaceCount = 0;
    int extremeAspectFaceCount = 0;
    double boundaryEdgeRatio = 0.0;
    double largestComponentFaceRatio = 0.0;
    double skinnyFaceRatio = 0.0;
    double highAspectFaceRatio = 0.0;
    double extremeAspectFaceRatio = 0.0;
    bool strictGatePassed = false;
};

struct MeshTriangleOptimizationOptions
{
    int maximumPasses = 4;
    double minimumWorstAspectImprovementRatio = 0.01;
    double maximumFeatureAngleDegrees = 45.0;
    double maximumNormalDeviationDegrees = 35.0;
    bool enableTangentialRelaxation = true;
    int tangentialRelaxationPasses = 3;
    double tangentialRelaxationLambda = 0.45;
    double tangentialMaximumDisplacementEdgeRatio = 0.20;
    bool enableIsotropicRemeshing = false;
    int isotropicRemeshingPasses = 2;
    double isotropicShortEdgeRatio = 0.25;
    double isotropicLongEdgeRatio = 2.0;
    double isotropicMaximumFaceGrowthRatio = 0.08;
    std::function<bool()> isCancelled;
};

struct MeshTriangleOptimizationStatistics
{
    int inputFaceCount = 0;
    int outputFaceCount = 0;
    int passCount = 0;
    int flippedEdgeCount = 0;
    int rejectedExistingDiagonalCount = 0;
    int rejectedFeatureEdgeCount = 0;
    int rejectedNormalCount = 0;
    int rejectedQualityCount = 0;
    int tangentialRelaxationPassCount = 0;
    int tangentialRelaxedVertexCount = 0;
    int isotropicRemeshingPassCount = 0;
    int isotropicCollapsedEdgeCount = 0;
    int isotropicSplitEdgeCount = 0;
    bool cancelled = false;
};

MeshTopologyQualityStatistics evaluateMeshTopologyQuality(
    const TriMesh &mesh,
    const MeshTopologyQualityThresholds &thresholds = {});

bool passesMeshTopologyQualityGate(
    const MeshTopologyQualityStatistics &statistics,
    const MeshTopologyQualityThresholds &thresholds = {});

MeshTriangleOptimizationStatistics optimizeTriangleQuality(
    TriMesh *mesh,
    const MeshTriangleOptimizationOptions &options = {});

} // namespace xjw::mesh
