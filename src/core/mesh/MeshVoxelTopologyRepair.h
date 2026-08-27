#pragma once

#include "MeshTypes.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>

namespace xjw::mesh
{

struct MeshVoxelTopologyRepairOptions
{
    int targetResolution = 160;
    // Try radii from zero upward and stop at the first strict genus-zero result.
    // A radius of four is required by the COLMAP building regression fixture;
    // keeping headroom up to the implementation clamp avoids a false failure
    // without forcing wider closing on inputs that already pass earlier.
    int maximumClosingRadius = 8;
    bool allowLargestComponentFallback = true;
    bool requireEnclosedInterior = true;
    bool preferSmoothExtraction = true;
    std::function<bool()> isCancelled;
};

struct MeshVoxelTopologyRepairStatistics
{
    int requestedResolution = 0;
    int effectiveResolution = 0;
    std::array<int, 3> gridDimensions{};
    float voxelSize = 0.0f;
    std::uint64_t validInputFaceCount = 0;
    std::uint64_t rejectedInputFaceCount = 0;
    std::uint64_t surfaceVoxelCount = 0;
    std::uint64_t enclosedInteriorCellCountBefore = 0;
    std::uint64_t enclosedInteriorCellCountAfter = 0;
    std::uint64_t occupiedCellCountBefore = 0;
    std::uint64_t occupiedCellCountAfter = 0;
    std::uint64_t discardedOccupiedCellCount = 0;
    int occupiedComponentCountBefore = 0;
    int occupiedComponentCountAfter = 0;
    int cubicalEulerBefore = 0;
    int cubicalEulerAfter = 0;
    int selectedClosingRadius = -1;
    std::uint64_t outputVertexCount = 0;
    std::uint64_t outputFaceCount = 0;
    std::uint64_t outputBoundaryEdgeCount = 0;
    std::uint64_t outputNonManifoldEdgeCount = 0;
    std::uint64_t outputNonManifoldVertexCount = 0;
    int outputComponentCount = 0;
    int outputSurfaceEulerCharacteristic = 0;
    bool usedLargestComponentFallback = false;
    bool smoothExtractionPreferred = false;
    bool smoothExtractionAvailable = false;
    bool smoothExtractionAttempted = false;
    bool smoothExtractionAccepted = false;
    bool smoothExtractionRejectedByTopology = false;
    bool cellBoundaryExtractionUsed = false;
    std::uint64_t smoothExtractionVertexCount = 0;
    std::uint64_t smoothExtractionFaceCount = 0;
    int smoothExtractionComponentCount = 0;
    int smoothExtractionBoundaryEdgeCount = 0;
    int smoothExtractionNonManifoldEdgeCount = 0;
    int smoothExtractionNonManifoldVertexCount = 0;
    int smoothExtractionSurfaceEulerCharacteristic = 0;
    double smoothExtractionHighAspectFaceRatio = 0.0;
    double smoothExtractionExtremeAspectFaceRatio = 0.0;
    bool smoothExtractionStrictGatePassed = false;
    bool smoothTriangleOptimizationAttempted = false;
    bool smoothTriangleOptimizationCancelled = false;
    bool smoothTriangleOptimizationAccepted = false;
    int smoothTriangleOptimizationPassCount = 0;
    int smoothTriangleOptimizationFlippedEdgeCount = 0;
    int smoothTriangleOptimizationTangentialPassCount = 0;
    int smoothTriangleOptimizationRelaxedVertexCount = 0;
    double smoothExtractionHighAspectFaceRatioBeforeOptimization = 0.0;
    double smoothExtractionExtremeAspectFaceRatioBeforeOptimization = 0.0;
};

struct MeshVoxelTopologyRepairResult
{
    bool ok = false;
    bool cancelled = false;
    std::string errorMessage;
    TriMesh mesh;
    MeshVoxelTopologyRepairStatistics statistics;
};

/**
 * @brief Repairs a closed-body mesh through a fail-closed cubical solid.
 *
 * Triangles are conservatively voxelized, exterior empty cells are found by a
 * six-neighbour boundary flood, and small morphological closings are tried in
 * increasing radius. A candidate is returned only when its cubical body is a
 * single component with Euler characteristic one. A locally averaged MC33
 * surface is preferred when it preserves the exact genus-zero topology;
 * otherwise the exact cubical cell boundary is used as a deterministic
 * fallback. Both paths require one closed two-manifold component with surface
 * Euler characteristic two.
 */
class MeshVoxelTopologyRepair
{
public:
    static MeshVoxelTopologyRepairResult repair(
        const TriMesh &mesh,
        const MeshVoxelTopologyRepairOptions &options = {});
};

} // namespace xjw::mesh
