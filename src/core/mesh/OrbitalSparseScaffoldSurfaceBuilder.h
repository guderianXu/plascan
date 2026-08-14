#pragma once

#include "MeshTopologyQuality.h"
#include "MeshVoxelTopologyRepair.h"
#include "ScreenedPoissonSurfaceBuilder.h"
#include "SparseOrbitalScaffoldBuilder.h"

#include <filesystem>
#include <string>

namespace xjw::mesh
{

struct OrbitalSparseScaffoldSurfaceOptions
{
    SparseOrbitalScaffoldOptions scaffold;
    ScreenedPoissonOptions poisson;
    MeshVoxelTopologyRepairOptions topologyRepair;
};

struct OrbitalSparseScaffoldSurfaceStatistics
{
    SparseOrbitalScaffoldStatistics scaffold;
    ScreenedPoissonStatistics poisson;
    MeshVoxelTopologyRepairStatistics topologyRepair;
    MeshTopologyQualityStatistics topologyBeforeRepair;
    MeshTopologyQualityStatistics topologyAfterComponentFilter;
    MeshTopologyQualityStatistics topologyAfterRepair;
    int poissonComponentCount = 0;
    double poissonLargestComponentFaceRatio = 0.0;
    int removedSatelliteComponentCount = 0;
    std::size_t componentFilteredVertexCount = 0;
    std::size_t componentFilteredFaceCount = 0;
    bool topologyRepairAttempted = false;
    bool topologyRepairApplied = false;
};

struct OrbitalSparseScaffoldSurfaceResult
{
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::filesystem::path sparsePlyPath;
    std::filesystem::path sparsePointsJsonPath;
    TriMesh mesh;
    OrbitalSparseScaffoldSurfaceStatistics statistics;
};

/**
 * Builds a fail-closed global carrier for a single orbital small body.
 *
 * Only SfM points carrying the required track/reprojection/triangulation
 * quality metadata are admitted. Screened Poisson supplies a smooth global
 * implicit surface; a voxel solid repair is mandatory whenever that surface
 * is not already one closed genus-zero two-manifold component.
 */
class OrbitalSparseScaffoldSurfaceBuilder
{
public:
    static OrbitalSparseScaffoldSurfaceResult build(
        const std::filesystem::path &mvsOutputPath,
        const std::filesystem::path &explicitSparsePly = {},
        const std::filesystem::path &explicitPointsJson = {},
        const OrbitalSparseScaffoldSurfaceOptions &options = {});
};

} // namespace xjw::mesh
