#include "OrbitalSparseScaffoldSurfaceBuilder.h"

#include "VisualHullReconstructor.h"

#include <utility>

namespace xjw::mesh
{
namespace
{

bool isExactSingleBody(const MeshTopologyQualityStatistics &quality)
{
    return quality.componentCount == 1 &&
        quality.boundaryEdgeCount == 0 &&
        quality.nonManifoldEdgeCount == 0 &&
        quality.nonManifoldVertexCount == 0 &&
        quality.closedTwoManifold &&
        quality.eulerCharacteristic == 2 &&
        quality.componentEulerCharacteristics == std::vector<int>{2};
}

} // namespace

OrbitalSparseScaffoldSurfaceResult
OrbitalSparseScaffoldSurfaceBuilder::build(
    const std::filesystem::path &mvsOutputPath,
    const std::filesystem::path &explicitSparsePly,
    const std::filesystem::path &explicitPointsJson,
    const OrbitalSparseScaffoldSurfaceOptions &options)
{
    OrbitalSparseScaffoldSurfaceResult result;
    SparseOrbitalScaffoldResult scaffold =
        SparseOrbitalScaffoldBuilder::build(
            mvsOutputPath,
            explicitSparsePly,
            explicitPointsJson,
            options.scaffold);
    result.sparsePlyPath = scaffold.sourcePath;
    result.sparsePointsJsonPath = scaffold.qualitySidecarPath;
    result.statistics.scaffold = scaffold.statistics;
    if (!scaffold.succeeded())
    {
        result.error = scaffold.error.empty()
            ? "failed to build the quality-filtered sparse scaffold"
            : scaffold.error;
        return result;
    }

    ScreenedPoissonResult poisson = ScreenedPoissonSurfaceBuilder::build(
        scaffold.points,
        scaffold.normals,
        options.poisson);
    result.statistics.poisson = poisson.statistics;
    if (!poisson.ok)
    {
        result.error = poisson.error.empty()
            ? "Screened Poisson failed for the sparse scaffold"
            : poisson.error;
        return result;
    }

    result.statistics.topologyBeforeRepair =
        evaluateMeshTopologyQuality(poisson.mesh);
    const MeshConnectivityStats connectivity =
        VisualHullReconstructor::analyzeConnectivity(poisson.mesh);
    result.statistics.poissonComponentCount = connectivity.componentCount;
    result.statistics.poissonLargestComponentFaceRatio =
        connectivity.largestComponentFaceRatio;
    if (connectivity.componentCount > 1)
    {
        if (!VisualHullReconstructor::retainLargestConnectedComponent(
                &poisson.mesh))
        {
            result.error =
                "failed to isolate the largest Screened Poisson component";
            return result;
        }
        result.statistics.removedSatelliteComponentCount =
            connectivity.componentCount - 1;
    }
    result.statistics.componentFilteredVertexCount = poisson.mesh.vertices.size();
    result.statistics.componentFilteredFaceCount = poisson.mesh.faces.size();
    result.statistics.topologyAfterComponentFilter =
        evaluateMeshTopologyQuality(poisson.mesh);
    if (isExactSingleBody(result.statistics.topologyAfterComponentFilter))
    {
        result.mesh = std::move(poisson.mesh);
        result.mesh.hasVertexColors = false;
        result.statistics.topologyAfterRepair =
            result.statistics.topologyAfterComponentFilter;
        result.ok = true;
        return result;
    }

    result.statistics.topologyRepairAttempted = true;
    MeshVoxelTopologyRepairResult repaired =
        MeshVoxelTopologyRepair::repair(poisson.mesh, options.topologyRepair);
    result.statistics.topologyRepair = repaired.statistics;
    result.cancelled = repaired.cancelled;
    if (!repaired.ok)
    {
        result.error = repaired.errorMessage.empty()
            ? "sparse scaffold topology repair failed"
            : repaired.errorMessage;
        return result;
    }

    result.statistics.topologyAfterRepair =
        evaluateMeshTopologyQuality(repaired.mesh);
    if (!isExactSingleBody(result.statistics.topologyAfterRepair))
    {
        result.error =
            "sparse scaffold repair did not produce one closed genus-zero body";
        return result;
    }

    result.mesh = std::move(repaired.mesh);
    result.mesh.hasVertexColors = false;
    result.statistics.topologyRepairApplied = true;
    result.ok = true;
    return result;
}

} // namespace xjw::mesh
