#include "ModelWorkflowInternals.h"
namespace xjw::mesh::workflow
{
    using namespace workflow_detail;

    FinalSurfaceDenoisingResult applyTopologyGuardedFinalSurfaceDenoising(TriMesh* mesh,
                                                                          int iterations,
                                                                          float lambda,
                                                                          float mu,
                                                                          float maximumDisplacement,
                                                                          float featureAngleDegrees,
                                                                          int boundaryProtectionRings,
                                                                          bool allowCarrierAreaRelaxation)
    {
        FinalSurfaceDenoisingResult result;
        if (mesh == nullptr || mesh->empty() || iterations <= 0 || lambda <= 0.0f || maximumDisplacement <= 0.0f)
        {
            return result;
        }

        result.attempted = true;
        result.carrierAreaRelaxationEnabled = allowCarrierAreaRelaxation;
        result.areaBefore = meshSurfaceArea(*mesh);
        result.absoluteVolumeBefore = meshAbsoluteOrientedVolume(*mesh);
        result.qualityBefore = evaluateMeshTopologyQuality(*mesh);
        const MeshTopologySignature topology_before = meshTopologySignature(result.qualityBefore);

        TriMesh candidate = *mesh;
        result.movedVertexCount = detail::smoothSurfaceVerticesTaubinProtected(
            &candidate, iterations, lambda, mu, maximumDisplacement, featureAngleDegrees, boundaryProtectionRings);
        detail::recomputeNormals(&candidate);
        result.areaAfter = meshSurfaceArea(candidate);
        result.absoluteVolumeAfter = meshAbsoluteOrientedVolume(candidate);
        result.qualityAfter = evaluateMeshTopologyQuality(candidate);
        const MeshTopologySignature topology_after = meshTopologySignature(result.qualityAfter);

        result.areaRatio = result.areaBefore > 1.0e-12 ? result.areaAfter / result.areaBefore : 0.0;
        result.absoluteVolumeRatio =
            result.absoluteVolumeBefore > 1.0e-12 ? result.absoluteVolumeAfter / result.absoluteVolumeBefore : 0.0;
        const bool mesh_structure_preserved =
            candidate.vertexCount() == mesh->vertexCount() && candidate.faceCount() == mesh->faceCount() &&
            hasSameFaceIndexBuffer(*mesh, candidate) && topology_after == topology_before &&
            result.qualityAfter.validFaceCount == result.qualityBefore.validFaceCount;
        const double high_aspect_ratio_tolerance =
            std::max(1.0e-6, 4.0 / std::max(1, result.qualityBefore.validFaceCount));
        const double extreme_aspect_ratio_tolerance =
            std::max(1.0e-6, 2.0 / std::max(1, result.qualityBefore.validFaceCount));
        const bool triangle_quality_not_worse =
            result.qualityAfter.highAspectFaceRatio <=
                result.qualityBefore.highAspectFaceRatio + high_aspect_ratio_tolerance &&
            result.qualityAfter.extremeAspectFaceRatio <=
                result.qualityBefore.extremeAspectFaceRatio + extreme_aspect_ratio_tolerance;
        const bool normal_quality_not_worse = result.qualityAfter.adjacentNormalAngleP90Degrees <=
                                                  result.qualityBefore.adjacentNormalAngleP90Degrees + 1.0e-6 &&
                                              result.qualityAfter.adjacentNormalAngleOver30Ratio <=
                                                  result.qualityBefore.adjacentNormalAngleOver30Ratio + 1.0e-6;
        const bool normal_quality_improved = result.qualityAfter.adjacentNormalAngleP90Degrees <=
                                                 result.qualityBefore.adjacentNormalAngleP90Degrees - 0.25 ||
                                             result.qualityAfter.adjacentNormalAngleOver30Ratio <=
                                                 result.qualityBefore.adjacentNormalAngleOver30Ratio - 0.001 ||
                                             result.qualityAfter.adjacentNormalAngleMedianDegrees <=
                                                 result.qualityBefore.adjacentNormalAngleMedianDegrees - 0.10;

        const bool closed_strict_topology_preserved =
            mesh_structure_preserved && result.qualityBefore.closedTwoManifold && result.qualityAfter.closedTwoManifold;
        const bool absolute_volume_preserved = std::isfinite(result.absoluteVolumeRatio) &&
                                               result.absoluteVolumeRatio >= 0.98 && result.absoluteVolumeRatio <= 1.02;
        const bool carrier_area_relaxation_eligible = allowCarrierAreaRelaxation && closed_strict_topology_preserved &&
                                                      absolute_volume_preserved && triangle_quality_not_worse &&
                                                      normal_quality_not_worse && normal_quality_improved;
        result.effectiveMinimumAreaRatio = carrier_area_relaxation_eligible ? 0.60 : 0.96;
        result.carrierAreaRelaxationApplied = carrier_area_relaxation_eligible &&
                                              result.areaRatio >= result.effectiveMinimumAreaRatio &&
                                              result.areaRatio < 0.96;
        const bool geometry_preserved =
            result.movedVertexCount > 0 && mesh_structure_preserved && std::isfinite(result.areaRatio) &&
            result.areaRatio >= result.effectiveMinimumAreaRatio && result.areaRatio <= 1.01;

        result.accepted =
            geometry_preserved && triangle_quality_not_worse && normal_quality_not_worse && normal_quality_improved;
        if (result.accepted)
        {
            *mesh = std::move(candidate);
        }
        return result;
    }
} // namespace xjw::mesh::workflow
