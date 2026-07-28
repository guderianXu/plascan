#pragma once

#include "DepthTsdfSurfaceBuilder.h"
#include "MeshTypes.h"

#include <QJsonObject>

#include <cstdint>
#include <vector>

namespace xjw::mesh
{

struct MeshBoundaryAttributionOptions
{
    int minimumSourceCount = 2;
    float maximumInverseDepthSpread = 0.015f;
    float minimumSurfaceWeightRatio = 0.10f;
    float maximumAbsoluteTsdf = 0.45f;
};

struct MeshBoundaryAttributionStatistics
{
    std::uint64_t boundaryEdgeCount = 0;
    std::uint64_t noObservationEdgeCount = 0;
    std::uint64_t insufficientSourceEdgeCount = 0;
    std::uint64_t depthSpreadRejectedEdgeCount = 0;
    std::uint64_t surfaceWeightRejectedEdgeCount = 0;
    std::uint64_t absoluteTsdfRejectedEdgeCount = 0;
    std::uint64_t supportGateRejectedEdgeCount = 0;
    std::uint64_t extractionOrPostprocessEdgeCount = 0;
    std::uint64_t unclassifiedEdgeCount = 0;
};

enum class MeshBoundaryAttributionReason : std::uint8_t
{
    None = 0,
    ExtractionOrPostprocess,
    SupportGateRejected,
    AbsoluteTsdfRejected,
    SurfaceWeightRejected,
    DepthSpreadRejected,
    InsufficientSource,
    NoObservation,
    Unclassified
};

MeshBoundaryAttributionStatistics attributeMeshBoundaryEdges(
    const TriMesh &mesh,
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<float> &surfaceObservationWeight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint16_t> &minimumInverseDepthSpread,
    const std::vector<std::uint8_t> &supported,
    const MeshBoundaryAttributionOptions &options = {},
    std::vector<MeshBoundaryAttributionReason> *vertexReasons = nullptr);

void applyMeshBoundaryAttributionColors(
    TriMesh *mesh,
    const std::vector<MeshBoundaryAttributionReason> &vertexReasons);

QJsonObject meshBoundaryAttributionToJson(
    const MeshBoundaryAttributionStatistics &statistics);

} // namespace xjw::mesh
