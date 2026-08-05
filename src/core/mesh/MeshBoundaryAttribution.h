#pragma once

#include "DepthTsdfSurfaceBuilder.h"
#include "MeshTypes.h"

#include <QJsonObject>

#include <array>
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

struct MeshBoundaryEdgeAttribution
{
    int firstVertex = -1;
    int secondVertex = -1;
    MeshBoundaryAttributionReason reason = MeshBoundaryAttributionReason::None;
    MeshBoundaryAttributionReason evidenceReason = MeshBoundaryAttributionReason::None;
    std::uint16_t sourceMask = 0;
    int supportedCornerCount = 0;
    int observedUnsupportedCornerCount = 0;
    int maximumSourceCount = 0;
    float inverseDepthSpread = 0.0f;
    float surfaceWeightRatio = 0.0f;
    float absoluteTsdf = 0.0f;
    std::array<float, 3> firstPoint{};
    std::array<float, 3> secondPoint{};
    std::array<float, 3> midpoint{};
    std::array<float, 3> normal{};
    float length = 0.0f;
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
    std::vector<MeshBoundaryAttributionReason> *vertexReasons = nullptr,
    std::vector<MeshBoundaryEdgeAttribution> *edgeAttributions = nullptr);

void applyMeshBoundaryAttributionColors(
    TriMesh *mesh,
    const std::vector<MeshBoundaryAttributionReason> &vertexReasons);

QJsonObject meshBoundaryAttributionToJson(
    const MeshBoundaryAttributionStatistics &statistics);

} // namespace xjw::mesh
