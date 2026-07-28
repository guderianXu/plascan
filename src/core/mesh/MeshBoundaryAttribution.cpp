#include "MeshBoundaryAttribution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace xjw::mesh
{
namespace
{

std::uint64_t edgeKey(int lhs, int rhs)
{
    const std::uint32_t low = static_cast<std::uint32_t>(std::min(lhs, rhs));
    const std::uint32_t high = static_cast<std::uint32_t>(std::max(lhs, rhs));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

int bitCount(std::uint16_t value)
{
    int count = 0;
    while (value != 0)
    {
        count += value & 1U;
        value >>= 1U;
    }
    return count;
}

std::size_t sampleIndex(const DepthTsdfLayout &layout, int x, int y, int z)
{
    const std::size_t samples_x =
        static_cast<std::size_t>(layout.cells[0] + 1);
    const std::size_t samples_y =
        static_cast<std::size_t>(layout.cells[1] + 1);
    return (static_cast<std::size_t>(z) * samples_y +
            static_cast<std::size_t>(y)) *
            samples_x +
        static_cast<std::size_t>(x);
}

std::array<int, 3> containingCell(const MeshVertex &point,
                                  const DepthTsdfLayout &layout)
{
    std::array<int, 3> cell{};
    const std::array<float, 3> coordinates{{point.x, point.y, point.z}};
    for (int axis = 0; axis < 3; ++axis)
    {
        const float voxel_size =
            std::max(layout.voxelSize[static_cast<std::size_t>(axis)],
                     std::numeric_limits<float>::epsilon());
        const float coordinate =
            (coordinates[static_cast<std::size_t>(axis)] -
             layout.boundsMin[static_cast<std::size_t>(axis)]) /
            voxel_size;
        cell[static_cast<std::size_t>(axis)] = std::clamp(
            static_cast<int>(std::floor(coordinate)),
            0,
            std::max(0, layout.cells[static_cast<std::size_t>(axis)] - 1));
    }
    return cell;
}

bool validInput(const DepthTsdfLayout &layout,
                std::size_t tsdfSize,
                std::size_t weightSize,
                std::size_t surfaceWeightSize,
                std::size_t sourceMaskSize,
                std::size_t spreadSize,
                std::size_t supportedSize)
{
    const std::size_t expected = static_cast<std::size_t>(layout.sampleCount);
    return layout.ok &&
        tsdfSize == expected &&
        weightSize == expected &&
        surfaceWeightSize == expected &&
        sourceMaskSize == expected &&
        spreadSize == expected &&
        supportedSize == expected;
}

} // namespace

MeshBoundaryAttributionStatistics attributeMeshBoundaryEdges(
    const TriMesh &mesh,
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<float> &surfaceObservationWeight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint16_t> &minimumInverseDepthSpread,
    const std::vector<std::uint8_t> &supported,
    const MeshBoundaryAttributionOptions &options,
    std::vector<MeshBoundaryAttributionReason> *vertexReasons)
{
    MeshBoundaryAttributionStatistics statistics;
    if (vertexReasons != nullptr)
    {
        vertexReasons->assign(
            mesh.vertices.size(),
            MeshBoundaryAttributionReason::None);
    }
    if (mesh.empty() ||
        !validInput(layout,
                    tsdf.size(),
                    weight.size(),
                    surfaceObservationWeight.size(),
                    geometrySourceMask.size(),
                    minimumInverseDepthSpread.size(),
                    supported.size()))
    {
        return statistics;
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh.faces.size() * 2U);
    for (const Triangle &face : mesh.faces)
    {
        if (face.v[0] < 0 || face.v[1] < 0 || face.v[2] < 0)
        {
            continue;
        }
        for (int edge = 0; edge < 3; ++edge)
        {
            const int lhs = face.v[edge];
            const int rhs = face.v[(edge + 1) % 3];
            if (lhs == rhs ||
                static_cast<std::size_t>(lhs) >= mesh.vertices.size() ||
                static_cast<std::size_t>(rhs) >= mesh.vertices.size())
            {
                continue;
            }
            ++edge_counts[edgeKey(lhs, rhs)];
        }
    }

    const int required_sources = std::max(1, options.minimumSourceCount);
    const float maximum_spread =
        std::max(0.0f, options.maximumInverseDepthSpread);
    const float minimum_surface_ratio =
        std::clamp(options.minimumSurfaceWeightRatio, 0.0f, 1.0f);
    const float maximum_absolute_tsdf =
        std::clamp(options.maximumAbsoluteTsdf, 0.0f, 1.0f);
    const auto assign_reason =
        [vertexReasons](std::size_t lhs,
                        std::size_t rhs,
                        MeshBoundaryAttributionReason reason)
    {
        if (vertexReasons == nullptr)
        {
            return;
        }
        for (const std::size_t vertex_index : {lhs, rhs})
        {
            MeshBoundaryAttributionReason &assigned =
                (*vertexReasons)[vertex_index];
            if (static_cast<std::uint8_t>(reason) >
                static_cast<std::uint8_t>(assigned))
            {
                assigned = reason;
            }
        }
    };
    for (const auto &[key, face_count] : edge_counts)
    {
        if (face_count != 1)
        {
            continue;
        }
        ++statistics.boundaryEdgeCount;
        const std::size_t lhs = static_cast<std::size_t>(key >> 32U);
        const std::size_t rhs =
            static_cast<std::size_t>(key & 0xffffffffULL);
        MeshVertex midpoint;
        midpoint.x = 0.5f * (mesh.vertices[lhs].x + mesh.vertices[rhs].x);
        midpoint.y = 0.5f * (mesh.vertices[lhs].y + mesh.vertices[rhs].y);
        midpoint.z = 0.5f * (mesh.vertices[lhs].z + mesh.vertices[rhs].z);
        const std::array<int, 3> cell = containingCell(midpoint, layout);

        int supported_corner_count = 0;
        bool has_observation = false;
        int maximum_source_count = 0;
        float minimum_spread = std::numeric_limits<float>::infinity();
        float maximum_surface_ratio = 0.0f;
        float minimum_absolute_tsdf = std::numeric_limits<float>::infinity();
        for (int dz = 0; dz <= 1; ++dz)
        {
            for (int dy = 0; dy <= 1; ++dy)
            {
                for (int dx = 0; dx <= 1; ++dx)
                {
                    const std::size_t index = sampleIndex(
                        layout,
                        cell[0] + dx,
                        cell[1] + dy,
                        cell[2] + dz);
                    supported_corner_count += supported[index] != 0 ? 1 : 0;
                    if (weight[index] <= 1.0e-6f)
                    {
                        continue;
                    }
                    has_observation = true;
                    maximum_source_count = std::max(
                        maximum_source_count,
                        bitCount(geometrySourceMask[index]));
                    const std::uint16_t encoded_spread =
                        minimumInverseDepthSpread[index];
                    if (encoded_spread !=
                        std::numeric_limits<std::uint16_t>::max())
                    {
                        minimum_spread = std::min(
                            minimum_spread,
                            static_cast<float>(encoded_spread) / 100000.0f);
                    }
                    maximum_surface_ratio = std::max(
                        maximum_surface_ratio,
                        surfaceObservationWeight[index] / weight[index]);
                    minimum_absolute_tsdf = std::min(
                        minimum_absolute_tsdf,
                        std::fabs(tsdf[index]));
                }
            }
        }

        if (supported_corner_count == 8)
        {
            ++statistics.extractionOrPostprocessEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::ExtractionOrPostprocess);
        }
        else if (supported_corner_count > 0)
        {
            ++statistics.supportGateRejectedEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::SupportGateRejected);
        }
        else if (!has_observation)
        {
            ++statistics.noObservationEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::NoObservation);
        }
        else if (maximum_source_count < required_sources)
        {
            ++statistics.insufficientSourceEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::InsufficientSource);
        }
        else if (!std::isfinite(minimum_spread) ||
                 minimum_spread > maximum_spread)
        {
            ++statistics.depthSpreadRejectedEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::DepthSpreadRejected);
        }
        else if (maximum_surface_ratio < minimum_surface_ratio)
        {
            ++statistics.surfaceWeightRejectedEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::SurfaceWeightRejected);
        }
        else if (minimum_absolute_tsdf > maximum_absolute_tsdf)
        {
            ++statistics.absoluteTsdfRejectedEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::AbsoluteTsdfRejected);
        }
        else
        {
            ++statistics.supportGateRejectedEdgeCount;
            assign_reason(
                lhs,
                rhs,
                MeshBoundaryAttributionReason::SupportGateRejected);
        }
    }
    const std::uint64_t classified =
        statistics.noObservationEdgeCount +
        statistics.insufficientSourceEdgeCount +
        statistics.depthSpreadRejectedEdgeCount +
        statistics.surfaceWeightRejectedEdgeCount +
        statistics.absoluteTsdfRejectedEdgeCount +
        statistics.supportGateRejectedEdgeCount +
        statistics.extractionOrPostprocessEdgeCount;
    statistics.unclassifiedEdgeCount =
        statistics.boundaryEdgeCount > classified
        ? statistics.boundaryEdgeCount - classified
        : 0;
    return statistics;
}

void applyMeshBoundaryAttributionColors(
    TriMesh *mesh,
    const std::vector<MeshBoundaryAttributionReason> &vertexReasons)
{
    if (mesh == nullptr || mesh->vertices.size() != vertexReasons.size())
    {
        return;
    }

    const auto color_for_reason = [](MeshBoundaryAttributionReason reason)
    {
        switch (reason)
        {
        case MeshBoundaryAttributionReason::ExtractionOrPostprocess:
            return std::array<std::uint8_t, 3>{{255, 128, 0}};
        case MeshBoundaryAttributionReason::SupportGateRejected:
            return std::array<std::uint8_t, 3>{{160, 64, 255}};
        case MeshBoundaryAttributionReason::AbsoluteTsdfRejected:
            return std::array<std::uint8_t, 3>{{255, 32, 32}};
        case MeshBoundaryAttributionReason::SurfaceWeightRejected:
            return std::array<std::uint8_t, 3>{{255, 224, 32}};
        case MeshBoundaryAttributionReason::DepthSpreadRejected:
            return std::array<std::uint8_t, 3>{{255, 32, 224}};
        case MeshBoundaryAttributionReason::InsufficientSource:
            return std::array<std::uint8_t, 3>{{32, 224, 255}};
        case MeshBoundaryAttributionReason::NoObservation:
            return std::array<std::uint8_t, 3>{{32, 96, 255}};
        case MeshBoundaryAttributionReason::Unclassified:
            return std::array<std::uint8_t, 3>{{255, 255, 255}};
        case MeshBoundaryAttributionReason::None:
        default:
            return std::array<std::uint8_t, 3>{{56, 56, 56}};
        }
    };

    mesh->hasVertexColors = true;
    for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
    {
        const std::array<std::uint8_t, 3> color =
            color_for_reason(vertexReasons[index]);
        mesh->vertices[index].r = color[0];
        mesh->vertices[index].g = color[1];
        mesh->vertices[index].b = color[2];
    }
}

QJsonObject meshBoundaryAttributionToJson(
    const MeshBoundaryAttributionStatistics &statistics)
{
    return {
        {QStringLiteral("boundary_edge_count"),
         static_cast<double>(statistics.boundaryEdgeCount)},
        {QStringLiteral("no_observation_edge_count"),
         static_cast<double>(statistics.noObservationEdgeCount)},
        {QStringLiteral("insufficient_source_edge_count"),
         static_cast<double>(statistics.insufficientSourceEdgeCount)},
        {QStringLiteral("depth_spread_rejected_edge_count"),
         static_cast<double>(statistics.depthSpreadRejectedEdgeCount)},
        {QStringLiteral("surface_weight_rejected_edge_count"),
         static_cast<double>(statistics.surfaceWeightRejectedEdgeCount)},
        {QStringLiteral("absolute_tsdf_rejected_edge_count"),
         static_cast<double>(statistics.absoluteTsdfRejectedEdgeCount)},
        {QStringLiteral("support_gate_rejected_edge_count"),
         static_cast<double>(statistics.supportGateRejectedEdgeCount)},
        {QStringLiteral("extraction_or_postprocess_edge_count"),
         static_cast<double>(statistics.extractionOrPostprocessEdgeCount)},
        {QStringLiteral("unclassified_edge_count"),
         static_cast<double>(statistics.unclassifiedEdgeCount)}};
}

} // namespace xjw::mesh
