#include "DepthTsdfInternals.h"
namespace xjw::mesh::tsdf_detail
{
    using namespace tsdf_detail;

    WeakBoundaryTipResult trimWeakBoundaryTips(TriMesh* mesh,
                                               const DepthTsdfLayout& layout,
                                               const std::vector<std::uint16_t>& support,
                                               const std::vector<std::uint8_t>& strongAdaptiveSupport,
                                               int minimum_support,
                                               int passes,
                                               bool enabled)
    {
        WeakBoundaryTipResult result;
        if (!mesh || mesh->faces.empty() || support.empty())
        {
            return result;
        }
        const auto edge_key = [](int first, int second)
        {
            const auto low = static_cast<std::uint32_t>(std::min(first, second));
            const auto high = static_cast<std::uint32_t>(std::max(first, second));
            return (static_cast<std::uint64_t>(low) << 32U) | high;
        };
        std::vector<std::uint8_t> counted_weak_vertices(mesh->vertices.size(), 0);
        const int maximum_passes = enabled ? std::clamp(passes, 1, 4) : 1;
        for (int pass = 0; pass < maximum_passes; ++pass)
        {
            std::unordered_map<std::uint64_t, int> edge_counts;
            edge_counts.reserve(mesh->faces.size() * 3);
            for (const Triangle& face : mesh->faces)
            {
                ++edge_counts[edge_key(face.v[0], face.v[1])];
                ++edge_counts[edge_key(face.v[1], face.v[2])];
                ++edge_counts[edge_key(face.v[2], face.v[0])];
            }
            std::vector<std::uint8_t> boundary_vertex(mesh->vertices.size(), 0);
            for (const Triangle& face : mesh->faces)
            {
                const std::array<std::array<int, 2>, 3> edges{
                    {{{face.v[0], face.v[1]}}, {{face.v[1], face.v[2]}}, {{face.v[2], face.v[0]}}}};
                for (const auto& edge : edges)
                {
                    if (edge_counts[edge_key(edge[0], edge[1])] == 1)
                    {
                        boundary_vertex[static_cast<std::size_t>(edge[0])] = 1;
                        boundary_vertex[static_cast<std::size_t>(edge[1])] = 1;
                    }
                }
            }
            std::vector<std::uint8_t> weak_vertex(mesh->vertices.size(), 0);
            for (std::size_t index = 0; index < mesh->vertices.size(); ++index)
            {
                if (!boundary_vertex[index])
                {
                    continue;
                }
                const MeshVertex& vertex = mesh->vertices[index];
                const int x =
                    std::clamp(static_cast<int>(std::lround((vertex.x - layout.boundsMin[0]) / layout.voxelSize[0])),
                               0,
                               layout.cells[0]);
                const int y =
                    std::clamp(static_cast<int>(std::lround((vertex.y - layout.boundsMin[1]) / layout.voxelSize[1])),
                               0,
                               layout.cells[1]);
                const int z =
                    std::clamp(static_cast<int>(std::lround((vertex.z - layout.boundsMin[2]) / layout.voxelSize[2])),
                               0,
                               layout.cells[2]);
                const std::size_t sample_index = sampleIndex(layout, x, y, z);
                const bool has_strong_adaptive_support =
                    sample_index < strongAdaptiveSupport.size() && strongAdaptiveSupport[sample_index] != 0;
                if (support[sample_index] < minimum_support && !has_strong_adaptive_support)
                {
                    weak_vertex[index] = 1;
                    if (!counted_weak_vertices[index])
                    {
                        counted_weak_vertices[index] = 1;
                        ++result.weakVertexCount;
                    }
                }
            }

            std::vector<std::uint8_t> remove_face(mesh->faces.size(), 0);
            int candidates = 0;
            for (std::size_t index = 0; index < mesh->faces.size(); ++index)
            {
                const Triangle& face = mesh->faces[index];
                const std::array<std::array<int, 2>, 3> edges{
                    {{{face.v[0], face.v[1]}}, {{face.v[1], face.v[2]}}, {{face.v[2], face.v[0]}}}};
                int boundary_edge_count = 0;
                for (const auto& edge : edges)
                {
                    boundary_edge_count += edge_counts[edge_key(edge[0], edge[1])] == 1;
                }
                const int weak_vertex_count = static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[0])]) +
                                              static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[1])]) +
                                              static_cast<int>(weak_vertex[static_cast<std::size_t>(face.v[2])]);
                const bool candidate =
                    DepthTsdfSurfaceBuilder::shouldTrimWeakBoundaryFace(boundary_edge_count, weak_vertex_count);
                if (candidate)
                {
                    remove_face[index] = 1;
                    ++candidates;
                }
            }
            result.candidateFaceCount += candidates;
            if (!enabled || candidates == 0)
            {
                break;
            }
            std::vector<Triangle> kept_faces;
            kept_faces.reserve(mesh->faces.size() - static_cast<std::size_t>(candidates));
            for (std::size_t index = 0; index < mesh->faces.size(); ++index)
            {
                if (!remove_face[index])
                {
                    kept_faces.push_back(mesh->faces[index]);
                }
            }
            mesh->faces = std::move(kept_faces);
            result.trimmedFaceCount += candidates;
        }
        return result;
    }
} // namespace xjw::mesh::tsdf_detail
