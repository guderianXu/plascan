#include "ConsistentIsoSurfaceExtractor.h"

#include "ConsistentIsoSurfaceInternal.h"
#include "IsoSurfaceTopology.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <plapoint/mesh/marching_cubes.h>

namespace xjw::mesh
{
ConsistentIsoSurfaceResult ConsistentIsoSurfaceExtractor::extract(
    const std::array<float, 3> &boundsMin,
    const std::array<float, 3> &boundsMax,
    const std::array<int, 3> &cells,
    const std::vector<float> &field,
    const std::vector<std::uint8_t> &extractionSupport,
    const ConsistentIsoSurfaceOptions &options)
{
    ConsistentIsoSurfaceResult result;
    try
    {
        const std::size_t expected_samples =
            detail::checkedIsoSurfaceSampleCount(cells);
        if (field.size() != expected_samples)
        {
            throw std::invalid_argument("Iso-surface field size does not match grid");
        }
        if (!extractionSupport.empty() &&
            extractionSupport.size() != expected_samples)
        {
            throw std::invalid_argument(
                "Iso-surface extraction support size does not match grid");
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(boundsMin[axis]) || !std::isfinite(boundsMax[axis]) ||
                !(boundsMin[axis] < boundsMax[axis]))
            {
                throw std::invalid_argument("Iso-surface bounds must be finite and ordered");
            }
        }
        if (!std::isfinite(options.isoLevel))
        {
            throw std::invalid_argument("Iso-surface level must be finite");
        }

        const std::array<float, 3> step{
            (boundsMax[0] - boundsMin[0]) / static_cast<float>(cells[0]),
            (boundsMax[1] - boundsMin[1]) / static_cast<float>(cells[1]),
            (boundsMax[2] - boundsMin[2]) / static_cast<float>(cells[2])};
        std::unordered_map<std::uint64_t, int> edge_vertices;
        std::unordered_set<std::uint64_t> ambiguous_faces;
        std::unordered_set<std::uint64_t> tied_faces;

        const std::uint64_t total_cells =
            static_cast<std::uint64_t>(cells[0]) *
            static_cast<std::uint64_t>(cells[1]) *
            static_cast<std::uint64_t>(cells[2]);
        for (int z = 0; z < cells[2]; ++z)
        {
            for (int y = 0; y < cells[1]; ++y)
            {
                for (int x = 0; x < cells[0]; ++x)
                {
                    ++result.statistics.visitedCellCount;
                    if ((result.statistics.visitedCellCount & 4095u) == 0u)
                    {
                        if (options.isCancelled && options.isCancelled())
                        {
                            result.cancelled = true;
                            result.errorMessage = "Iso-surface extraction cancelled";
                            return result;
                        }
                        if (options.progress)
                        {
                            options.progress(
                                static_cast<double>(result.statistics.visitedCellCount) /
                                static_cast<double>(total_cells));
                        }
                    }

                    std::array<float, 8> values{};
                    std::array<MeshVertex, 8> points{};
                    std::array<bool, 12> active_edges{};
                    bool has_inside = false;
                    bool has_outside = false;
                    for (int corner = 0; corner < 8; ++corner)
                    {
                        const int cx = x + detail::kIsoSurfaceCorners[corner][0];
                        const int cy = y + detail::kIsoSurfaceCorners[corner][1];
                        const int cz = z + detail::kIsoSurfaceCorners[corner][2];
                        const std::size_t sample_index =
                            detail::isoSurfaceSampleIndex(cells, cx, cy, cz);
                        const float value =
                            extractionSupport.empty() ||
                                extractionSupport[sample_index] != 0u
                            ? field[sample_index]
                            : options.isoLevel + 1.0f;
                        if (!std::isfinite(value))
                        {
                            throw std::invalid_argument("Iso-surface samples must be finite");
                        }
                        values[static_cast<std::size_t>(corner)] = value;
                        has_inside = has_inside || value < options.isoLevel;
                        has_outside = has_outside || value >= options.isoLevel;
                        points[static_cast<std::size_t>(corner)].x =
                            boundsMin[0] + static_cast<float>(cx) * step[0];
                        points[static_cast<std::size_t>(corner)].y =
                            boundsMin[1] + static_cast<float>(cy) * step[1];
                        points[static_cast<std::size_t>(corner)].z =
                            boundsMin[2] + static_cast<float>(cz) * step[2];
                    }
                    if (!has_inside || !has_outside)
                    {
                        continue;
                    }
                    ++result.statistics.activeCellCount;

                    for (int edge = 0; edge < 12; ++edge)
                    {
                        const int first = detail::kIsoSurfaceEdgeCorners[edge][0];
                        const int second = detail::kIsoSurfaceEdgeCorners[edge][1];
                        active_edges[static_cast<std::size_t>(edge)] =
                            (values[first] < options.isoLevel) !=
                            (values[second] < options.isoLevel);
                    }

                    std::array<std::vector<int>, 12> adjacency;
                    int ambiguous_face_count = 0;
                    for (int face = 0; face < 6; ++face)
                    {
                        std::array<int, 4> crossings{};
                        int crossing_count = 0;
                        for (int face_edge = 0; face_edge < 4; ++face_edge)
                        {
                            const int edge =
                                detail::kIsoSurfaceFaceEdges[face][face_edge];
                            if (active_edges[static_cast<std::size_t>(edge)])
                            {
                                crossings[static_cast<std::size_t>(crossing_count++)] =
                                    face_edge;
                            }
                        }
                        if (crossing_count == 2)
                        {
                            detail::addIsoSurfaceConnection(
                                &adjacency,
                                detail::kIsoSurfaceFaceEdges[face][crossings[0]],
                                detail::kIsoSurfaceFaceEdges[face][crossings[1]]);
                        }
                        else if (crossing_count == 4)
                        {
                            ++ambiguous_face_count;
                            std::array<float, 4> face_values{};
                            for (int index = 0; index < 4; ++index)
                            {
                                face_values[static_cast<std::size_t>(index)] =
                                    values[
                                        detail::kIsoSurfaceFaceCorners[face][index]];
                            }
                            const GridFaceKey key =
                                detail::isoSurfaceFaceKey(x, y, z, face);
                            const IsoSurfaceFaceDecision decision =
                                decideIsoSurfaceFace(
                                    face_values,
                                    options.isoLevel,
                                    key,
                                    options.ambiguityEpsilon);
                            const std::uint64_t encoded_key =
                                encodeGridFaceKey(key, cells);
                            ambiguous_faces.insert(encoded_key);
                            if (decision.usedTieBreak)
                            {
                                tied_faces.insert(encoded_key);
                            }
                            if (decision.connectEdge01And23)
                            {
                                detail::addIsoSurfaceConnection(
                                    &adjacency,
                                    detail::kIsoSurfaceFaceEdges[face][0],
                                    detail::kIsoSurfaceFaceEdges[face][1]);
                                detail::addIsoSurfaceConnection(
                                    &adjacency,
                                    detail::kIsoSurfaceFaceEdges[face][2],
                                    detail::kIsoSurfaceFaceEdges[face][3]);
                            }
                            else
                            {
                                detail::addIsoSurfaceConnection(
                                    &adjacency,
                                    detail::kIsoSurfaceFaceEdges[face][0],
                                    detail::kIsoSurfaceFaceEdges[face][3]);
                                detail::addIsoSurfaceConnection(
                                    &adjacency,
                                    detail::kIsoSurfaceFaceEdges[face][1],
                                    detail::kIsoSurfaceFaceEdges[face][2]);
                            }
                        }
                    }

                    bool valid_adjacency = true;
                    for (int edge = 0; edge < 12; ++edge)
                    {
                        if (active_edges[static_cast<std::size_t>(edge)] &&
                            adjacency[static_cast<std::size_t>(edge)].size() != 2u)
                        {
                            valid_adjacency = false;
                        }
                    }
                    if (!valid_adjacency)
                    {
                        ++result.statistics.unresolvedCellCount;
                        continue;
                    }

                    std::array<int, 12> local_vertex_indices{};
                    local_vertex_indices.fill(-1);
                    for (int edge = 0; edge < 12; ++edge)
                    {
                        if (!active_edges[static_cast<std::size_t>(edge)])
                        {
                            continue;
                        }
                        const std::uint64_t key =
                            detail::isoSurfaceEdgeKey(cells, x, y, z, edge);
                        const auto existing = edge_vertices.find(key);
                        if (existing != edge_vertices.end())
                        {
                            local_vertex_indices[static_cast<std::size_t>(edge)] =
                                existing->second;
                            ++result.statistics.edgeVertexCacheHitCount;
                            continue;
                        }
                        const int vertex_index =
                            static_cast<int>(result.mesh.vertices.size());
                        result.mesh.vertices.push_back(
                            detail::interpolateIsoSurfaceVertex(
                                points, values, edge, options.isoLevel));
                        edge_vertices.emplace(key, vertex_index);
                        local_vertex_indices[static_cast<std::size_t>(edge)] =
                            vertex_index;
                        ++result.statistics.edgeVertexCacheMissCount;
                    }

                    const std::array<double, 3> gradient{
                        ((values[1] + values[2] + values[5] + values[6]) -
                         (values[0] + values[3] + values[4] + values[7])) /
                            (4.0 * step[0]),
                        ((values[2] + values[3] + values[6] + values[7]) -
                         (values[0] + values[1] + values[4] + values[5])) /
                            (4.0 * step[1]),
                        ((values[4] + values[5] + values[6] + values[7]) -
                         (values[0] + values[1] + values[2] + values[3])) /
                            (4.0 * step[2])};
                    const int topology_loop_count =
                        detail::countIsoSurfaceLoops(active_edges, adjacency);
                    if (topology_loop_count < 1)
                    {
                        ++result.statistics.unresolvedCellCount;
                        continue;
                    }
                    int cube_index = 0;
                    for (int corner = 0; corner < 8; ++corner)
                    {
                        if (values[corner] < options.isoLevel)
                        {
                            cube_index |= 1 << corner;
                        }
                    }
                    const bool classic_topology_matches =
                        ambiguous_face_count == 0 &&
                        topology_loop_count == 1 &&
                        detail::classicIsoSurfaceTriangulationMatchesFaceGraph(
                            cube_index,
                            active_edges,
                            adjacency);
                    if (classic_topology_matches)
                    {
                        const auto &triangles =
                            plapoint::mesh::detail::triTable(cube_index);
                        for (std::size_t index = 0;
                             index + 2u < triangles.size();
                             index += 3u)
                        {
                            detail::appendOrientedIsoSurfaceFace(
                                &result.mesh,
                                local_vertex_indices[static_cast<std::size_t>(
                                    triangles[index])],
                                local_vertex_indices[static_cast<std::size_t>(
                                    triangles[index + 1u])],
                                local_vertex_indices[static_cast<std::size_t>(
                                    triangles[index + 2u])],
                                gradient,
                                &result.statistics);
                        }
                        continue;
                    }
                    ++result.statistics.topologyAdjustedCellCount;
                    std::array<bool, 12> visited_edges{};
                    int loop_count = 0;
                    for (int start = 0; start < 12; ++start)
                    {
                        if (!active_edges[static_cast<std::size_t>(start)] ||
                            visited_edges[static_cast<std::size_t>(start)])
                        {
                            continue;
                        }
                        std::vector<int> loop;
                        int previous = -1;
                        int current = start;
                        do
                        {
                            if (loop.size() > 12u ||
                                visited_edges[static_cast<std::size_t>(current)])
                            {
                                loop.clear();
                                break;
                            }
                            loop.push_back(current);
                            visited_edges[static_cast<std::size_t>(current)] = true;
                            const auto &neighbors =
                                adjacency[static_cast<std::size_t>(current)];
                            const int next =
                                neighbors[0] == previous ? neighbors[1] : neighbors[0];
                            previous = current;
                            current = next;
                        }
                        while (current != start);
                        if (loop.size() < 3u)
                        {
                            ++result.statistics.unresolvedCellCount;
                            continue;
                        }
                        ++loop_count;
                        if (loop.size() == 3u)
                        {
                            detail::appendOrientedIsoSurfaceFace(
                                &result.mesh,
                                local_vertex_indices[static_cast<std::size_t>(loop[0])],
                                local_vertex_indices[static_cast<std::size_t>(loop[1])],
                                local_vertex_indices[static_cast<std::size_t>(loop[2])],
                                gradient,
                                &result.statistics);
                            continue;
                        }

                        MeshVertex center;
                        for (int edge : loop)
                        {
                            const MeshVertex &vertex = result.mesh.vertices[
                                static_cast<std::size_t>(
                                    local_vertex_indices[
                                        static_cast<std::size_t>(edge)])];
                            center.x += vertex.x;
                            center.y += vertex.y;
                            center.z += vertex.z;
                        }
                        const float inverse_count =
                            1.0f / static_cast<float>(loop.size());
                        center.x *= inverse_count;
                        center.y *= inverse_count;
                        center.z *= inverse_count;
                        const int center_index =
                            static_cast<int>(result.mesh.vertices.size());
                        result.mesh.vertices.push_back(center);
                        ++result.statistics.interiorLoopVertexCount;
                        for (std::size_t index = 0; index < loop.size(); ++index)
                        {
                            detail::appendOrientedIsoSurfaceFace(
                                &result.mesh,
                                center_index,
                                local_vertex_indices[
                                    static_cast<std::size_t>(loop[index])],
                                local_vertex_indices[static_cast<std::size_t>(
                                    loop[(index + 1u) % loop.size()])],
                                gradient,
                                &result.statistics);
                        }
                    }
                    if (loop_count > 1)
                    {
                        ++result.statistics.multipleLoopCellCount;
                    }
                }
            }
        }
        result.statistics.uniqueAmbiguousFaceCount = ambiguous_faces.size();
        result.statistics.deciderTieCount = tied_faces.size();
        if (options.progress)
        {
            options.progress(1.0);
        }
        result.ok = result.statistics.unresolvedCellCount == 0u;
        if (!result.ok)
        {
            result.errorMessage = "Iso-surface extraction left unresolved cells";
        }
    }
    catch (const std::exception &exception)
    {
        result.errorMessage = exception.what();
    }
    return result;
}

} // namespace xjw::mesh
