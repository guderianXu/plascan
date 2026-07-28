#include "ConsistentIsoSurfaceInternal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#include <plapoint/mesh/marching_cubes.h>

namespace xjw::mesh::detail
{

std::size_t checkedIsoSurfaceSampleCount(const std::array<int, 3> &cells)
{
    std::size_t count = 1;
    for (int cell_count : cells)
    {
        if (cell_count <= 0)
        {
            throw std::invalid_argument("Iso-surface grid resolution must be positive");
        }
        const std::size_t samples = static_cast<std::size_t>(cell_count) + 1u;
        if (count > std::numeric_limits<std::size_t>::max() / samples)
        {
            throw std::invalid_argument("Iso-surface grid sample count overflows");
        }
        count *= samples;
    }
    return count;
}

std::size_t isoSurfaceSampleIndex(const std::array<int, 3> &cells,
                                  int x,
                                  int y,
                                  int z)
{
    const std::size_t sx = static_cast<std::size_t>(cells[0]) + 1u;
    const std::size_t sy = static_cast<std::size_t>(cells[1]) + 1u;
    return (static_cast<std::size_t>(z) * sy + static_cast<std::size_t>(y)) * sx +
           static_cast<std::size_t>(x);
}

std::uint64_t isoSurfaceEdgeKey(const std::array<int, 3> &cells,
                                int cellX,
                                int cellY,
                                int cellZ,
                                int edge)
{
    static constexpr int direction[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};
    static constexpr int offset[12][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {0, 0, 1},
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    const std::uint64_t sx = static_cast<std::uint64_t>(cells[0]) + 1u;
    const std::uint64_t sy = static_cast<std::uint64_t>(cells[1]) + 1u;
    const std::uint64_t x = static_cast<std::uint64_t>(cellX + offset[edge][0]);
    const std::uint64_t y = static_cast<std::uint64_t>(cellY + offset[edge][1]);
    const std::uint64_t z = static_cast<std::uint64_t>(cellZ + offset[edge][2]);
    return ((z * sy + y) * sx + x) * 3u +
           static_cast<std::uint64_t>(direction[edge]);
}

GridFaceKey isoSurfaceFaceKey(int cellX, int cellY, int cellZ, int face)
{
    switch (face)
    {
    case 0:
        return {GridAxis::Z, cellX, cellY, cellZ};
    case 1:
        return {GridAxis::Z, cellX, cellY, cellZ + 1};
    case 2:
        return {GridAxis::Y, cellX, cellY, cellZ};
    case 3:
        return {GridAxis::Y, cellX, cellY + 1, cellZ};
    case 4:
        return {GridAxis::X, cellX, cellY, cellZ};
    default:
        return {GridAxis::X, cellX + 1, cellY, cellZ};
    }
}

void addIsoSurfaceConnection(std::array<std::vector<int>, 12> *adjacency,
                             int first,
                             int second)
{
    auto &first_neighbors = (*adjacency)[static_cast<std::size_t>(first)];
    auto &second_neighbors = (*adjacency)[static_cast<std::size_t>(second)];
    if (std::find(first_neighbors.begin(), first_neighbors.end(), second) ==
        first_neighbors.end())
    {
        first_neighbors.push_back(second);
    }
    if (std::find(second_neighbors.begin(), second_neighbors.end(), first) ==
        second_neighbors.end())
    {
        second_neighbors.push_back(first);
    }
}

int countIsoSurfaceLoops(
    const std::array<bool, 12> &activeEdges,
    const std::array<std::vector<int>, 12> &adjacency)
{
    std::array<bool, 12> visited{};
    int loop_count = 0;
    for (int start = 0; start < 12; ++start)
    {
        if (!activeEdges[static_cast<std::size_t>(start)] ||
            visited[static_cast<std::size_t>(start)])
        {
            continue;
        }
        ++loop_count;
        int previous = -1;
        int current = start;
        int step_count = 0;
        do
        {
            const auto &neighbors =
                adjacency[static_cast<std::size_t>(current)];
            if (neighbors.size() != 2u ||
                visited[static_cast<std::size_t>(current)] ||
                ++step_count > 12)
            {
                return -1;
            }
            visited[static_cast<std::size_t>(current)] = true;
            const int next =
                neighbors[0] == previous ? neighbors[1] : neighbors[0];
            previous = current;
            current = next;
        }
        while (current != start);
    }
    return loop_count;
}

bool classicIsoSurfaceTriangulationMatchesFaceGraph(
    int cubeIndex,
    const std::array<bool, 12> &activeEdges,
    const std::array<std::vector<int>, 12> &adjacency)
{
    using Edge = std::pair<int, int>;
    const auto make_edge = [](int first, int second) -> Edge
    {
        return {std::min(first, second), std::max(first, second)};
    };
    std::set<Edge> expected_boundary;
    for (int edge = 0; edge < 12; ++edge)
    {
        if (!activeEdges[static_cast<std::size_t>(edge)])
        {
            continue;
        }
        for (int neighbor : adjacency[static_cast<std::size_t>(edge)])
        {
            expected_boundary.insert(make_edge(edge, neighbor));
        }
    }

    const auto &triangles = plapoint::mesh::detail::triTable(cubeIndex);
    std::map<Edge, int> edge_incidence;
    std::set<int> used_vertices;
    for (std::size_t index = 0; index + 2u < triangles.size(); index += 3u)
    {
        const int first = triangles[index];
        const int second = triangles[index + 1u];
        const int third = triangles[index + 2u];
        if (first == second || second == third || third == first ||
            !activeEdges[static_cast<std::size_t>(first)] ||
            !activeEdges[static_cast<std::size_t>(second)] ||
            !activeEdges[static_cast<std::size_t>(third)])
        {
            return false;
        }
        used_vertices.insert(first);
        used_vertices.insert(second);
        used_vertices.insert(third);
        ++edge_incidence[make_edge(first, second)];
        ++edge_incidence[make_edge(second, third)];
        ++edge_incidence[make_edge(third, first)];
    }

    std::set<Edge> actual_boundary;
    for (const auto &[edge, incidence] : edge_incidence)
    {
        if (incidence == 1)
        {
            actual_boundary.insert(edge);
        }
        else if (incidence != 2)
        {
            return false;
        }
    }
    if (actual_boundary != expected_boundary)
    {
        return false;
    }

    const int vertex_count = static_cast<int>(used_vertices.size());
    const int edge_count = static_cast<int>(edge_incidence.size());
    const int face_count = static_cast<int>(triangles.size() / 3u);
    return vertex_count - edge_count + face_count == 1;
}

MeshVertex interpolateIsoSurfaceVertex(
    const std::array<MeshVertex, 8> &points,
    const std::array<float, 8> &values,
    int edge,
    float isoLevel)
{
    const int first = kIsoSurfaceEdgeCorners[edge][0];
    const int second = kIsoSurfaceEdgeCorners[edge][1];
    const float denominator = values[second] - values[first];
    float t = 0.5f;
    if (std::abs(denominator) > 1.0e-12f)
    {
        t = (isoLevel - values[first]) / denominator;
    }
    t = std::clamp(t, 0.0f, 1.0f);
    MeshVertex vertex;
    vertex.x = points[first].x + t * (points[second].x - points[first].x);
    vertex.y = points[first].y + t * (points[second].y - points[first].y);
    vertex.z = points[first].z + t * (points[second].z - points[first].z);
    return vertex;
}

void appendOrientedIsoSurfaceFace(
    TriMesh *mesh,
    int first,
    int second,
    int third,
    const std::array<double, 3> &gradient,
    ConsistentIsoSurfaceStatistics *statistics)
{
    const MeshVertex &p0 = mesh->vertices[static_cast<std::size_t>(first)];
    const MeshVertex &p1 = mesh->vertices[static_cast<std::size_t>(second)];
    const MeshVertex &p2 = mesh->vertices[static_cast<std::size_t>(third)];
    const double ux = static_cast<double>(p1.x) - p0.x;
    const double uy = static_cast<double>(p1.y) - p0.y;
    const double uz = static_cast<double>(p1.z) - p0.z;
    const double vx = static_cast<double>(p2.x) - p0.x;
    const double vy = static_cast<double>(p2.y) - p0.y;
    const double vz = static_cast<double>(p2.z) - p0.z;
    const double nx = uy * vz - uz * vy;
    const double ny = uz * vx - ux * vz;
    const double nz = ux * vy - uy * vx;
    const double squared_area = 0.25 * (nx * nx + ny * ny + nz * nz);
    if (first == second || second == third || third == first ||
        squared_area <= 1.0e-24)
    {
        ++statistics->rejectedDegenerateFaceCount;
        return;
    }
    const double dot =
        nx * gradient[0] + ny * gradient[1] + nz * gradient[2];
    Triangle face;
    face.v[0] = first;
    face.v[1] = dot < 0.0 ? third : second;
    face.v[2] = dot < 0.0 ? second : third;
    mesh->faces.push_back(face);
}

} // namespace xjw::mesh::detail
