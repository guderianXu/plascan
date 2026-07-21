#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/mesh/mesh_processing.h>

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

using PlaMesh = plapoint::PointCloud<float, plamatrix::Device::CPU>;

struct VertexCell
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const VertexCell &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VertexCellHash
{
    std::size_t operator()(const VertexCell &cell) const
    {
        std::size_t seed = std::hash<std::int64_t>{}(cell.x);
        seed ^= std::hash<std::int64_t>{}(cell.y) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= std::hash<std::int64_t>{}(cell.z) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

PlaMesh toPlaMesh(const TriMesh &mesh)
{
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> points(
        static_cast<plamatrix::Index>(mesh.vertices.size()), 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(
        static_cast<plamatrix::Index>(mesh.vertices.size()), 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(mesh.vertices.size()), 3);

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        const MeshVertex &vertex = mesh.vertices[i];
        points.setValue(row, 0, vertex.x);
        points.setValue(row, 1, vertex.y);
        points.setValue(row, 2, vertex.z);
        normals.setValue(row, 0, vertex.nx);
        normals.setValue(row, 1, vertex.ny);
        normals.setValue(row, 2, vertex.nz);
        colors.setValue(row, 0, vertex.r);
        colors.setValue(row, 1, vertex.g);
        colors.setValue(row, 2, vertex.b);
    }

    PlaMesh out(std::move(points));
    out.setNormals(std::move(normals));
    out.setColors(std::move(colors));

    plamatrix::DenseMatrix<int, plamatrix::Device::CPU> faces(
        static_cast<plamatrix::Index>(mesh.faces.size()), 3);
    for (std::size_t i = 0; i < mesh.faces.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        faces.setValue(row, 0, mesh.faces[i].v[0]);
        faces.setValue(row, 1, mesh.faces[i].v[1]);
        faces.setValue(row, 2, mesh.faces[i].v[2]);
    }
    out.setFaces(std::move(faces));
    return out;
}

void assignFromPlaMesh(const PlaMesh &source, TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    mesh->vertices.reserve(source.size());
    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        MeshVertex vertex;
        vertex.x = source.points().getValue(row, 0);
        vertex.y = source.points().getValue(row, 1);
        vertex.z = source.points().getValue(row, 2);
        if (source.hasNormals())
        {
            vertex.nx = source.normals()->getValue(row, 0);
            vertex.ny = source.normals()->getValue(row, 1);
            vertex.nz = source.normals()->getValue(row, 2);
        }
        else
        {
            vertex.nx = 0.0f;
            vertex.ny = 0.0f;
            vertex.nz = 1.0f;
        }
        if (source.hasColors())
        {
            vertex.r = source.colors()->getValue(row, 0);
            vertex.g = source.colors()->getValue(row, 1);
            vertex.b = source.colors()->getValue(row, 2);
        }
        mesh->vertices.push_back(vertex);
    }

    if (source.hasFaces())
    {
        mesh->faces.reserve(static_cast<std::size_t>(source.faces()->rows()));
        for (plamatrix::Index r = 0; r < source.faces()->rows(); ++r)
        {
            Triangle triangle;
            triangle.v[0] = source.faces()->getValue(r, 0);
            triangle.v[1] = source.faces()->getValue(r, 1);
            triangle.v[2] = source.faces()->getValue(r, 2);
            mesh->faces.push_back(triangle);
        }
    }
}

std::uint64_t edgeKey(int a, int b)
{
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

std::vector<std::vector<int>> findFaceComponents(const TriMesh &mesh)
{
    if (mesh.faces.empty())
    {
        return {};
    }

    std::unordered_map<std::uint64_t, std::vector<int>> edgeToFaces;
    edgeToFaces.reserve(mesh.faces.size() * 3);
    for (std::size_t faceIndex = 0; faceIndex < mesh.faces.size(); ++faceIndex)
    {
        const Triangle &face = mesh.faces[faceIndex];
        edgeToFaces[edgeKey(face.v[0], face.v[1])].push_back(static_cast<int>(faceIndex));
        edgeToFaces[edgeKey(face.v[1], face.v[2])].push_back(static_cast<int>(faceIndex));
        edgeToFaces[edgeKey(face.v[2], face.v[0])].push_back(static_cast<int>(faceIndex));
    }

    std::vector<std::vector<int>> adjacency(mesh.faces.size());
    for (const auto &entry : edgeToFaces)
    {
        const auto &edgeFaces = entry.second;
        for (std::size_t i = 0; i < edgeFaces.size(); ++i)
        {
            for (std::size_t j = i + 1; j < edgeFaces.size(); ++j)
            {
                adjacency[static_cast<std::size_t>(edgeFaces[i])].push_back(edgeFaces[j]);
                adjacency[static_cast<std::size_t>(edgeFaces[j])].push_back(edgeFaces[i]);
            }
        }
    }

    std::vector<std::uint8_t> visited(mesh.faces.size(), 0);
    std::vector<std::vector<int>> components;
    for (std::size_t start = 0; start < mesh.faces.size(); ++start)
    {
        if (visited[start])
        {
            continue;
        }

        std::vector<int> component;
        std::queue<int> queue;
        visited[start] = 1;
        queue.push(static_cast<int>(start));
        while (!queue.empty())
        {
            const int faceIndex = queue.front();
            queue.pop();
            component.push_back(faceIndex);
            for (int neighbor : adjacency[static_cast<std::size_t>(faceIndex)])
            {
                const auto neighborIndex = static_cast<std::size_t>(neighbor);
                if (!visited[neighborIndex])
                {
                    visited[neighborIndex] = 1;
                    queue.push(neighbor);
                }
            }
        }

        components.push_back(std::move(component));
    }

    return components;
}

void keepLargestConnectedComponent(TriMesh *mesh)
{
    if (!mesh || mesh->faces.empty())
    {
        return;
    }

    const std::vector<std::vector<int>> components = findFaceComponents(*mesh);
    const auto largest = std::max_element(
        components.cbegin(),
        components.cend(),
        [](const auto &left, const auto &right) { return left.size() < right.size(); });
    if (largest == components.cend())
    {
        return;
    }
    const std::vector<int> &largestComponent = *largest;

    if (largestComponent.empty())
    {
        return;
    }

    std::vector<int> oldToNew(mesh->vertices.size(), -1);
    std::vector<MeshVertex> compactVertices;
    std::vector<Triangle> compactFaces;
    compactFaces.reserve(largestComponent.size());
    for (int faceIndex : largestComponent)
    {
        Triangle remapped;
        const Triangle &source = mesh->faces[static_cast<std::size_t>(faceIndex)];
        for (int corner = 0; corner < 3; ++corner)
        {
            const int oldIndex = source.v[corner];
            if (oldIndex < 0 || static_cast<std::size_t>(oldIndex) >= mesh->vertices.size())
            {
                return;
            }
            if (oldToNew[static_cast<std::size_t>(oldIndex)] < 0)
            {
                oldToNew[static_cast<std::size_t>(oldIndex)] = static_cast<int>(compactVertices.size());
                compactVertices.push_back(mesh->vertices[static_cast<std::size_t>(oldIndex)]);
            }
            remapped.v[corner] = oldToNew[static_cast<std::size_t>(oldIndex)];
        }
        compactFaces.push_back(remapped);
    }

    mesh->vertices = std::move(compactVertices);
    mesh->faces = std::move(compactFaces);
}

} // namespace

void weldCoincidentVertices(TriMesh *mesh, float relativeTolerance)
{
    if (!mesh || mesh->vertices.empty() || relativeTolerance <= 0.0f)
    {
        return;
    }

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float min_z = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    float max_z = std::numeric_limits<float>::lowest();
    for (const MeshVertex &vertex : mesh->vertices)
    {
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) || !std::isfinite(vertex.z))
        {
            return;
        }
        min_x = std::min(min_x, vertex.x);
        min_y = std::min(min_y, vertex.y);
        min_z = std::min(min_z, vertex.z);
        max_x = std::max(max_x, vertex.x);
        max_y = std::max(max_y, vertex.y);
        max_z = std::max(max_z, vertex.z);
    }

    const float span = std::max({max_x - min_x, max_y - min_y, max_z - min_z, 1.0e-6f});
    const float tolerance = std::max(span * relativeTolerance, 1.0e-8f);
    const float tolerance_squared = tolerance * tolerance;

    std::unordered_map<VertexCell, std::vector<int>, VertexCellHash> cells;
    cells.reserve(mesh->vertices.size());
    std::vector<MeshVertex> welded_vertices;
    welded_vertices.reserve(mesh->vertices.size());
    std::vector<int> old_to_new(mesh->vertices.size(), -1);

    auto cell_for = [&](const MeshVertex &vertex) -> VertexCell
    {
        return VertexCell{
            static_cast<std::int64_t>(std::floor((vertex.x - min_x) / tolerance)),
            static_cast<std::int64_t>(std::floor((vertex.y - min_y) / tolerance)),
            static_cast<std::int64_t>(std::floor((vertex.z - min_z) / tolerance))};
    };

    for (std::size_t old_index = 0; old_index < mesh->vertices.size(); ++old_index)
    {
        const MeshVertex &vertex = mesh->vertices[old_index];
        const VertexCell cell = cell_for(vertex);
        int match = -1;
        for (int dz = -1; dz <= 1 && match < 0; ++dz)
        {
            for (int dy = -1; dy <= 1 && match < 0; ++dy)
            {
                for (int dx = -1; dx <= 1 && match < 0; ++dx)
                {
                    const VertexCell neighbor{cell.x + dx, cell.y + dy, cell.z + dz};
                    const auto it = cells.find(neighbor);
                    if (it == cells.end())
                    {
                        continue;
                    }
                    for (int candidate : it->second)
                    {
                        const MeshVertex &existing = welded_vertices[static_cast<std::size_t>(candidate)];
                        const float delta_x = vertex.x - existing.x;
                        const float delta_y = vertex.y - existing.y;
                        const float delta_z = vertex.z - existing.z;
                        if (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z <=
                            tolerance_squared)
                        {
                            match = candidate;
                            break;
                        }
                    }
                }
            }
        }

        if (match < 0)
        {
            match = static_cast<int>(welded_vertices.size());
            welded_vertices.push_back(vertex);
            cells[cell].push_back(match);
        }
        old_to_new[old_index] = match;
    }

    for (Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const int old_index = face.v[corner];
            if (old_index < 0 || static_cast<std::size_t>(old_index) >= old_to_new.size())
            {
                return;
            }
            face.v[corner] = old_to_new[static_cast<std::size_t>(old_index)];
        }
    }
    mesh->vertices = std::move(welded_vertices);
}

void removeDegenerateFaces(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }
    assignFromPlaMesh(plapoint::mesh::removeDegenerateFaces(toPlaMesh(*mesh), 5.0e-9f), mesh);
}

int compactReferencedVertices(TriMesh *mesh)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty())
    {
        return 0;
    }

    std::vector<std::uint8_t> referenced(mesh->vertices.size(), 0);
    for (const Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const int vertex_index = face.v[corner];
            if (vertex_index < 0 || static_cast<std::size_t>(vertex_index) >= mesh->vertices.size())
            {
                return 0;
            }
            referenced[static_cast<std::size_t>(vertex_index)] = 1;
        }
    }

    const int removed_count = static_cast<int>(std::count(
        referenced.cbegin(), referenced.cend(), std::uint8_t{0}));
    if (removed_count <= 0)
    {
        return 0;
    }

    std::vector<int> old_to_new(mesh->vertices.size(), -1);
    std::vector<MeshVertex> compact_vertices;
    compact_vertices.reserve(mesh->vertices.size() - static_cast<std::size_t>(removed_count));
    for (std::size_t old_index = 0; old_index < mesh->vertices.size(); ++old_index)
    {
        if (!referenced[old_index])
        {
            continue;
        }
        old_to_new[old_index] = static_cast<int>(compact_vertices.size());
        compact_vertices.push_back(mesh->vertices[old_index]);
    }

    for (Triangle &face : mesh->faces)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            face.v[corner] = old_to_new[static_cast<std::size_t>(face.v[corner])];
        }
    }
    mesh->vertices = std::move(compact_vertices);
    return removed_count;
}

void removeSmallConnectedComponents(TriMesh *mesh,
                                    int minFaces,
                                    float minimumLargestComponentRatio)
{
    if (!mesh || mesh->faces.empty())
    {
        return;
    }
    if (minFaces <= 1 && minimumLargestComponentRatio <= 0.0f)
    {
        return;
    }

    int effective_minimum_faces = std::max(2, minFaces);
    if (minimumLargestComponentRatio > 0.0f)
    {
        const std::vector<std::vector<int>> components = findFaceComponents(*mesh);
        const auto largest = std::max_element(
            components.cbegin(),
            components.cend(),
            [](const auto &left, const auto &right) { return left.size() < right.size(); });
        if (largest != components.cend())
        {
            const int relative_minimum = static_cast<int>(std::ceil(
                static_cast<double>(largest->size()) *
                std::clamp(minimumLargestComponentRatio, 0.0f, 1.0f)));
            effective_minimum_faces = std::max(effective_minimum_faces, relative_minimum);
        }
    }
    PlaMesh filtered = plapoint::mesh::removeSmallConnectedComponents(
        toPlaMesh(*mesh),
        static_cast<std::size_t>(effective_minimum_faces));
    if (filtered.hasFaces() && filtered.faces()->rows() > 0)
    {
        assignFromPlaMesh(filtered, mesh);
        return;
    }

    keepLargestConnectedComponent(mesh);
}

int fillSmallBoundaryHoles(TriMesh *mesh,
                           int maxBoundaryEdges,
                           float maxBoundaryDiameter)
{
    if (!mesh || mesh->faces.empty() || mesh->vertices.empty() || maxBoundaryEdges < 3)
    {
        return 0;
    }

    const auto edge_key = [](int first, int second) {
        const auto low = static_cast<std::uint32_t>(std::min(first, second));
        const auto high = static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
    };

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh->faces.size() * 3);
    for (const Triangle &face : mesh->faces)
    {
        ++edge_counts[edge_key(face.v[0], face.v[1])];
        ++edge_counts[edge_key(face.v[1], face.v[2])];
        ++edge_counts[edge_key(face.v[2], face.v[0])];
    }

    std::vector<std::vector<int>> boundary_neighbors(mesh->vertices.size());
    for (const Triangle &face : mesh->faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            if (edge_counts[edge_key(edge[0], edge[1])] == 1)
            {
                boundary_neighbors[static_cast<std::size_t>(edge[0])].push_back(edge[1]);
                boundary_neighbors[static_cast<std::size_t>(edge[1])].push_back(edge[0]);
            }
        }
    }

    std::unordered_map<std::uint64_t, bool> visited_edges;
    visited_edges.reserve(edge_counts.size());
    std::vector<Triangle> added_faces;
    int filled_holes = 0;
    for (int start_vertex = 0; start_vertex < static_cast<int>(boundary_neighbors.size()); ++start_vertex)
    {
        const auto &start_neighbors = boundary_neighbors[static_cast<std::size_t>(start_vertex)];
        if (start_neighbors.size() != 2)
        {
            continue;
        }

        for (const int first_neighbor : start_neighbors)
        {
            if (visited_edges[edge_key(start_vertex, first_neighbor)])
            {
                continue;
            }

            std::vector<int> loop{start_vertex};
            int previous = start_vertex;
            int current = first_neighbor;
            bool closed = false;
            visited_edges[edge_key(previous, current)] = true;
            while (static_cast<int>(loop.size()) <= maxBoundaryEdges)
            {
                if (current == start_vertex)
                {
                    closed = true;
                    break;
                }
                loop.push_back(current);

                const auto &current_neighbors = boundary_neighbors[static_cast<std::size_t>(current)];
                if (current_neighbors.size() != 2)
                {
                    break;
                }
                const int next = current_neighbors[0] == previous
                                     ? current_neighbors[1]
                                     : current_neighbors[0];
                const std::uint64_t next_key = edge_key(current, next);
                if (next != start_vertex && visited_edges[next_key])
                {
                    break;
                }
                previous = current;
                current = next;
                visited_edges[next_key] = true;
            }

            if (!closed || loop.size() < 3 || static_cast<int>(loop.size()) > maxBoundaryEdges)
            {
                continue;
            }
            if (maxBoundaryDiameter > 0.0f)
            {
                const float maximum_squared = maxBoundaryDiameter * maxBoundaryDiameter;
                bool diameter_exceeded = false;
                for (std::size_t first = 0; first < loop.size() && !diameter_exceeded; ++first)
                {
                    const MeshVertex &a = mesh->vertices[static_cast<std::size_t>(loop[first])];
                    for (std::size_t second = first + 1; second < loop.size(); ++second)
                    {
                        const MeshVertex &b = mesh->vertices[static_cast<std::size_t>(loop[second])];
                        const float dx = a.x - b.x;
                        const float dy = a.y - b.y;
                        const float dz = a.z - b.z;
                        if (dx * dx + dy * dy + dz * dz > maximum_squared)
                        {
                            diameter_exceeded = true;
                            break;
                        }
                    }
                }
                if (diameter_exceeded)
                {
                    continue;
                }
            }

            MeshVertex center;
            int red_sum = 0;
            int green_sum = 0;
            int blue_sum = 0;
            for (const int vertex_index : loop)
            {
                const MeshVertex &vertex = mesh->vertices[static_cast<std::size_t>(vertex_index)];
                center.x += vertex.x;
                center.y += vertex.y;
                center.z += vertex.z;
                red_sum += vertex.r;
                green_sum += vertex.g;
                blue_sum += vertex.b;
            }
            const float inverse_loop_size = 1.0f / static_cast<float>(loop.size());
            center.x *= inverse_loop_size;
            center.y *= inverse_loop_size;
            center.z *= inverse_loop_size;
            center.r = static_cast<std::uint8_t>(red_sum / static_cast<int>(loop.size()));
            center.g = static_cast<std::uint8_t>(green_sum / static_cast<int>(loop.size()));
            center.b = static_cast<std::uint8_t>(blue_sum / static_cast<int>(loop.size()));
            const int center_index = static_cast<int>(mesh->vertices.size());
            mesh->vertices.push_back(center);

            for (std::size_t edge_index = 0; edge_index < loop.size(); ++edge_index)
            {
                Triangle face;
                face.v[0] = loop[edge_index];
                face.v[1] = loop[(edge_index + 1) % loop.size()];
                face.v[2] = center_index;
                added_faces.push_back(face);
            }
            ++filled_holes;
        }
    }

    mesh->faces.insert(mesh->faces.end(), added_faces.begin(), added_faces.end());
    return filled_holes;
}

int smoothOpenBoundaryVertices(TriMesh *mesh,
                               int iterations,
                               float lambda,
                               float maximumDisplacement)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() ||
        iterations <= 0 || lambda <= 0.0f || maximumDisplacement <= 0.0f)
    {
        return 0;
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh->faces.size() * 3);
    for (const Triangle &face : mesh->faces)
    {
        ++edge_counts[edgeKey(face.v[0], face.v[1])];
        ++edge_counts[edgeKey(face.v[1], face.v[2])];
        ++edge_counts[edgeKey(face.v[2], face.v[0])];
    }

    std::vector<std::vector<int>> boundary_neighbors(mesh->vertices.size());
    for (const Triangle &face : mesh->faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            if (edge_counts[edgeKey(edge[0], edge[1])] == 1)
            {
                boundary_neighbors[static_cast<std::size_t>(edge[0])].push_back(edge[1]);
                boundary_neighbors[static_cast<std::size_t>(edge[1])].push_back(edge[0]);
            }
        }
    }

    lambda = std::clamp(lambda, 0.0f, 1.0f);
    std::vector<std::uint8_t> moved(mesh->vertices.size(), 0);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const std::vector<MeshVertex> source = mesh->vertices;
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            const auto &neighbors = boundary_neighbors[index];
            if (neighbors.size() != 2)
            {
                continue;
            }

            const MeshVertex &first = source[static_cast<std::size_t>(neighbors[0])];
            const MeshVertex &second = source[static_cast<std::size_t>(neighbors[1])];
            const MeshVertex &current = source[index];
            float dx = ((first.x + second.x) * 0.5f - current.x) * lambda;
            float dy = ((first.y + second.y) * 0.5f - current.y) * lambda;
            float dz = ((first.z + second.z) * 0.5f - current.z) * lambda;
            const float displacement = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (displacement > maximumDisplacement)
            {
                const float scale = maximumDisplacement / displacement;
                dx *= scale;
                dy *= scale;
                dz *= scale;
            }
            mesh->vertices[index].x = current.x + dx;
            mesh->vertices[index].y = current.y + dy;
            mesh->vertices[index].z = current.z + dz;
            moved[index] = 1;
        }
    }

    return static_cast<int>(std::count(moved.cbegin(), moved.cend(), std::uint8_t{1}));
}

void simplifyVoxelMeshAdaptive(TriMesh *mesh,
                               const ReconstructionConfig &config,
                               float voxelStep)
{
    if (!mesh || mesh->empty() || voxelStep <= 0.0f)
    {
        return;
    }

    const int fallbackTarget = std::clamp((config.resolution * config.resolution) / 2, 3000, 30000);
    const int targetFaceCount = std::clamp(config.simplifyTargetFaces > 0 ? config.simplifyTargetFaces : fallbackTarget,
                                           2000,
                                           120000);
    if (mesh->faceCount() <= targetFaceCount)
    {
        return;
    }

    PlaMesh current = toPlaMesh(*mesh);
    float clusterSize = voxelStep * std::clamp(config.voxelSimplifyFactor, 1.0f, 4.0f);
    const int minFacesForComponent = std::max(24, config.minComponentFaces / 3);

    for (int iteration = 0; iteration < 8; ++iteration)
    {
        const int beforeFaces = current.hasFaces() ? static_cast<int>(current.faces()->rows()) : 0;
        current = plapoint::mesh::voxelClusterSimplify(current, std::max(clusterSize, voxelStep));
        current = plapoint::mesh::removeDegenerateFaces(current, 5.0e-9f);
        current = plapoint::mesh::removeSmallConnectedComponents(
            current,
            static_cast<std::size_t>(minFacesForComponent));

        const int afterFaces = current.hasFaces() ? static_cast<int>(current.faces()->rows()) : 0;
        if (afterFaces <= 0 || afterFaces <= targetFaceCount)
        {
            break;
        }
        if (afterFaces >= static_cast<int>(beforeFaces * 0.97f))
        {
            clusterSize *= 1.45f;
        }
        else
        {
            clusterSize *= 1.25f;
        }
    }

    assignFromPlaMesh(current, mesh);
}

void taubinSmooth(TriMesh *mesh, int iterations, float lambda)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() || iterations <= 0)
    {
        return;
    }

    lambda = std::clamp(lambda, 0.0f, 1.0f);
    const float mu = -std::clamp(lambda * 0.53f, 0.01f, 0.53f);
    assignFromPlaMesh(plapoint::mesh::taubinSmooth(toPlaMesh(*mesh), iterations, lambda, mu), mesh);
}

void recomputeNormals(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }
    auto withNormals = plapoint::mesh::recomputeVertexNormals(toPlaMesh(*mesh));
    withNormals = plapoint::mesh::orientNormalsOutwardFromCentroid(withNormals);
    assignFromPlaMesh(withNormals, mesh);
}

} // namespace detail
} // namespace mesh
} // namespace xjw
