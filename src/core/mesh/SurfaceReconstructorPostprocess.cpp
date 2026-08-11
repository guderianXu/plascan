#include "SurfaceReconstructorPostprocess.h"

#include "BoundaryAwareVoxelSimplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
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

struct TriangleKey
{
    std::array<int, 3> vertices{};

    bool operator==(const TriangleKey &other) const
    {
        return vertices == other.vertices;
    }
};

struct TriangleKeyHash
{
    std::size_t operator()(const TriangleKey &key) const
    {
        std::size_t seed = std::hash<int>{}(key.vertices[0]);
        seed ^= std::hash<int>{}(key.vertices[1]) + 0x9e3779b9u +
            (seed << 6U) + (seed >> 2U);
        seed ^= std::hash<int>{}(key.vertices[2]) + 0x9e3779b9u +
            (seed << 6U) + (seed >> 2U);
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

void weldCoincidentVertices(TriMesh *mesh,
                            float relativeTolerance,
                            float absoluteTolerance)
{
    if (!mesh || mesh->vertices.empty() ||
        (relativeTolerance <= 0.0f && absoluteTolerance <= 0.0f))
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
    const float tolerance = std::max(
        {span * std::max(0.0f, relativeTolerance),
         std::max(0.0f, absoluteTolerance),
         1.0e-8f});
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

void removeDegenerateFaces(TriMesh *mesh, float minimumArea)
{
    if (!mesh)
    {
        return;
    }
    assignFromPlaMesh(
        plapoint::mesh::removeDegenerateFaces(
            toPlaMesh(*mesh),
            std::max(0.0f, minimumArea)),
        mesh);
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

int removeDuplicateFaces(TriMesh *mesh)
{
    if (!mesh || mesh->faces.empty())
    {
        return 0;
    }
    std::unordered_set<TriangleKey, TriangleKeyHash> unique_faces;
    unique_faces.reserve(mesh->faces.size());
    std::vector<Triangle> kept_faces;
    kept_faces.reserve(mesh->faces.size());
    for (const Triangle &face : mesh->faces)
    {
        TriangleKey key{{face.v[0], face.v[1], face.v[2]}};
        std::sort(key.vertices.begin(), key.vertices.end());
        if (unique_faces.insert(key).second)
        {
            kept_faces.push_back(face);
        }
    }
    const int removed_count = mesh->faceCount() - static_cast<int>(kept_faces.size());
    mesh->faces = std::move(kept_faces);
    return removed_count;
}

int removeNonManifoldFaces(TriMesh *mesh)
{
    if (!mesh || mesh->faces.empty())
    {
        return 0;
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh->faces.size() * 2);
    for (const Triangle &face : mesh->faces)
    {
        ++edge_counts[edgeKey(face.v[0], face.v[1])];
        ++edge_counts[edgeKey(face.v[1], face.v[2])];
        ++edge_counts[edgeKey(face.v[2], face.v[0])];
    }

    std::unordered_map<std::uint64_t, std::vector<int>> non_manifold_edge_faces;
    for (const auto &entry : edge_counts)
    {
        if (entry.second > 2)
        {
            non_manifold_edge_faces.emplace(entry.first, std::vector<int>{});
        }
    }
    if (non_manifold_edge_faces.empty())
    {
        return 0;
    }
    for (std::size_t face_index = 0; face_index < mesh->faces.size(); ++face_index)
    {
        const Triangle &face = mesh->faces[face_index];
        for (const std::uint64_t key : {edgeKey(face.v[0], face.v[1]),
                                        edgeKey(face.v[1], face.v[2]),
                                        edgeKey(face.v[2], face.v[0])})
        {
            const auto found = non_manifold_edge_faces.find(key);
            if (found != non_manifold_edge_faces.end())
            {
                found->second.push_back(static_cast<int>(face_index));
            }
        }
    }

    const auto face_normal = [mesh](int face_index)
    {
        const Triangle &face = mesh->faces[static_cast<std::size_t>(face_index)];
        const MeshVertex &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];
        const double ab_x = b.x - a.x;
        const double ab_y = b.y - a.y;
        const double ab_z = b.z - a.z;
        const double ac_x = c.x - a.x;
        const double ac_y = c.y - a.y;
        const double ac_z = c.z - a.z;
        std::array<double, 3> normal{{ab_y * ac_z - ab_z * ac_y,
                                      ab_z * ac_x - ab_x * ac_z,
                                      ab_x * ac_y - ab_y * ac_x}};
        const double magnitude = std::sqrt(normal[0] * normal[0] +
                                           normal[1] * normal[1] +
                                           normal[2] * normal[2]);
        if (magnitude > 1.0e-18)
        {
            normal[0] /= magnitude;
            normal[1] /= magnitude;
            normal[2] /= magnitude;
        }
        return normal;
    };

    std::vector<std::uint8_t> remove_face(mesh->faces.size(), 0);
    for (const auto &entry : non_manifold_edge_faces)
    {
        const std::vector<int> &faces = entry.second;
        if (faces.size() <= 2)
        {
            continue;
        }
        std::size_t best_first = 0;
        std::size_t best_second = 1;
        double best_alignment = -std::numeric_limits<double>::infinity();
        for (std::size_t first = 0; first < faces.size(); ++first)
        {
            const auto first_normal = face_normal(faces[first]);
            for (std::size_t second = first + 1; second < faces.size(); ++second)
            {
                const auto second_normal = face_normal(faces[second]);
                const double alignment = first_normal[0] * second_normal[0] +
                    first_normal[1] * second_normal[1] +
                    first_normal[2] * second_normal[2];
                if (alignment > best_alignment)
                {
                    best_alignment = alignment;
                    best_first = first;
                    best_second = second;
                }
            }
        }
        for (std::size_t index = 0; index < faces.size(); ++index)
        {
            if (index != best_first && index != best_second)
            {
                remove_face[static_cast<std::size_t>(faces[index])] = 1;
            }
        }
    }

    std::vector<Triangle> kept_faces;
    kept_faces.reserve(mesh->faces.size());
    for (std::size_t face_index = 0; face_index < mesh->faces.size(); ++face_index)
    {
        if (!remove_face[face_index])
        {
            kept_faces.push_back(mesh->faces[face_index]);
        }
    }
    const int removed_count = mesh->faceCount() - static_cast<int>(kept_faces.size());
    mesh->faces = std::move(kept_faces);
    return removed_count;
}

int splitPinchedBoundaryVertices(TriMesh *mesh)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty())
    {
        return 0;
    }
    std::vector<std::vector<int>> incident_faces(mesh->vertices.size());
    for (std::size_t face_index = 0; face_index < mesh->faces.size(); ++face_index)
    {
        const Triangle &face = mesh->faces[face_index];
        for (int corner = 0; corner < 3; ++corner)
        {
            incident_faces[static_cast<std::size_t>(face.v[corner])].push_back(
                static_cast<int>(face_index));
        }
    }

    const std::size_t original_vertex_count = mesh->vertices.size();
    const std::vector<Triangle> original_faces = mesh->faces;
    int split_vertex_count = 0;
    for (std::size_t vertex_index = 0; vertex_index < original_vertex_count; ++vertex_index)
    {
        const std::vector<int> &faces = incident_faces[vertex_index];
        if (faces.size() < 2)
        {
            continue;
        }

        std::vector<std::pair<int, std::size_t>> spokes;
        spokes.reserve(faces.size() * 2);
        for (std::size_t local_face_index = 0; local_face_index < faces.size();
             ++local_face_index)
        {
            const Triangle &face = original_faces[static_cast<std::size_t>(
                faces[local_face_index])];
            for (const int face_vertex : face.v)
            {
                if (face_vertex != static_cast<int>(vertex_index))
                {
                    spokes.emplace_back(face_vertex, local_face_index);
                }
            }
        }
        std::sort(spokes.begin(), spokes.end(), [](const auto &left, const auto &right)
        {
            return left.first < right.first;
        });
        std::vector<std::vector<std::size_t>> adjacent_faces(faces.size());
        for (std::size_t start = 0; start < spokes.size();)
        {
            std::size_t end = start + 1;
            while (end < spokes.size() && spokes[end].first == spokes[start].first)
            {
                ++end;
            }
            if (end - start == 2)
            {
                const std::size_t first_face = spokes[start].second;
                const std::size_t second_face = spokes[start + 1].second;
                adjacent_faces[first_face].push_back(second_face);
                adjacent_faces[second_face].push_back(first_face);
            }
            start = end;
        }

        std::vector<int> component(faces.size(), -1);
        int component_count = 0;
        for (std::size_t seed = 0; seed < faces.size(); ++seed)
        {
            if (component[seed] >= 0)
            {
                continue;
            }
            std::queue<std::size_t> pending;
            pending.push(seed);
            component[seed] = component_count;
            while (!pending.empty())
            {
                const std::size_t local_face_index = pending.front();
                pending.pop();
                for (const std::size_t adjacent_local : adjacent_faces[local_face_index])
                {
                    if (component[adjacent_local] < 0)
                    {
                        component[adjacent_local] = component_count;
                        pending.push(adjacent_local);
                    }
                }
            }
            ++component_count;
        }

        for (int component_index = 1; component_index < component_count; ++component_index)
        {
            const int duplicate_index = static_cast<int>(mesh->vertices.size());
            mesh->vertices.push_back(mesh->vertices[vertex_index]);
            for (std::size_t local_face_index = 0; local_face_index < faces.size(); ++local_face_index)
            {
                if (component[local_face_index] != component_index)
                {
                    continue;
                }
                Triangle &face = mesh->faces[static_cast<std::size_t>(faces[local_face_index])];
                for (int &face_vertex : face.v)
                {
                    if (face_vertex == static_cast<int>(vertex_index))
                    {
                        face_vertex = duplicate_index;
                    }
                }
            }
            ++split_vertex_count;
        }
    }
    return split_vertex_count;
}

namespace
{

struct ProjectedBoundaryPoint
{
    double x = 0.0;
    double y = 0.0;
};

double boundaryCross(const ProjectedBoundaryPoint &first,
                     const ProjectedBoundaryPoint &second,
                     const ProjectedBoundaryPoint &third)
{
    return (second.x - first.x) * (third.y - first.y) -
        (second.y - first.y) * (third.x - first.x);
}

bool pointInsideBoundaryTriangle(const ProjectedBoundaryPoint &point,
                                 const ProjectedBoundaryPoint &first,
                                 const ProjectedBoundaryPoint &second,
                                 const ProjectedBoundaryPoint &third,
                                 double orientation)
{
    constexpr double epsilon = 1.0e-12;
    return boundaryCross(first, second, point) * orientation >= -epsilon &&
        boundaryCross(second, third, point) * orientation >= -epsilon &&
        boundaryCross(third, first, point) * orientation >= -epsilon;
}

double boundaryTriangleAspect(const TriMesh &mesh,
                              int first,
                              int second,
                              int third)
{
    const MeshVertex &a = mesh.vertices[static_cast<std::size_t>(first)];
    const MeshVertex &b = mesh.vertices[static_cast<std::size_t>(second)];
    const MeshVertex &c = mesh.vertices[static_cast<std::size_t>(third)];
    const double ab_x = b.x - a.x;
    const double ab_y = b.y - a.y;
    const double ab_z = b.z - a.z;
    const double ac_x = c.x - a.x;
    const double ac_y = c.y - a.y;
    const double ac_z = c.z - a.z;
    const double bc_x = c.x - b.x;
    const double bc_y = c.y - b.y;
    const double bc_z = c.z - b.z;
    const double cross_x = ab_y * ac_z - ab_z * ac_y;
    const double cross_y = ab_z * ac_x - ab_x * ac_z;
    const double cross_z = ab_x * ac_y - ab_y * ac_x;
    const double doubled_area = std::sqrt(
        cross_x * cross_x + cross_y * cross_y + cross_z * cross_z);
    const double longest_squared = std::max({
        ab_x * ab_x + ab_y * ab_y + ab_z * ab_z,
        ac_x * ac_x + ac_y * ac_y + ac_z * ac_z,
        bc_x * bc_x + bc_y * bc_y + bc_z * bc_z});
    return doubled_area > 1.0e-20
        ? longest_squared / doubled_area
        : std::numeric_limits<double>::infinity();
}

using PatchVector = std::array<double, 3>;

PatchVector subtractPatchVector(const PatchVector &left,
                                const PatchVector &right)
{
    return {{left[0] - right[0],
             left[1] - right[1],
             left[2] - right[2]}};
}

PatchVector addPatchVector(const PatchVector &left,
                           const PatchVector &right)
{
    return {{left[0] + right[0],
             left[1] + right[1],
             left[2] + right[2]}};
}

PatchVector scalePatchVector(const PatchVector &value, double scale)
{
    return {{value[0] * scale, value[1] * scale, value[2] * scale}};
}

double dotPatchVector(const PatchVector &left, const PatchVector &right)
{
    return left[0] * right[0] + left[1] * right[1] +
        left[2] * right[2];
}

PatchVector crossPatchVector(const PatchVector &left,
                             const PatchVector &right)
{
    return {{left[1] * right[2] - left[2] * right[1],
             left[2] * right[0] - left[0] * right[2],
             left[0] * right[1] - left[1] * right[0]}};
}

bool normalizePatchVector(PatchVector *value)
{
    if (!value)
    {
        return false;
    }
    const double magnitude = std::sqrt(dotPatchVector(*value, *value));
    if (!std::isfinite(magnitude) || magnitude <= 1.0e-12)
    {
        return false;
    }
    *value = scalePatchVector(*value, 1.0 / magnitude);
    return true;
}

PatchVector patchPosition(const MeshVertex &vertex)
{
    return {{vertex.x, vertex.y, vertex.z}};
}

struct BoundaryPatchFrame
{
    PatchVector origin{};
    PatchVector tangent{};
    PatchVector bitangent{};
    PatchVector normal{};
    double scale = 0.0;
};

bool buildBoundaryPatchFrame(const TriMesh &mesh,
                             const std::vector<int> &loop,
                             BoundaryPatchFrame *frame)
{
    if (!frame || loop.size() < 3)
    {
        return false;
    }

    PatchVector origin{};
    PatchVector newell_normal{};
    PatchVector average_normal{};
    for (std::size_t index = 0; index < loop.size(); ++index)
    {
        const MeshVertex &current =
            mesh.vertices[static_cast<std::size_t>(loop[index])];
        const MeshVertex &next = mesh.vertices[
            static_cast<std::size_t>(loop[(index + 1) % loop.size()])];
        origin = addPatchVector(origin, patchPosition(current));
        newell_normal[0] += (current.y - next.y) * (current.z + next.z);
        newell_normal[1] += (current.z - next.z) * (current.x + next.x);
        newell_normal[2] += (current.x - next.x) * (current.y + next.y);
        average_normal = addPatchVector(
            average_normal, {{current.nx, current.ny, current.nz}});
    }
    origin = scalePatchVector(origin, 1.0 / static_cast<double>(loop.size()));
    if (!normalizePatchVector(&newell_normal))
    {
        newell_normal = average_normal;
        if (!normalizePatchVector(&newell_normal))
        {
            return false;
        }
    }
    if (normalizePatchVector(&average_normal) &&
        dotPatchVector(newell_normal, average_normal) < 0.0)
    {
        newell_normal = scalePatchVector(newell_normal, -1.0);
    }

    PatchVector tangent{};
    double maximum_tangent_length = 0.0;
    for (const int vertex_index : loop)
    {
        const PatchVector offset = subtractPatchVector(
            patchPosition(mesh.vertices[static_cast<std::size_t>(vertex_index)]),
            origin);
        const PatchVector planar = subtractPatchVector(
            offset,
            scalePatchVector(newell_normal,
                             dotPatchVector(offset, newell_normal)));
        const double length = dotPatchVector(planar, planar);
        if (length > maximum_tangent_length)
        {
            maximum_tangent_length = length;
            tangent = planar;
        }
    }
    if (!normalizePatchVector(&tangent))
    {
        return false;
    }
    PatchVector bitangent = crossPatchVector(newell_normal, tangent);
    if (!normalizePatchVector(&bitangent))
    {
        return false;
    }

    double scale = 0.0;
    for (const int vertex_index : loop)
    {
        const PatchVector offset = subtractPatchVector(
            patchPosition(mesh.vertices[static_cast<std::size_t>(vertex_index)]),
            origin);
        const double u = dotPatchVector(offset, tangent);
        const double v = dotPatchVector(offset, bitangent);
        scale = std::max(scale, std::sqrt(u * u + v * v));
    }
    if (!std::isfinite(scale) || scale <= 1.0e-8)
    {
        return false;
    }
    frame->origin = origin;
    frame->tangent = tangent;
    frame->bitangent = bitangent;
    frame->normal = newell_normal;
    frame->scale = scale;
    return true;
}

std::array<double, 3> boundaryPatchCoordinates(
    const BoundaryPatchFrame &frame,
    const MeshVertex &vertex)
{
    const PatchVector offset = subtractPatchVector(
        patchPosition(vertex), frame.origin);
    return {{dotPatchVector(offset, frame.tangent) / frame.scale,
             dotPatchVector(offset, frame.bitangent) / frame.scale,
             dotPatchVector(offset, frame.normal) / frame.scale}};
}

bool solveBoundaryPatchSystem(double matrix[6][7],
                              std::array<double, 6> *solution)
{
    if (!solution)
    {
        return false;
    }
    for (int pivot = 0; pivot < 6; ++pivot)
    {
        int best_row = pivot;
        for (int row = pivot + 1; row < 6; ++row)
        {
            if (std::abs(matrix[row][pivot]) >
                std::abs(matrix[best_row][pivot]))
            {
                best_row = row;
            }
        }
        if (!std::isfinite(matrix[best_row][pivot]) ||
            std::abs(matrix[best_row][pivot]) <= 1.0e-12)
        {
            return false;
        }
        if (best_row != pivot)
        {
            for (int column = pivot; column < 7; ++column)
            {
                std::swap(matrix[pivot][column], matrix[best_row][column]);
            }
        }
        const double inverse_pivot = 1.0 / matrix[pivot][pivot];
        for (int column = pivot; column < 7; ++column)
        {
            matrix[pivot][column] *= inverse_pivot;
        }
        for (int row = 0; row < 6; ++row)
        {
            if (row == pivot)
            {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (int column = pivot; column < 7; ++column)
            {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    for (int row = 0; row < 6; ++row)
    {
        (*solution)[static_cast<std::size_t>(row)] = matrix[row][6];
    }
    return true;
}

struct BoundaryPatchQuadric
{
    std::array<double, 6> coefficients{};
    double minimumHeight = -0.1;
    double maximumHeight = 0.1;
};

double evaluateBoundaryPatchQuadric(const BoundaryPatchQuadric &quadric,
                                    double u,
                                    double v)
{
    return quadric.coefficients[0] * u * u +
        quadric.coefficients[1] * u * v +
        quadric.coefficients[2] * v * v +
        quadric.coefficients[3] * u +
        quadric.coefficients[4] * v +
        quadric.coefficients[5];
}

BoundaryPatchQuadric fitBoundaryPatchQuadric(
    const TriMesh &mesh,
    const std::vector<int> &loop,
    const BoundaryPatchFrame &frame)
{
    BoundaryPatchQuadric quadric;
    std::vector<double> sample_weights(mesh.vertices.size(), 0.0);
    for (const int vertex_index : loop)
    {
        sample_weights[static_cast<std::size_t>(vertex_index)] = 2.0;
    }
    for (const Triangle &face : mesh.faces)
    {
        bool touches_boundary = false;
        for (const int vertex_index : face.v)
        {
            if (vertex_index >= 0 &&
                static_cast<std::size_t>(vertex_index) < sample_weights.size() &&
                sample_weights[static_cast<std::size_t>(vertex_index)] >= 2.0)
            {
                touches_boundary = true;
                break;
            }
        }
        if (!touches_boundary)
        {
            continue;
        }
        for (const int vertex_index : face.v)
        {
            if (vertex_index >= 0 &&
                static_cast<std::size_t>(vertex_index) < sample_weights.size())
            {
                sample_weights[static_cast<std::size_t>(vertex_index)] =
                    std::max(sample_weights[static_cast<std::size_t>(vertex_index)],
                             1.0);
            }
        }
    }

    double normal_matrix[6][7]{};
    double minimum_height = std::numeric_limits<double>::infinity();
    double maximum_height = -std::numeric_limits<double>::infinity();
    int sample_count = 0;
    for (std::size_t vertex_index = 0;
         vertex_index < sample_weights.size();
         ++vertex_index)
    {
        const double weight = sample_weights[vertex_index];
        if (weight <= 0.0)
        {
            continue;
        }
        const auto coordinates = boundaryPatchCoordinates(
            frame, mesh.vertices[vertex_index]);
        const double features[6]{
            coordinates[0] * coordinates[0],
            coordinates[0] * coordinates[1],
            coordinates[1] * coordinates[1],
            coordinates[0],
            coordinates[1],
            1.0};
        for (int row = 0; row < 6; ++row)
        {
            for (int column = 0; column < 6; ++column)
            {
                normal_matrix[row][column] +=
                    weight * features[row] * features[column];
            }
            normal_matrix[row][6] +=
                weight * features[row] * coordinates[2];
        }
        minimum_height = std::min(minimum_height, coordinates[2]);
        maximum_height = std::max(maximum_height, coordinates[2]);
        ++sample_count;
    }

    if (sample_count >= 6)
    {
        double diagonal_scale = 0.0;
        for (int diagonal = 0; diagonal < 6; ++diagonal)
        {
            diagonal_scale += normal_matrix[diagonal][diagonal];
        }
        const double regularization =
            std::max(1.0e-10, diagonal_scale * 1.0e-8);
        for (int diagonal = 0; diagonal < 6; ++diagonal)
        {
            normal_matrix[diagonal][diagonal] += regularization;
        }
        solveBoundaryPatchSystem(normal_matrix, &quadric.coefficients);
    }

    if (std::isfinite(minimum_height) && std::isfinite(maximum_height))
    {
        const double margin = std::max(
            0.10, 2.0 * std::max(0.0, maximum_height - minimum_height));
        quadric.minimumHeight = minimum_height - margin;
        quadric.maximumHeight = maximum_height + margin;
    }
    return quadric;
}

MeshVertex makeBoundaryPatchVertex(const TriMesh &mesh,
                                   const std::vector<int> &loop,
                                   const BoundaryPatchFrame &frame,
                                   const BoundaryPatchQuadric &quadric,
                                   double loop_parameter,
                                   double radius,
                                   const std::vector<double> &edge_lengths,
                                   double perimeter)
{
    const double target_length =
        std::clamp(loop_parameter, 0.0, 1.0) * perimeter;
    std::size_t edge_index = 0;
    double accumulated = 0.0;
    while (edge_index + 1 < edge_lengths.size() &&
           accumulated + edge_lengths[edge_index] < target_length)
    {
        accumulated += edge_lengths[edge_index];
        ++edge_index;
    }
    const double edge_length = edge_lengths[edge_index];
    const double alpha = edge_length > 1.0e-12
        ? std::clamp((target_length - accumulated) / edge_length, 0.0, 1.0)
        : 0.0;
    const MeshVertex &first = mesh.vertices[
        static_cast<std::size_t>(loop[edge_index])];
    const MeshVertex &second = mesh.vertices[
        static_cast<std::size_t>(loop[(edge_index + 1) % loop.size()])];
    const auto first_coordinates = boundaryPatchCoordinates(frame, first);
    const auto second_coordinates = boundaryPatchCoordinates(frame, second);
    const double boundary_u =
        first_coordinates[0] * (1.0 - alpha) + second_coordinates[0] * alpha;
    const double boundary_v =
        first_coordinates[1] * (1.0 - alpha) + second_coordinates[1] * alpha;
    const double boundary_height =
        first_coordinates[2] * (1.0 - alpha) + second_coordinates[2] * alpha;
    const double fit_boundary_height = evaluateBoundaryPatchQuadric(
        quadric, boundary_u, boundary_v);
    const double u = boundary_u * radius;
    const double v = boundary_v * radius;
    double height = evaluateBoundaryPatchQuadric(quadric, u, v) +
        radius * (boundary_height - fit_boundary_height);
    height = std::clamp(
        height, quadric.minimumHeight, quadric.maximumHeight);

    const PatchVector position = addPatchVector(
        frame.origin,
        scalePatchVector(
            addPatchVector(
                addPatchVector(
                    scalePatchVector(frame.tangent, u),
                    scalePatchVector(frame.bitangent, v)),
                scalePatchVector(frame.normal, height)),
            frame.scale));

    const double derivative_u =
        2.0 * quadric.coefficients[0] * u +
        quadric.coefficients[1] * v + quadric.coefficients[3];
    const double derivative_v =
        quadric.coefficients[1] * u +
        2.0 * quadric.coefficients[2] * v + quadric.coefficients[4];
    PatchVector fitted_normal = addPatchVector(
        frame.normal,
        addPatchVector(
            scalePatchVector(frame.tangent, -derivative_u),
            scalePatchVector(frame.bitangent, -derivative_v)));
    normalizePatchVector(&fitted_normal);
    PatchVector boundary_normal{{
        first.nx * (1.0 - alpha) + second.nx * alpha,
        first.ny * (1.0 - alpha) + second.ny * alpha,
        first.nz * (1.0 - alpha) + second.nz * alpha}};
    if (!normalizePatchVector(&boundary_normal))
    {
        boundary_normal = fitted_normal;
    }
    if (dotPatchVector(boundary_normal, fitted_normal) < 0.0)
    {
        fitted_normal = scalePatchVector(fitted_normal, -1.0);
    }
    PatchVector blended_normal = addPatchVector(
        scalePatchVector(boundary_normal, radius),
        scalePatchVector(fitted_normal, 1.0 - radius));
    normalizePatchVector(&blended_normal);

    MeshVertex vertex;
    vertex.x = static_cast<float>(position[0]);
    vertex.y = static_cast<float>(position[1]);
    vertex.z = static_cast<float>(position[2]);
    vertex.nx = static_cast<float>(blended_normal[0]);
    vertex.ny = static_cast<float>(blended_normal[1]);
    vertex.nz = static_cast<float>(blended_normal[2]);
    vertex.r = static_cast<std::uint8_t>(std::lround(
        first.r * (1.0 - alpha) + second.r * alpha));
    vertex.g = static_cast<std::uint8_t>(std::lround(
        first.g * (1.0 - alpha) + second.g * alpha));
    vertex.b = static_cast<std::uint8_t>(std::lround(
        first.b * (1.0 - alpha) + second.b * alpha));
    return vertex;
}

void appendBoundaryPatchAnnulus(const std::vector<int> &outer,
                                const std::vector<double> &outer_parameters,
                                const std::vector<int> &inner,
                                const std::vector<double> &inner_parameters,
                                std::vector<Triangle> *faces)
{
    if (!faces || outer.size() < 3 || inner.size() < 3)
    {
        return;
    }
    std::size_t outer_index = 0;
    std::size_t inner_index = 0;
    constexpr double epsilon = 1.0e-10;
    while (outer_index < outer.size() || inner_index < inner.size())
    {
        const double next_outer = outer_index + 1 < outer.size()
            ? outer_parameters[outer_index + 1]
            : 1.0;
        const double next_inner = inner_index + 1 < inner.size()
            ? inner_parameters[inner_index + 1]
            : 1.0;
        const int outer_current = outer[outer_index % outer.size()];
        const int inner_current = inner[inner_index % inner.size()];
        if (std::abs(next_outer - next_inner) <= epsilon)
        {
            const int outer_next = outer[(outer_index + 1) % outer.size()];
            const int inner_next = inner[(inner_index + 1) % inner.size()];
            faces->push_back(Triangle{{outer_current, outer_next, inner_current}});
            faces->push_back(Triangle{{outer_next, inner_next, inner_current}});
            ++outer_index;
            ++inner_index;
        }
        else if (next_outer < next_inner)
        {
            const int outer_next = outer[(outer_index + 1) % outer.size()];
            faces->push_back(Triangle{{outer_current, outer_next, inner_current}});
            ++outer_index;
        }
        else
        {
            const int inner_next = inner[(inner_index + 1) % inner.size()];
            faces->push_back(Triangle{{outer_current, inner_next, inner_current}});
            ++inner_index;
        }
    }
}

bool triangulateBoundaryLoop(const TriMesh &mesh,
                             const std::vector<int> &loop,
                             const std::unordered_map<std::uint64_t, int> &
                                 existingEdges,
                             std::vector<Triangle> *triangles)
{
    if (!triangles || loop.size() < 3)
    {
        return false;
    }

    double normal_x = 0.0;
    double normal_y = 0.0;
    double normal_z = 0.0;
    for (std::size_t index = 0; index < loop.size(); ++index)
    {
        const MeshVertex &current =
            mesh.vertices[static_cast<std::size_t>(loop[index])];
        const MeshVertex &next = mesh.vertices[
            static_cast<std::size_t>(loop[(index + 1) % loop.size()])];
        normal_x += (current.y - next.y) * (current.z + next.z);
        normal_y += (current.z - next.z) * (current.x + next.x);
        normal_z += (current.x - next.x) * (current.y + next.y);
    }
    const double absolute_x = std::abs(normal_x);
    const double absolute_y = std::abs(normal_y);
    const double absolute_z = std::abs(normal_z);
    int dropped_axis = 2;
    if (absolute_x >= absolute_y && absolute_x >= absolute_z)
    {
        dropped_axis = 0;
    }
    else if (absolute_y >= absolute_z)
    {
        dropped_axis = 1;
    }

    std::vector<ProjectedBoundaryPoint> projected(loop.size());
    for (std::size_t index = 0; index < loop.size(); ++index)
    {
        const MeshVertex &vertex =
            mesh.vertices[static_cast<std::size_t>(loop[index])];
        if (dropped_axis == 0)
        {
            projected[index] = {vertex.y, vertex.z};
        }
        else if (dropped_axis == 1)
        {
            projected[index] = {vertex.x, vertex.z};
        }
        else
        {
            projected[index] = {vertex.x, vertex.y};
        }
    }
    double signed_area = 0.0;
    for (std::size_t index = 0; index < projected.size(); ++index)
    {
        const ProjectedBoundaryPoint &current = projected[index];
        const ProjectedBoundaryPoint &next =
            projected[(index + 1) % projected.size()];
        signed_area += current.x * next.y - next.x * current.y;
    }
    if (std::abs(signed_area) <= 1.0e-16)
    {
        return false;
    }
    const double orientation = signed_area > 0.0 ? 1.0 : -1.0;

    std::vector<int> remaining(loop.size());
    for (std::size_t index = 0; index < remaining.size(); ++index)
    {
        remaining[index] = static_cast<int>(index);
    }
    std::vector<Triangle> candidate;
    candidate.reserve(loop.size() - 2);
    while (remaining.size() > 3)
    {
        int best_position = -1;
        double best_aspect = std::numeric_limits<double>::infinity();
        for (std::size_t position = 0; position < remaining.size(); ++position)
        {
            const int previous = remaining[
                (position + remaining.size() - 1) % remaining.size()];
            const int current = remaining[position];
            const int next = remaining[(position + 1) % remaining.size()];
            const bool closes_original_boundary_edge =
                (static_cast<std::size_t>(previous + 1) % loop.size() ==
                     static_cast<std::size_t>(next)) ||
                (static_cast<std::size_t>(next + 1) % loop.size() ==
                     static_cast<std::size_t>(previous));
            if (!closes_original_boundary_edge &&
                existingEdges.find(edgeKey(
                    loop[static_cast<std::size_t>(previous)],
                    loop[static_cast<std::size_t>(next)])) !=
                    existingEdges.cend())
            {
                continue;
            }
            if (boundaryCross(
                    projected[static_cast<std::size_t>(previous)],
                    projected[static_cast<std::size_t>(current)],
                    projected[static_cast<std::size_t>(next)]) *
                    orientation <= 1.0e-14)
            {
                continue;
            }
            bool contains_vertex = false;
            for (const int other : remaining)
            {
                if (other == previous || other == current || other == next)
                {
                    continue;
                }
                if (pointInsideBoundaryTriangle(
                        projected[static_cast<std::size_t>(other)],
                        projected[static_cast<std::size_t>(previous)],
                        projected[static_cast<std::size_t>(current)],
                        projected[static_cast<std::size_t>(next)],
                        orientation))
                {
                    contains_vertex = true;
                    break;
                }
            }
            if (contains_vertex)
            {
                continue;
            }
            const double aspect = boundaryTriangleAspect(
                mesh,
                loop[static_cast<std::size_t>(previous)],
                loop[static_cast<std::size_t>(current)],
                loop[static_cast<std::size_t>(next)]);
            if (aspect < best_aspect)
            {
                best_aspect = aspect;
                best_position = static_cast<int>(position);
            }
        }
        if (best_position < 0 || !std::isfinite(best_aspect))
        {
            return false;
        }
        const std::size_t position = static_cast<std::size_t>(best_position);
        const int previous = remaining[
            (position + remaining.size() - 1) % remaining.size()];
        const int current = remaining[position];
        const int next = remaining[(position + 1) % remaining.size()];
        candidate.push_back(Triangle{{
            loop[static_cast<std::size_t>(previous)],
            loop[static_cast<std::size_t>(current)],
            loop[static_cast<std::size_t>(next)]}});
        remaining.erase(remaining.begin() + best_position);
    }
    candidate.push_back(Triangle{{
        loop[static_cast<std::size_t>(remaining[0])],
        loop[static_cast<std::size_t>(remaining[1])],
        loop[static_cast<std::size_t>(remaining[2])]}});
    triangles->insert(
        triangles->end(), candidate.cbegin(), candidate.cend());
    return true;
}

} // namespace

bool fillCurvatureAwareBoundaryPatch(TriMesh *mesh,
                                     const std::vector<int> &boundaryLoop)
{
    if (!mesh || boundaryLoop.size() < 6 || mesh->vertices.empty())
    {
        return false;
    }
    for (const int vertex_index : boundaryLoop)
    {
        if (vertex_index < 0 ||
            static_cast<std::size_t>(vertex_index) >= mesh->vertices.size())
        {
            return false;
        }
    }

    BoundaryPatchFrame frame;
    if (!buildBoundaryPatchFrame(*mesh, boundaryLoop, &frame))
    {
        return false;
    }
    const BoundaryPatchQuadric quadric = fitBoundaryPatchQuadric(
        *mesh, boundaryLoop, frame);

    std::vector<double> boundary_edge_lengths(boundaryLoop.size(), 0.0);
    std::vector<double> boundary_parameters(boundaryLoop.size(), 0.0);
    double perimeter = 0.0;
    for (std::size_t index = 0; index < boundaryLoop.size(); ++index)
    {
        const PatchVector current = patchPosition(mesh->vertices[
            static_cast<std::size_t>(boundaryLoop[index])]);
        const PatchVector next = patchPosition(mesh->vertices[
            static_cast<std::size_t>(boundaryLoop[(index + 1) %
                                                   boundaryLoop.size()])]);
        boundary_parameters[index] = perimeter;
        const PatchVector edge = subtractPatchVector(next, current);
        boundary_edge_lengths[index] = std::sqrt(dotPatchVector(edge, edge));
        perimeter += boundary_edge_lengths[index];
    }
    if (!std::isfinite(perimeter) || perimeter <= 1.0e-8)
    {
        return false;
    }
    for (double &parameter : boundary_parameters)
    {
        parameter /= perimeter;
    }

    const int first_ring_count = std::max(
        3, static_cast<int>(std::ceil(boundaryLoop.size() * 0.45)));
    const bool use_second_ring = boundaryLoop.size() >= 18;
    const int second_ring_count = use_second_ring
        ? std::max(3, static_cast<int>(std::ceil(boundaryLoop.size() * 0.20)))
        : 0;
    const std::size_t original_vertex_count = mesh->vertices.size();
    std::vector<MeshVertex> new_vertices;
    new_vertices.reserve(static_cast<std::size_t>(
        first_ring_count + second_ring_count + 1));
    std::vector<int> first_ring;
    std::vector<int> second_ring;
    std::vector<double> first_parameters;
    std::vector<double> second_parameters;

    const auto build_ring = [&](int count,
                                double radius,
                                std::vector<int> *indices,
                                std::vector<double> *parameters)
    {
        indices->reserve(static_cast<std::size_t>(count));
        parameters->reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const double parameter =
                static_cast<double>(index) / static_cast<double>(count);
            parameters->push_back(parameter);
            indices->push_back(static_cast<int>(
                original_vertex_count + new_vertices.size()));
            new_vertices.push_back(makeBoundaryPatchVertex(
                *mesh,
                boundaryLoop,
                frame,
                quadric,
                parameter,
                radius,
                boundary_edge_lengths,
                perimeter));
        }
    };
    build_ring(first_ring_count, 0.65, &first_ring, &first_parameters);
    if (use_second_ring)
    {
        build_ring(second_ring_count, 0.32, &second_ring, &second_parameters);
    }

    std::vector<Triangle> new_faces;
    new_faces.reserve(boundaryLoop.size() +
                      first_ring.size() * 2 +
                      second_ring.size() * 2);
    appendBoundaryPatchAnnulus(
        boundaryLoop,
        boundary_parameters,
        first_ring,
        first_parameters,
        &new_faces);
    const std::vector<int> &last_ring = use_second_ring
        ? second_ring
        : first_ring;
    if (use_second_ring)
    {
        appendBoundaryPatchAnnulus(
            first_ring,
            first_parameters,
            second_ring,
            second_parameters,
            &new_faces);
    }

    TriMesh inner_cap_mesh;
    std::vector<int> inner_cap_loop(last_ring.size());
    inner_cap_mesh.vertices.reserve(last_ring.size());
    for (std::size_t index = 0; index < last_ring.size(); ++index)
    {
        inner_cap_loop[index] = static_cast<int>(index);
        inner_cap_mesh.vertices.push_back(new_vertices[
            static_cast<std::size_t>(last_ring[index]) -
            original_vertex_count]);
    }
    std::vector<Triangle> inner_cap_faces;
    const std::unordered_map<std::uint64_t, int> no_existing_edges;
    if (!triangulateBoundaryLoop(
            inner_cap_mesh,
            inner_cap_loop,
            no_existing_edges,
            &inner_cap_faces))
    {
        return false;
    }
    for (Triangle face : inner_cap_faces)
    {
        for (int &vertex_index : face.v)
        {
            vertex_index = last_ring[static_cast<std::size_t>(vertex_index)];
        }
        new_faces.push_back(face);
    }

    const auto vertex_at = [&](int vertex_index) -> const MeshVertex &
    {
        if (static_cast<std::size_t>(vertex_index) < original_vertex_count)
        {
            return mesh->vertices[static_cast<std::size_t>(vertex_index)];
        }
        return new_vertices[static_cast<std::size_t>(vertex_index) -
                            original_vertex_count];
    };
    const double minimum_doubled_area =
        frame.scale * frame.scale * 1.0e-12;
    for (const Triangle &face : new_faces)
    {
        if (face.v[0] == face.v[1] || face.v[1] == face.v[2] ||
            face.v[2] == face.v[0])
        {
            return false;
        }
        const PatchVector first = patchPosition(vertex_at(face.v[0]));
        const PatchVector second = patchPosition(vertex_at(face.v[1]));
        const PatchVector third = patchPosition(vertex_at(face.v[2]));
        const PatchVector doubled_area = crossPatchVector(
            subtractPatchVector(second, first),
            subtractPatchVector(third, first));
        if (!std::isfinite(dotPatchVector(doubled_area, doubled_area)) ||
            std::sqrt(dotPatchVector(doubled_area, doubled_area)) <=
                minimum_doubled_area)
        {
            return false;
        }
    }

    mesh->vertices.insert(
        mesh->vertices.end(), new_vertices.cbegin(), new_vertices.cend());
    mesh->faces.insert(mesh->faces.end(), new_faces.cbegin(), new_faces.cend());
    return true;
}

int fillSmallBoundaryHoles(TriMesh *mesh,
                           int maxBoundaryEdges,
                           float maxBoundaryDiameter,
                           const std::vector<std::uint8_t> *
                               protectedBoundaryVertices,
                           int *protectedHoleCount,
                           bool useQualityTriangulation)
{
    if (protectedHoleCount)
    {
        *protectedHoleCount = 0;
    }
    if (!mesh || mesh->faces.empty() || mesh->vertices.empty() || maxBoundaryEdges < 3)
    {
        return 0;
    }
    const bool has_boundary_protection =
        protectedBoundaryVertices &&
        protectedBoundaryVertices->size() == mesh->vertices.size();

    const auto edge_key = [](int first, int second) {
        const auto low = static_cast<std::uint32_t>(std::min(first, second));
        const auto high = static_cast<std::uint32_t>(std::max(first, second));
        return (static_cast<std::uint64_t>(low) << 32U) | static_cast<std::uint64_t>(high);
    };

    std::unordered_map<std::uint64_t, int> edge_counts;
    std::unordered_map<std::uint64_t, std::array<int, 2>> edge_directions;
    edge_counts.reserve(mesh->faces.size() * 3);
    for (const Triangle &face : mesh->faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            const std::uint64_t key = edge_key(edge[0], edge[1]);
            if (++edge_counts[key] == 1)
            {
                edge_directions[key] = edge;
            }
        }
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
            if (useQualityTriangulation)
            {
                int matching_face_directions = 0;
                for (std::size_t edge_index = 0;
                     edge_index < loop.size();
                     ++edge_index)
                {
                    const int first = loop[edge_index];
                    const int second =
                        loop[(edge_index + 1) % loop.size()];
                    const auto direction = edge_directions.find(
                        edge_key(first, second));
                    if (direction != edge_directions.cend() &&
                        direction->second[0] == first &&
                        direction->second[1] == second)
                    {
                        ++matching_face_directions;
                    }
                }
                if (matching_face_directions * 2 >=
                    static_cast<int>(loop.size()))
                {
                    std::reverse(loop.begin(), loop.end());
                }
            }
            if (has_boundary_protection)
            {
                const bool protects_silhouette = std::any_of(
                    loop.cbegin(), loop.cend(), [&](int vertex_index)
                    {
                        return (*protectedBoundaryVertices)[
                            static_cast<std::size_t>(vertex_index)] != 0;
                    });
                if (protects_silhouette)
                {
                    if (protectedHoleCount)
                    {
                        ++(*protectedHoleCount);
                    }
                    continue;
                }
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

            if (useQualityTriangulation && loop.size() >= 24)
            {
                if (fillCurvatureAwareBoundaryPatch(mesh, loop))
                {
                    ++filled_holes;
                }
                // Large non-planar holes are never collapsed into a single
                // centroid fan.  If the curvature-aware patch cannot be
                // constructed, preserving the opening is safer than adding a
                // star-shaped geometric artifact.
                continue;
            }
            if (useQualityTriangulation &&
                triangulateBoundaryLoop(
                    *mesh, loop, edge_counts, &added_faces))
            {
                ++filled_holes;
                continue;
            }

            MeshVertex center;
            int red_sum = 0;
            int green_sum = 0;
            int blue_sum = 0;
            for (const int vertex_index : loop)
            {
                const MeshVertex &vertex =
                    mesh->vertices[static_cast<std::size_t>(vertex_index)];
                center.x += vertex.x;
                center.y += vertex.y;
                center.z += vertex.z;
                red_sum += vertex.r;
                green_sum += vertex.g;
                blue_sum += vertex.b;
            }
            const float inverse_loop_size = 1.0f /
                static_cast<float>(loop.size());
            center.x *= inverse_loop_size;
            center.y *= inverse_loop_size;
            center.z *= inverse_loop_size;
            center.r = static_cast<std::uint8_t>(
                red_sum / static_cast<int>(loop.size()));
            center.g = static_cast<std::uint8_t>(
                green_sum / static_cast<int>(loop.size()));
            center.b = static_cast<std::uint8_t>(
                blue_sum / static_cast<int>(loop.size()));
            const int center_index = static_cast<int>(mesh->vertices.size());
            mesh->vertices.push_back(center);
            for (std::size_t edge_index = 0; edge_index < loop.size(); ++edge_index)
            {
                added_faces.push_back(Triangle{{
                    loop[edge_index],
                    loop[(edge_index + 1) % loop.size()],
                    center_index}});
            }
            ++filled_holes;
        }
    }

    mesh->faces.insert(mesh->faces.end(), added_faces.begin(), added_faces.end());
    return filled_holes;
}

int collapseTinyBoundaryLoops(TriMesh *mesh,
                              int maximumBoundaryEdges,
                              float maximumBoundaryDiameter,
                              float maximumCollapseEdgeLength)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() ||
        maximumBoundaryEdges < 3 || maximumBoundaryDiameter <= 0.0f ||
        maximumCollapseEdgeLength <= 0.0f)
    {
        return 0;
    }

    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh->faces.size() * 3);
    std::vector<std::unordered_set<int>> vertex_neighbors(mesh->vertices.size());
    for (const Triangle &face : mesh->faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            ++edge_counts[edgeKey(edge[0], edge[1])];
            vertex_neighbors[static_cast<std::size_t>(edge[0])].insert(edge[1]);
            vertex_neighbors[static_cast<std::size_t>(edge[1])].insert(edge[0]);
        }
    }

    std::vector<std::vector<int>> boundary_neighbors(mesh->vertices.size());
    for (const auto &[key, count] : edge_counts)
    {
        if (count != 1)
        {
            continue;
        }
        const int first = static_cast<int>(key >> 32U);
        const int second = static_cast<int>(key & 0xffffffffU);
        boundary_neighbors[static_cast<std::size_t>(first)].push_back(second);
        boundary_neighbors[static_cast<std::size_t>(second)].push_back(first);
    }

    std::unordered_set<std::uint64_t> visited_edges;
    visited_edges.reserve(edge_counts.size());
    std::vector<std::vector<int>> loops;
    for (int start = 0;
         start < static_cast<int>(boundary_neighbors.size());
         ++start)
    {
        if (boundary_neighbors[static_cast<std::size_t>(start)].size() != 2)
        {
            continue;
        }
        for (const int first_neighbor :
             boundary_neighbors[static_cast<std::size_t>(start)])
        {
            if (visited_edges.find(edgeKey(start, first_neighbor)) !=
                visited_edges.cend())
            {
                continue;
            }

            std::vector<int> loop{start};
            int previous = start;
            int current = first_neighbor;
            bool closed = false;
            visited_edges.insert(edgeKey(previous, current));
            while (static_cast<int>(loop.size()) <= maximumBoundaryEdges)
            {
                if (current == start)
                {
                    closed = true;
                    break;
                }
                loop.push_back(current);
                const std::vector<int> &neighbors =
                    boundary_neighbors[static_cast<std::size_t>(current)];
                if (neighbors.size() != 2)
                {
                    break;
                }
                const int next =
                    neighbors[0] == previous ? neighbors[1] : neighbors[0];
                const std::uint64_t next_key = edgeKey(current, next);
                if (next != start &&
                    visited_edges.find(next_key) != visited_edges.cend())
                {
                    break;
                }
                previous = current;
                current = next;
                visited_edges.insert(next_key);
            }
            if (closed && loop.size() >= 3 &&
                static_cast<int>(loop.size()) <= maximumBoundaryEdges)
            {
                loops.push_back(std::move(loop));
            }
        }
    }
    std::stable_sort(
        loops.begin(), loops.end(),
        [](const std::vector<int> &first, const std::vector<int> &second)
        {
            return first.size() < second.size();
        });

    const float maximum_diameter_squared =
        maximumBoundaryDiameter * maximumBoundaryDiameter;
    const float maximum_edge_squared =
        maximumCollapseEdgeLength * maximumCollapseEdgeLength;
    std::vector<int> replacements(mesh->vertices.size());
    for (std::size_t index = 0; index < replacements.size(); ++index)
    {
        replacements[index] = static_cast<int>(index);
    }
    std::vector<std::uint8_t> blocked_vertices(mesh->vertices.size(), 0);
    int collapsed_edge_count = 0;
    for (const std::vector<int> &loop : loops)
    {
        bool diameter_exceeded = false;
        for (std::size_t first = 0;
             first < loop.size() && !diameter_exceeded;
             ++first)
        {
            const MeshVertex &a =
                mesh->vertices[static_cast<std::size_t>(loop[first])];
            for (std::size_t second = first + 1;
                 second < loop.size();
                 ++second)
            {
                const MeshVertex &b =
                    mesh->vertices[static_cast<std::size_t>(loop[second])];
                const float dx = a.x - b.x;
                const float dy = a.y - b.y;
                const float dz = a.z - b.z;
                if (dx * dx + dy * dy + dz * dz >
                    maximum_diameter_squared)
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

        struct CollapseCandidate
        {
            int first = -1;
            int second = -1;
            float lengthSquared = std::numeric_limits<float>::infinity();
        };
        std::vector<CollapseCandidate> candidates;
        candidates.reserve(loop.size());
        for (std::size_t edge_index = 0;
             edge_index < loop.size();
             ++edge_index)
        {
            const int first = loop[edge_index];
            const int second = loop[(edge_index + 1) % loop.size()];
            const MeshVertex &a =
                mesh->vertices[static_cast<std::size_t>(first)];
            const MeshVertex &b =
                mesh->vertices[static_cast<std::size_t>(second)];
            const float dx = a.x - b.x;
            const float dy = a.y - b.y;
            const float dz = a.z - b.z;
            candidates.push_back(
                {first, second, dx * dx + dy * dy + dz * dz});
        }
        std::sort(
            candidates.begin(), candidates.end(),
            [](const CollapseCandidate &first,
               const CollapseCandidate &second)
            {
                return first.lengthSquared < second.lengthSquared;
            });

        for (const CollapseCandidate &candidate : candidates)
        {
            if (candidate.lengthSquared > maximum_edge_squared)
            {
                break;
            }
            int common_neighbor_count = 0;
            const auto &first_neighbors = vertex_neighbors[
                static_cast<std::size_t>(candidate.first)];
            const auto &second_neighbors = vertex_neighbors[
                static_cast<std::size_t>(candidate.second)];
            const auto &smaller =
                first_neighbors.size() <= second_neighbors.size()
                ? first_neighbors
                : second_neighbors;
            const auto &larger =
                first_neighbors.size() <= second_neighbors.size()
                ? second_neighbors
                : first_neighbors;
            for (const int neighbor : smaller)
            {
                if (larger.find(neighbor) != larger.cend())
                {
                    ++common_neighbor_count;
                }
            }
            if (common_neighbor_count > 2)
            {
                continue;
            }

            std::vector<int> neighborhood{
                candidate.first, candidate.second};
            neighborhood.insert(
                neighborhood.end(),
                first_neighbors.cbegin(),
                first_neighbors.cend());
            neighborhood.insert(
                neighborhood.end(),
                second_neighbors.cbegin(),
                second_neighbors.cend());
            const bool overlaps_previous = std::any_of(
                neighborhood.cbegin(), neighborhood.cend(),
                [&blocked_vertices](int vertex)
                {
                    return blocked_vertices[
                        static_cast<std::size_t>(vertex)] != 0;
                });
            if (overlaps_previous)
            {
                continue;
            }

            MeshVertex &survivor = mesh->vertices[
                static_cast<std::size_t>(candidate.first)];
            const MeshVertex &removed = mesh->vertices[
                static_cast<std::size_t>(candidate.second)];
            survivor.x = 0.5f * (survivor.x + removed.x);
            survivor.y = 0.5f * (survivor.y + removed.y);
            survivor.z = 0.5f * (survivor.z + removed.z);
            survivor.nx = 0.5f * (survivor.nx + removed.nx);
            survivor.ny = 0.5f * (survivor.ny + removed.ny);
            survivor.nz = 0.5f * (survivor.nz + removed.nz);
            survivor.r = static_cast<std::uint8_t>(
                (static_cast<int>(survivor.r) + removed.r) / 2);
            survivor.g = static_cast<std::uint8_t>(
                (static_cast<int>(survivor.g) + removed.g) / 2);
            survivor.b = static_cast<std::uint8_t>(
                (static_cast<int>(survivor.b) + removed.b) / 2);
            replacements[static_cast<std::size_t>(candidate.second)] =
                candidate.first;
            for (const int vertex : neighborhood)
            {
                blocked_vertices[static_cast<std::size_t>(vertex)] = 1;
            }
            ++collapsed_edge_count;
            break;
        }
    }
    if (collapsed_edge_count == 0)
    {
        return 0;
    }

    std::vector<Triangle> kept_faces;
    kept_faces.reserve(mesh->faces.size());
    for (Triangle face : mesh->faces)
    {
        for (int &vertex : face.v)
        {
            vertex = replacements[static_cast<std::size_t>(vertex)];
        }
        if (face.v[0] != face.v[1] &&
            face.v[1] != face.v[2] &&
            face.v[2] != face.v[0])
        {
            kept_faces.push_back(face);
        }
    }
    mesh->faces = std::move(kept_faces);
    removeDuplicateFaces(mesh);
    compactReferencedVertices(mesh);
    return collapsed_edge_count;
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

int smoothSurfaceVerticesNormalAware(TriMesh *mesh,
                                     int iterations,
                                     float lambda,
                                     float maximumDisplacement,
                                     float maximumNormalAngleDegrees,
                                     int boundaryProtectionRings)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() ||
        iterations <= 0 || lambda <= 0.0f || maximumDisplacement <= 0.0f)
    {
        return 0;
    }

    std::vector<std::vector<int>> neighbors(mesh->vertices.size());
    std::unordered_map<std::uint64_t, int> edge_counts;
    edge_counts.reserve(mesh->faces.size() * 3);
    for (const Triangle &face : mesh->faces)
    {
        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            ++edge_counts[edgeKey(edge[0], edge[1])];
            neighbors[static_cast<std::size_t>(edge[0])].push_back(edge[1]);
            neighbors[static_cast<std::size_t>(edge[1])].push_back(edge[0]);
        }
    }
    for (auto &adjacent : neighbors)
    {
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    }

    std::vector<std::uint8_t> protected_vertex(mesh->vertices.size(), 0);
    for (const auto &entry : edge_counts)
    {
        if (entry.second != 1)
        {
            continue;
        }
        const int first = static_cast<int>(entry.first >> 32U);
        const int second = static_cast<int>(entry.first & 0xffffffffU);
        protected_vertex[static_cast<std::size_t>(first)] = 1;
        protected_vertex[static_cast<std::size_t>(second)] = 1;
    }
    for (int ring = 0; ring < std::max(0, boundaryProtectionRings); ++ring)
    {
        std::vector<std::uint8_t> expanded = protected_vertex;
        for (std::size_t index = 0; index < protected_vertex.size(); ++index)
        {
            if (!protected_vertex[index])
            {
                continue;
            }
            for (const int neighbor : neighbors[index])
            {
                expanded[static_cast<std::size_t>(neighbor)] = 1;
            }
        }
        protected_vertex = std::move(expanded);
    }

    lambda = std::clamp(lambda, 0.0f, 1.0f);
    const float clamped_angle = std::clamp(maximumNormalAngleDegrees, 1.0f, 89.0f);
    const float minimum_normal_dot = std::cos(
        clamped_angle * 3.14159265358979323846f / 180.0f);
    const std::vector<MeshVertex> original = mesh->vertices;
    std::vector<std::uint8_t> moved(mesh->vertices.size(), 0);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        recomputeNormals(mesh);
        const std::vector<MeshVertex> source = mesh->vertices;
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            if (protected_vertex[index] || neighbors[index].size() < 3)
            {
                continue;
            }
            const MeshVertex &current = source[index];
            const float normal_length = std::sqrt(
                current.nx * current.nx + current.ny * current.ny + current.nz * current.nz);
            if (!std::isfinite(normal_length) || normal_length <= 1.0e-6f)
            {
                continue;
            }
            const float nx = current.nx / normal_length;
            const float ny = current.ny / normal_length;
            const float nz = current.nz / normal_length;
            float weighted_x = 0.0f;
            float weighted_y = 0.0f;
            float weighted_z = 0.0f;
            float weight_sum = 0.0f;
            for (const int neighbor_index : neighbors[index])
            {
                const MeshVertex &neighbor = source[static_cast<std::size_t>(neighbor_index)];
                const float neighbor_length = std::sqrt(
                    neighbor.nx * neighbor.nx + neighbor.ny * neighbor.ny +
                    neighbor.nz * neighbor.nz);
                if (!std::isfinite(neighbor_length) || neighbor_length <= 1.0e-6f)
                {
                    continue;
                }
                const float normal_dot =
                    (nx * neighbor.nx + ny * neighbor.ny + nz * neighbor.nz) /
                    neighbor_length;
                if (normal_dot < minimum_normal_dot)
                {
                    continue;
                }
                const float weight = std::max(
                    0.05f,
                    (normal_dot - minimum_normal_dot) /
                        std::max(1.0f - minimum_normal_dot, 1.0e-6f));
                weighted_x += neighbor.x * weight;
                weighted_y += neighbor.y * weight;
                weighted_z += neighbor.z * weight;
                weight_sum += weight;
            }
            if (weight_sum <= 1.0e-6f)
            {
                continue;
            }
            const float mean_x = weighted_x / weight_sum;
            const float mean_y = weighted_y / weight_sum;
            const float mean_z = weighted_z / weight_sum;
            float normal_displacement =
                ((mean_x - current.x) * nx +
                 (mean_y - current.y) * ny +
                 (mean_z - current.z) * nz) * lambda;
            normal_displacement = std::clamp(
                normal_displacement, -maximumDisplacement, maximumDisplacement);

            MeshVertex candidate = current;
            candidate.x += normal_displacement * nx;
            candidate.y += normal_displacement * ny;
            candidate.z += normal_displacement * nz;
            const MeshVertex &origin = original[index];
            float offset_x = candidate.x - origin.x;
            float offset_y = candidate.y - origin.y;
            float offset_z = candidate.z - origin.z;
            const float total_displacement = std::sqrt(
                offset_x * offset_x + offset_y * offset_y + offset_z * offset_z);
            if (total_displacement > maximumDisplacement)
            {
                const float scale = maximumDisplacement / total_displacement;
                offset_x *= scale;
                offset_y *= scale;
                offset_z *= scale;
                candidate.x = origin.x + offset_x;
                candidate.y = origin.y + offset_y;
                candidate.z = origin.z + offset_z;
            }
            if (std::abs(normal_displacement) > 1.0e-8f)
            {
                mesh->vertices[index].x = candidate.x;
                mesh->vertices[index].y = candidate.y;
                mesh->vertices[index].z = candidate.z;
                moved[index] = 1;
            }
        }
    }
    return static_cast<int>(std::count(moved.cbegin(), moved.cend(), std::uint8_t{1}));
}

int smoothSurfaceVerticesTaubinProtected(TriMesh *mesh,
                                         int iterations,
                                         float lambda,
                                         float mu,
                                         float maximumDisplacement,
                                         float featureAngleDegrees,
                                         int boundaryProtectionRings)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() ||
        iterations <= 0 || !std::isfinite(lambda) || !std::isfinite(mu) ||
        maximumDisplacement <= 0.0f)
    {
        return 0;
    }

    struct EdgeFaces
    {
        int first = -1;
        int second = -1;
        int count = 0;
    };

    std::vector<std::vector<int>> neighbors(mesh->vertices.size());
    std::unordered_map<std::uint64_t, EdgeFaces> edge_faces;
    edge_faces.reserve(mesh->faces.size() * 3);
    std::vector<std::array<float, 3>> face_normals(mesh->faces.size());
    for (std::size_t face_index = 0; face_index < mesh->faces.size(); ++face_index)
    {
        const Triangle &face = mesh->faces[face_index];
        const MeshVertex &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];
        const float ab_x = b.x - a.x;
        const float ab_y = b.y - a.y;
        const float ab_z = b.z - a.z;
        const float ac_x = c.x - a.x;
        const float ac_y = c.y - a.y;
        const float ac_z = c.z - a.z;
        float nx = ab_y * ac_z - ab_z * ac_y;
        float ny = ab_z * ac_x - ab_x * ac_z;
        float nz = ab_x * ac_y - ab_y * ac_x;
        const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (length > 1.0e-12f)
        {
            nx /= length;
            ny /= length;
            nz /= length;
        }
        face_normals[face_index] = {nx, ny, nz};

        const std::array<std::array<int, 2>, 3> edges{{
            {{face.v[0], face.v[1]}},
            {{face.v[1], face.v[2]}},
            {{face.v[2], face.v[0]}}
        }};
        for (const auto &edge : edges)
        {
            neighbors[static_cast<std::size_t>(edge[0])].push_back(edge[1]);
            neighbors[static_cast<std::size_t>(edge[1])].push_back(edge[0]);
            EdgeFaces &incident = edge_faces[edgeKey(edge[0], edge[1])];
            if (incident.count == 0)
            {
                incident.first = static_cast<int>(face_index);
            }
            else if (incident.count == 1)
            {
                incident.second = static_cast<int>(face_index);
            }
            ++incident.count;
        }
    }
    for (auto &adjacent : neighbors)
    {
        std::sort(adjacent.begin(), adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    }

    std::vector<std::uint8_t> protected_vertex(mesh->vertices.size(), 0);
    std::vector<int> sharp_edge_degree(mesh->vertices.size(), 0);
    const float clamped_feature_angle = std::clamp(featureAngleDegrees, 1.0f, 179.0f);
    const float feature_dot = std::cos(
        clamped_feature_angle * 3.14159265358979323846f / 180.0f);
    for (const auto &entry : edge_faces)
    {
        const int first_vertex = static_cast<int>(entry.first >> 32U);
        const int second_vertex = static_cast<int>(entry.first & 0xffffffffU);
        const EdgeFaces &incident = entry.second;
        if (incident.count != 2)
        {
            protected_vertex[static_cast<std::size_t>(first_vertex)] = 1;
            protected_vertex[static_cast<std::size_t>(second_vertex)] = 1;
            continue;
        }
        const auto &first_normal = face_normals[static_cast<std::size_t>(incident.first)];
        const auto &second_normal = face_normals[static_cast<std::size_t>(incident.second)];
        const float normal_dot =
            first_normal[0] * second_normal[0] +
            first_normal[1] * second_normal[1] +
            first_normal[2] * second_normal[2];
        if (normal_dot < feature_dot)
        {
            ++sharp_edge_degree[static_cast<std::size_t>(first_vertex)];
            ++sharp_edge_degree[static_cast<std::size_t>(second_vertex)];
        }
    }
    for (std::size_t index = 0; index < sharp_edge_degree.size(); ++index)
    {
        // A coherent crease contributes at least two sharp incident edges. A
        // single sharp edge is usually a marching-cubes spike and remains
        // eligible for denoising.
        if (sharp_edge_degree[index] >= 2)
        {
            protected_vertex[index] = 1;
        }
    }
    for (int ring = 0; ring < std::max(0, boundaryProtectionRings); ++ring)
    {
        std::vector<std::uint8_t> expanded = protected_vertex;
        for (std::size_t index = 0; index < protected_vertex.size(); ++index)
        {
            if (!protected_vertex[index])
            {
                continue;
            }
            for (const int neighbor : neighbors[index])
            {
                expanded[static_cast<std::size_t>(neighbor)] = 1;
            }
        }
        protected_vertex = std::move(expanded);
    }

    lambda = std::clamp(lambda, 0.0f, 1.0f);
    mu = std::clamp(mu, -1.0f, 0.0f);
    const std::vector<MeshVertex> original = mesh->vertices;
    std::vector<std::uint8_t> moved(mesh->vertices.size(), 0);
    const auto apply_laplacian_step = [&](float factor)
    {
        const std::vector<MeshVertex> source = mesh->vertices;
        for (std::size_t index = 0; index < source.size(); ++index)
        {
            const auto &adjacent = neighbors[index];
            if (protected_vertex[index] || adjacent.size() < 3)
            {
                continue;
            }
            const MeshVertex &current = source[index];
            float mean_edge_length = 0.0f;
            for (const int neighbor_index : adjacent)
            {
                const MeshVertex &neighbor = source[static_cast<std::size_t>(neighbor_index)];
                const float dx = neighbor.x - current.x;
                const float dy = neighbor.y - current.y;
                const float dz = neighbor.z - current.z;
                mean_edge_length += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            mean_edge_length /= static_cast<float>(adjacent.size());
            const float maximum_neighbor_distance = mean_edge_length * 2.5f;
            float mean_x = 0.0f;
            float mean_y = 0.0f;
            float mean_z = 0.0f;
            int accepted_neighbor_count = 0;
            for (const int neighbor_index : adjacent)
            {
                const MeshVertex &neighbor = source[static_cast<std::size_t>(neighbor_index)];
                const float dx = neighbor.x - current.x;
                const float dy = neighbor.y - current.y;
                const float dz = neighbor.z - current.z;
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance > maximum_neighbor_distance)
                {
                    continue;
                }
                mean_x += neighbor.x;
                mean_y += neighbor.y;
                mean_z += neighbor.z;
                ++accepted_neighbor_count;
            }
            if (accepted_neighbor_count < 3)
            {
                continue;
            }
            const float inverse_count = 1.0f / static_cast<float>(accepted_neighbor_count);
            MeshVertex candidate = current;
            candidate.x += factor * (mean_x * inverse_count - current.x);
            candidate.y += factor * (mean_y * inverse_count - current.y);
            candidate.z += factor * (mean_z * inverse_count - current.z);

            const MeshVertex &origin = original[index];
            float offset_x = candidate.x - origin.x;
            float offset_y = candidate.y - origin.y;
            float offset_z = candidate.z - origin.z;
            const float displacement = std::sqrt(
                offset_x * offset_x + offset_y * offset_y + offset_z * offset_z);
            if (displacement > maximumDisplacement)
            {
                const float scale = maximumDisplacement / displacement;
                offset_x *= scale;
                offset_y *= scale;
                offset_z *= scale;
                candidate.x = origin.x + offset_x;
                candidate.y = origin.y + offset_y;
                candidate.z = origin.z + offset_z;
            }
            const float step_x = candidate.x - current.x;
            const float step_y = candidate.y - current.y;
            const float step_z = candidate.z - current.z;
            if (step_x * step_x + step_y * step_y + step_z * step_z > 1.0e-16f)
            {
                mesh->vertices[index].x = candidate.x;
                mesh->vertices[index].y = candidate.y;
                mesh->vertices[index].z = candidate.z;
                moved[index] = 1;
            }
        }
    };

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        apply_laplacian_step(lambda);
        apply_laplacian_step(mu);
    }
    return static_cast<int>(std::count(moved.cbegin(), moved.cend(), std::uint8_t{1}));
}

void simplifyVoxelMeshAdaptive(TriMesh *mesh,
                               const ReconstructionConfig &config,
                               float voxelStep)
{
    simplifyVoxelMeshAdaptive(mesh, config, voxelStep, false);
}

void simplifyVoxelMeshAdaptive(TriMesh *mesh,
                               const ReconstructionConfig &config,
                               float voxelStep,
                               bool preserveOpenBoundaries,
                               int minimumProtectedBoundaryVertices,
                               float maximumCollapsibleBoundaryDiameter,
                               float maximumNormalClusterAngleDegrees,
                               const std::vector<std::uint8_t> *
                                   protectedBoundaryVertices)
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
    std::vector<std::uint8_t> current_boundary_protection;
    if (protectedBoundaryVertices &&
        protectedBoundaryVertices->size() == mesh->vertices.size())
    {
        current_boundary_protection = *protectedBoundaryVertices;
    }
    float clusterSize = voxelStep * std::clamp(config.voxelSimplifyFactor, 1.0f, 4.0f);
    const int minFacesForComponent = std::max(24, config.minComponentFaces / 3);

    for (int iteration = 0; iteration < 8; ++iteration)
    {
        const int beforeFaces = current.hasFaces() ? static_cast<int>(current.faces()->rows()) : 0;
        if (preserveOpenBoundaries)
        {
            TriMesh boundary_aware_mesh;
            assignFromPlaMesh(current, &boundary_aware_mesh);
            std::vector<std::uint8_t> next_boundary_protection;
            boundary_aware_mesh = voxelClusterSimplifyPreservingOpenBoundaries(
                boundary_aware_mesh,
                std::max(clusterSize, voxelStep),
                minimumProtectedBoundaryVertices,
                maximumCollapsibleBoundaryDiameter,
                maximumNormalClusterAngleDegrees,
                current_boundary_protection.empty()
                    ? nullptr : &current_boundary_protection,
                current_boundary_protection.empty()
                    ? nullptr : &next_boundary_protection);
            current_boundary_protection = std::move(next_boundary_protection);
            current = toPlaMesh(boundary_aware_mesh);
        }
        else
        {
            current = plapoint::mesh::voxelClusterSimplify(
                current, std::max(clusterSize, voxelStep));
        }
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
