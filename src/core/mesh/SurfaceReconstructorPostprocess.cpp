#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

void keepLargestConnectedComponent(TriMesh *mesh)
{
    if (!mesh || mesh->faces.empty())
    {
        return;
    }

    std::unordered_map<std::uint64_t, std::vector<int>> edgeToFaces;
    edgeToFaces.reserve(mesh->faces.size() * 3);
    for (std::size_t faceIndex = 0; faceIndex < mesh->faces.size(); ++faceIndex)
    {
        const Triangle &face = mesh->faces[faceIndex];
        edgeToFaces[edgeKey(face.v[0], face.v[1])].push_back(static_cast<int>(faceIndex));
        edgeToFaces[edgeKey(face.v[1], face.v[2])].push_back(static_cast<int>(faceIndex));
        edgeToFaces[edgeKey(face.v[2], face.v[0])].push_back(static_cast<int>(faceIndex));
    }

    std::vector<std::vector<int>> adjacency(mesh->faces.size());
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

    std::vector<std::uint8_t> visited(mesh->faces.size(), 0);
    std::vector<int> largestComponent;
    for (std::size_t start = 0; start < mesh->faces.size(); ++start)
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

        if (component.size() > largestComponent.size())
        {
            largestComponent = std::move(component);
        }
    }

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

void removeDegenerateFaces(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }
    assignFromPlaMesh(plapoint::mesh::removeDegenerateFaces(toPlaMesh(*mesh), 5.0e-9f), mesh);
}

void removeSmallConnectedComponents(TriMesh *mesh, int minFaces)
{
    if (!mesh || mesh->faces.empty() || minFaces <= 1)
    {
        return;
    }
    PlaMesh filtered = plapoint::mesh::removeSmallConnectedComponents(
        toPlaMesh(*mesh),
        static_cast<std::size_t>(minFaces));
    if (filtered.hasFaces() && filtered.faces()->rows() > 0)
    {
        assignFromPlaMesh(filtered, mesh);
        return;
    }

    keepLargestConnectedComponent(mesh);
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
