#include "SurfaceReconstructorPostprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct VertexClusterKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VertexClusterKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VertexClusterKeyHash
{
    std::size_t operator()(const VertexClusterKey &key) const
    {
        const std::size_t h1 = std::hash<int>{}(key.x);
        const std::size_t h2 = std::hash<int>{}(key.y);
        const std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 7);
    }
};

float dot(const Vec3 &left, const Vec3 &right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

float norm(const Vec3 &value)
{
    return std::sqrt(dot(value, value));
}

Vec3 normalize(const Vec3 &value)
{
    const float length = norm(value);
    if (length < 1e-12f)
    {
        return {0.0f, 0.0f, 1.0f};
    }
    return {value.x / length, value.y / length, value.z / length};
}

void simplifyMeshByVertexClustering(TriMesh *mesh, float clusterSize)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() || clusterSize <= 1e-6f)
    {
        return;
    }

    float minX = mesh->vertices[0].x;
    float minY = mesh->vertices[0].y;
    float minZ = mesh->vertices[0].z;
    for (const MeshVertex &vertex : mesh->vertices)
    {
        minX = std::min(minX, vertex.x);
        minY = std::min(minY, vertex.y);
        minZ = std::min(minZ, vertex.z);
    }

    struct ClusterAccumulator
    {
        double sumX = 0.0;
        double sumY = 0.0;
        double sumZ = 0.0;
        double sumR = 0.0;
        double sumG = 0.0;
        double sumB = 0.0;
        int count = 0;
        int newIndex = -1;
    };

    std::unordered_map<VertexClusterKey, ClusterAccumulator, VertexClusterKeyHash> clusters;
    clusters.reserve(mesh->vertices.size());

    std::vector<int> vertexRemap(mesh->vertices.size(), -1);
    for (int index = 0; index < static_cast<int>(mesh->vertices.size()); ++index)
    {
        const MeshVertex &vertex = mesh->vertices[static_cast<std::size_t>(index)];
        VertexClusterKey key;
        key.x = static_cast<int>(std::floor((vertex.x - minX) / clusterSize));
        key.y = static_cast<int>(std::floor((vertex.y - minY) / clusterSize));
        key.z = static_cast<int>(std::floor((vertex.z - minZ) / clusterSize));

        ClusterAccumulator &cluster = clusters[key];
        cluster.sumX += vertex.x;
        cluster.sumY += vertex.y;
        cluster.sumZ += vertex.z;
        cluster.sumR += vertex.r;
        cluster.sumG += vertex.g;
        cluster.sumB += vertex.b;
        ++cluster.count;
    }

    std::vector<MeshVertex> simplifiedVertices;
    simplifiedVertices.reserve(clusters.size());
    for (auto &entry : clusters)
    {
        ClusterAccumulator &cluster = entry.second;
        if (cluster.count <= 0)
        {
            continue;
        }

        MeshVertex vertex;
        const double invCount = 1.0 / static_cast<double>(cluster.count);
        vertex.x = static_cast<float>(cluster.sumX * invCount);
        vertex.y = static_cast<float>(cluster.sumY * invCount);
        vertex.z = static_cast<float>(cluster.sumZ * invCount);
        vertex.r = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(cluster.sumR * invCount)), 0, 255));
        vertex.g = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(cluster.sumG * invCount)), 0, 255));
        vertex.b = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(cluster.sumB * invCount)), 0, 255));
        vertex.nx = 0.0f;
        vertex.ny = 0.0f;
        vertex.nz = 1.0f;

        cluster.newIndex = static_cast<int>(simplifiedVertices.size());
        simplifiedVertices.push_back(vertex);
    }

    for (int index = 0; index < static_cast<int>(mesh->vertices.size()); ++index)
    {
        const MeshVertex &vertex = mesh->vertices[static_cast<std::size_t>(index)];
        VertexClusterKey key;
        key.x = static_cast<int>(std::floor((vertex.x - minX) / clusterSize));
        key.y = static_cast<int>(std::floor((vertex.y - minY) / clusterSize));
        key.z = static_cast<int>(std::floor((vertex.z - minZ) / clusterSize));

        auto it = clusters.find(key);
        if (it != clusters.end())
        {
            vertexRemap[static_cast<std::size_t>(index)] = it->second.newIndex;
        }
    }

    std::vector<Triangle> simplifiedFaces;
    simplifiedFaces.reserve(mesh->faces.size());
    for (const Triangle &face : mesh->faces)
    {
        Triangle mapped;
        mapped.v[0] = vertexRemap[static_cast<std::size_t>(face.v[0])];
        mapped.v[1] = vertexRemap[static_cast<std::size_t>(face.v[1])];
        mapped.v[2] = vertexRemap[static_cast<std::size_t>(face.v[2])];
        if (mapped.v[0] < 0 || mapped.v[1] < 0 || mapped.v[2] < 0)
        {
            continue;
        }
        if (mapped.v[0] == mapped.v[1] || mapped.v[1] == mapped.v[2] || mapped.v[0] == mapped.v[2])
        {
            continue;
        }
        simplifiedFaces.push_back(mapped);
    }

    mesh->vertices.swap(simplifiedVertices);
    mesh->faces.swap(simplifiedFaces);
}

void compactMeshVertices(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    std::vector<int> remap(mesh->vertices.size(), -1);
    std::vector<MeshVertex> compactVertices;
    compactVertices.reserve(mesh->vertices.size());

    for (Triangle &face : mesh->faces)
    {
        for (int &vertexIndex : face.v)
        {
            if (remap[static_cast<std::size_t>(vertexIndex)] < 0)
            {
                remap[static_cast<std::size_t>(vertexIndex)] = static_cast<int>(compactVertices.size());
                compactVertices.push_back(mesh->vertices[static_cast<std::size_t>(vertexIndex)]);
            }
            vertexIndex = remap[static_cast<std::size_t>(vertexIndex)];
        }
    }

    mesh->vertices.swap(compactVertices);
}

} // namespace

void removeDegenerateFaces(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    std::vector<Triangle> filtered;
    filtered.reserve(mesh->faces.size());
    for (const Triangle &face : mesh->faces)
    {
        if (face.v[0] == face.v[1] || face.v[1] == face.v[2] || face.v[0] == face.v[2])
        {
            continue;
        }

        const MeshVertex &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const MeshVertex &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const MeshVertex &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];
        const Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
        const Vec3 n{e1.y * e2.z - e1.z * e2.y,
                     e1.z * e2.x - e1.x * e2.z,
                     e1.x * e2.y - e1.y * e2.x};
        if (norm(n) < 1e-8f)
        {
            continue;
        }

        filtered.push_back(face);
    }
    mesh->faces.swap(filtered);
}

void removeSmallConnectedComponents(TriMesh *mesh, int minFaces)
{
    if (!mesh || mesh->faces.empty() || minFaces <= 1)
    {
        return;
    }

    std::unordered_map<std::uint64_t, std::vector<int>> edgeToFaces;
    edgeToFaces.reserve(mesh->faces.size() * 3);
    auto makeEdgeKey = [](int a, int b) {
        const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
        const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
        return (static_cast<std::uint64_t>(lo) << 32) | hi;
    };

    for (int faceIndex = 0; faceIndex < static_cast<int>(mesh->faces.size()); ++faceIndex)
    {
        const Triangle &face = mesh->faces[static_cast<std::size_t>(faceIndex)];
        edgeToFaces[makeEdgeKey(face.v[0], face.v[1])].push_back(faceIndex);
        edgeToFaces[makeEdgeKey(face.v[1], face.v[2])].push_back(faceIndex);
        edgeToFaces[makeEdgeKey(face.v[2], face.v[0])].push_back(faceIndex);
    }

    std::vector<std::vector<int>> adjacency(mesh->faces.size());
    for (const auto &entry : edgeToFaces)
    {
        const std::vector<int> &faces = entry.second;
        for (std::size_t i = 0; i < faces.size(); ++i)
        {
            for (std::size_t j = i + 1; j < faces.size(); ++j)
            {
                adjacency[static_cast<std::size_t>(faces[i])].push_back(faces[j]);
                adjacency[static_cast<std::size_t>(faces[j])].push_back(faces[i]);
            }
        }
    }

    std::vector<uint8_t> visited(mesh->faces.size(), 0);
    std::vector<Triangle> keptFaces;
    keptFaces.reserve(mesh->faces.size());
    std::deque<int> queue;
    std::vector<int> componentFaces;

    for (int seed = 0; seed < static_cast<int>(mesh->faces.size()); ++seed)
    {
        if (visited[static_cast<std::size_t>(seed)] != 0)
        {
            continue;
        }

        componentFaces.clear();
        queue.clear();
        queue.push_back(seed);
        visited[static_cast<std::size_t>(seed)] = 1;

        while (!queue.empty())
        {
            const int current = queue.front();
            queue.pop_front();
            componentFaces.push_back(current);
            for (int next : adjacency[static_cast<std::size_t>(current)])
            {
                if (visited[static_cast<std::size_t>(next)] == 0)
                {
                    visited[static_cast<std::size_t>(next)] = 1;
                    queue.push_back(next);
                }
            }
        }

        if (static_cast<int>(componentFaces.size()) < minFaces)
        {
            continue;
        }

        for (int faceIndex : componentFaces)
        {
            keptFaces.push_back(mesh->faces[static_cast<std::size_t>(faceIndex)]);
        }
    }

    mesh->faces.swap(keptFaces);
    compactMeshVertices(mesh);
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

    float clusterSize = voxelStep * std::clamp(config.voxelSimplifyFactor, 1.0f, 4.0f);
    const int minFacesForComponent = std::max(24, config.minComponentFaces / 3);

    for (int iteration = 0; iteration < 8; ++iteration)
    {
        const int beforeFaces = mesh->faceCount();
        simplifyMeshByVertexClustering(mesh, std::max(clusterSize, voxelStep));
        removeDegenerateFaces(mesh);
        removeSmallConnectedComponents(mesh, minFacesForComponent);
        if (mesh->empty())
        {
            break;
        }

        const int afterFaces = mesh->faceCount();
        if (afterFaces <= targetFaceCount)
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
}

void taubinSmooth(TriMesh *mesh, int iterations, float lambda)
{
    if (!mesh || mesh->vertices.empty() || mesh->faces.empty() || iterations <= 0)
    {
        return;
    }

    lambda = std::clamp(lambda, 0.0f, 1.0f);
    const float mu = -std::clamp(lambda * 0.53f, 0.01f, 0.53f);

    const int vertexCount = static_cast<int>(mesh->vertices.size());
    std::vector<int> adjacencyData;
    std::vector<int> adjacencyBegin(static_cast<std::size_t>(vertexCount) + 1, 0);

    {
        std::vector<int> degree(vertexCount, 0);
        for (const auto &face : mesh->faces)
        {
            degree[static_cast<std::size_t>(face.v[0])] += 2;
            degree[static_cast<std::size_t>(face.v[1])] += 2;
            degree[static_cast<std::size_t>(face.v[2])] += 2;
        }
        adjacencyBegin[0] = 0;
        for (int index = 0; index < vertexCount; ++index)
        {
            adjacencyBegin[static_cast<std::size_t>(index + 1)] = adjacencyBegin[static_cast<std::size_t>(index)] + degree[static_cast<std::size_t>(index)];
        }
        adjacencyData.resize(static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(vertexCount)]));
        std::fill(degree.begin(), degree.end(), 0);

        for (const auto &face : mesh->faces)
        {
            const int a = face.v[0];
            const int b = face.v[1];
            const int c = face.v[2];
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(a)] + degree[static_cast<std::size_t>(a)]++)] = b;
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(a)] + degree[static_cast<std::size_t>(a)]++)] = c;
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(b)] + degree[static_cast<std::size_t>(b)]++)] = a;
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(b)] + degree[static_cast<std::size_t>(b)]++)] = c;
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(c)] + degree[static_cast<std::size_t>(c)]++)] = a;
            adjacencyData[static_cast<std::size_t>(adjacencyBegin[static_cast<std::size_t>(c)] + degree[static_cast<std::size_t>(c)]++)] = b;
        }
    }

    std::vector<Vec3> newPositions(static_cast<std::size_t>(vertexCount));
    const auto smoothPass = [&](float weight) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int index = 0; index < vertexCount; ++index)
        {
            const int begin = adjacencyBegin[static_cast<std::size_t>(index)];
            const int end = adjacencyBegin[static_cast<std::size_t>(index + 1)];
            if (begin == end)
            {
                newPositions[static_cast<std::size_t>(index)] = {mesh->vertices[static_cast<std::size_t>(index)].x,
                                                                mesh->vertices[static_cast<std::size_t>(index)].y,
                                                                mesh->vertices[static_cast<std::size_t>(index)].z};
                continue;
            }

            float cx = 0.0f;
            float cy = 0.0f;
            float cz = 0.0f;
            for (int adjacencyIndex = begin; adjacencyIndex < end; ++adjacencyIndex)
            {
                const int neighbor = adjacencyData[static_cast<std::size_t>(adjacencyIndex)];
                cx += mesh->vertices[static_cast<std::size_t>(neighbor)].x;
                cy += mesh->vertices[static_cast<std::size_t>(neighbor)].y;
                cz += mesh->vertices[static_cast<std::size_t>(neighbor)].z;
            }

            const float inv = 1.0f / static_cast<float>(end - begin);
            cx *= inv;
            cy *= inv;
            cz *= inv;
            newPositions[static_cast<std::size_t>(index)].x = mesh->vertices[static_cast<std::size_t>(index)].x + (cx - mesh->vertices[static_cast<std::size_t>(index)].x) * weight;
            newPositions[static_cast<std::size_t>(index)].y = mesh->vertices[static_cast<std::size_t>(index)].y + (cy - mesh->vertices[static_cast<std::size_t>(index)].y) * weight;
            newPositions[static_cast<std::size_t>(index)].z = mesh->vertices[static_cast<std::size_t>(index)].z + (cz - mesh->vertices[static_cast<std::size_t>(index)].z) * weight;
        }

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int index = 0; index < vertexCount; ++index)
        {
            mesh->vertices[static_cast<std::size_t>(index)].x = newPositions[static_cast<std::size_t>(index)].x;
            mesh->vertices[static_cast<std::size_t>(index)].y = newPositions[static_cast<std::size_t>(index)].y;
            mesh->vertices[static_cast<std::size_t>(index)].z = newPositions[static_cast<std::size_t>(index)].z;
        }
    };

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        smoothPass(lambda);
        smoothPass(mu);
    }
}

void recomputeNormals(TriMesh *mesh)
{
    if (!mesh)
    {
        return;
    }

    const int vertexCount = static_cast<int>(mesh->vertices.size());
    const int faceCount = static_cast<int>(mesh->faces.size());

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int index = 0; index < vertexCount; ++index)
    {
        mesh->vertices[static_cast<std::size_t>(index)].nx = 0.0f;
        mesh->vertices[static_cast<std::size_t>(index)].ny = 0.0f;
        mesh->vertices[static_cast<std::size_t>(index)].nz = 0.0f;
    }

    std::vector<float> fnx(static_cast<std::size_t>(vertexCount), 0.0f);
    std::vector<float> fny(static_cast<std::size_t>(vertexCount), 0.0f);
    std::vector<float> fnz(static_cast<std::size_t>(vertexCount), 0.0f);
    for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex)
    {
        const auto &face = mesh->faces[static_cast<std::size_t>(faceIndex)];
        const auto &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const auto &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const auto &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];
        const Vec3 e1{b.x - a.x, b.y - a.y, b.z - a.z};
        const Vec3 e2{c.x - a.x, c.y - a.y, c.z - a.z};
        const float nx = e1.y * e2.z - e1.z * e2.y;
        const float ny = e1.z * e2.x - e1.x * e2.z;
        const float nz = e1.x * e2.y - e1.y * e2.x;
        fnx[static_cast<std::size_t>(face.v[0])] += nx;
        fny[static_cast<std::size_t>(face.v[0])] += ny;
        fnz[static_cast<std::size_t>(face.v[0])] += nz;
        fnx[static_cast<std::size_t>(face.v[1])] += nx;
        fny[static_cast<std::size_t>(face.v[1])] += ny;
        fnz[static_cast<std::size_t>(face.v[1])] += nz;
        fnx[static_cast<std::size_t>(face.v[2])] += nx;
        fny[static_cast<std::size_t>(face.v[2])] += ny;
        fnz[static_cast<std::size_t>(face.v[2])] += nz;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int index = 0; index < vertexCount; ++index)
    {
        const Vec3 normal = normalize({fnx[static_cast<std::size_t>(index)],
                                       fny[static_cast<std::size_t>(index)],
                                       fnz[static_cast<std::size_t>(index)]});
        mesh->vertices[static_cast<std::size_t>(index)].nx = normal.x;
        mesh->vertices[static_cast<std::size_t>(index)].ny = normal.y;
        mesh->vertices[static_cast<std::size_t>(index)].nz = normal.z;
    }

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (int index = 0; index < vertexCount; ++index)
    {
        cx += mesh->vertices[static_cast<std::size_t>(index)].x;
        cy += mesh->vertices[static_cast<std::size_t>(index)].y;
        cz += mesh->vertices[static_cast<std::size_t>(index)].z;
    }
    cx /= std::max(1, vertexCount);
    cy /= std::max(1, vertexCount);
    cz /= std::max(1, vertexCount);

    double dotSum = 0.0;
    for (int index = 0; index < vertexCount; ++index)
    {
        const auto &vertex = mesh->vertices[static_cast<std::size_t>(index)];
        dotSum += static_cast<double>(vertex.nx) * (vertex.x - cx)
                + static_cast<double>(vertex.ny) * (vertex.y - cy)
                + static_cast<double>(vertex.nz) * (vertex.z - cz);
    }

    if (dotSum < 0.0)
    {
        for (int index = 0; index < vertexCount; ++index)
        {
            mesh->vertices[static_cast<std::size_t>(index)].nx = -mesh->vertices[static_cast<std::size_t>(index)].nx;
            mesh->vertices[static_cast<std::size_t>(index)].ny = -mesh->vertices[static_cast<std::size_t>(index)].ny;
            mesh->vertices[static_cast<std::size_t>(index)].nz = -mesh->vertices[static_cast<std::size_t>(index)].nz;
        }
        for (int faceIndex = 0; faceIndex < faceCount; ++faceIndex)
        {
            std::swap(mesh->faces[static_cast<std::size_t>(faceIndex)].v[1],
                      mesh->faces[static_cast<std::size_t>(faceIndex)].v[2]);
        }
    }
}

} // namespace detail
} // namespace mesh
} // namespace xjw
