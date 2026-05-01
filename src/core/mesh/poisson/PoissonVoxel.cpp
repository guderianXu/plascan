#include "PoissonVoxel.h"

#include "PoissonCommon.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <vector>

namespace xjw
{
namespace mesh
{
namespace poisson
{

namespace
{

using common::Vec3;
using common::index3D;
using common::inBounds3D;

struct SpatialGrid
{
    float minX = 0.0f;
    float minY = 0.0f;
    float minZ = 0.0f;
    float cellSize = 1.0f;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    std::vector<std::vector<int>> cells;
    const std::vector<detail::PointXYZRGB> *points = nullptr;

    void build(const std::vector<detail::PointXYZRGB> &inputPoints, float requestedCellSize)
    {
        points = &inputPoints;
        cellSize = requestedCellSize;
        if (inputPoints.empty())
        {
            return;
        }

        minX = inputPoints[0].x;
        minY = inputPoints[0].y;
        minZ = inputPoints[0].z;
        float maxX = minX;
        float maxY = minY;
        float maxZ = minZ;
        for (const auto &point : inputPoints)
        {
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
            minZ = std::min(minZ, point.z);
            maxZ = std::max(maxZ, point.z);
        }

        nx = std::max(1, static_cast<int>(std::ceil((maxX - minX) / cellSize)) + 1);
        ny = std::max(1, static_cast<int>(std::ceil((maxY - minY) / cellSize)) + 1);
        nz = std::max(1, static_cast<int>(std::ceil((maxZ - minZ) / cellSize)) + 1);
        cells.assign(static_cast<std::size_t>(nx) * ny * nz, {});

        for (int index = 0; index < static_cast<int>(inputPoints.size()); ++index)
        {
            const auto &point = inputPoints[static_cast<std::size_t>(index)];
            const int cx = std::clamp(static_cast<int>((point.x - minX) / cellSize), 0, nx - 1);
            const int cy = std::clamp(static_cast<int>((point.y - minY) / cellSize), 0, ny - 1);
            const int cz = std::clamp(static_cast<int>((point.z - minZ) / cellSize), 0, nz - 1);
            cells[(static_cast<std::size_t>(cz) * nx * ny) + cy * nx + cx].push_back(index);
        }
    }

    detail::PointXYZRGB nearest(const Vec3 &query) const
    {
        if (!points || points->empty())
        {
            return {};
        }

        const int cx = std::clamp(static_cast<int>((query.x - minX) / cellSize), 0, nx - 1);
        const int cy = std::clamp(static_cast<int>((query.y - minY) / cellSize), 0, ny - 1);
        const int cz = std::clamp(static_cast<int>((query.z - minZ) / cellSize), 0, nz - 1);

        float bestDistance2 = std::numeric_limits<float>::max();
        int bestIndex = 0;
        for (int radius = 0; radius <= std::max({nx, ny, nz}); ++radius)
        {
            bool found = false;
            const int x0 = std::max(0, cx - radius);
            const int x1 = std::min(nx - 1, cx + radius);
            const int y0 = std::max(0, cy - radius);
            const int y1 = std::min(ny - 1, cy + radius);
            const int z0 = std::max(0, cz - radius);
            const int z1 = std::min(nz - 1, cz + radius);

            for (int iz = z0; iz <= z1; ++iz)
            {
                for (int iy = y0; iy <= y1; ++iy)
                {
                    for (int ix = x0; ix <= x1; ++ix)
                    {
                        const bool shell = (radius == 0)
                                           || ix == x0 || ix == x1 || iy == y0 || iy == y1 || iz == z0 || iz == z1;
                        if (!shell)
                        {
                            continue;
                        }

                        for (int pointIndex : cells[(static_cast<std::size_t>(iz) * nx * ny) + iy * nx + ix])
                        {
                            const auto &point = (*points)[static_cast<std::size_t>(pointIndex)];
                            const float distance2 = (point.x - query.x) * (point.x - query.x)
                                                    + (point.y - query.y) * (point.y - query.y)
                                                    + (point.z - query.z) * (point.z - query.z);
                            if (distance2 < bestDistance2)
                            {
                                bestDistance2 = distance2;
                                bestIndex = pointIndex;
                                found = true;
                            }
                        }
                    }
                }
            }

            if (found)
            {
                break;
            }
        }

        return (*points)[static_cast<std::size_t>(bestIndex)];
    }
};

struct QuantizedVertexKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const QuantizedVertexKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedVertexKeyHash
{
    std::size_t operator()(const QuantizedVertexKey &key) const
    {
        std::size_t seed = 1469598103934665603ull;
        seed ^= static_cast<std::size_t>(key.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= static_cast<std::size_t>(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

Vec3 normalizeSafe(const Vec3 &v)
{
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f)
    {
        return Vec3{0.0f, 0.0f, 1.0f};
    }
    const float inv = 1.0f / len;
    return Vec3{v.x * inv, v.y * inv, v.z * inv};
}

Vec3 operator+(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator/(const Vec3 &v, float s)
{
    const float inv = 1.0f / std::max(1e-8f, s);
    return Vec3{v.x * inv, v.y * inv, v.z * inv};
}

Vec3 cross3(const Vec3 &a, const Vec3 &b)
{
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

int addOrGetInterpolatedVertex(std::unordered_map<QuantizedVertexKey, int, QuantizedVertexKeyHash> *vertexMap,
                               TriMesh *mesh,
                               const SpatialGrid &grid,
                               const VoxelGrid &voxelGrid,
                               const Vec3 &position)
{
    const float quant = std::max(1e-6f, voxelGrid.step * 0.09f);
    const float invQuant = 1.0f / quant;
    QuantizedVertexKey key;
    key.x = static_cast<int>(std::lround((position.x - voxelGrid.minX) * invQuant));
    key.y = static_cast<int>(std::lround((position.y - voxelGrid.minY) * invQuant));
    key.z = static_cast<int>(std::lround((position.z - voxelGrid.minZ) * invQuant));

    const auto it = vertexMap->find(key);
    if (it != vertexMap->end())
    {
        return it->second;
    }

    MeshVertex vertex;
    vertex.x = position.x;
    vertex.y = position.y;
    vertex.z = position.z;
    const detail::PointXYZRGB nearestPoint = grid.nearest(position);
    vertex.r = nearestPoint.r;
    vertex.g = nearestPoint.g;
    vertex.b = nearestPoint.b;
    vertex.nx = 0.0f;
    vertex.ny = 0.0f;
    vertex.nz = 1.0f;

    const int index = static_cast<int>(mesh->vertices.size());
    mesh->vertices.push_back(vertex);
    (*vertexMap)[key] = index;
    return index;
}

void orientTriangle(TriMesh *mesh, int *a, int *b, int *c, const Vec3 &desiredDir)
{
    const auto &va = mesh->vertices[static_cast<std::size_t>(*a)];
    const auto &vb = mesh->vertices[static_cast<std::size_t>(*b)];
    const auto &vc = mesh->vertices[static_cast<std::size_t>(*c)];
    const Vec3 p0{va.x, va.y, va.z};
    const Vec3 p1{vb.x, vb.y, vb.z};
    const Vec3 p2{vc.x, vc.y, vc.z};
    const Vec3 n = cross3(p1 - p0, p2 - p0);
    if (common::dot(n, desiredDir) < 0.0f)
    {
        std::swap(*b, *c);
    }
}

int addOrGetVertex(std::unordered_map<std::uint64_t, int> *vertexMap,
                   TriMesh *mesh,
                   const SpatialGrid &grid,
                   int vx,
                   int vy,
                   int vz,
                   const VoxelGrid &voxelGrid)
{
    const std::uint64_t key = (static_cast<std::uint64_t>(vx) << 42)
                              | (static_cast<std::uint64_t>(vy) << 21)
                              | static_cast<std::uint64_t>(vz);
    const auto it = vertexMap->find(key);
    if (it != vertexMap->end())
    {
        return it->second;
    }

    MeshVertex vertex;
    vertex.x = voxelGrid.minX + static_cast<float>(vx) * voxelGrid.step;
    vertex.y = voxelGrid.minY + static_cast<float>(vy) * voxelGrid.step;
    vertex.z = voxelGrid.minZ + static_cast<float>(vz) * voxelGrid.step;
    const detail::PointXYZRGB nearestPoint = grid.nearest(Vec3{vertex.x, vertex.y, vertex.z});
    vertex.r = nearestPoint.r;
    vertex.g = nearestPoint.g;
    vertex.b = nearestPoint.b;
    vertex.nx = 0.0f;
    vertex.ny = 0.0f;
    vertex.nz = 1.0f;

    const int index = static_cast<int>(mesh->vertices.size());
    mesh->vertices.push_back(vertex);
    (*vertexMap)[key] = index;
    return index;
}

void addFaceTriangles(TriMesh *mesh, int a, int b, int c, int d)
{
    Triangle first;
    first.v[0] = a;
    first.v[1] = b;
    first.v[2] = c;
    mesh->faces.push_back(first);

    Triangle second;
    second.v[0] = a;
    second.v[1] = c;
    second.v[2] = d;
    mesh->faces.push_back(second);
}

void extractIsoSurfaceMarchingTetra(const VoxelGrid &voxelGrid,
                                    const SpatialGrid &spatialGrid,
                                    TriMesh *mesh)
{
    static constexpr int kCubeCorners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    static constexpr int kTetrahedra[6][4] = {
        {0, 5, 1, 6},
        {0, 1, 2, 6},
        {0, 2, 3, 6},
        {0, 3, 7, 6},
        {0, 7, 4, 6},
        {0, 4, 5, 6}};
    static constexpr int kTetEdges[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    std::unordered_map<QuantizedVertexKey, int, QuantizedVertexKeyHash> vertexMap;
    vertexMap.reserve(static_cast<std::size_t>(voxelGrid.nx) * voxelGrid.ny);

    auto scalarAt = [&](int x, int y, int z) -> float {
        return voxelGrid.scalarField[index3D(x, y, z, voxelGrid.nx, voxelGrid.ny)];
    };

    const float iso = voxelGrid.isoLevel;
    const float minArea2 = voxelGrid.step * voxelGrid.step * 1.2e-3f;
    for (int z = 0; z < voxelGrid.nz - 1; ++z)
    {
        for (int y = 0; y < voxelGrid.ny - 1; ++y)
        {
            for (int x = 0; x < voxelGrid.nx - 1; ++x)
            {
                std::array<Vec3, 8> cornerPos;
                std::array<float, 8> cornerVal;
                bool hasBelow = false;
                bool hasAbove = false;

                for (int c = 0; c < 8; ++c)
                {
                    const int gx = x + kCubeCorners[c][0];
                    const int gy = y + kCubeCorners[c][1];
                    const int gz = z + kCubeCorners[c][2];
                    cornerPos[c] = Vec3{
                        voxelGrid.minX + static_cast<float>(gx) * voxelGrid.step,
                        voxelGrid.minY + static_cast<float>(gy) * voxelGrid.step,
                        voxelGrid.minZ + static_cast<float>(gz) * voxelGrid.step};
                    cornerVal[c] = scalarAt(gx, gy, gz);
                    hasBelow = hasBelow || (cornerVal[c] < iso);
                    hasAbove = hasAbove || (cornerVal[c] >= iso);
                }

                if (!(hasBelow && hasAbove))
                {
                    continue;
                }

                for (int t = 0; t < 6; ++t)
                {
                    std::array<Vec3, 4> p;
                    std::array<float, 4> v;
                    std::array<bool, 4> inside;
                    int insideCount = 0;
                    Vec3 insideCenter;
                    Vec3 outsideCenter;

                    for (int i = 0; i < 4; ++i)
                    {
                        const int corner = kTetrahedra[t][i];
                        p[i] = cornerPos[corner];
                        v[i] = cornerVal[corner];
                        inside[i] = (v[i] >= iso);
                        if (inside[i])
                        {
                            insideCenter = insideCenter + p[i];
                            ++insideCount;
                        }
                        else
                        {
                            outsideCenter = outsideCenter + p[i];
                        }
                    }

                    if (insideCount == 0 || insideCount == 4)
                    {
                        continue;
                    }

                    insideCenter = insideCenter / static_cast<float>(insideCount);
                    outsideCenter = outsideCenter / static_cast<float>(4 - insideCount);
                    const Vec3 desiredDir = normalizeSafe(outsideCenter - insideCenter);

                    std::vector<Vec3> intersections;
                    intersections.reserve(4);
                    for (int e = 0; e < 6; ++e)
                    {
                        const int a = kTetEdges[e][0];
                        const int b = kTetEdges[e][1];
                        if (inside[a] == inside[b])
                        {
                            continue;
                        }
                        const float denom = (v[b] - v[a]);
                        if (std::fabs(denom) < 1e-7f)
                        {
                            continue;
                        }
                        const float alpha = std::clamp((iso - v[a]) / denom, 0.0f, 1.0f);
                        intersections.push_back(p[a] + (p[b] - p[a]) * alpha);
                    }

                    if (intersections.size() < 3)
                    {
                        continue;
                    }

                    if (intersections.size() == 4)
                    {
                        const Vec3 centroid = (intersections[0] + intersections[1] + intersections[2] + intersections[3]) / 4.0f;
                        Vec3 axisA = cross3(desiredDir, Vec3{0.0f, 0.0f, 1.0f});
                        if (common::norm(axisA) < 1e-6f)
                        {
                            axisA = cross3(desiredDir, Vec3{0.0f, 1.0f, 0.0f});
                        }
                        axisA = normalizeSafe(axisA);
                        const Vec3 axisB = normalizeSafe(cross3(desiredDir, axisA));

                        std::sort(intersections.begin(), intersections.end(), [&](const Vec3 &lhs, const Vec3 &rhs) {
                            const Vec3 aRel = lhs - centroid;
                            const Vec3 bRel = rhs - centroid;
                            const float aAng = std::atan2(common::dot(aRel, axisB), common::dot(aRel, axisA));
                            const float bAng = std::atan2(common::dot(bRel, axisB), common::dot(bRel, axisA));
                            return aAng < bAng;
                        });
                    }

                    std::array<int, 4> ids = {-1, -1, -1, -1};
                    for (std::size_t i = 0; i < intersections.size(); ++i)
                    {
                        ids[i] = addOrGetInterpolatedVertex(&vertexMap, mesh, spatialGrid, voxelGrid, intersections[i]);
                    }

                    if (intersections.size() == 3)
                    {
                        int a = ids[0];
                        int b = ids[1];
                        int c = ids[2];
                        orientTriangle(mesh, &a, &b, &c, desiredDir);
                        Triangle triangle;
                        triangle.v[0] = a;
                        triangle.v[1] = b;
                        triangle.v[2] = c;
                        const auto &v0 = mesh->vertices[static_cast<std::size_t>(triangle.v[0])];
                        const auto &v1 = mesh->vertices[static_cast<std::size_t>(triangle.v[1])];
                        const auto &v2 = mesh->vertices[static_cast<std::size_t>(triangle.v[2])];
                        const Vec3 e1{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
                        const Vec3 e2{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
                        const Vec3 cross = cross3(e1, e2);
                        const float area2 = common::dot(cross, cross);
                        if (area2 >= minArea2)
                        {
                            mesh->faces.push_back(triangle);
                        }
                    }
                    else
                    {
                        int a0 = ids[0];
                        int b0 = ids[1];
                        int c0 = ids[2];
                        orientTriangle(mesh, &a0, &b0, &c0, desiredDir);
                        Triangle first;
                        first.v[0] = a0;
                        first.v[1] = b0;
                        first.v[2] = c0;
                        const auto &f0 = mesh->vertices[static_cast<std::size_t>(first.v[0])];
                        const auto &f1 = mesh->vertices[static_cast<std::size_t>(first.v[1])];
                        const auto &f2 = mesh->vertices[static_cast<std::size_t>(first.v[2])];
                        const Vec3 fe1{f1.x - f0.x, f1.y - f0.y, f1.z - f0.z};
                        const Vec3 fe2{f2.x - f0.x, f2.y - f0.y, f2.z - f0.z};
                        const float firstArea2 = common::dot(cross3(fe1, fe2), cross3(fe1, fe2));
                        if (firstArea2 >= minArea2)
                        {
                            mesh->faces.push_back(first);
                        }

                        int a1 = ids[0];
                        int b1 = ids[2];
                        int c1 = ids[3];
                        orientTriangle(mesh, &a1, &b1, &c1, desiredDir);
                        Triangle second;
                        second.v[0] = a1;
                        second.v[1] = b1;
                        second.v[2] = c1;
                        const auto &s0 = mesh->vertices[static_cast<std::size_t>(second.v[0])];
                        const auto &s1 = mesh->vertices[static_cast<std::size_t>(second.v[1])];
                        const auto &s2 = mesh->vertices[static_cast<std::size_t>(second.v[2])];
                        const Vec3 se1{s1.x - s0.x, s1.y - s0.y, s1.z - s0.z};
                        const Vec3 se2{s2.x - s0.x, s2.y - s0.y, s2.z - s0.z};
                        const float secondArea2 = common::dot(cross3(se1, se2), cross3(se1, se2));
                        if (secondArea2 >= minArea2)
                        {
                            mesh->faces.push_back(second);
                        }
                    }
                }
            }
        }
    }
}

} // namespace

bool PoissonVoxelPipeline::shouldUseVoxelReconstruction(const std::vector<detail::PointXYZRGB> &points) const
{
    if (points.size() < 1000)
    {
        return false;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;
    float minZ = points[0].z;
    float maxZ = points[0].z;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
        minZ = std::min(minZ, point.z);
        maxZ = std::max(maxZ, point.z);
    }

    const float spanX = std::max(1e-6f, maxX - minX);
    const float spanY = std::max(1e-6f, maxY - minY);
    const float spanZ = std::max(1e-6f, maxZ - minZ);
    const float maxSpan = std::max({spanX, spanY, spanZ});
    const float minSpan = std::min({spanX, spanY, spanZ});
    const float flatRatio = minSpan / std::max(1e-6f, maxSpan);
    const float zRatio = spanZ / std::max(spanX, spanY);

    return flatRatio > 0.22f && zRatio > 0.28f;
}

VoxelGrid PoissonVoxelPipeline::buildVoxelGrid(const std::vector<detail::PointXYZRGB> &points,
                                                const ReconstructionConfig &config) const
{
    const std::vector<cv::Vec3f> emptyNormals;
    return buildVoxelGrid(points, emptyNormals, config);
}

VoxelGrid PoissonVoxelPipeline::buildVoxelGrid(const std::vector<detail::PointXYZRGB> &points,
                                                const std::vector<cv::Vec3f> &normals,
                                                const ReconstructionConfig &config) const
{
    VoxelGrid voxelGrid;
    if (points.empty())
    {
        return voxelGrid;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;
    float minZ = points[0].z;
    float maxZ = points[0].z;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
        minZ = std::min(minZ, point.z);
        maxZ = std::max(maxZ, point.z);
    }

    float spanX = std::max(1e-6f, maxX - minX);
    float spanY = std::max(1e-6f, maxY - minY);
    float spanZ = std::max(1e-6f, maxZ - minZ);
    const float maxSpan = std::max({spanX, spanY, spanZ});

    minX -= spanX * config.padding;
    maxX += spanX * config.padding;
    minY -= spanY * config.padding;
    maxY += spanY * config.padding;
    minZ -= spanZ * config.padding;
    maxZ += spanZ * config.padding;
    spanX = std::max(1e-6f, maxX - minX);
    spanY = std::max(1e-6f, maxY - minY);
    spanZ = std::max(1e-6f, maxZ - minZ);

    const bool useNormals = !normals.empty() && normals.size() == points.size();

    int baseResolution = std::clamp(config.resolution, 96, 512);
    if (useNormals)
    {
        const int depth = std::clamp(config.poissonDepth, 7, 10);
        const int depthResolution = (1 << depth);
        baseResolution = std::clamp(baseResolution + std::max(0, config.poissonDepth - 7) * 32, 128, 768);
        const int densityResolution = std::clamp(static_cast<int>(std::cbrt(static_cast<double>(points.size())) * 3.4), 112, 768);
        baseResolution = std::max(baseResolution, depthResolution);
        baseResolution = std::max(baseResolution, densityResolution);
    }
    voxelGrid.nx = std::max(12, static_cast<int>(std::round(baseResolution * spanX / maxSpan))) + 2;
    voxelGrid.ny = std::max(12, static_cast<int>(std::round(baseResolution * spanY / maxSpan))) + 2;
    voxelGrid.nz = std::max(12, static_cast<int>(std::round(baseResolution * spanZ / maxSpan))) + 2;
    voxelGrid.minX = minX;
    voxelGrid.minY = minY;
    voxelGrid.minZ = minZ;
    voxelGrid.step = maxSpan / std::max(8, baseResolution - 1);
    voxelGrid.occupied.assign(static_cast<std::size_t>(voxelGrid.nx) * voxelGrid.ny * voxelGrid.nz, 0);

    const float pointWeight = std::clamp(config.poissonPointWeight, 0.0f, 8.0f);
    std::vector<float> scoreField;
    if (useNormals)
    {
        scoreField.assign(voxelGrid.occupied.size(), 0.0f);
    }

    auto accumulateScore = [&](int gx, int gy, int gz, float delta) {
        if (!inBounds3D(gx, gy, gz, voxelGrid.nx, voxelGrid.ny, voxelGrid.nz))
        {
            return;
        }
        scoreField[index3D(gx, gy, gz, voxelGrid.nx, voxelGrid.ny)] += delta;
    };

    const float centerScore = std::clamp(1.20f + pointWeight * 0.26f, 1.20f, 3.40f);
    const float neighborScore = std::clamp(0.22f + pointWeight * 0.06f, 0.18f, 0.90f);
    const float insideScore = std::clamp(0.50f + pointWeight * 0.24f, 0.45f, 2.40f);
    const float outsideScore = -std::clamp(0.40f + pointWeight * 0.15f, 0.35f, 1.80f);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto &point = points[i];
        const int cx = std::clamp(static_cast<int>(std::floor((point.x - voxelGrid.minX) / voxelGrid.step)), 1, voxelGrid.nx - 2);
        const int cy = std::clamp(static_cast<int>(std::floor((point.y - voxelGrid.minY) / voxelGrid.step)), 1, voxelGrid.ny - 2);
        const int cz = std::clamp(static_cast<int>(std::floor((point.z - voxelGrid.minZ) / voxelGrid.step)), 1, voxelGrid.nz - 2);

        if (useNormals)
        {
            accumulateScore(cx, cy, cz, centerScore);
            accumulateScore(cx - 1, cy, cz, neighborScore);
            accumulateScore(cx + 1, cy, cz, neighborScore);
            accumulateScore(cx, cy - 1, cz, neighborScore);
            accumulateScore(cx, cy + 1, cz, neighborScore);
            accumulateScore(cx, cy, cz - 1, neighborScore);
            accumulateScore(cx, cy, cz + 1, neighborScore);

            cv::Vec3f n = normals[i];
            const float nlen = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (nlen > 1e-6f)
            {
                n /= nlen;
                const float shift = voxelGrid.step * std::clamp(0.60f + pointWeight * 0.06f, 0.55f, 1.05f);

                const int insideX = std::clamp(static_cast<int>(std::floor((point.x - shift * n[0] - voxelGrid.minX) / voxelGrid.step)), 1, voxelGrid.nx - 2);
                const int insideY = std::clamp(static_cast<int>(std::floor((point.y - shift * n[1] - voxelGrid.minY) / voxelGrid.step)), 1, voxelGrid.ny - 2);
                const int insideZ = std::clamp(static_cast<int>(std::floor((point.z - shift * n[2] - voxelGrid.minZ) / voxelGrid.step)), 1, voxelGrid.nz - 2);

                const int outsideX = std::clamp(static_cast<int>(std::floor((point.x + shift * n[0] - voxelGrid.minX) / voxelGrid.step)), 1, voxelGrid.nx - 2);
                const int outsideY = std::clamp(static_cast<int>(std::floor((point.y + shift * n[1] - voxelGrid.minY) / voxelGrid.step)), 1, voxelGrid.ny - 2);
                const int outsideZ = std::clamp(static_cast<int>(std::floor((point.z + shift * n[2] - voxelGrid.minZ) / voxelGrid.step)), 1, voxelGrid.nz - 2);

                accumulateScore(insideX, insideY, insideZ, insideScore);
                accumulateScore(outsideX, outsideY, outsideZ, outsideScore);
            }
            continue;
        }

        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (std::abs(dx) + std::abs(dy) + std::abs(dz) > 1)
                    {
                        continue;
                    }
                    const int nx = cx + dx;
                    const int ny = cy + dy;
                    const int nz = cz + dz;
                    if (!inBounds3D(nx, ny, nz, voxelGrid.nx, voxelGrid.ny, voxelGrid.nz))
                    {
                        continue;
                    }
                    voxelGrid.occupied[index3D(nx, ny, nz, voxelGrid.nx, voxelGrid.ny)] = 1;
                }
            }
        }
    }

    if (useNormals)
    {
        // Smooth the signed score field before thresholding to reduce jagged triangles.
        std::vector<float> smoothed(scoreField.size(), 0.0f);
        const float smoothCenterWeight = std::clamp(0.72f - pointWeight * 0.03f, 0.56f, 0.74f);
        const float smoothNeighborWeight = (1.0f - smoothCenterWeight) / 6.0f;
        for (int z = 1; z < voxelGrid.nz - 1; ++z)
        {
            for (int y = 1; y < voxelGrid.ny - 1; ++y)
            {
                for (int x = 1; x < voxelGrid.nx - 1; ++x)
                {
                    const std::size_t center = index3D(x, y, z, voxelGrid.nx, voxelGrid.ny);
                    float value = scoreField[center] * smoothCenterWeight;
                    value += scoreField[index3D(x - 1, y, z, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    value += scoreField[index3D(x + 1, y, z, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    value += scoreField[index3D(x, y - 1, z, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    value += scoreField[index3D(x, y + 1, z, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    value += scoreField[index3D(x, y, z - 1, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    value += scoreField[index3D(x, y, z + 1, voxelGrid.nx, voxelGrid.ny)] * smoothNeighborWeight;
                    smoothed[center] = value;
                }
            }
        }
        scoreField.swap(smoothed);
        voxelGrid.scalarField = scoreField;

        std::vector<float> surfaceSamples;
        surfaceSamples.reserve(points.size());
        for (const auto &point : points)
        {
            const int sx = std::clamp(static_cast<int>(std::floor((point.x - voxelGrid.minX) / voxelGrid.step)), 1, voxelGrid.nx - 2);
            const int sy = std::clamp(static_cast<int>(std::floor((point.y - voxelGrid.minY) / voxelGrid.step)), 1, voxelGrid.ny - 2);
            const int sz = std::clamp(static_cast<int>(std::floor((point.z - voxelGrid.minZ) / voxelGrid.step)), 1, voxelGrid.nz - 2);
            surfaceSamples.push_back(scoreField[index3D(sx, sy, sz, voxelGrid.nx, voxelGrid.ny)]);
        }
        float isoThreshold = 0.6f;
        if (!surfaceSamples.empty())
        {
            const std::size_t mid = surfaceSamples.size() / 2;
            const std::size_t p80 = std::min(surfaceSamples.size() - 1,
                                             static_cast<std::size_t>(surfaceSamples.size() * 0.80f));
            using Diff = std::vector<float>::difference_type;
            std::nth_element(surfaceSamples.begin(), surfaceSamples.begin() + static_cast<Diff>(mid), surfaceSamples.end());
            const float medianSurface = surfaceSamples[mid];
            std::nth_element(surfaceSamples.begin(), surfaceSamples.begin() + static_cast<Diff>(p80), surfaceSamples.end());
            const float highSurface = surfaceSamples[p80];
            const float blendedSurface = medianSurface * 0.30f + highSurface * 0.20f;
            const float weightBias = std::clamp(1.0f + (pointWeight - 4.0f) * 0.04f, 0.86f, 1.18f);
            isoThreshold = std::clamp(blendedSurface * weightBias, 0.10f, 1.50f);
        }

        voxelGrid.isoLevel = isoThreshold;
        for (std::size_t idx = 0; idx < scoreField.size(); ++idx)
        {
            voxelGrid.occupied[idx] = scoreField[idx] > isoThreshold ? 1 : 0;
        }
    }

    std::vector<std::uint8_t> relaxed = voxelGrid.occupied;
    for (int z = 1; z < voxelGrid.nz - 1; ++z)
    {
        for (int y = 1; y < voxelGrid.ny - 1; ++y)
        {
            for (int x = 1; x < voxelGrid.nx - 1; ++x)
            {
                const std::size_t center = index3D(x, y, z, voxelGrid.nx, voxelGrid.ny);
                int n6 = 0;
                n6 += voxelGrid.occupied[index3D(x - 1, y, z, voxelGrid.nx, voxelGrid.ny)] != 0;
                n6 += voxelGrid.occupied[index3D(x + 1, y, z, voxelGrid.nx, voxelGrid.ny)] != 0;
                n6 += voxelGrid.occupied[index3D(x, y - 1, z, voxelGrid.nx, voxelGrid.ny)] != 0;
                n6 += voxelGrid.occupied[index3D(x, y + 1, z, voxelGrid.nx, voxelGrid.ny)] != 0;
                n6 += voxelGrid.occupied[index3D(x, y, z - 1, voxelGrid.nx, voxelGrid.ny)] != 0;
                n6 += voxelGrid.occupied[index3D(x, y, z + 1, voxelGrid.nx, voxelGrid.ny)] != 0;

                const int addThreshold = useNormals ? 5 : 4;
                const int removeThreshold = useNormals ? 0 : 1;

                if (voxelGrid.occupied[center] == 0 && n6 >= addThreshold)
                {
                    relaxed[center] = 1;
                }
                else if (voxelGrid.occupied[center] != 0 && n6 <= removeThreshold)
                {
                    relaxed[center] = 0;
                }
            }
        }
    }
    voxelGrid.occupied.swap(relaxed);

    std::vector<std::uint8_t> outside(voxelGrid.occupied.size(), 0);
    std::deque<std::array<int, 3>> queue;
    auto tryPushOutside = [&](int x, int y, int z) {
        const std::size_t index = index3D(x, y, z, voxelGrid.nx, voxelGrid.ny);
        if (outside[index] != 0 || voxelGrid.occupied[index] != 0)
        {
            return;
        }
        outside[index] = 1;
        queue.push_back({x, y, z});
    };

    for (int z = 0; z < voxelGrid.nz; ++z)
    {
        for (int y = 0; y < voxelGrid.ny; ++y)
        {
            tryPushOutside(0, y, z);
            tryPushOutside(voxelGrid.nx - 1, y, z);
        }
    }
    for (int z = 0; z < voxelGrid.nz; ++z)
    {
        for (int x = 0; x < voxelGrid.nx; ++x)
        {
            tryPushOutside(x, 0, z);
            tryPushOutside(x, voxelGrid.ny - 1, z);
        }
    }
    for (int y = 0; y < voxelGrid.ny; ++y)
    {
        for (int x = 0; x < voxelGrid.nx; ++x)
        {
            tryPushOutside(x, y, 0);
            tryPushOutside(x, y, voxelGrid.nz - 1);
        }
    }

    const int offsets[6][3] = {
        {-1, 0, 0}, {1, 0, 0},
        {0, -1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}};
    while (!queue.empty())
    {
        const auto current = queue.front();
        queue.pop_front();
        for (const auto &offset : offsets)
        {
            const int nx = current[0] + offset[0];
            const int ny = current[1] + offset[1];
            const int nz = current[2] + offset[2];
            if (!inBounds3D(nx, ny, nz, voxelGrid.nx, voxelGrid.ny, voxelGrid.nz))
            {
                continue;
            }
            const std::size_t nIdx = index3D(nx, ny, nz, voxelGrid.nx, voxelGrid.ny);
            if (outside[nIdx] != 0 || voxelGrid.occupied[nIdx] != 0)
            {
                continue;
            }
            outside[nIdx] = 1;
            queue.push_back({nx, ny, nz});
        }
    }

    // For normal-guided reconstruction we keep shells open to avoid over-filled solids.
    if (!useNormals)
    {
        for (int z = 1; z < voxelGrid.nz - 1; ++z)
        {
            for (int y = 1; y < voxelGrid.ny - 1; ++y)
            {
                for (int x = 1; x < voxelGrid.nx - 1; ++x)
                {
                    const std::size_t index = index3D(x, y, z, voxelGrid.nx, voxelGrid.ny);
                    if (voxelGrid.occupied[index] == 0 && outside[index] == 0)
                    {
                        voxelGrid.occupied[index] = 1;
                    }
                }
            }
        }
    }

    return voxelGrid;
}

void PoissonVoxelPipeline::voxelGridToMesh(const VoxelGrid &voxelGrid,
                                           const std::vector<detail::PointXYZRGB> &points,
                                           TriMesh *mesh) const
{
    if (!mesh)
    {
        return;
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    if (!voxelGrid.valid())
    {
        return;
    }

    SpatialGrid spatialGrid;
    spatialGrid.build(points, voxelGrid.step * 1.8f);
    if (voxelGrid.hasScalarField())
    {
        extractIsoSurfaceMarchingTetra(voxelGrid, spatialGrid, mesh);
        return;
    }

    std::unordered_map<std::uint64_t, int> vertexMap;
    vertexMap.reserve(static_cast<std::size_t>(voxelGrid.nx) * voxelGrid.ny);

    auto isOccupied = [&](int x, int y, int z) {
        if (!inBounds3D(x, y, z, voxelGrid.nx, voxelGrid.ny, voxelGrid.nz))
        {
            return false;
        }
        return voxelGrid.occupied[index3D(x, y, z, voxelGrid.nx, voxelGrid.ny)] != 0;
    };

    for (int z = 1; z < voxelGrid.nz - 1; ++z)
    {
        for (int y = 1; y < voxelGrid.ny - 1; ++y)
        {
            for (int x = 1; x < voxelGrid.nx - 1; ++x)
            {
                if (!isOccupied(x, y, z))
                {
                    continue;
                }

                const int corner[8] = {
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x, y, z, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x + 1, y, z, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x + 1, y + 1, z, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x, y + 1, z, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x, y, z + 1, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x + 1, y, z + 1, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x + 1, y + 1, z + 1, voxelGrid),
                    addOrGetVertex(&vertexMap, mesh, spatialGrid, x, y + 1, z + 1, voxelGrid)};

                if (!isOccupied(x - 1, y, z))
                {
                    addFaceTriangles(mesh, corner[0], corner[7], corner[3], corner[4]);
                }
                if (!isOccupied(x + 1, y, z))
                {
                    addFaceTriangles(mesh, corner[1], corner[2], corner[6], corner[5]);
                }
                if (!isOccupied(x, y - 1, z))
                {
                    addFaceTriangles(mesh, corner[0], corner[1], corner[5], corner[4]);
                }
                if (!isOccupied(x, y + 1, z))
                {
                    addFaceTriangles(mesh, corner[3], corner[7], corner[6], corner[2]);
                }
                if (!isOccupied(x, y, z - 1))
                {
                    addFaceTriangles(mesh, corner[0], corner[3], corner[2], corner[1]);
                }
                if (!isOccupied(x, y, z + 1))
                {
                    addFaceTriangles(mesh, corner[4], corner[5], corner[6], corner[7]);
                }
            }
        }
    }
}

} // namespace poisson
} // namespace mesh
} // namespace xjw
