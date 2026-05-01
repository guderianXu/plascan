#include "PoissonLite.h"

#include "../SurfaceReconstructorPostprocess.h"
#include "PoissonCommon.h"
#include "PoissonVoxel.h"

#include <algorithm>
#include <array>
#include <cmath>
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

struct PointSupportGrid
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
        cellSize = std::max(1e-6f, requestedCellSize);
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
            cells[common::index3D(cx, cy, cz, nx, ny)].push_back(index);
        }
    }

    float nearestDistanceSq(float x, float y, float z) const
    {
        if (!points || points->empty())
        {
            return std::numeric_limits<float>::max();
        }

        const int cx = std::clamp(static_cast<int>((x - minX) / cellSize), 0, nx - 1);
        const int cy = std::clamp(static_cast<int>((y - minY) / cellSize), 0, ny - 1);
        const int cz = std::clamp(static_cast<int>((z - minZ) / cellSize), 0, nz - 1);

        float bestDistance2 = std::numeric_limits<float>::max();
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

                        for (int pointIndex : cells[common::index3D(ix, iy, iz, nx, ny)])
                        {
                            const auto &point = (*points)[static_cast<std::size_t>(pointIndex)];
                            const float dx = point.x - x;
                            const float dy = point.y - y;
                            const float dz = point.z - z;
                            const float d2 = dx * dx + dy * dy + dz * dz;
                            if (d2 < bestDistance2)
                            {
                                bestDistance2 = d2;
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

        return bestDistance2;
    }
};

void trimUnsupportedFaces(const std::vector<detail::PointXYZRGB> &points,
                          float voxelStep,
                          TriMesh *mesh)
{
    if (!mesh || mesh->empty() || points.empty())
    {
        return;
    }

    PointSupportGrid grid;
    grid.build(points, voxelStep * 2.0f);

    std::vector<float> sampleDistances;
    sampleDistances.reserve(mesh->vertices.size());
    for (const auto &vertex : mesh->vertices)
    {
        sampleDistances.push_back(std::sqrt(grid.nearestDistanceSq(vertex.x, vertex.y, vertex.z)));
    }
    float adaptiveThreshold = voxelStep * 1.6f;
    if (!sampleDistances.empty())
    {
        const std::size_t p90Index = static_cast<std::size_t>(sampleDistances.size() * 0.90f);
        std::nth_element(sampleDistances.begin(), sampleDistances.begin() + std::min(p90Index, sampleDistances.size() - 1), sampleDistances.end());
        const float p90 = sampleDistances[std::min(p90Index, sampleDistances.size() - 1)];
        adaptiveThreshold = std::clamp(p90 * 1.35f, voxelStep * 1.3f, voxelStep * 2.8f);
    }
    const float distThresholdSq = adaptiveThreshold * adaptiveThreshold;
    std::vector<Triangle> keptFaces;
    keptFaces.reserve(mesh->faces.size());

    for (const auto &face : mesh->faces)
    {
        const auto &v0 = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const auto &v1 = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const auto &v2 = mesh->vertices[static_cast<std::size_t>(face.v[2])];

        const float d0 = grid.nearestDistanceSq(v0.x, v0.y, v0.z);
        const float d1 = grid.nearestDistanceSq(v1.x, v1.y, v1.z);
        const float d2 = grid.nearestDistanceSq(v2.x, v2.y, v2.z);

        // COLMAP's trimmer removes low-confidence surfaces.
        // Here we approximate it by removing faces whose all vertices lack point support.
        if (d0 > distThresholdSq && d1 > distThresholdSq && d2 > distThresholdSq)
        {
            continue;
        }

        keptFaces.push_back(face);
    }

    mesh->faces.swap(keptFaces);
    const int dynamicMinComponent = std::clamp(static_cast<int>(mesh->faceCount() / 1200), 64, 700);
    detail::removeSmallConnectedComponents(mesh, dynamicMinComponent);
}

float sampleScalarNearest(const VoxelGrid &voxelGrid, const MeshVertex &vertex)
{
    if (!voxelGrid.hasScalarField())
    {
        return 0.0f;
    }

    const int gx = std::clamp(static_cast<int>(std::floor((vertex.x - voxelGrid.minX) / voxelGrid.step)),
                              0,
                              voxelGrid.nx - 1);
    const int gy = std::clamp(static_cast<int>(std::floor((vertex.y - voxelGrid.minY) / voxelGrid.step)),
                              0,
                              voxelGrid.ny - 1);
    const int gz = std::clamp(static_cast<int>(std::floor((vertex.z - voxelGrid.minZ) / voxelGrid.step)),
                              0,
                              voxelGrid.nz - 1);
    return voxelGrid.scalarField[common::index3D(gx, gy, gz, voxelGrid.nx, voxelGrid.ny)];
}

void trimLowConfidenceFaces(const VoxelGrid &voxelGrid,
                            const ReconstructionConfig &config,
                            TriMesh *mesh)
{
    if (!mesh || mesh->empty() || !voxelGrid.hasScalarField())
    {
        return;
    }

    std::vector<float> vertexScores;
    vertexScores.reserve(mesh->vertices.size());
    for (const auto &vertex : mesh->vertices)
    {
        vertexScores.push_back(sampleScalarNearest(voxelGrid, vertex));
    }

    if (vertexScores.empty())
    {
        return;
    }

    std::vector<float> sortedScores = vertexScores;
    const float trimValue = std::clamp(config.poissonTrim, 0.0f, 12.0f);
    const float trimQuantile = std::clamp(0.03f + trimValue * 0.022f, 0.03f, 0.40f);
    const std::size_t thresholdIndex = std::min(sortedScores.size() - 1,
                                                static_cast<std::size_t>(sortedScores.size() * trimQuantile));
    std::nth_element(sortedScores.begin(),
                     sortedScores.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                     sortedScores.end());
    const float trimThreshold = sortedScores[thresholdIndex];

    std::vector<Triangle> keptFaces;
    keptFaces.reserve(mesh->faces.size());
    for (const auto &face : mesh->faces)
    {
        const float s0 = vertexScores[static_cast<std::size_t>(face.v[0])];
        const float s1 = vertexScores[static_cast<std::size_t>(face.v[1])];
        const float s2 = vertexScores[static_cast<std::size_t>(face.v[2])];
        if (s0 < trimThreshold && s1 < trimThreshold && s2 < trimThreshold)
        {
            continue;
        }
        keptFaces.push_back(face);
    }

    mesh->faces.swap(keptFaces);
    const int componentFloor = std::clamp(static_cast<int>(mesh->faceCount() / 1400), 48, 620);
    detail::removeSmallConnectedComponents(mesh, componentFloor);
}

int dominantNormalSector(const MeshVertex &vertex)
{
    const float ax = std::abs(vertex.nx);
    const float ay = std::abs(vertex.ny);
    const float az = std::abs(vertex.nz);
    if (ax >= ay && ax >= az)
    {
        return vertex.nx >= 0.0f ? 0 : 1;
    }
    if (ay >= ax && ay >= az)
    {
        return vertex.ny >= 0.0f ? 2 : 3;
    }
    return vertex.nz >= 0.0f ? 4 : 5;
}

int curvatureBand(float curvature)
{
    if (curvature < 0.10f)
    {
        return 0;
    }
    if (curvature < 0.24f)
    {
        return 1;
    }
    return 2;
}

void simplifyNormalAware(TriMesh *mesh, int targetFaces, float baseStep)
{
    if (!mesh || mesh->empty() || mesh->faceCount() <= targetFaces)
    {
        return;
    }

    std::vector<float> curvature;
    auto updateCurvature = [&]() {
        detail::recomputeNormals(mesh);
        std::vector<std::vector<int>> adjacency(mesh->vertices.size());
        adjacency.reserve(mesh->vertices.size());
        for (const auto &face : mesh->faces)
        {
            const int a = face.v[0];
            const int b = face.v[1];
            const int c = face.v[2];
            adjacency[static_cast<std::size_t>(a)].push_back(b);
            adjacency[static_cast<std::size_t>(a)].push_back(c);
            adjacency[static_cast<std::size_t>(b)].push_back(a);
            adjacency[static_cast<std::size_t>(b)].push_back(c);
            adjacency[static_cast<std::size_t>(c)].push_back(a);
            adjacency[static_cast<std::size_t>(c)].push_back(b);
        }

        curvature.assign(mesh->vertices.size(), 0.0f);
        for (std::size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            const auto &vertex = mesh->vertices[i];
            const float nlen = std::sqrt(vertex.nx * vertex.nx + vertex.ny * vertex.ny + vertex.nz * vertex.nz);
            if (nlen < 1e-6f || adjacency[i].empty())
            {
                continue;
            }

            common::Vec3 n0{vertex.nx / nlen, vertex.ny / nlen, vertex.nz / nlen};
            common::Vec3 nAvg{0.0f, 0.0f, 0.0f};
            for (int nb : adjacency[i])
            {
                const auto &nv = mesh->vertices[static_cast<std::size_t>(nb)];
                const float nn = std::sqrt(nv.nx * nv.nx + nv.ny * nv.ny + nv.nz * nv.nz);
                if (nn > 1e-6f)
                {
                    nAvg.x += nv.nx / nn;
                    nAvg.y += nv.ny / nn;
                    nAvg.z += nv.nz / nn;
                }
            }
            nAvg = common::normalize(nAvg);
            curvature[i] = std::clamp(1.0f - common::dot(n0, nAvg), 0.0f, 1.0f);
        }
    };

    updateCurvature();

    struct ClusterAccum
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double nx = 0.0;
        double ny = 0.0;
        double nz = 0.0;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        int count = 0;
    };

    float minX = mesh->vertices[0].x;
    float minY = mesh->vertices[0].y;
    float minZ = mesh->vertices[0].z;
    for (const auto &vertex : mesh->vertices)
    {
        minX = std::min(minX, vertex.x);
        minY = std::min(minY, vertex.y);
        minZ = std::min(minZ, vertex.z);
    }

    const float faceRatio = static_cast<float>(mesh->faceCount()) / std::max(1.0f, static_cast<float>(targetFaces));
    float clusterStep = std::max(baseStep * std::clamp(0.55f + faceRatio * 0.06f, 0.55f, 1.00f), 1e-5f);
    for (int pass = 0; pass < 8 && mesh->faceCount() > targetFaces; ++pass)
    {
        if (pass > 0 && (pass % 2) == 0)
        {
            updateCurvature();
        }

        std::unordered_map<common::OrientedVoxelKey, int, common::OrientedVoxelKeyHash> keyToIndex;
        std::vector<ClusterAccum> accumulators;
        std::vector<int> remap(mesh->vertices.size(), -1);
        keyToIndex.reserve(mesh->vertices.size());
        accumulators.reserve(mesh->vertices.size());

        for (std::size_t i = 0; i < mesh->vertices.size(); ++i)
        {
            const auto &vertex = mesh->vertices[i];
            common::OrientedVoxelKey key;
            const int band = curvatureBand(curvature[i]);
            key.detailBand = band;
            const float localStep = band == 0 ? clusterStep * 1.28f : (band == 1 ? clusterStep : clusterStep * 0.66f);
            key.x = static_cast<int>(std::floor((vertex.x - minX) / localStep));
            key.y = static_cast<int>(std::floor((vertex.y - minY) / localStep));
            key.z = static_cast<int>(std::floor((vertex.z - minZ) / localStep));
            key.normalSector = dominantNormalSector(vertex);

            auto it = keyToIndex.find(key);
            int clusterIndex = -1;
            if (it == keyToIndex.end())
            {
                clusterIndex = static_cast<int>(accumulators.size());
                keyToIndex.emplace(key, clusterIndex);
                accumulators.push_back({});
            }
            else
            {
                clusterIndex = it->second;
            }

            remap[i] = clusterIndex;
            auto &acc = accumulators[static_cast<std::size_t>(clusterIndex)];
            acc.x += vertex.x;
            acc.y += vertex.y;
            acc.z += vertex.z;
            acc.nx += vertex.nx;
            acc.ny += vertex.ny;
            acc.nz += vertex.nz;
            acc.r += vertex.r;
            acc.g += vertex.g;
            acc.b += vertex.b;
            ++acc.count;
        }

        std::vector<MeshVertex> newVertices(accumulators.size());
        for (std::size_t i = 0; i < accumulators.size(); ++i)
        {
            const auto &acc = accumulators[i];
            const float inv = acc.count > 0 ? (1.0f / static_cast<float>(acc.count)) : 1.0f;
            auto &vertex = newVertices[i];
            vertex.x = static_cast<float>(acc.x * inv);
            vertex.y = static_cast<float>(acc.y * inv);
            vertex.z = static_cast<float>(acc.z * inv);
            vertex.nx = static_cast<float>(acc.nx * inv);
            vertex.ny = static_cast<float>(acc.ny * inv);
            vertex.nz = static_cast<float>(acc.nz * inv);
            vertex.r = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.r * inv)), 0, 255));
            vertex.g = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.g * inv)), 0, 255));
            vertex.b = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.b * inv)), 0, 255));
        }

        std::vector<Triangle> newFaces;
        newFaces.reserve(mesh->faces.size());
        for (const auto &face : mesh->faces)
        {
            const int a = remap[static_cast<std::size_t>(face.v[0])];
            const int b = remap[static_cast<std::size_t>(face.v[1])];
            const int c = remap[static_cast<std::size_t>(face.v[2])];
            if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c)
            {
                continue;
            }
            Triangle simplified;
            simplified.v[0] = a;
            simplified.v[1] = b;
            simplified.v[2] = c;
            newFaces.push_back(simplified);
        }

        mesh->vertices.swap(newVertices);
        mesh->faces.swap(newFaces);
        detail::removeDegenerateFaces(mesh);

        if (mesh->faceCount() <= targetFaces)
        {
            break;
        }
        clusterStep *= 1.16f;
    }
}

void pruneLowQualityTriangles(TriMesh *mesh, float baseStep)
{
    if (!mesh || mesh->faces.empty())
    {
        return;
    }

    const float minEdge2 = std::max(1e-10f, baseStep * baseStep * 0.020f);
    const float minArea2 = std::max(1e-12f, baseStep * baseStep * 8.0e-4f);
    const float maxAspect = 10.0f;
    const float maxAspectSq = maxAspect * maxAspect;

    std::vector<Triangle> filtered;
    filtered.reserve(mesh->faces.size());

    for (const auto &face : mesh->faces)
    {
        const auto &a = mesh->vertices[static_cast<std::size_t>(face.v[0])];
        const auto &b = mesh->vertices[static_cast<std::size_t>(face.v[1])];
        const auto &c = mesh->vertices[static_cast<std::size_t>(face.v[2])];

        const common::Vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
        const common::Vec3 bc{c.x - b.x, c.y - b.y, c.z - b.z};
        const common::Vec3 ca{a.x - c.x, a.y - c.y, a.z - c.z};
        const float l2ab = common::dot(ab, ab);
        const float l2bc = common::dot(bc, bc);
        const float l2ca = common::dot(ca, ca);
        const float minL2 = std::min({l2ab, l2bc, l2ca});
        const float maxL2 = std::max({l2ab, l2bc, l2ca});
        if (minL2 < minEdge2)
        {
            continue;
        }

        const common::Vec3 ac{c.x - a.x, c.y - a.y, c.z - a.z};
        const common::Vec3 areaV = common::Vec3{
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x};
        const float area2 = common::dot(areaV, areaV);
        if (area2 < minArea2)
        {
            continue;
        }

        if (maxL2 > maxAspectSq * std::max(minL2, 1e-12f) && area2 < (minArea2 * 6.0f))
        {
            continue;
        }

        filtered.push_back(face);
    }

    mesh->faces.swap(filtered);
}

} // namespace

bool PoissonLiteReconstructor::validateInput(const std::vector<detail::PointXYZRGB> &points,
                                             const std::vector<cv::Vec3f> &normals,
                                             TriMesh *mesh,
                                             std::string *errorMessage) const
{
    (void)points;
    (void)normals;

    if (!mesh)
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson-Lite 重建失败：输出网格指针为空";
        }
        return false;
    }

    return true;
}

bool PoissonLiteReconstructor::reconstruct(const std::vector<detail::PointXYZRGB> &points,
                                           const std::vector<cv::Vec3f> &normals,
                                           const ReconstructionConfig &config,
                                           TriMesh *mesh,
                                           std::string *errorMessage) const
{
    if (!validateInput(points, normals, mesh, errorMessage))
    {
        return false;
    }

    mesh->vertices.clear();
    mesh->faces.clear();

    if (points.size() < 120)
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson-Lite 输入点过少";
        }
        return false;
    }

    if (!normals.empty() && normals.size() != points.size())
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson-Lite 输入法线数量与点数量不一致";
        }
        return false;
    }

    // The previous third-party Poisson integration was removed.
    // Keep a compact internal approximation pipeline based on occupancy extraction.
    const PoissonVoxelPipeline pipeline;
    VoxelGrid voxelGrid = pipeline.buildVoxelGrid(points, normals, config);
    if (!voxelGrid.valid())
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson-Lite 重建失败：体素场构建失败";
        }
        return false;
    }

    pipeline.voxelGridToMesh(voxelGrid, points, mesh);

    trimUnsupportedFaces(points, voxelGrid.step, mesh);
    trimLowConfidenceFaces(voxelGrid, config, mesh);
    pruneLowQualityTriangles(mesh, voxelGrid.step);

    detail::removeDegenerateFaces(mesh);
    const int poissonTargetFaces = std::clamp(std::max(config.simplifyTargetFaces, 10000), 10000, 90000);
    if (mesh->faceCount() > poissonTargetFaces)
    {
        simplifyNormalAware(mesh, poissonTargetFaces, voxelGrid.step);
        if (mesh->faceCount() > static_cast<int>(poissonTargetFaces * 1.05f))
        {
            ReconstructionConfig simplifyConfig = config;
            simplifyConfig.simplifyTargetFaces = poissonTargetFaces;
            simplifyConfig.voxelSimplifyFactor = std::clamp(config.voxelSimplifyFactor * 1.18f, 1.0f, 2.6f);
            detail::simplifyVoxelMeshAdaptive(mesh, simplifyConfig, voxelGrid.step);
            if (mesh->faceCount() > static_cast<int>(poissonTargetFaces * 1.10f))
            {
                simplifyNormalAware(mesh, poissonTargetFaces, voxelGrid.step * 1.12f);
            }
        }

        if (mesh->faceCount() > static_cast<int>(poissonTargetFaces * 1.15f))
        {
            ReconstructionConfig aggressiveConfig = config;
            aggressiveConfig.simplifyTargetFaces = poissonTargetFaces;
            aggressiveConfig.voxelSimplifyFactor = std::clamp(config.voxelSimplifyFactor * 1.45f, 1.2f, 3.0f);
            detail::simplifyVoxelMeshAdaptive(mesh, aggressiveConfig, voxelGrid.step * 1.05f);
            simplifyNormalAware(mesh, poissonTargetFaces, voxelGrid.step * 1.20f);
        }
    }

    pruneLowQualityTriangles(mesh, voxelGrid.step);

    // Keep detail: Poisson branch avoids extra smoothing because global postprocess also smooths.
    const int smoothIters = 0;
    const float smoothLambda = 0.0f;
    if (smoothIters > 0)
    {
        detail::taubinSmooth(mesh, smoothIters, smoothLambda);
    }
    detail::recomputeNormals(mesh);

    if (mesh->empty())
    {
        if (errorMessage)
        {
            *errorMessage = "Poisson-Lite 重建失败：网格为空";
        }
        return false;
    }

    return true;
}

} // namespace poisson
} // namespace mesh
} // namespace xjw
