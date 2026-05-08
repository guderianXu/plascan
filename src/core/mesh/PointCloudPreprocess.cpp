#include "PointCloudPreprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

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

struct VoxelKey
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey &key) const
    {
        const std::size_t h1 = std::hash<int>{}(key.x);
        const std::size_t h2 = std::hash<int>{}(key.y);
        const std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 7);
    }
};

inline float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float norm(const Vec3 &v)
{
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3 &v)
{
    const float n = norm(v);
    if (n < 1e-12f)
    {
        return {0.0f, 0.0f, 1.0f};
    }
    return {v.x / n, v.y / n, v.z / n};
}

struct NeighborGrid
{
    float minX = 0.0f;
    float minY = 0.0f;
    float minZ = 0.0f;
    float cellSize = 1.0f;
    int nx = 1;
    int ny = 1;
    int nz = 1;
    std::vector<std::vector<int>> bins;

    void build(const std::vector<PointXYZRGB> &points, float requestedCellSize)
    {
        if (points.empty())
        {
            bins.clear();
            return;
        }

        minX = points[0].x;
        minY = points[0].y;
        minZ = points[0].z;
        float maxX = minX;
        float maxY = minY;
        float maxZ = minZ;
        for (const auto &point : points)
        {
            minX = std::min(minX, point.x);
            minY = std::min(minY, point.y);
            minZ = std::min(minZ, point.z);
            maxX = std::max(maxX, point.x);
            maxY = std::max(maxY, point.y);
            maxZ = std::max(maxZ, point.z);
        }

        cellSize = std::max(1e-6f, requestedCellSize);
        nx = std::max(1, static_cast<int>(std::floor((maxX - minX) / cellSize)) + 1);
        ny = std::max(1, static_cast<int>(std::floor((maxY - minY) / cellSize)) + 1);
        nz = std::max(1, static_cast<int>(std::floor((maxZ - minZ) / cellSize)) + 1);
        bins.assign(static_cast<std::size_t>(nx) * ny * nz, {});

        for (int index = 0; index < static_cast<int>(points.size()); ++index)
        {
            const auto &point = points[static_cast<std::size_t>(index)];
            const int cx = std::clamp(static_cast<int>(std::floor((point.x - minX) / cellSize)), 0, nx - 1);
            const int cy = std::clamp(static_cast<int>(std::floor((point.y - minY) / cellSize)), 0, ny - 1);
            const int cz = std::clamp(static_cast<int>(std::floor((point.z - minZ) / cellSize)), 0, nz - 1);
            bins[(static_cast<std::size_t>(cz) * ny + cy) * nx + cx].push_back(index);
        }
    }

    std::vector<int> gatherKnn(const std::vector<PointXYZRGB> &points,
                               int queryIndex,
                               int k) const
    {
        std::vector<int> candidates;
        if (points.empty() || queryIndex < 0
            || queryIndex >= static_cast<int>(points.size()) || k <= 0)
        {
            return candidates;
        }

        const auto &queryPoint = points[static_cast<std::size_t>(queryIndex)];
        const int cx = std::clamp(static_cast<int>(std::floor((queryPoint.x - minX) / cellSize)), 0, nx - 1);
        const int cy = std::clamp(static_cast<int>(std::floor((queryPoint.y - minY) / cellSize)), 0, ny - 1);
        const int cz = std::clamp(static_cast<int>(std::floor((queryPoint.z - minZ) / cellSize)), 0, nz - 1);

        std::vector<std::pair<float, int>> distWithIndex;
        distWithIndex.reserve(static_cast<std::size_t>(k * 6));

        for (int radius = 0; radius <= 6; ++radius)
        {
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
                        const bool onShell = (radius == 0)
                            || ix == x0 || ix == x1
                            || iy == y0 || iy == y1
                            || iz == z0 || iz == z1;
                        if (!onShell)
                        {
                            continue;
                        }

                        const auto &bucket = bins[(static_cast<std::size_t>(iz) * ny + iy) * nx + ix];
                        for (int idx : bucket)
                        {
                            if (idx == queryIndex)
                            {
                                continue;
                            }
                            const auto &point = points[static_cast<std::size_t>(idx)];
                            const float dx = point.x - queryPoint.x;
                            const float dy = point.y - queryPoint.y;
                            const float dz = point.z - queryPoint.z;
                            const float d2 = dx * dx + dy * dy + dz * dz;
                            distWithIndex.emplace_back(d2, idx);
                        }
                    }
                }
            }

            if (static_cast<int>(distWithIndex.size()) >= k)
            {
                break;
            }
        }

        if (distWithIndex.empty())
        {
            return candidates;
        }

        const int keep = std::min(k, static_cast<int>(distWithIndex.size()));
        std::nth_element(distWithIndex.begin(), distWithIndex.begin() + keep - 1,
                         distWithIndex.end(),
                         [](const auto &a, const auto &b) { return a.first < b.first; });
        std::sort(distWithIndex.begin(), distWithIndex.begin() + keep,
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        candidates.reserve(static_cast<std::size_t>(keep));
        for (int i = 0; i < keep; ++i)
        {
            candidates.push_back(distWithIndex[static_cast<std::size_t>(i)].second);
        }
        return candidates;
    }
};

} // namespace

float estimateBaseVoxelStep(const std::vector<PointXYZRGB> &points,
                            int resolution)
{
    if (points.empty())
    {
        return 1e-3f;
    }

    float minX = points[0].x;
    float minY = points[0].y;
    float minZ = points[0].z;
    float maxX = minX;
    float maxY = minY;
    float maxZ = minZ;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        minZ = std::min(minZ, point.z);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
        maxZ = std::max(maxZ, point.z);
    }

    const float maxSpan = std::max({maxX - minX, maxY - minY, maxZ - minZ, 1e-6f});
    return std::max(1e-5f, maxSpan / std::max(16, resolution));
}

std::vector<PointXYZRGB> voxelDownsamplePoints(const std::vector<PointXYZRGB> &points,
                                                float voxelSize)
{
    if (points.empty() || voxelSize <= 1e-6f)
    {
        return points;
    }

    struct Accumulator
    {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        int count = 0;
    };

    float minX = points[0].x;
    float minY = points[0].y;
    float minZ = points[0].z;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        minZ = std::min(minZ, point.z);
    }

    std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> voxels;
    voxels.reserve(points.size() / 2 + 1);

    for (const auto &point : points)
    {
        VoxelKey key;
        key.x = static_cast<int>(std::floor((point.x - minX) / voxelSize));
        key.y = static_cast<int>(std::floor((point.y - minY) / voxelSize));
        key.z = static_cast<int>(std::floor((point.z - minZ) / voxelSize));

        auto &acc = voxels[key];
        acc.x += point.x;
        acc.y += point.y;
        acc.z += point.z;
        acc.r += point.r;
        acc.g += point.g;
        acc.b += point.b;
        ++acc.count;
    }

    std::vector<PointXYZRGB> downsampled;
    downsampled.reserve(voxels.size());
    for (const auto &entry : voxels)
    {
        const auto &acc = entry.second;
        if (acc.count <= 0)
        {
            continue;
        }

        const float invCount = 1.0f / static_cast<float>(acc.count);
        PointXYZRGB point;
        point.x = static_cast<float>(acc.x * invCount);
        point.y = static_cast<float>(acc.y * invCount);
        point.z = static_cast<float>(acc.z * invCount);
        point.r = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.r * invCount)), 0, 255));
        point.g = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.g * invCount)), 0, 255));
        point.b = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(acc.b * invCount)), 0, 255));
        downsampled.push_back(point);
    }

    return downsampled;
}

std::vector<PointXYZRGB> statisticalDenoisePoints(const std::vector<PointXYZRGB> &points,
                                                    int k,
                                                    float stdMul,
                                                    float gridCellSize)
{
    if (points.size() < 64 || k < 4)
    {
        return points;
    }

    NeighborGrid grid;
    grid.build(points, std::max(1e-5f, gridCellSize));

    std::vector<float> meanDistances(points.size(), 0.0f);
    std::vector<uint8_t> valid(points.size(), 0);

    double globalMean = 0.0;
    int validCount = 0;
    for (int index = 0; index < static_cast<int>(points.size()); ++index)
    {
        const auto neighbors = grid.gatherKnn(points, index, k);
        if (neighbors.size() < 4)
        {
            continue;
        }

        const auto &queryPoint = points[static_cast<std::size_t>(index)];
        double sum = 0.0;
        for (int neighborIndex : neighbors)
        {
            const auto &neighbor = points[static_cast<std::size_t>(neighborIndex)];
            const double dx = static_cast<double>(neighbor.x - queryPoint.x);
            const double dy = static_cast<double>(neighbor.y - queryPoint.y);
            const double dz = static_cast<double>(neighbor.z - queryPoint.z);
            sum += std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        const float meanDist = static_cast<float>(sum / neighbors.size());
        meanDistances[static_cast<std::size_t>(index)] = meanDist;
        valid[static_cast<std::size_t>(index)] = 1;
        globalMean += meanDist;
        ++validCount;
    }

    if (validCount < 8)
    {
        return points;
    }

    globalMean /= static_cast<double>(validCount);
    double variance = 0.0;
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
    {
        if (valid[static_cast<std::size_t>(i)] == 0)
        {
            continue;
        }
        const double delta = meanDistances[static_cast<std::size_t>(i)] - globalMean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(std::max(1, validCount - 1));
    const double sigma = std::sqrt(std::max(0.0, variance));
    const double threshold = globalMean + std::max(0.2f, stdMul) * sigma;

    std::vector<PointXYZRGB> filtered;
    filtered.reserve(points.size());
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
    {
        if (valid[static_cast<std::size_t>(i)] == 0
            || meanDistances[static_cast<std::size_t>(i)] <= threshold)
        {
            filtered.push_back(points[static_cast<std::size_t>(i)]);
        }
    }

    return filtered.size() >= 100 ? filtered : points;
}

} // namespace detail
} // namespace mesh
} // namespace xjw
