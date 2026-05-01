#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace xjw
{
namespace mesh
{
namespace poisson
{
namespace common
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

struct OrientedVoxelKey
{
    int x = 0;
    int y = 0;
    int z = 0;
    int normalSector = 0;
    int detailBand = 0;

    bool operator==(const OrientedVoxelKey &other) const
    {
        return x == other.x && y == other.y && z == other.z
               && normalSector == other.normalSector
               && detailBand == other.detailBand;
    }
};

struct OrientedVoxelKeyHash
{
    std::size_t operator()(const OrientedVoxelKey &key) const
    {
        const std::size_t h1 = std::hash<int>{}(key.x);
        const std::size_t h2 = std::hash<int>{}(key.y);
        const std::size_t h3 = std::hash<int>{}(key.z);
        const std::size_t h4 = std::hash<int>{}(key.normalSector);
        const std::size_t h5 = std::hash<int>{}(key.detailBand);
        return h1 ^ (h2 << 1) ^ (h3 << 7) ^ (h4 << 11) ^ (h5 << 15);
    }
};

inline Vec3 operator*(const Vec3 &v, float s)
{
    return {v.x * s, v.y * s, v.z * s};
}

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

inline bool inBounds3D(int x, int y, int z, int nx, int ny, int nz)
{
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
}

inline std::size_t index3D(int x, int y, int z, int nx, int ny)
{
    return static_cast<std::size_t>((z * ny + y) * nx + x);
}

} // namespace common
} // namespace poisson
} // namespace mesh
} // namespace xjw
