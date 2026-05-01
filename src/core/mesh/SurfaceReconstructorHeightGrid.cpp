#include "SurfaceReconstructorHeightGrid.h"

/**
 * @file SurfaceReconstructorHeightGrid.cpp
 * @brief 2.5D 高程格网重建实现。
 *
 * 该实现用于航测/地形等近似单值高度场场景：
 * 点云 -> 高程格网 -> 补洞 -> 规则剖分网格。
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
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

inline float dot(const Vec3 &a, const Vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float norm(const Vec3 &vector)
{
    return std::sqrt(dot(vector, vector));
}

inline Vec3 normalize(const Vec3 &vector)
{
    const float length = norm(vector);
    if (length < 1e-12f)
    {
        return {0.0f, 0.0f, 1.0f};
    }
    return {vector.x / length, vector.y / length, vector.z / length};
}

/**
 * @brief 仅用于高程格网三角化阶段的最近邻颜色查询。
 */
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
    const std::vector<PointXYZRGB> *points = nullptr;

    void build(const std::vector<PointXYZRGB> &inputPoints, float requestedCellSize)
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

    PointXYZRGB nearest(const Vec3 &query) const
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

float medianValue(std::vector<float> values, float fallback)
{
    if (values.empty())
    {
        return fallback;
    }

    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if ((values.size() % 2U) == 0U && middle > 0U)
    {
        std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
        result = 0.5f * (result + values[middle - 1]);
    }
    return result;
}

float estimateHeightJumpThreshold(const HeightGrid &heightGrid)
{
    std::vector<float> diffs;
    diffs.reserve(static_cast<std::size_t>(heightGrid.nx) * heightGrid.ny * 2U);
    for (int iy = 0; iy < heightGrid.ny; ++iy)
    {
        for (int ix = 0; ix < heightGrid.nx; ++ix)
        {
            if (!heightGrid.vd(ix, iy))
            {
                continue;
            }

            if (ix + 1 < heightGrid.nx && heightGrid.vd(ix + 1, iy))
            {
                diffs.push_back(std::abs(heightGrid.at(ix, iy) - heightGrid.at(ix + 1, iy)));
            }
            if (iy + 1 < heightGrid.ny && heightGrid.vd(ix, iy + 1))
            {
                diffs.push_back(std::abs(heightGrid.at(ix, iy) - heightGrid.at(ix, iy + 1)));
            }
        }
    }

    const float cellStep = std::max(1e-6f, std::sqrt(heightGrid.stepX * heightGrid.stepX + heightGrid.stepY * heightGrid.stepY));
    const float medianDiff = medianValue(std::move(diffs), cellStep * 0.5f);
    return std::max(cellStep * 1.5f, medianDiff * 4.0f);
}

bool isTriangleReliable(const HeightGrid &heightGrid,
                        int ax,
                        int ay,
                        int bx,
                        int by,
                        int cx,
                        int cy,
                        float maxHeightJump,
                        int maxFillPass)
{
    const int ia = ay * heightGrid.nx + ax;
    const int ib = by * heightGrid.nx + bx;
    const int ic = cy * heightGrid.nx + cx;
    if (!heightGrid.valid[static_cast<std::size_t>(ia)]
        || !heightGrid.valid[static_cast<std::size_t>(ib)]
        || !heightGrid.valid[static_cast<std::size_t>(ic)])
    {
        return false;
    }

    if (std::max({static_cast<int>(heightGrid.fillPass[static_cast<std::size_t>(ia)]),
                  static_cast<int>(heightGrid.fillPass[static_cast<std::size_t>(ib)]),
                  static_cast<int>(heightGrid.fillPass[static_cast<std::size_t>(ic)])}) > maxFillPass)
    {
        return false;
    }

    const float za = heightGrid.heights[static_cast<std::size_t>(ia)];
    const float zb = heightGrid.heights[static_cast<std::size_t>(ib)];
    const float zc = heightGrid.heights[static_cast<std::size_t>(ic)];
    if (std::abs(za - zb) > maxHeightJump
        || std::abs(za - zc) > maxHeightJump
        || std::abs(zb - zc) > maxHeightJump)
    {
        return false;
    }

    const Vec3 a{heightGrid.minX + ax * heightGrid.stepX, heightGrid.minY + ay * heightGrid.stepY, za};
    const Vec3 b{heightGrid.minX + bx * heightGrid.stepX, heightGrid.minY + by * heightGrid.stepY, zb};
    const Vec3 c{heightGrid.minX + cx * heightGrid.stepX, heightGrid.minY + cy * heightGrid.stepY, zc};
    const Vec3 normal{
        (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y),
        (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z),
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)};
    const Vec3 normalized = normalize(normal);
    return std::abs(normalized.z) >= 0.08f;
}

} // namespace

HeightGrid buildHeightGrid(const std::vector<PointXYZRGB> &points,
                           const ReconstructionConfig &config)
{
    HeightGrid heightGrid;
    if (points.empty())
    {
        return heightGrid;
    }

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;
    for (const auto &point : points)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    float dx = std::max(1e-6f, maxX - minX);
    float dy = std::max(1e-6f, maxY - minY);
    minX -= dx * config.padding;
    maxX += dx * config.padding;
    minY -= dy * config.padding;
    maxY += dy * config.padding;
    dx = maxX - minX;
    dy = maxY - minY;

    const int resolution = std::clamp(config.resolution, 32, 1024);
    const float maxSpan = std::max(dx, dy);
    heightGrid.nx = std::max(8, static_cast<int>(std::round(resolution * dx / maxSpan)));
    heightGrid.ny = std::max(8, static_cast<int>(std::round(resolution * dy / maxSpan)));
    heightGrid.minX = minX;
    heightGrid.minY = minY;
    heightGrid.stepX = dx / std::max(1, heightGrid.nx - 1);
    heightGrid.stepY = dy / std::max(1, heightGrid.ny - 1);

    const int n = heightGrid.nx * heightGrid.ny;
    heightGrid.heights.assign(static_cast<std::size_t>(n), 0.0f);
    heightGrid.sumW.assign(static_cast<std::size_t>(n), 0.0f);
    heightGrid.valid.assign(static_cast<std::size_t>(n), 0);
    heightGrid.fillPass.assign(static_cast<std::size_t>(n), 0);

#ifdef _OPENMP
    const int threadCount = omp_get_max_threads();
    std::vector<std::vector<float>> threadHeights(static_cast<std::size_t>(threadCount), std::vector<float>(static_cast<std::size_t>(n), 0.0f));
    std::vector<std::vector<float>> threadWeights(static_cast<std::size_t>(threadCount), std::vector<float>(static_cast<std::size_t>(n), 0.0f));

    #pragma omp parallel for schedule(static)
    for (int pointIndex = 0; pointIndex < static_cast<int>(points.size()); ++pointIndex)
    {
        const auto &point = points[static_cast<std::size_t>(pointIndex)];
        const int threadId = omp_get_thread_num();
        const float gx = (point.x - heightGrid.minX) / heightGrid.stepX;
        const float gy = (point.y - heightGrid.minY) / heightGrid.stepY;
        const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, heightGrid.nx - 2);
        const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, heightGrid.ny - 2);
        const float tx = std::clamp(gx - ix, 0.0f, 1.0f);
        const float ty = std::clamp(gy - iy, 0.0f, 1.0f);

        for (int dyv = 0; dyv <= 1; ++dyv)
        {
            for (int dxv = 0; dxv <= 1; ++dxv)
            {
                const float w = (dxv ? tx : 1.0f - tx) * (dyv ? ty : 1.0f - ty);
                const int cellIndex = (iy + dyv) * heightGrid.nx + (ix + dxv);
                threadHeights[static_cast<std::size_t>(threadId)][static_cast<std::size_t>(cellIndex)] += point.z * w;
                threadWeights[static_cast<std::size_t>(threadId)][static_cast<std::size_t>(cellIndex)] += w;
            }
        }
    }

    for (int cellIndex = 0; cellIndex < n; ++cellIndex)
    {
        for (int threadId = 0; threadId < threadCount; ++threadId)
        {
            heightGrid.heights[static_cast<std::size_t>(cellIndex)] += threadHeights[static_cast<std::size_t>(threadId)][static_cast<std::size_t>(cellIndex)];
            heightGrid.sumW[static_cast<std::size_t>(cellIndex)] += threadWeights[static_cast<std::size_t>(threadId)][static_cast<std::size_t>(cellIndex)];
        }

        if (heightGrid.sumW[static_cast<std::size_t>(cellIndex)] > 1e-9f)
        {
            heightGrid.heights[static_cast<std::size_t>(cellIndex)] /= heightGrid.sumW[static_cast<std::size_t>(cellIndex)];
            heightGrid.valid[static_cast<std::size_t>(cellIndex)] = 1;
        }
    }
#else
    for (const auto &point : points)
    {
        const float gx = (point.x - heightGrid.minX) / heightGrid.stepX;
        const float gy = (point.y - heightGrid.minY) / heightGrid.stepY;
        const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, heightGrid.nx - 2);
        const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, heightGrid.ny - 2);
        const float tx = std::clamp(gx - ix, 0.0f, 1.0f);
        const float ty = std::clamp(gy - iy, 0.0f, 1.0f);

        for (int dyv = 0; dyv <= 1; ++dyv)
        {
            for (int dxv = 0; dxv <= 1; ++dxv)
            {
                const float w = (dxv ? tx : 1.0f - tx) * (dyv ? ty : 1.0f - ty);
                const int cellIndex = (iy + dyv) * heightGrid.nx + (ix + dxv);
                heightGrid.heights[static_cast<std::size_t>(cellIndex)] += point.z * w;
                heightGrid.sumW[static_cast<std::size_t>(cellIndex)] += w;
            }
        }
    }

    for (int cellIndex = 0; cellIndex < n; ++cellIndex)
    {
        if (heightGrid.sumW[static_cast<std::size_t>(cellIndex)] > 1e-9f)
        {
            heightGrid.heights[static_cast<std::size_t>(cellIndex)] /= heightGrid.sumW[static_cast<std::size_t>(cellIndex)];
            heightGrid.valid[static_cast<std::size_t>(cellIndex)] = 1;
        }
    }
#endif

    return heightGrid;
}

void fillHoles(HeightGrid *heightGrid, int maxPasses)
{
    if (heightGrid == nullptr)
    {
        return;
    }

    const int nx = heightGrid->nx;
    const int ny = heightGrid->ny;
    std::vector<float> tmpHeights = heightGrid->heights;
    std::vector<std::uint8_t> tmpValid = heightGrid->valid;
    std::vector<std::uint16_t> tmpFillPass = heightGrid->fillPass;

    for (int pass = 0; pass < maxPasses; ++pass)
    {
        bool changed = false;
        for (int iy = 0; iy < ny; ++iy)
        {
            for (int ix = 0; ix < nx; ++ix)
            {
                const int cellIndex = iy * nx + ix;
                if (heightGrid->valid[static_cast<std::size_t>(cellIndex)])
                {
                    tmpHeights[static_cast<std::size_t>(cellIndex)] = heightGrid->heights[static_cast<std::size_t>(cellIndex)];
                    tmpValid[static_cast<std::size_t>(cellIndex)] = 1;
                    tmpFillPass[static_cast<std::size_t>(cellIndex)] = heightGrid->fillPass[static_cast<std::size_t>(cellIndex)];
                    continue;
                }

                float sum = 0.0f;
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        const int nx2 = ix + dx;
                        const int ny2 = iy + dy;
                        if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny)
                        {
                            continue;
                        }

                        const int neighborIndex = ny2 * nx + nx2;
                        if (heightGrid->valid[static_cast<std::size_t>(neighborIndex)])
                        {
                            sum += heightGrid->heights[static_cast<std::size_t>(neighborIndex)];
                            ++count;
                        }
                    }
                }

                if (count > 0)
                {
                    tmpHeights[static_cast<std::size_t>(cellIndex)] = sum / static_cast<float>(count);
                    tmpValid[static_cast<std::size_t>(cellIndex)] = 1;
                    tmpFillPass[static_cast<std::size_t>(cellIndex)] = static_cast<std::uint16_t>(pass + 1);
                    changed = true;
                }
            }
        }

        heightGrid->heights.swap(tmpHeights);
        heightGrid->valid.swap(tmpValid);
        heightGrid->fillPass.swap(tmpFillPass);
        tmpHeights = heightGrid->heights;
        tmpValid = heightGrid->valid;
        tmpFillPass = heightGrid->fillPass;
        if (!changed)
        {
            break;
        }
    }
}

void heightGridToMesh(const HeightGrid &heightGrid,
                      const std::vector<PointXYZRGB> &points,
                      const ReconstructionConfig &config,
                      TriMesh *mesh)
{
    if (mesh == nullptr)
    {
        return;
    }

    mesh->vertices.clear();
    mesh->faces.clear();
    if (heightGrid.nx < 2 || heightGrid.ny < 2)
    {
        return;
    }

    SpatialGrid spatialGrid;
    spatialGrid.build(points, (heightGrid.stepX + heightGrid.stepY) * 1.5f);

    const int nx = heightGrid.nx;
    const int ny = heightGrid.ny;
    const float maxHeightJump = estimateHeightJumpThreshold(heightGrid);
    const int maxFillPass = std::max(0, config.holeFillPasses);
    std::vector<int> vertexIndices(static_cast<std::size_t>(nx * ny), -1);

    for (int iy = 0; iy < ny; ++iy)
    {
        for (int ix = 0; ix < nx; ++ix)
        {
            if (!heightGrid.vd(ix, iy))
            {
                continue;
            }

            const Vec3 position{heightGrid.minX + ix * heightGrid.stepX,
                                heightGrid.minY + iy * heightGrid.stepY,
                                heightGrid.at(ix, iy)};
            const PointXYZRGB colorPoint = spatialGrid.nearest(position);
            MeshVertex vertex;
            vertex.x = position.x;
            vertex.y = position.y;
            vertex.z = position.z;
            vertex.nx = 0.0f;
            vertex.ny = 0.0f;
            vertex.nz = 1.0f;
            vertex.r = colorPoint.r;
            vertex.g = colorPoint.g;
            vertex.b = colorPoint.b;
            vertexIndices[static_cast<std::size_t>(iy * nx + ix)] = static_cast<int>(mesh->vertices.size());
            mesh->vertices.push_back(vertex);
        }
    }

    for (int iy = 0; iy < ny - 1; ++iy)
    {
        for (int ix = 0; ix < nx - 1; ++ix)
        {
            const int i00 = vertexIndices[static_cast<std::size_t>(iy * nx + ix)];
            const int i10 = vertexIndices[static_cast<std::size_t>(iy * nx + ix + 1)];
            const int i01 = vertexIndices[static_cast<std::size_t>((iy + 1) * nx + ix)];
            const int i11 = vertexIndices[static_cast<std::size_t>((iy + 1) * nx + ix + 1)];
            if (i00 < 0 || i10 < 0 || i01 < 0 || i11 < 0)
            {
                continue;
            }

            const float diagMain = std::abs(heightGrid.at(ix, iy) - heightGrid.at(ix + 1, iy + 1));
            const float diagAlt = std::abs(heightGrid.at(ix + 1, iy) - heightGrid.at(ix, iy + 1));

            if (diagMain <= diagAlt)
            {
                if (isTriangleReliable(heightGrid, ix, iy, ix + 1, iy, ix + 1, iy + 1, maxHeightJump, maxFillPass))
                {
                    Triangle triangle;
                    triangle.v[0] = i00;
                    triangle.v[1] = i10;
                    triangle.v[2] = i11;
                    mesh->faces.push_back(triangle);
                }
                if (isTriangleReliable(heightGrid, ix, iy, ix + 1, iy + 1, ix, iy + 1, maxHeightJump, maxFillPass))
                {
                    Triangle triangle;
                    triangle.v[0] = i00;
                    triangle.v[1] = i11;
                    triangle.v[2] = i01;
                    mesh->faces.push_back(triangle);
                }
            }
            else
            {
                if (isTriangleReliable(heightGrid, ix, iy, ix + 1, iy, ix, iy + 1, maxHeightJump, maxFillPass))
                {
                    Triangle triangle;
                    triangle.v[0] = i00;
                    triangle.v[1] = i10;
                    triangle.v[2] = i01;
                    mesh->faces.push_back(triangle);
                }
                if (isTriangleReliable(heightGrid, ix + 1, iy, ix + 1, iy + 1, ix, iy + 1, maxHeightJump, maxFillPass))
                {
                    Triangle triangle;
                    triangle.v[0] = i10;
                    triangle.v[1] = i11;
                    triangle.v[2] = i01;
                    mesh->faces.push_back(triangle);
                }
            }
        }
    }
}

} // namespace detail
} // namespace mesh
} // namespace xjw
