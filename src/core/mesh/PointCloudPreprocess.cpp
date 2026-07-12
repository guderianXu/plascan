#include "PointCloudPreprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>

namespace xjw
{
namespace mesh
{
namespace detail
{

namespace
{

using PlaCloud = plapoint::PointCloud<float, plamatrix::Device::CPU>;

PlaCloud toPlaCloud(const std::vector<PointXYZRGB> &points)
{
    const bool hasNormals = !points.empty() &&
                            std::all_of(points.begin(), points.end(), [](const PointXYZRGB &point) {
                                return point.hasNormal;
                            });
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> xyz(
        static_cast<plamatrix::Index>(points.size()), 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(points.size()), 3);
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> normals(
        hasNormals ? static_cast<plamatrix::Index>(points.size()) : 0, 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        xyz.setValue(row, 0, points[i].x);
        xyz.setValue(row, 1, points[i].y);
        xyz.setValue(row, 2, points[i].z);
        colors.setValue(row, 0, points[i].r);
        colors.setValue(row, 1, points[i].g);
        colors.setValue(row, 2, points[i].b);
        if (hasNormals)
        {
            normals.setValue(row, 0, points[i].nx);
            normals.setValue(row, 1, points[i].ny);
            normals.setValue(row, 2, points[i].nz);
        }
    }

    PlaCloud cloud(std::move(xyz));
    cloud.setColors(std::move(colors));
    if (hasNormals)
    {
        cloud.setNormals(std::move(normals));
    }
    return cloud;
}

std::vector<PointXYZRGB> fromPlaCloud(const PlaCloud &cloud)
{
    std::vector<PointXYZRGB> points;
    points.reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        PointXYZRGB point;
        point.x = cloud.points().getValue(row, 0);
        point.y = cloud.points().getValue(row, 1);
        point.z = cloud.points().getValue(row, 2);
        if (cloud.hasNormals())
        {
            point.hasNormal = true;
            point.nx = cloud.normals()->getValue(row, 0);
            point.ny = cloud.normals()->getValue(row, 1);
            point.nz = cloud.normals()->getValue(row, 2);
        }
        if (cloud.hasColors())
        {
            point.r = cloud.colors()->getValue(row, 0);
            point.g = cloud.colors()->getValue(row, 1);
            point.b = cloud.colors()->getValue(row, 2);
        }
        points.push_back(point);
    }
    return points;
}

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
                                                float voxelSize,
                                                plapoint::ProcessingDevice device)
{
    if (points.empty() || voxelSize <= 1e-6f)
    {
        return points;
    }

    const auto inputCloud = std::make_shared<PlaCloud>(toPlaCloud(points));
    PlaCloud output = plapoint::voxelDownsample(*inputCloud, voxelSize, device);
    return fromPlaCloud(output);
}

std::vector<PointXYZRGB> statisticalDenoisePoints(const std::vector<PointXYZRGB> &points,
                                                    int k,
                                                    float stdMul,
                                                    float gridCellSize,
                                                    plapoint::ProcessingDevice device)
{
    (void)gridCellSize;
    if (points.size() < 64 || k < 4)
    {
        return points;
    }

    const auto inputCloud = std::make_shared<PlaCloud>(toPlaCloud(points));
    PlaCloud output = plapoint::statisticalOutlierRemoval(
        *inputCloud, k, std::max(0.2f, stdMul), device);
    const auto filtered = fromPlaCloud(output);
    return filtered.size() >= 100 ? filtered : points;
}

std::size_t removeInvalidPoissonPoints(std::vector<PointXYZRGB> *points)
{
    if (!points)
    {
        return 0;
    }

    const std::size_t original_size = points->size();
    points->erase(std::remove_if(points->begin(), points->end(), [](PointXYZRGB &point)
    {
        const bool finite_position = std::isfinite(point.x) &&
                                     std::isfinite(point.y) &&
                                     std::isfinite(point.z);
        const float normal_length = std::sqrt(point.nx * point.nx +
                                              point.ny * point.ny +
                                              point.nz * point.nz);
        if (!finite_position || !point.hasNormal ||
            !std::isfinite(normal_length) || normal_length <= 1.0e-8f)
        {
            return true;
        }
        point.nx /= normal_length;
        point.ny /= normal_length;
        point.nz /= normal_length;
        return false;
    }), points->end());
    return original_size - points->size();
}

} // namespace detail
} // namespace mesh
} // namespace xjw
