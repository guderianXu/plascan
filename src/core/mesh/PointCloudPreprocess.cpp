#include "PointCloudPreprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/statistical_outlier_removal.h>
#include <plapoint/filters/voxel_grid.h>
#include <plapoint/search/kdtree.h>

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
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> xyz(
        static_cast<plamatrix::Index>(points.size()), 3);
    plamatrix::DenseMatrix<std::uint8_t, plamatrix::Device::CPU> colors(
        static_cast<plamatrix::Index>(points.size()), 3);
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        const auto row = static_cast<plamatrix::Index>(i);
        xyz.setValue(row, 0, points[i].x);
        xyz.setValue(row, 1, points[i].y);
        xyz.setValue(row, 2, points[i].z);
        colors.setValue(row, 0, points[i].r);
        colors.setValue(row, 1, points[i].g);
        colors.setValue(row, 2, points[i].b);
    }

    PlaCloud cloud(std::move(xyz));
    cloud.setColors(std::move(colors));
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
                                                float voxelSize)
{
    if (points.empty() || voxelSize <= 1e-6f)
    {
        return points;
    }

    const auto inputCloud = std::make_shared<PlaCloud>(toPlaCloud(points));
    plapoint::VoxelGrid<float, plamatrix::Device::CPU> voxelGrid;
    voxelGrid.setInputCloud(inputCloud);
    voxelGrid.setLeafSize(voxelSize, voxelSize, voxelSize);

    PlaCloud output;
    voxelGrid.filter(output);
    return fromPlaCloud(output);
}

std::vector<PointXYZRGB> statisticalDenoisePoints(const std::vector<PointXYZRGB> &points,
                                                    int k,
                                                    float stdMul,
                                                    float gridCellSize)
{
    (void)gridCellSize;
    if (points.size() < 64 || k < 4)
    {
        return points;
    }

    const auto inputCloud = std::make_shared<PlaCloud>(toPlaCloud(points));
    auto tree = std::make_shared<plapoint::search::KdTree<float, plamatrix::Device::CPU>>();
    tree->setInputCloud(inputCloud);
    tree->build();

    plapoint::StatisticalOutlierRemoval<float, plamatrix::Device::CPU> filter;
    filter.setInputCloud(inputCloud);
    filter.setSearchMethod(tree);
    filter.setMeanK(k);
    filter.setStddevMulThresh(std::max(0.2f, stdMul));

    PlaCloud output;
    filter.filter(output);
    const auto filtered = fromPlaCloud(output);
    return filtered.size() >= 100 ? filtered : points;
}

} // namespace detail
} // namespace mesh
} // namespace xjw
