#include "PointCloudPreprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <memory>
#include <utility>

#include <plamatrix/dense/dense_matrix.h>
#include <plapoint/core/point_cloud.h>
#include <plapoint/filters/preprocessing.h>
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

class DisjointSet
{
public:
    explicit DisjointSet(std::size_t size)
        : _parents(size), _ranks(size, 0)
    {
        std::iota(_parents.begin(), _parents.end(), 0);
    }

    std::size_t find(std::size_t index)
    {
        while (_parents[index] != index)
        {
            _parents[index] = _parents[_parents[index]];
            index = _parents[index];
        }
        return index;
    }

    void unite(std::size_t left, std::size_t right)
    {
        left = find(left);
        right = find(right);
        if (left == right)
        {
            return;
        }
        if (_ranks[left] < _ranks[right])
        {
            std::swap(left, right);
        }
        _parents[right] = left;
        if (_ranks[left] == _ranks[right])
        {
            ++_ranks[left];
        }
    }

private:
    std::vector<std::size_t> _parents;
    std::vector<std::uint8_t> _ranks;
};

float squaredDistance(const PointXYZRGB &left, const PointXYZRGB &right)
{
    const float dx = left.x - right.x;
    const float dy = left.y - right.y;
    const float dz = left.z - right.z;
    return dx * dx + dy * dy + dz * dz;
}

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

PoissonPointComponentStats removeSmallPoissonPointComponents(
    std::vector<PointXYZRGB> *points,
    std::size_t minComponentPoints,
    float neighborDistanceFactor)
{
    PoissonPointComponentStats stats;
    if (!points || points->empty())
    {
        return stats;
    }

    stats.retainedPointCount = points->size();
    if (points->size() < 2 || minComponentPoints <= 1)
    {
        stats.componentCount = 1;
        return stats;
    }

    const std::size_t point_count = points->size();
    plamatrix::DenseMatrix<float, plamatrix::Device::CPU> coordinates(
        static_cast<plamatrix::Index>(point_count), 3);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const auto row = static_cast<plamatrix::Index>(index);
        coordinates(row, 0) = (*points)[index].x;
        coordinates(row, 1) = (*points)[index].y;
        coordinates(row, 2) = (*points)[index].z;
    }

    const auto cloud = std::make_shared<PlaCloud>(std::move(coordinates));
    std::shared_ptr<const PlaCloud> const_cloud = cloud;
    plapoint::search::KdTree<float, plamatrix::Device::CPU> tree;
    tree.setInputCloud(const_cloud);
    tree.build();

    const int neighbor_count = std::min(12, static_cast<int>(point_count));
    std::vector<float> local_spacing(point_count, 0.0f);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const PointXYZRGB &point = (*points)[index];
        const plamatrix::Vec3<float> query{point.x, point.y, point.z};
        const std::vector<int> neighbors = tree.nearestKSearch(query, neighbor_count);
        std::array<float, 12> distances{};
        int distance_count = 0;
        for (const int neighbor_index : neighbors)
        {
            if (neighbor_index < 0 ||
                static_cast<std::size_t>(neighbor_index) == index ||
                distance_count >= static_cast<int>(distances.size()))
            {
                continue;
            }
            const float distance_squared = squaredDistance(point, (*points)[static_cast<std::size_t>(neighbor_index)]);
            if (std::isfinite(distance_squared) && distance_squared > 1.0e-16f)
            {
                distances[static_cast<std::size_t>(distance_count++)] = std::sqrt(distance_squared);
            }
        }
        if (distance_count > 0)
        {
            std::sort(distances.begin(), distances.begin() + distance_count);
            local_spacing[index] = distances[static_cast<std::size_t>(std::min(3, distance_count - 1) / 2)];
        }
    }

    std::vector<float> positive_spacing;
    positive_spacing.reserve(point_count);
    std::copy_if(local_spacing.begin(), local_spacing.end(), std::back_inserter(positive_spacing), [](float value)
    {
        return std::isfinite(value) && value > 0.0f;
    });
    if (positive_spacing.empty())
    {
        stats.componentCount = 1;
        return stats;
    }
    const auto median_it = positive_spacing.begin() + static_cast<std::ptrdiff_t>(positive_spacing.size() / 2);
    std::nth_element(positive_spacing.begin(), median_it, positive_spacing.end());
    stats.medianNeighborSpacing = *median_it;

    DisjointSet components(point_count);
    const float distance_factor = std::max(1.5f, neighborDistanceFactor);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const PointXYZRGB &point = (*points)[index];
        const plamatrix::Vec3<float> query{point.x, point.y, point.z};
        const std::vector<int> neighbors = tree.nearestKSearch(query, neighbor_count);
        for (const int neighbor_index : neighbors)
        {
            if (neighbor_index < 0 || static_cast<std::size_t>(neighbor_index) == index)
            {
                continue;
            }
            const std::size_t neighbor = static_cast<std::size_t>(neighbor_index);
            const float distance_squared = squaredDistance(point, (*points)[neighbor]);
            if (!std::isfinite(distance_squared))
            {
                continue;
            }
            if (distance_squared <= 1.0e-16f)
            {
                components.unite(index, neighbor);
                continue;
            }
            const float local_limit = distance_factor * std::min(local_spacing[index], local_spacing[neighbor]);
            if (local_limit > 0.0f && distance_squared <= local_limit * local_limit)
            {
                components.unite(index, neighbor);
            }
        }
    }

    std::vector<std::size_t> component_sizes(point_count, 0);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        ++component_sizes[components.find(index)];
    }
    std::size_t largest_root = 0;
    std::size_t largest_size = 0;
    for (std::size_t root = 0; root < point_count; ++root)
    {
        if (component_sizes[root] == 0)
        {
            continue;
        }
        ++stats.componentCount;
        if (component_sizes[root] > largest_size)
        {
            largest_size = component_sizes[root];
            largest_root = root;
        }
    }
    if (stats.componentCount <= 1)
    {
        return stats;
    }

    std::vector<PointXYZRGB> retained;
    retained.reserve(point_count);
    for (std::size_t index = 0; index < point_count; ++index)
    {
        const std::size_t root = components.find(index);
        if (root == largest_root || component_sizes[root] >= minComponentPoints)
        {
            retained.push_back((*points)[index]);
        }
        else
        {
            ++stats.removedPointCount;
        }
    }
    for (std::size_t root = 0; root < point_count; ++root)
    {
        if (component_sizes[root] > 0 && root != largest_root && component_sizes[root] < minComponentPoints)
        {
            ++stats.removedComponentCount;
        }
    }
    points->swap(retained);
    stats.retainedPointCount = points->size();
    return stats;
}

} // namespace detail
} // namespace mesh
} // namespace xjw
