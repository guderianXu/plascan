#pragma once

// =============================================================================
// 文件: KDTree3D.h
// 功能: 提供基于通用 KDTree 的三维兼容包装类型。
// 说明:
//   - 该文件位于 common/spatial 下，作为项目内三维空间索引的统一入口；
//   - 保留历史上的 KDTree3D API，供 MVS 过滤与点云构建流程继续复用；
//   - 具体索引实现复用通用模板 xjw::common::spatial::KDTree<3, float>。
// =============================================================================

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "spatial/KDTree.h"

namespace xjw
{
namespace common
{
namespace spatial
{

class KDTree3D
{
public:
    KDTree3D() = default;

    struct Point3
    {
        float v[3];
    };

    void build(const float *coords, int pointCount)
    {
        std::vector<CommonTree3D::Point> points;
        points.reserve(static_cast<std::size_t>(std::max(0, pointCount)));
        for (int index = 0; index < pointCount; ++index)
        {
            typename CommonTree3D::Point point;
            point.coords = {coords[index * 3], coords[index * 3 + 1], coords[index * 3 + 2]};
            point.index = index;
            points.push_back(point);
        }
        _tree.build(points);
    }

    void build(const std::vector<std::array<float, 3>> &points)
    {
        _tree.build(points);
    }

    float knnMeanDist(int queryIndex, int k) const
    {
        const std::vector<CommonNeighbor> neighbors =
            _tree.kNearestByPointIndex(static_cast<std::size_t>(queryIndex), static_cast<std::size_t>(k));
        if (neighbors.empty())
        {
            return 1e9f;
        }

        float sum = 0.0f;
        for (const CommonNeighbor &neighbor : neighbors)
        {
            sum += std::sqrt(neighbor.distanceSquared);
        }
        return sum / static_cast<float>(neighbors.size());
    }

    std::vector<float> knnDists2(int queryIndex, int k) const
    {
        const std::vector<CommonNeighbor> neighbors =
            _tree.kNearestByPointIndex(static_cast<std::size_t>(queryIndex), static_cast<std::size_t>(k));

        std::vector<float> result;
        result.reserve(neighbors.size());
        for (const CommonNeighbor &neighbor : neighbors)
        {
            result.push_back(neighbor.distanceSquared);
        }
        return result;
    }

    int radiusCount(int queryIndex, float radius, int earlyStop = std::numeric_limits<int>::max()) const
    {
        return _tree.radiusCountByPointIndex(static_cast<std::size_t>(queryIndex), radius, earlyStop);
    }

    int size() const
    {
        return static_cast<int>(_tree.size());
    }

private:
    using CommonTree3D = KDTree<3, float>;
    using CommonNeighbor = CommonTree3D::Neighbor;

private:
    CommonTree3D _tree;
};

} // namespace spatial
} // namespace common
} // namespace xjw