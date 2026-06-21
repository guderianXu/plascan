#pragma once

// ============================================================
// 文件：KDTree2D.h
// 功能：提供二维兼容包装类型，底层复用 PlaPoint 通用空间 KDTree。
// ============================================================

#include <plapoint/search/spatial_kdtree.h>

#include <array>
#include <cstddef>
#include <vector>

namespace xjw
{
namespace common
{
namespace spatial
{

template <std::size_t Dimension, typename Scalar = double>
using KDPoint = plapoint::search::SpatialPoint<Dimension, Scalar>;

template <typename Scalar>
using KDNeighbor = plapoint::search::SpatialNeighbor<Scalar>;

template <std::size_t Dimension, typename Scalar = double>
using KDTree = plapoint::search::SpatialKdTree<Dimension, Scalar>;

using KDTree2Dd = KDTree<2, double>;
using KDTree2Df = KDTree<2, float>;
using KDTree3Dd = KDTree<3, double>;
using KDTree3Df = KDTree<3, float>;

// ============================================================
// KDTree2D 历史包装类型
// ============================================================

struct KDPoint2D
{
    double x = 0.0;
    double y = 0.0;
    int index = -1;
};

class KDTree2D
{
public:
    KDTree2D() = default;

    explicit KDTree2D(const std::vector<KDPoint2D> &points)
    {
        build(points);
    }

    void build(const std::vector<KDPoint2D> &points)
    {
        std::vector<CommonTree2D::Point> commonPoints;
        commonPoints.reserve(points.size());
        for (const KDPoint2D &point : points)
        {
            commonPoints.push_back(CommonTree2D::Point{{point.x, point.y}, point.index});
        }
        _tree.build(commonPoints);
    }

    bool empty() const
    {
        return _tree.empty();
    }

    int nearest(double x, double y, double *outDistance = nullptr) const
    {
        return _tree.nearest({x, y}, outDistance);
    }

    std::vector<int> radiusSearch(double x, double y, double radius) const
    {
        return _tree.radiusSearch({x, y}, radius);
    }

private:
    using CommonTree2D = KDTree<2, double>;

private:
    CommonTree2D _tree;
};

} // namespace spatial
} // namespace common
} // namespace xjw
