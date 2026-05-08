#pragma once

// ============================================================
// 文件：KDTree2D.h
// 功能：提供基于通用 KDTree 的二维兼容包装类型。
// 说明：
//   - 该文件位于 common/spatial 下，作为项目内二维空间索引的统一入口；
//   - 保留历史上的 KDPoint2D / KDTree2D 接口，减少上层业务改动；
//   - 具体索引实现复用通用模板 xjw::common::spatial::KDTree<2, double>。
// ============================================================

#include <vector>

#include "KDTree.h"

namespace xjw
{
namespace common
{
namespace spatial
{

// ============================================================
// 结构体：KDPoint2D
// 描述：二维空间点，记录 XY 坐标与原始点索引。
// ============================================================
struct KDPoint2D
{
    double x = 0.0;
    double y = 0.0;
    int index = -1;
};

// ============================================================
// 类：KDTree2D
// 描述：二维 KD 树兼容包装类，供 overlap / DEM 等模块直接使用。
// ============================================================
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