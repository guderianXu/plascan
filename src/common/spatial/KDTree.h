#pragma once

// =============================================================================
// 文件: spatial/KDTree.h
// 模块: common / spatial
// 说明:
//   提供一个通用、头文件实现的 K-D 树（KD-Tree）空间索引模块。
//   该实现面向低维空间点集（2D / 3D / N 维）查询，适合点云处理、重叠度分析、
//   邻域统计、最近邻检索等需要空间索引的公共场景。
//
//   设计目标:
//   1. 通用性: 使用模板参数 Dimension 支持任意固定维度。
//   2. 轻依赖: 仅依赖 C++17 标准库，不依赖 Qt / OpenCV / PCL。
//   3. 可复用: 放在 src/common，供 core / gui / tools 多模块共享。
//   4. 易接入: 既支持直接传入坐标数组，也支持以“点索引”为查询入口。
//   5. 线程安全: 建树完成后为只读结构，可被多线程并发查询。
//
//   复杂度:
//   - 建树: 平均 O(n log n)
//   - 最近邻: 平均 O(log n)，最坏 O(n)
//   - K 近邻: 平均 O(log n + k log k)
//   - 半径查询: 平均 O(log n + m)，m 为命中点数
//
//   适用场景:
//   - 点云最近邻 / 邻域搜索
//   - 稀疏/稠密点云半径统计
//   - 2D 地理点 / 影像中心点的空间近邻查询
//   - 需要公共空间索引但又不希望绑定第三方库的场景
// =============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

namespace xjw::common::spatial
{

/**
 * @brief K-D 树中的一个点。
 *
 * `coords` 保存固定维度坐标，`index` 保存该点在原始输入数组中的索引。
 * 查询接口统一返回该 `index`，调用方可以据此回溯自己的原始数据结构。
 */
template <std::size_t Dimension, typename Scalar = double>
struct KDPoint
{
    static_assert(Dimension > 0, "KDPoint Dimension must be greater than zero");
    static_assert(std::is_arithmetic<Scalar>::value, "KDPoint Scalar must be arithmetic");

    std::array<Scalar, Dimension> coords{};
    int index = -1;
};

/**
 * @brief 邻居搜索结果。
 *
 * `index` 为原始点索引，`distanceSquared` 为与查询点的平方距离。
 * 使用平方距离可以避免不必要的开方开销，调用方如需真实欧氏距离，
 * 可在外部自行对该值取平方根。
 */
template <typename Scalar>
struct KDNeighbor
{
    int index = -1;
    Scalar distanceSquared = Scalar(0);
};

/**
 * @brief 通用固定维度 K-D 树。
 *
 * 模板参数:
 * - Dimension: 空间维度，例如 2 表示二维，3 表示三维。
 * - Scalar:    坐标标量类型，通常为 float 或 double。
 *
 * 使用方式示例:
 * @code
 * using Tree3D = xjw::common::spatial::KDTree<3, double>;
 * Tree3D tree;
 * std::vector<Tree3D::Point> pts;
 * pts.push_back({{0.0, 0.0, 0.0}, 0});
 * pts.push_back({{1.0, 0.0, 0.0}, 1});
 * tree.build(pts);
 * auto nearest = tree.nearest({0.2, 0.0, 0.0});
 * @endcode
 */
template <std::size_t Dimension, typename Scalar = double>
class KDTree
{
public:
    static_assert(Dimension > 0, "KDTree Dimension must be greater than zero");
    static_assert(std::is_arithmetic<Scalar>::value, "KDTree Scalar must be arithmetic");

    using Point = KDPoint<Dimension, Scalar>;
    using CoordinateArray = std::array<Scalar, Dimension>;
    using Neighbor = KDNeighbor<Scalar>;

public:
    KDTree() = default;

    /// @brief 直接用带原始索引的点集建树。
    explicit KDTree(const std::vector<Point> &points)
    {
        build(points);
    }

    /// @brief 用纯坐标数组建树，索引自动按输入顺序赋值。
    explicit KDTree(const std::vector<CoordinateArray> &coords)
    {
        build(coords);
    }

    /**
     * @brief 用显式点集建树。
     *
     * 调用后，树内部会完整拷贝输入点，后续查询不依赖调用方持有原数组。
     */
    void build(const std::vector<Point> &points)
    {
        _points = points;
        rebuild();
    }

    /**
     * @brief 用纯坐标数组建树，并自动生成顺序索引。
     *
     * 输入第 i 个坐标会自动被赋予 index=i，便于调用方按输入顺序回溯。
     */
    void build(const std::vector<CoordinateArray> &coords)
    {
        _points.clear();
        _points.reserve(coords.size());

        assert(coords.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
               "KDTree: point count exceeds int range");
        for (std::size_t index = 0; index < coords.size(); ++index)
        {
            Point point;
            point.coords = coords[index];
            point.index = static_cast<int>(index);
            _points.push_back(point);
        }

        rebuild();
    }

    /// @brief 清空树和内部点集。
    void clear()
    {
        _points.clear();
        _nodes.clear();
        _root = -1;
    }

    /// @brief 判断树是否为空。
    bool empty() const
    {
        return _points.empty();
    }

    /// @brief 返回点数量。
    std::size_t size() const
    {
        return _points.size();
    }

    /// @brief 只读访问内部点集。
    const std::vector<Point> &points() const
    {
        return _points;
    }

    /**
     * @brief 查询与指定坐标最近的点。
     *
     * @param query       查询坐标。
     * @param outDistance 可选输出，返回欧氏距离而非平方距离。
     * @return 最近点的原始索引；空树时返回 -1。
     */
    int nearest(const CoordinateArray &query, Scalar *outDistance = nullptr) const
    {
        if (_root < 0)
        {
            return -1;
        }

        int bestPoint = -1;
        Scalar bestDistSq = std::numeric_limits<Scalar>::max();
        nearestRecursive(_root, query, -1, bestPoint, bestDistSq);

        if (outDistance)
        {
            *outDistance = std::sqrt(bestDistSq);
        }

        return bestPoint < 0 ? -1 : _points[static_cast<std::size_t>(bestPoint)].index;
    }

    /**
     * @brief 以“内部点下标”为查询对象，查找最近邻。
     *
     * 该接口适合点云自身邻域分析，会自动跳过查询点自身。
     * 参数使用内部点数组下标，而不是原始 index 值。
     */
    int nearestByPointIndex(std::size_t pointIndex, Scalar *outDistance = nullptr) const
    {
        if (pointIndex >= _points.size())
        {
            return -1;
        }

        return nearestSkipping(_points[pointIndex].coords, static_cast<int>(pointIndex), outDistance);
    }

    /**
     * @brief 查询 k 个最近邻。
     *
     * 返回结果按距离从近到远排序。若树为空或 k=0，则返回空数组。
     */
    std::vector<Neighbor> kNearest(const CoordinateArray &query, std::size_t k) const
    {
        return kNearestSkipping(query, k, -1);
    }

    /**
     * @brief 以内部点下标为查询对象，查询 k 个最近邻并跳过自身。
     */
    std::vector<Neighbor> kNearestByPointIndex(std::size_t pointIndex, std::size_t k) const
    {
        if (pointIndex >= _points.size())
        {
            return {};
        }

        return kNearestSkipping(_points[pointIndex].coords, k, static_cast<int>(pointIndex));
    }

    /**
     * @brief 半径查询，返回落入半径内的原始索引集合。
     *
     * 查询结果顺序不保证按距离排序，如需排序应由调用方自行处理。
     */
    std::vector<int> radiusSearch(const CoordinateArray &query, Scalar radius) const
    {
        std::vector<int> result;
        radiusSearchInternal(query, radius, -1, &result, nullptr, std::numeric_limits<int>::max());
        return result;
    }

    /**
     * @brief 以内部点下标为查询对象执行半径查询，并跳过自身。
     */
    std::vector<int> radiusSearchByPointIndex(std::size_t pointIndex, Scalar radius) const
    {
        if (pointIndex >= _points.size())
        {
            return {};
        }

        std::vector<int> result;
        radiusSearchInternal(_points[pointIndex].coords,
                             radius,
                             static_cast<int>(pointIndex),
                             &result,
                             nullptr,
                             std::numeric_limits<int>::max());
        return result;
    }

    /**
     * @brief 统计半径内邻居数量，并支持达到 earlyStop 后提前终止。
     *
     * 该接口适合离群点过滤或密度估计，避免生成完整结果数组。
     */
    int radiusCount(const CoordinateArray &query, Scalar radius, int earlyStop = std::numeric_limits<int>::max()) const
    {
        int count = 0;
        radiusSearchInternal(query, radius, -1, nullptr, &count, earlyStop);
        return count;
    }

    /**
     * @brief 以内部点下标为查询对象统计半径内邻居数，并跳过自身。
     */
    int radiusCountByPointIndex(std::size_t pointIndex,
                                Scalar radius,
                                int earlyStop = std::numeric_limits<int>::max()) const
    {
        if (pointIndex >= _points.size())
        {
            return 0;
        }

        int count = 0;
        radiusSearchInternal(_points[pointIndex].coords,
                             radius,
                             static_cast<int>(pointIndex),
                             nullptr,
                             &count,
                             earlyStop);
        return count;
    }

private:
    /**
     * @brief K-D 树节点。
     *
     * `point` 保存当前节点对应的点在 `_points` 中的下标；
     * `axis` 保存划分轴；`left/right` 保存左右子树在 `_nodes` 中的下标。
     * 采用“数组 + 整数下标”形式可减少碎片，并提高缓存局部性。
     */
    struct Node
    {
        int point = -1;
        int axis = 0;
        int left = -1;
        int right = -1;
    };

    /**
     * @brief K 近邻查询的堆元素。
     *
     * 使用“大根堆”维护当前最差的候选邻居，便于快速剪枝。
     */
    struct HeapEntry
    {
        Scalar distanceSquared = Scalar(0);
        int point = -1;

        bool operator<(const HeapEntry &other) const
        {
            return distanceSquared < other.distanceSquared;
        }
    };

private:
    void rebuild()
    {
        _nodes.clear();
        _root = -1;
        if (_points.empty())
        {
            return;
        }

        std::vector<int> indices(_points.size());
        std::iota(indices.begin(), indices.end(), 0);
        _nodes.reserve(_points.size());
        _root = buildRecursive(indices, 0, static_cast<int>(indices.size()));
    }

    /**
     * @brief 递归建树。
     *
     * 每次在当前子数组上选择“跨度最大”的维度作为划分轴，再使用
     * `std::nth_element` 选取中位数点作为当前节点，以尽量保持树平衡。
     */
    int buildRecursive(std::vector<int> &indices, int begin, int end)
    {
        if (begin >= end)
        {
            return -1;
        }

        const int axis = selectSplitAxis(indices, begin, end);
        const int mid = (begin + end) / 2;

        std::nth_element(indices.begin() + begin,
                         indices.begin() + mid,
                         indices.begin() + end,
                         [this, axis](int lhs, int rhs) {
                             return _points[static_cast<std::size_t>(lhs)].coords[static_cast<std::size_t>(axis)] <
                                    _points[static_cast<std::size_t>(rhs)].coords[static_cast<std::size_t>(axis)];
                         });

        Node node;
        node.point = indices[mid];
        node.axis = axis;

        const int nodeIndex = static_cast<int>(_nodes.size());
        _nodes.push_back(node);
        _nodes[static_cast<std::size_t>(nodeIndex)].left = buildRecursive(indices, begin, mid);
        _nodes[static_cast<std::size_t>(nodeIndex)].right = buildRecursive(indices, mid + 1, end);
        return nodeIndex;
    }

    /// @brief 选择当前子数组中跨度最大的维度作为划分轴。
    int selectSplitAxis(const std::vector<int> &indices, int begin, int end) const
    {
        CoordinateArray minValues{};
        CoordinateArray maxValues{};
        minValues.fill(std::numeric_limits<Scalar>::max());
        maxValues.fill(std::numeric_limits<Scalar>::lowest());

        for (int i = begin; i < end; ++i)
        {
            const Point &point = _points[static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])];
            for (std::size_t axis = 0; axis < Dimension; ++axis)
            {
                minValues[axis] = std::min(minValues[axis], point.coords[axis]);
                maxValues[axis] = std::max(maxValues[axis], point.coords[axis]);
            }
        }

        int bestAxis = 0;
        Scalar bestExtent = maxValues[0] - minValues[0];
        for (std::size_t axis = 1; axis < Dimension; ++axis)
        {
            const Scalar extent = maxValues[axis] - minValues[axis];
            if (extent > bestExtent)
            {
                bestExtent = extent;
                bestAxis = static_cast<int>(axis);
            }
        }

        return bestAxis;
    }

    /// @brief 计算两点的欧氏距离平方。
    static Scalar distanceSquared(const CoordinateArray &lhs, const CoordinateArray &rhs)
    {
        Scalar result = Scalar(0);
        for (std::size_t axis = 0; axis < Dimension; ++axis)
        {
            const Scalar delta = lhs[axis] - rhs[axis];
            result += delta * delta;
        }
        return result;
    }

    int nearestSkipping(const CoordinateArray &query, int skipPoint, Scalar *outDistance) const
    {
        if (_root < 0)
        {
            return -1;
        }

        int bestPoint = -1;
        Scalar bestDistSq = std::numeric_limits<Scalar>::max();
        nearestRecursive(_root, query, skipPoint, bestPoint, bestDistSq);

        if (outDistance)
        {
            *outDistance = bestPoint >= 0 ? std::sqrt(bestDistSq) : Scalar(0);
        }

        return bestPoint < 0 ? -1 : _points[static_cast<std::size_t>(bestPoint)].index;
    }

    /**
     * @brief 递归最近邻搜索。
     *
     * 搜索策略:
     * 1. 先访问查询点所在侧子树，尽快收缩当前最优距离。
     * 2. 再根据查询点到划分超平面的距离决定是否需要访问另一侧。
     */
    void nearestRecursive(int nodeIndex,
                          const CoordinateArray &query,
                          int skipPoint,
                          int &bestPoint,
                          Scalar &bestDistSq) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        const Node &node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point &point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (currentDistSq < bestDistSq)
            {
                bestDistSq = currentDistSq;
                bestPoint = node.point;
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        nearestRecursive(nearChild, query, skipPoint, bestPoint, bestDistSq);

        if (delta * delta < bestDistSq)
        {
            nearestRecursive(farChild, query, skipPoint, bestPoint, bestDistSq);
        }
    }

    std::vector<Neighbor> kNearestSkipping(const CoordinateArray &query, std::size_t k, int skipPoint) const
    {
        if (_root < 0 || k == 0)
        {
            return {};
        }

        std::priority_queue<HeapEntry> heap;
        kNearestRecursive(_root, query, k, skipPoint, heap);

        std::vector<Neighbor> result(heap.size());
        for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i)
        {
            const HeapEntry entry = heap.top();
            heap.pop();
            result[static_cast<std::size_t>(i)] = Neighbor{
                _points[static_cast<std::size_t>(entry.point)].index,
                entry.distanceSquared
            };
        }
        return result;
    }

    /**
     * @brief 递归 K 近邻搜索。
     *
     * 通过固定大小的大根堆维护当前最差候选；当远侧子树与当前最优超球体
     * 相交时，再继续搜索远侧。
     */
    void kNearestRecursive(int nodeIndex,
                           const CoordinateArray &query,
                           std::size_t k,
                           int skipPoint,
                           std::priority_queue<HeapEntry> &heap) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        const Node &node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point &point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (heap.size() < k)
            {
                heap.push(HeapEntry{currentDistSq, node.point});
            }
            else if (currentDistSq < heap.top().distanceSquared)
            {
                heap.pop();
                heap.push(HeapEntry{currentDistSq, node.point});
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        kNearestRecursive(nearChild, query, k, skipPoint, heap);

        if (heap.size() < k || delta * delta < heap.top().distanceSquared)
        {
            kNearestRecursive(farChild, query, k, skipPoint, heap);
        }
    }

    void radiusSearchInternal(const CoordinateArray &query,
                              Scalar radius,
                              int skipPoint,
                              std::vector<int> *results,
                              int *count,
                              int earlyStop) const
    {
        if (_root < 0 || radius <= Scalar(0))
        {
            return;
        }

        const Scalar radiusSq = radius * radius;
        radiusRecursive(_root, query, radiusSq, skipPoint, results, count, earlyStop);
    }

    /**
     * @brief 递归半径搜索。
     *
     * 当调用方只关心“数量”时，`results` 为 nullptr，仅累计计数；
     * 当达到 `earlyStop` 时可提前终止，适合密度判定和离群点过滤。
     */
    void radiusRecursive(int nodeIndex,
                         const CoordinateArray &query,
                         Scalar radiusSq,
                         int skipPoint,
                         std::vector<int> *results,
                         int *count,
                         int earlyStop) const
    {
        if (nodeIndex < 0)
        {
            return;
        }

        if (count && *count >= earlyStop)
        {
            return;
        }

        const Node &node = _nodes[static_cast<std::size_t>(nodeIndex)];
        const Point &point = _points[static_cast<std::size_t>(node.point)];

        if (node.point != skipPoint)
        {
            const Scalar currentDistSq = distanceSquared(query, point.coords);
            if (currentDistSq <= radiusSq)
            {
                if (results)
                {
                    results->push_back(point.index);
                }

                if (count)
                {
                    ++(*count);
                    if (*count >= earlyStop)
                    {
                        return;
                    }
                }
            }
        }

        const std::size_t axis = static_cast<std::size_t>(node.axis);
        const Scalar delta = query[axis] - point.coords[axis];
        const int nearChild = (delta < Scalar(0)) ? node.left : node.right;
        const int farChild = (delta < Scalar(0)) ? node.right : node.left;

        radiusRecursive(nearChild, query, radiusSq, skipPoint, results, count, earlyStop);

        if ((!count || *count < earlyStop) && delta * delta <= radiusSq)
        {
            radiusRecursive(farChild, query, radiusSq, skipPoint, results, count, earlyStop);
        }
    }

private:
    std::vector<Point> _points;
    std::vector<Node> _nodes;
    int _root = -1;
};

using KDTree2Dd = KDTree<2, double>;
using KDTree2Df = KDTree<2, float>;
using KDTree3Dd = KDTree<3, double>;
using KDTree3Df = KDTree<3, float>;

} // namespace xjw::common::spatial