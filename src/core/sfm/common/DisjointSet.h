#pragma once

/**
 * @file DisjointSet.h
 * @brief 多视轨迹构建使用的确定性并查集。
 *
 * 除路径压缩和按秩合并外，同秩时固定选择较小根索引，保证相同匹配输入在不同
 * 线程数和标准库实现下产生稳定组件编号，便于缓存、测试和结果对比。
 */

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace xjw
{
namespace detail
{

/// SfM 内部工具，不暴露到公共模块 API。
class DisjointSet
{
public:
    struct MergeResult
    {
        int root = -1; ///< 合并后的代表根。
        int absorbedRoot = -1; ///< 被吸收的旧根；未合并时等于 root。
        bool merged = false; ///< 两元素原先属于不同集合时为 true。
    };

    explicit DisjointSet(int size = 0)
        : _parent(static_cast<std::size_t>(std::max(0, size))),
          _rank(static_cast<std::size_t>(std::max(0, size)), 0)
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    /// 追加一个独立集合并返回其连续索引。
    int add()
    {
        const int index = static_cast<int>(_parent.size());
        _parent.push_back(index);
        _rank.push_back(0);
        return index;
    }

    /// 查找代表根并对访问路径执行压缩。
    int find(int index)
    {
        assert(index >= 0 && index < static_cast<int>(_parent.size()));
        int root = index;
        while (_parent[static_cast<std::size_t>(root)] != root)
        {
            root = _parent[static_cast<std::size_t>(root)];
        }
        while (_parent[static_cast<std::size_t>(index)] != index)
        {
            const int next = _parent[static_cast<std::size_t>(index)];
            _parent[static_cast<std::size_t>(index)] = root;
            index = next;
        }
        return root;
    }

    /// 按秩合并两个集合；同秩时以较小根保证确定性。
    MergeResult unite(int left, int right)
    {
        int leftRoot = find(left);
        int rightRoot = find(right);
        if (leftRoot == rightRoot)
        {
            return {leftRoot, leftRoot, false};
        }

        const int leftRank = _rank[static_cast<std::size_t>(leftRoot)];
        const int rightRank = _rank[static_cast<std::size_t>(rightRoot)];
        if (leftRank < rightRank || (leftRank == rightRank && rightRoot < leftRoot))
        {
            std::swap(leftRoot, rightRoot);
        }

        _parent[static_cast<std::size_t>(rightRoot)] = leftRoot;
        if (leftRank == rightRank)
        {
            ++_rank[static_cast<std::size_t>(leftRoot)];
        }
        return {leftRoot, rightRoot, true};
    }

private:
    std::vector<int> _parent;
    std::vector<int> _rank;
};

} // namespace detail
} // namespace xjw
