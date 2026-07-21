#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

namespace xjw
{
namespace detail
{

class DisjointSet
{
public:
    struct MergeResult
    {
        int root = -1;
        int absorbedRoot = -1;
        bool merged = false;
    };

    explicit DisjointSet(int size = 0)
        : _parent(static_cast<std::size_t>(std::max(0, size))),
          _rank(static_cast<std::size_t>(std::max(0, size)), 0)
    {
        std::iota(_parent.begin(), _parent.end(), 0);
    }

    int add()
    {
        const int index = static_cast<int>(_parent.size());
        _parent.push_back(index);
        _rank.push_back(0);
        return index;
    }

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
