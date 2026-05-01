// =============================================================================
// 文件: ObservationNetworkBuilder.cpp
// =============================================================================

#include "ObservationNetworkBuilder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace xjw
{

// ===========================================================================
// 公共入口
// ===========================================================================

ObservationNetwork ObservationNetworkBuilder::build(const std::vector<std::string> &nodeNames,
                                                    const std::vector<MatchEdge> &edges,
                                                    const std::vector<GpsCoord> &gps,
                                                    const ObservationNetworkConfig &cfg)
{
    const int n = static_cast<int>(nodeNames.size());
    const bool hasOverlapData =
        std::any_of(edges.begin(), edges.end(), [](const MatchEdge &e) { return e.overlapScore > 0.0; });
    const bool useOverlapFilter = cfg.minOverlap > 0.0 && hasOverlapData;

    // 第一步：按最少匹配数过滤
    std::vector<MatchEdge> filtered;
    filtered.reserve(edges.size());
    for (const auto &e : edges)
    {
        if (e.numMatches >= cfg.minMatches && (!useOverlapFilter || e.overlapScore >= cfg.minOverlap))
        {
            filtered.push_back(e);
        }
    }

    // 第二步：选择算法
    std::vector<NetworkEdge> selected;
    switch (cfg.algorithm)
    {
    case ObservationNetworkConfig::Complete:
        selected = runComplete(n, filtered, cfg);
        break;
    case ObservationNetworkConfig::KNN:
        selected = runKNN(n, filtered, cfg);
        break;
    case ObservationNetworkConfig::MST:
        selected = runMST(n, filtered, cfg);
        break;
    case ObservationNetworkConfig::Spatial:
        selected = runSpatial(n, filtered, cfg);
        break;
    case ObservationNetworkConfig::KDTree:
        selected = runKDTree(n, filtered, gps, cfg);
        break;
    }

    // 第三步：选做弱边剪枝
    if (cfg.pruneWeak)
    {
        pruneEdges(selected, filtered, cfg);
    }

    // 第四步：归一化权重
    normalizeWeights(selected);

    // 组装结果
    ObservationNetwork net;
    net.nodeNames = nodeNames;
    net.edges = std::move(selected);
    net.degrees = computeDegrees(n, net.edges);
    return net;
}

// ===========================================================================
// Complete 算法 — 保留所有满足过滤条件的边
// ===========================================================================

std::vector<NetworkEdge> ObservationNetworkBuilder::runComplete(int /*n*/, const std::vector<MatchEdge> &edges,
                                                                const ObservationNetworkConfig & /*cfg*/)
{
    std::vector<NetworkEdge> out;
    out.reserve(edges.size());
    for (const auto &e : edges)
    {
        out.push_back({e.idx0, e.idx1, e.numMatches, 0.0});
    }
    return out;
}

// ===========================================================================
// KNN 算法 — 每个节点保留匹配数最多的前 K 个邻居
// ===========================================================================

std::vector<NetworkEdge> ObservationNetworkBuilder::runKNN(int n, const std::vector<MatchEdge> &edges,
                                                           const ObservationNetworkConfig &cfg)
{
    // 邻接列表：node -> [(numMatches, neighborIdx)]
    std::vector<std::vector<std::pair<int, int>>> adj(n);
    for (const auto &e : edges)
    {
        adj[e.idx0].emplace_back(e.numMatches, e.idx1);
        adj[e.idx1].emplace_back(e.numMatches, e.idx0);
    }

    // 每个节点只保留 top-K
    const int k = cfg.k;
    std::vector<std::vector<int>> kept(n);
    for (int i = 0; i < n; ++i)
    {
        auto &nbrs = adj[i];
        if ((int)nbrs.size() > k)
        {
            std::partial_sort(nbrs.begin(), nbrs.begin() + k, nbrs.end(),
                              [](const auto &a, const auto &b)
                              {
                                  return a.first > b.first; // 降序
                              });
            nbrs.resize(k);
        }
        for (const auto &[cnt, j] : nbrs)
        {
            int a = std::min(i, j), b = std::max(i, j);
            kept[a].push_back(b);
        }
    }

    // 去重（双向选择可能重复）
    std::unordered_map<long long, int> edgeMap;
    for (const auto &e : edges)
    {
        int a = std::min(e.idx0, e.idx1);
        int b = std::max(e.idx0, e.idx1);
        long long key = ((long long)a << 32) | (unsigned)b;
        edgeMap[key] = e.numMatches;
    }

    std::vector<NetworkEdge> out;
    std::unordered_map<long long, bool> added;
    for (int i = 0; i < n; ++i)
    {
        std::sort(kept[i].begin(), kept[i].end());
        kept[i].erase(std::unique(kept[i].begin(), kept[i].end()), kept[i].end());
        for (int j : kept[i])
        {
            int a = std::min(i, j), b = std::max(i, j);
            long long key = ((long long)a << 32) | (unsigned)b;
            if (!added.count(key))
            {
                added[key] = true;
                int nm = edgeMap.count(key) ? edgeMap[key] : 0;
                out.push_back({a, b, nm, 0.0});
            }
        }
    }
    return out;
}

// ===========================================================================
// MST 算法 — Kruskal 最大生成树（最大化连通性）
// ===========================================================================

ObservationNetworkBuilder::UnionFind::UnionFind(int n) : parent(n), rankValues(n, 0)
{
    std::iota(parent.begin(), parent.end(), 0);
}

int ObservationNetworkBuilder::UnionFind::find(int x)
{
    while (parent[x] != x)
    {
        parent[x] = parent[parent[x]]; // 路径压缩
        x = parent[x];
    }
    return x;
}

bool ObservationNetworkBuilder::UnionFind::unite(int a, int b)
{
    a = find(a);
    b = find(b);
    if (a == b)
    {
        return false;
    }
    if (rankValues[a] < rankValues[b])
    {
        std::swap(a, b);
    }
    parent[b] = a;
    if (rankValues[a] == rankValues[b])
    {
        ++rankValues[a];
    }
    return true;
}

std::vector<NetworkEdge> ObservationNetworkBuilder::runMST(int n, const std::vector<MatchEdge> &edges,
                                                           const ObservationNetworkConfig & /*cfg*/)
{
    // 按匹配数降序排序（最大生成树）
    std::vector<const MatchEdge *> sorted;
    sorted.reserve(edges.size());
    for (const auto &e : edges)
    {
        sorted.push_back(&e);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const MatchEdge *a, const MatchEdge *b) { return a->numMatches > b->numMatches; });

    UnionFind uf(n);
    std::vector<NetworkEdge> out;
    out.reserve(n - 1);
    for (const auto *e : sorted)
    {
        if ((int)out.size() >= n - 1)
        {
            break;
        }
        if (uf.unite(e->idx0, e->idx1))
        {
            out.push_back({e->idx0, e->idx1, e->numMatches, 0.0});
        }
    }
    return out;
}

// ===========================================================================
// Spatial 算法 — 滑动窗口（按图像序列索引）
// ===========================================================================

std::vector<NetworkEdge> ObservationNetworkBuilder::runSpatial(int /*n*/, const std::vector<MatchEdge> &edges,
                                                               const ObservationNetworkConfig &cfg)
{
    const int k = cfg.k;
    std::vector<NetworkEdge> out;
    for (const auto &e : edges)
    {
        if (std::abs(e.idx0 - e.idx1) <= k)
        {
            out.push_back({e.idx0, e.idx1, e.numMatches, 0.0});
        }
    }
    return out;
}

// ===========================================================================
// KDTree 算法 — 在 GPS 或序列坐标空间中构建 2D KDTree，
//               找到空间最近的 K 个邻居后过滤有效匹配边
// ===========================================================================

void ObservationNetworkBuilder::buildKD(std::vector<KDNode> &nodes, int lo, int hi, int depth)
{
    if (hi - lo <= 1)
    {
        return;
    }
    int axis = depth % 2;
    int mid = (lo + hi) / 2;
    std::nth_element(nodes.begin() + lo, nodes.begin() + mid, nodes.begin() + hi,
                     [axis](const KDNode &a, const KDNode &b) { return axis == 0 ? a.x < b.x : a.y < b.y; });
    buildKD(nodes, lo, mid, depth + 1);
    buildKD(nodes, mid + 1, hi, depth + 1);
}

void ObservationNetworkBuilder::queryKD(const std::vector<KDNode> &nodes, int lo, int hi, int depth, double qx,
                                        double qy, int queryIdx, int k, std::vector<std::pair<double, int>> &result)
{
    if (hi - lo <= 0)
    {
        return;
    }

    int axis = depth % 2;
    int mid = (lo + hi) / 2;

    const KDNode &node = nodes[mid];
    if (node.index != queryIdx)
    {
        double dx = node.x - qx, dy = node.y - qy;
        double d2 = dx * dx + dy * dy;
        if ((int)result.size() < k)
        {
            result.emplace_back(d2, node.index);
            if ((int)result.size() == k)
                std::make_heap(result.begin(), result.end()); // max-heap on distance
        }
        else if (d2 < result.front().first)
        {
            std::pop_heap(result.begin(), result.end());
            result.back() = {d2, node.index};
            std::push_heap(result.begin(), result.end());
        }
    }

    // 判断哪侧更近
    double splitVal = (axis == 0) ? node.x : node.y;
    double qVal = (axis == 0) ? qx : qy;
    int nearSide = (qVal <= splitVal) ? 0 : 1; // 0=left,1=right

    auto recurse = [&](int side)
    {
        if (side == 0 && lo < mid)
        {
            queryKD(nodes, lo, mid, depth + 1, qx, qy, queryIdx, k, result);
        }
        else if (side == 1 && mid + 1 < hi)
        {
            queryKD(nodes, mid + 1, hi, depth + 1, qx, qy, queryIdx, k, result);
        }
    };

    recurse(nearSide);

    // 检查另一侧是否可能更近
    double planeDist = qVal - splitVal;
    double maxDist2 = result.empty() ? std::numeric_limits<double>::max() : result.front().first;
    if (planeDist * planeDist < maxDist2)
    {
        recurse(1 - nearSide);
    }
}

std::vector<NetworkEdge> ObservationNetworkBuilder::runKDTree(int n, const std::vector<MatchEdge> &edges,
                                                              const std::vector<GpsCoord> &gps,
                                                              const ObservationNetworkConfig &cfg)
{
    // 构建带坐标的节点列表
    bool hasGps =
        ((int)gps.size() == n) && std::any_of(gps.begin(), gps.end(), [](const GpsCoord &g) { return g.valid; });

    std::vector<KDNode> kdNodes(n);
    for (int i = 0; i < n; ++i)
    {
        kdNodes[i].index = i;
        if (hasGps && gps[i].valid)
        {
            kdNodes[i].x = gps[i].lat;
            kdNodes[i].y = gps[i].lon;
        }
        else
        {
            // 无 GPS：使用序列索引作为 1D 坐标，y=0
            kdNodes[i].x = static_cast<double>(i);
            kdNodes[i].y = 0.0;
        }
    }

    // 构建 KD 树
    buildKD(kdNodes, 0, n, 0);

    // 构建匹配查找映射: (min,max) -> numMatches
    std::unordered_map<long long, int> matchMap;
    for (const auto &e : edges)
    {
        int a = std::min(e.idx0, e.idx1);
        int b = std::max(e.idx0, e.idx1);
        long long key = ((long long)a << 32) | (unsigned)b;
        matchMap[key] = e.numMatches;
    }

    // 为每个节点查询 K 近邻，收集有匹配的边
    std::unordered_map<long long, bool> added;
    std::vector<NetworkEdge> out;
    const int k = cfg.k;

    for (int i = 0; i < n; ++i)
    {
        std::vector<std::pair<double, int>> nbrs;
        nbrs.reserve(k + 1);
        queryKD(kdNodes, 0, n, 0, kdNodes[i].x, kdNodes[i].y, i, k, nbrs);

        for (const auto &[distanceSquared, j] : nbrs)
        {
            (void)distanceSquared;
            int a = std::min(i, j), b = std::max(i, j);
            long long key = ((long long)a << 32) | (unsigned)b;
            if (added.count(key))
            {
                continue;
            }
            auto it = matchMap.find(key);
            if (it == matchMap.end())
            {
                continue; // 无实际匹配则跳过
            }
            added[key] = true;
            out.push_back({a, b, it->second, 0.0});
        }
    }
    return out;
}

// ===========================================================================
// 工具函数
// ===========================================================================

void ObservationNetworkBuilder::pruneEdges(std::vector<NetworkEdge> &edges, const std::vector<MatchEdge> &src,
                                           const ObservationNetworkConfig &cfg)
{
    // 建立 (a,b)->inlierRatio 映射
    std::unordered_map<long long, double> inlierMap;
    for (const auto &e : src)
    {
        int a = std::min(e.idx0, e.idx1);
        int b = std::max(e.idx0, e.idx1);
        long long key = ((long long)a << 32) | (unsigned)b;
        inlierMap[key] = e.inlierRatio;
    }

    edges.erase(std::remove_if(edges.begin(), edges.end(),
                               [&](const NetworkEdge &ne)
                               {
                                   int a = std::min(ne.idx0, ne.idx1);
                                   int b = std::max(ne.idx0, ne.idx1);
                                   long long key = ((long long)a << 32) | (unsigned)b;
                                   auto it = inlierMap.find(key);
                                   if (it == inlierMap.end())
                                   {
                                       return false;
                                   }
                                   return it->second < cfg.pruneThresh;
                               }),
                edges.end());
}

void ObservationNetworkBuilder::normalizeWeights(std::vector<NetworkEdge> &edges)
{
    if (edges.empty())
    {
        return;
    }
    int maxM = 0;
    for (const auto &e : edges)
    {
        maxM = std::max(maxM, e.numMatches);
    }
    if (maxM == 0)
    {
        return;
    }
    double inv = 1.0 / maxM;
    for (auto &e : edges)
    {
        e.weight = std::min(1.0, e.numMatches * inv);
    }
}

std::vector<int> ObservationNetworkBuilder::computeDegrees(int n, const std::vector<NetworkEdge> &edges)
{
    std::vector<int> deg(n, 0);
    for (const auto &e : edges)
    {
        ++deg[e.idx0];
        ++deg[e.idx1];
    }
    return deg;
}

} // namespace xjw
