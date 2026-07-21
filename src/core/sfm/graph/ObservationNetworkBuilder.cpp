// =============================================================================
// 文件: ObservationNetworkBuilder.cpp
// =============================================================================

#include "ObservationNetworkBuilder.h"
#include "common/DisjointSet.h"

#include <plapoint/search/spatial_kdtree.h>

#include <algorithm>
#include <cmath>
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

std::vector<MatchEdge> ObservationNetworkBuilder::selectStrongConnectedCore(
    int numNodes,
    const std::vector<MatchEdge> &edges,
    int strongMinMatches)
{
    if (numNodes <= 1 || edges.empty())
    {
        return edges;
    }

    detail::DisjointSet disjointSet(numNodes);
    std::vector<MatchEdge> strongEdges;
    strongEdges.reserve(edges.size());
    for (const MatchEdge &edge : edges)
    {
        if (edge.idx0 < 0 || edge.idx0 >= numNodes || edge.idx1 < 0 || edge.idx1 >= numNodes ||
            edge.numMatches < strongMinMatches)
        {
            continue;
        }
        strongEdges.push_back(edge);
        disjointSet.unite(edge.idx0, edge.idx1);
    }

    if (strongEdges.empty())
    {
        return edges;
    }

    const int root = disjointSet.find(0);
    for (int node = 1; node < numNodes; ++node)
    {
        if (disjointSet.find(node) != root)
        {
            return edges;
        }
    }
    return strongEdges;
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

    detail::DisjointSet disjointSet(n);
    std::vector<NetworkEdge> out;
    out.reserve(n - 1);
    for (const auto *e : sorted)
    {
        if ((int)out.size() >= n - 1)
        {
            break;
        }
        if (disjointSet.unite(e->idx0, e->idx1).merged)
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

std::vector<NetworkEdge> ObservationNetworkBuilder::runKDTree(int n, const std::vector<MatchEdge> &edges,
                                                              const std::vector<GpsCoord> &gps,
                                                              const ObservationNetworkConfig &cfg)
{
    // 构建带坐标的节点列表
    bool hasGps =
        ((int)gps.size() == n) && std::any_of(gps.begin(), gps.end(), [](const GpsCoord &g) { return g.valid; });

    using SpatialTree = plapoint::search::SpatialKdTree<2, double>;
    std::vector<SpatialTree::Point> spatialPoints(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        SpatialTree::Point &point = spatialPoints[static_cast<std::size_t>(i)];
        point.index = i;
        if (hasGps && gps[i].valid)
        {
            point.coords = {gps[i].lat, gps[i].lon};
        }
        else
        {
            // 无 GPS：使用序列索引作为 1D 坐标，y=0
            point.coords = {static_cast<double>(i), 0.0};
        }
    }

    const SpatialTree spatialTree(spatialPoints);

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
    const std::size_t k = static_cast<std::size_t>(std::max(0, cfg.k));

    for (int i = 0; i < n; ++i)
    {
        std::vector<SpatialTree::Neighbor> neighbors =
            spatialTree.kNearestByPointIndex(static_cast<std::size_t>(i), k);
        std::sort(neighbors.begin(), neighbors.end(), [](const auto &left, const auto &right)
        {
            if (left.distanceSquared != right.distanceSquared)
            {
                return left.distanceSquared < right.distanceSquared;
            }
            return left.index < right.index;
        });

        for (const SpatialTree::Neighbor &neighbor : neighbors)
        {
            const int j = neighbor.index;
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
