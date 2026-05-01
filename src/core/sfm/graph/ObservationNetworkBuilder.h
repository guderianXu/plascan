#pragma once
// =============================================================================
// 文件: ObservationNetworkBuilder.h
// 模块: core/sfm
// 说明:
//   从图像对两两匹配结果构建"观测网络图"（Observation Network Graph）。
//
//   支持算法:
//     Complete  — 所有满足阈值的边全部保留（密集图，适合小数据集）
//     KNN       — 每个节点保留匹配数最多的前 K 条边（暴力 KNN）
//     MST       — 以匹配数为权重的最大生成树（保证连通性的稀疏图）
//     Spatial   — 按图像采集顺序（序列索引）滑动窗口 ±K 连接
//     KDTree    — 在GPS坐标空间（或序列索引空间）构建KD树，
//                 查找空间最近的 K 个邻居后保留有匹配的边
// =============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xjw
{

// ---------------------------------------------------------------------------
// 数据结构
// ---------------------------------------------------------------------------

/// 候选输入边（来自两两匹配结果）
struct MatchEdge
{
    int idx0;            ///< 图像列表中的索引
    int idx1;            ///< 图像列表中的索引（idx0 < idx1）
    int numMatches;      ///< 特征匹配点数
    double overlapScore; ///< 可选重叠率，无则置 0
    double inlierRatio;  ///< 内点率，值域 [0,1]；无则置 1.0
};

/// 网络输出边
struct NetworkEdge
{
    int idx0, idx1;
    int numMatches;
    double weight; ///< 归一化权重 [0,1]（用于可视化着色）
};

/// 图像的可选 GPS 坐标（WGS-84）
struct GpsCoord
{
    double lat = 0.0;
    double lon = 0.0;
    bool valid = false;
};

/// 构建好的观测网络
struct ObservationNetwork
{
    std::vector<std::string> nodeNames; ///< 每个节点对应的图像文件名
    std::vector<NetworkEdge> edges;
    std::vector<int> degrees; ///< 每个节点的度数
    int numNodes() const
    {
        return static_cast<int>(nodeNames.size());
    }

    int numEdges() const
    {
        return static_cast<int>(edges.size());
    }
};

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------

struct ObservationNetworkConfig
{
    enum Algorithm
    {
        Complete,
        KNN,
        MST,
        Spatial,
        KDTree
    };

    Algorithm algorithm = KNN;
    int k = 20;               ///< KNN/Spatial/KDTree 的邻居数
    int minMatches = 30;      ///< 建边最少匹配数
    double minOverlap = 0.10; ///< 建边最低重叠率（0 = 不检查）
    bool pruneWeak = true;
    double pruneThresh = 0.15; ///< 内点率低于此值的边被剪枝
};

// ---------------------------------------------------------------------------
// 构建器
// ---------------------------------------------------------------------------

class ObservationNetworkBuilder
{
  public:
    /**
     * @brief 从候选边集合构建观测网络。
     *
     * @param nodeNames  节点名称（图像文件基名）
     * @param edges      所有候选输入边（来自匹配结果）
     * @param gps        可选 GPS 坐标列表（长度需与 nodeNames 一致，或为空）
     * @param cfg        构建配置
     * @return 构建完成的 ObservationNetwork
     */
    static ObservationNetwork build(const std::vector<std::string> &nodeNames, const std::vector<MatchEdge> &edges,
                                    const std::vector<GpsCoord> &gps, const ObservationNetworkConfig &cfg);

  private:
    // ── 各算法实现 ──
    static std::vector<NetworkEdge> runComplete(int n, const std::vector<MatchEdge> &edges,
                                                const ObservationNetworkConfig &cfg);

    static std::vector<NetworkEdge> runKNN(int n, const std::vector<MatchEdge> &edges,
                                           const ObservationNetworkConfig &cfg);

    static std::vector<NetworkEdge> runMST(int n, const std::vector<MatchEdge> &edges,
                                           const ObservationNetworkConfig &cfg);

    static std::vector<NetworkEdge> runSpatial(int n, const std::vector<MatchEdge> &edges,
                                               const ObservationNetworkConfig &cfg);

    static std::vector<NetworkEdge> runKDTree(int n, const std::vector<MatchEdge> &edges,
                                              const std::vector<GpsCoord> &gps, const ObservationNetworkConfig &cfg);

    // ── 工具 ──
    /// 剪枝内点率过低的边
    static void pruneEdges(std::vector<NetworkEdge> &edges, const std::vector<MatchEdge> &src,
                           const ObservationNetworkConfig &cfg);

    /// 归一化权重 → [0,1]
    static void normalizeWeights(std::vector<NetworkEdge> &edges);

    /// 计算每节点度数
    static std::vector<int> computeDegrees(int n, const std::vector<NetworkEdge> &edges);

    // ── KDTree 辅助（内部实现，2D 点集） ──
    struct KDNode
    {
        double x, y; ///< 2D 坐标
        int index;   ///< 对应 nodeNames 的下标
    };

    /// 构建 2D KDTree，返回重排后的节点数组；depth 参数用于递归
    static void buildKD(std::vector<KDNode> &nodes, int lo, int hi, int depth);

    /// 在 KDTree 中查找 (qx,qy) 的 k 个最近邻（不含 queryIdx 自身）
    static void queryKD(const std::vector<KDNode> &nodes, int lo, int hi, int depth, double qx, double qy, int queryIdx,
                        int k, std::vector<std::pair<double, int>> &result);

    // ── Union-Find（MST 用） ──
    struct UnionFind
    {
        std::vector<int> parent;
        std::vector<int> rankValues;
        explicit UnionFind(int n);
        int find(int x);
        bool unite(int a, int b);
    };
};

} // namespace xjw
