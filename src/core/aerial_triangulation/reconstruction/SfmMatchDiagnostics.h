#pragma once

/**
 * @file SfmMatchDiagnostics.h
 * @brief 匹配图连通性诊断和已知位姿引导匹配候选规划。
 *
 * 这里刻意区分两张图：
 * 1. candidateGraph：候选对图，表示工作流计划检查哪些影像对；
 * 2. actualMatchGraph：实际匹配图，只包含已经载入且存在几何内点的影像对。
 *
 * 候选图连通不代表 SfM 可解。只有 actualMatchGraph 的连通分量、边密度和内点数
 * 才能说明影像之间存在可用于恢复位姿的观测。所有函数均为无状态纯计算，便于 CLI、
 * GUI 和测试共享完全一致的诊断语义。
 */

#include <QSet>
#include <QString>
#include <QStringList>
#include <QMap>
#include <QPair>
#include <QVector>

#include <algorithm>
#include <functional>

namespace xjw::aerial_triangulation
{

struct SfmMatchDiagnosticPair
{
    int imageA = -1; ///< 第一张影像在当前 SfM 输入中的稳定整数标识。
    int imageB = -1; ///< 第二张影像在当前 SfM 输入中的稳定整数标识。
    int matchCount = 0; ///< 已通过匹配文件几何验证的内点数。
    bool loaded = false; ///< 是否已完成该 pair 的缓存/匹配文件检查。
    bool skippedByNoMatchCache = false; ///< 是否被“已确认无匹配”负缓存直接跳过。
};

/// 无向影像匹配图的基础拓扑统计。
struct SfmMatchGraphStats
{
    int nodeCount = 0; ///< 纳入诊断的影像节点数，包含孤立影像。
    int edgeCount = 0; ///< 去重、去自环后的无向边数。
    int componentCount = 0; ///< 连通分量数；完整网络通常期望为 1。
    int largestComponentSize = 0; ///< 最大连通分量包含的影像数。
    int isolatedNodeCount = 0; ///< 度为 0 的影像数。
    QVector<int> componentSizes; ///< 各分量大小，按降序排列。
};

/// 一轮匹配缓存扫描完成后的分类统计和两张匹配图。
struct SfmMatchDiagnostics
{
    int totalPairs = 0; ///< 计划检查的 pair 总数。
    int actualMatchPairs = 0; ///< 已载入且 matchCount > 0 的 pair 数。
    int noMatchCacheSkippedPairs = 0; ///< 命中负缓存的 pair 数。
    int pendingPairs = 0; ///< 尚未载入、当前状态未知的 pair 数。
    int emptyLoadedPairs = 0; ///< 已载入但没有几何内点的 pair 数。
    SfmMatchGraphStats candidateGraph; ///< 由 totalPairs 构成的候选图。
    SfmMatchGraphStats actualMatchGraph; ///< 仅由 actualMatchPairs 构成的有效匹配图。
};

/**
 * @brief 已知相机位姿下的二次引导匹配规划参数。
 *
 * 引导匹配只补救弱边或缺失边，不重复计算已经具有足量内点的健康边。若
 * registeredImageIds 非空，只有两端都已注册的 pair 才能利用基础矩阵定义极线带。
 */
struct SfmGuidedMatchPlannerOptions
{
    QSet<int> registeredImageIds; ///< 空集合表示调用方允许所有影像参与。
    int minSeedMatches = 80; ///< 达到该内点数的边作为健康种子，不进入补匹配。
    int maxHealthyMatches = 60; ///< 不高于该值的已载入边视为弱边。
    int maxCandidates = 2000; ///< 计划最多保留的补匹配 pair 数；0 表示禁用。
};

/// 一条需要用当前相机几何重新匹配的影像对。
struct SfmGuidedMatchCandidate
{
    int imageA = -1; ///< 第一张影像标识。
    int imageB = -1; ///< 第二张影像标识。
    int matchCount = 0; ///< 当前已有几何内点数。
    QString reason; ///< 稳定机器可读原因，例如 weak_geometric_inliers。
    double priorityScore = 0.0; ///< 越大越先补救；只用于排序，不是概率。
    bool canUseEpipolarBand = false; ///< 两端位姿可用，可将搜索限制在极线带内。
};

/// 引导匹配候选及规划过程中的过滤计数。
struct SfmGuidedMatchPlan
{
    QVector<SfmGuidedMatchCandidate> candidates; ///< 按 priorityScore 降序排列。
    int seedPairCount = 0; ///< 达到 minSeedMatches 的健康种子边数。
    int skippedHealthyPairs = 0; ///< 因已有足够内点而跳过的 pair 数。
    int skippedUnregisteredPairs = 0; ///< 因至少一端无位姿而无法极线引导的 pair 数。
};

/**
 * @brief 分析指定节点和无向边构成的匹配图。
 *
 * 边会先规范化为 min(id):max(id)，并去除自环、未知节点和重复边。即使 imageIds
 * 中的节点没有边，也会作为大小为 1 的连通分量进入统计。
 */
inline SfmMatchGraphStats analyzeSfmMatchGraph(const QVector<int> &imageIds,
                                               const QVector<QPair<int, int>> &edges)
{
    SfmMatchGraphStats stats;
    QSet<int> nodes;
    for (const int id : imageIds)
    {
        nodes.insert(id);
    }

    QMap<int, QVector<int>> adjacency;
    for (const int id : nodes)
    {
        adjacency.insert(id, QVector<int>());
    }

    QSet<QString> uniqueEdges;
    for (const QPair<int, int> &edge : edges)
    {
        if (edge.first == edge.second || !nodes.contains(edge.first) || !nodes.contains(edge.second))
        {
            continue;
        }

        const int a = std::min(edge.first, edge.second);
        const int b = std::max(edge.first, edge.second);
        const QString key = QStringLiteral("%1:%2").arg(a).arg(b);
        if (uniqueEdges.contains(key))
        {
            continue;
        }
        uniqueEdges.insert(key);

        adjacency[a].append(b);
        adjacency[b].append(a);
    }

    QSet<int> visited;
    for (const int start : nodes)
    {
        if (visited.contains(start))
        {
            continue;
        }

        QVector<int> stack;
        stack.append(start);
        visited.insert(start);
        int size = 0;

        while (!stack.isEmpty())
        {
            const int current = stack.takeLast();
            ++size;

            for (const int next : adjacency.value(current))
            {
                if (!visited.contains(next))
                {
                    visited.insert(next);
                    stack.append(next);
                }
            }
        }

        stats.componentSizes.append(size);
    }

    std::sort(stats.componentSizes.begin(), stats.componentSizes.end(), std::greater<int>());
    stats.nodeCount = static_cast<int>(nodes.size());
    stats.edgeCount = static_cast<int>(uniqueEdges.size());
    stats.componentCount = static_cast<int>(stats.componentSizes.size());
    stats.largestComponentSize = stats.componentSizes.isEmpty() ? 0 : stats.componentSizes.front();
    stats.isolatedNodeCount = 0;
    for (const int size : stats.componentSizes)
    {
        if (size == 1)
        {
            ++stats.isolatedNodeCount;
        }
    }

    return stats;
}

/**
 * @brief 将 pair 的缓存状态分类，并同时计算候选图和实际匹配图。
 *
 * skippedByNoMatchCache 只有在 loaded=true 时才计入负缓存；loaded=false 始终视为
 * pending。matchCount 只有在成功载入且未命中负缓存时才形成 actualMatchGraph 边。
 */
inline SfmMatchDiagnostics analyzeSfmMatchDiagnostics(
    const QVector<int> &imageIds,
    const QVector<SfmMatchDiagnosticPair> &pairs)
{
    SfmMatchDiagnostics diagnostics;
    QVector<QPair<int, int>> candidateEdges;
    QVector<QPair<int, int>> actualEdges;
    candidateEdges.reserve(pairs.size());
    actualEdges.reserve(pairs.size());

    diagnostics.totalPairs = static_cast<int>(pairs.size());
    for (const SfmMatchDiagnosticPair &pair : pairs)
    {
        candidateEdges.append(qMakePair(pair.imageA, pair.imageB));

        if (!pair.loaded)
        {
            ++diagnostics.pendingPairs;
        }
        else if (pair.skippedByNoMatchCache)
        {
            ++diagnostics.noMatchCacheSkippedPairs;
        }
        else if (pair.matchCount > 0)
        {
            ++diagnostics.actualMatchPairs;
            actualEdges.append(qMakePair(pair.imageA, pair.imageB));
        }
        else
        {
            ++diagnostics.emptyLoadedPairs;
        }
    }

    diagnostics.candidateGraph = analyzeSfmMatchGraph(imageIds, candidateEdges);
    diagnostics.actualMatchGraph = analyzeSfmMatchGraph(imageIds, actualEdges);
    return diagnostics;
}

inline bool sfmGuidedMatchingHasRegisteredPose(const SfmGuidedMatchPlannerOptions &options, int imageId)
{
    return options.registeredImageIds.isEmpty() || options.registeredImageIds.contains(imageId);
}

/**
 * @brief 根据缓存状态和当前注册位姿生成二次引导匹配计划。
 *
 * 优先级顺序为：负缓存边、弱几何内点边、尚未扫描的边。负缓存可以在相机几何
 * 已知后重新尝试，因为极线约束可能使原先的全局描述子搜索变得可解。函数不会
 * 修改匹配缓存，只返回确定性的候选顺序。
 */
inline SfmGuidedMatchPlan planSfmGuidedMatching(const QVector<int> &imageIds,
                                                const QVector<SfmMatchDiagnosticPair> &pairs,
                                                const SfmGuidedMatchPlannerOptions &options)
{
    Q_UNUSED(imageIds);

    SfmGuidedMatchPlan plan;
    const int minSeedMatches = std::max(1, options.minSeedMatches);
    const int maxHealthyMatches = std::max(0, options.maxHealthyMatches);
    const int maxCandidates = std::max(0, options.maxCandidates);

    for (const SfmMatchDiagnosticPair &pair : pairs)
    {
        if (pair.loaded && !pair.skippedByNoMatchCache && pair.matchCount >= minSeedMatches)
        {
            ++plan.seedPairCount;
            ++plan.skippedHealthyPairs;
            continue;
        }

        if (!sfmGuidedMatchingHasRegisteredPose(options, pair.imageA) ||
            !sfmGuidedMatchingHasRegisteredPose(options, pair.imageB))
        {
            ++plan.skippedUnregisteredPairs;
            continue;
        }

        SfmGuidedMatchCandidate candidate;
        candidate.imageA = pair.imageA;
        candidate.imageB = pair.imageB;
        candidate.matchCount = pair.matchCount;
        candidate.canUseEpipolarBand = true;

        if (pair.skippedByNoMatchCache)
        {
            candidate.reason = QStringLiteral("skipped_no_match_cache");
            candidate.priorityScore = 200.0;
        }
        else if (!pair.loaded)
        {
            candidate.reason = QStringLiteral("pending_match_file");
            candidate.priorityScore = 120.0;
        }
        else if (pair.matchCount <= maxHealthyMatches)
        {
            candidate.reason = QStringLiteral("weak_geometric_inliers");
            candidate.priorityScore = 160.0 - static_cast<double>(std::max(0, pair.matchCount));
        }
        else
        {
            ++plan.skippedHealthyPairs;
            continue;
        }

        plan.candidates.append(candidate);
    }

    std::sort(plan.candidates.begin(), plan.candidates.end(),
              [](const SfmGuidedMatchCandidate &lhs, const SfmGuidedMatchCandidate &rhs)
    {
        if (lhs.priorityScore != rhs.priorityScore)
        {
            return lhs.priorityScore > rhs.priorityScore;
        }
        if (lhs.imageA != rhs.imageA)
        {
            return lhs.imageA < rhs.imageA;
        }
        return lhs.imageB < rhs.imageB;
    });

    if (maxCandidates > 0 && plan.candidates.size() > maxCandidates)
    {
        plan.candidates.resize(maxCandidates);
    }
    else if (maxCandidates == 0)
    {
        plan.candidates.clear();
    }

    return plan;
}

/// 把降序分量大小压缩成日志字段；超出 maxCount 的部分以省略号表示。
inline QString formatSfmComponentSizes(const QVector<int> &componentSizes, int maxCount = 8)
{
    QStringList parts;
    const int count = std::min(maxCount, static_cast<int>(componentSizes.size()));
    for (int i = 0; i < count; ++i)
    {
        parts.append(QString::number(componentSizes.at(i)));
    }
    if (componentSizes.size() > maxCount)
    {
        parts.append(QStringLiteral("..."));
    }
    return parts.join(QStringLiteral(","));
}

} // namespace xjw::aerial_triangulation
