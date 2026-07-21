#pragma once

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
    int imageA = -1;
    int imageB = -1;
    int matchCount = 0;
    bool loaded = false;
    bool skippedByNoMatchCache = false;
};

struct SfmMatchGraphStats
{
    int nodeCount = 0;
    int edgeCount = 0;
    int componentCount = 0;
    int largestComponentSize = 0;
    int isolatedNodeCount = 0;
    QVector<int> componentSizes;
};

struct SfmMatchDiagnostics
{
    int totalPairs = 0;
    int actualMatchPairs = 0;
    int noMatchCacheSkippedPairs = 0;
    int pendingPairs = 0;
    int emptyLoadedPairs = 0;
    SfmMatchGraphStats candidateGraph;
    SfmMatchGraphStats actualMatchGraph;
};

struct SfmGuidedMatchPlannerOptions
{
    QSet<int> registeredImageIds;
    int minSeedMatches = 80;
    int maxHealthyMatches = 60;
    int maxCandidates = 2000;
};

struct SfmGuidedMatchCandidate
{
    int imageA = -1;
    int imageB = -1;
    int matchCount = 0;
    QString reason;
    double priorityScore = 0.0;
    bool canUseEpipolarBand = false;
};

struct SfmGuidedMatchPlan
{
    QVector<SfmGuidedMatchCandidate> candidates;
    int seedPairCount = 0;
    int skippedHealthyPairs = 0;
    int skippedUnregisteredPairs = 0;
};

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

inline SfmMatchDiagnostics analyzeSfmMatchDiagnostics(const QVector<int> &imageIds,
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
