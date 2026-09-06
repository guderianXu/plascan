#pragma once

#include <QString>
#include <QStringList>

#include <vector>

namespace xjw
{
    namespace matchphotos
    {

        // 候选来源采用累积方式，而不是后来的来源覆盖先前来源。
        // 同一影像对可能同时来自通用预选、参考预选或显式序列等多个信号。
        enum class PairSource
        {
            Manual,
            Exhaustive,
            SequenceWindow,
            PlaMatchGeneric,
            PlaMatchReferenceSource,
            PlaMatchReferenceEstimated,
            PlaMatchReferenceSequential,
            GuidedRematch
        };

        struct ImagePair
        {
            // 指向 PairSelectionInput::images 的下标。
            // normalized() 会把较小下标放在前面，保证影像对键和诊断输出稳定。
            int indexA = -1;
            int indexB = -1;

            bool isValid(int imageCount) const;
            ImagePair normalized() const;
        };

        struct PairCandidate
        {
            // pairKey 是下游匹配代码使用的稳定身份标识。
            // 下标形式的影像对保留给内存中的规划、排序和诊断报告使用。
            ImagePair pair;
            QString pairKey;

            // 当策略限制 maxPairs 时，优先级更高的影像对会先进入匹配。
            // 各类分数字段用于保留“为什么选择这个影像对”的证据。
            std::vector<PairSource> sources;
            double priorityScore = 0.0;
            double sequenceScore = 0.0;
            QString detail;
        };

        struct PairSelectionResult
        {
            // restrictPairs=false 表示调用方可以执行全量两两匹配。
            // 为 true 时，allowedPairKeys 就是传给匹配阶段的精确候选列表。
            bool restrictPairs = false;
            int imageCount = 0;
            int allPairCount = 0;
            std::vector<PairCandidate> candidates;
            QStringList allowedPairKeys;
            QString detail;
        };

        QString pairSourceId(PairSource source);
        QString pairSourceDisplayName(PairSource source);
        QString canonicalImagePath(const QString& path);
        QString makePairKey(const QString& pathA, const QString& pathB);
        QString makePairKey(const QStringList& images, int indexA, int indexB);
        void appendPairSource(PairCandidate* candidate, PairSource source);

    } // namespace matchphotos
} // namespace xjw
