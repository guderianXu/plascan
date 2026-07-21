#include "PairSelector.h"

#include <QSet>

#include <algorithm>

namespace xjw
{
namespace matchphotos
{
namespace
{

int allPairCount(int imageCount)
{
    return imageCount > 1 ? imageCount * (imageCount - 1) / 2 : 0;
}

PairCandidate *findCandidate(std::vector<PairCandidate> *candidates, const QString &pairKey)
{
    if (!candidates || pairKey.isEmpty())
    {
        return nullptr;
    }

    for (PairCandidate &candidate : *candidates)
    {
        if (candidate.pairKey == pairKey)
        {
            return &candidate;
        }
    }
    return nullptr;
}

PairCandidate *addOrUpdateCandidate(PairSelectionResult *result,
                                    const QStringList &images,
                                    int indexA,
                                    int indexB,
                                    PairSource source,
                                    double priorityScore)
{
    if (!result)
    {
        return nullptr;
    }

    const ImagePair pair{indexA, indexB};
    if (!pair.isValid(images.size()))
    {
        return nullptr;
    }

    const QString pairKey = makePairKey(images, indexA, indexB);
    if (pairKey.isEmpty())
    {
        return nullptr;
    }

    // 所有候选生成器最终都汇聚到这里。每个影像对只保留一条记录，
    // 这样后续来源可以继续提升优先级并补充自己的诊断分数。
    PairCandidate *candidate = findCandidate(&result->candidates, pairKey);
    if (!candidate)
    {
        PairCandidate created;
        created.pair = pair.normalized();
        created.pairKey = pairKey;
        result->candidates.push_back(created);
        candidate = &result->candidates.back();
    }

    appendPairSource(candidate, source);
    candidate->priorityScore += std::max(0.0, priorityScore);
    return candidate;
}

void addManualCandidates(PairSelectionResult *result, const QStringList &pairKeys)
{
    if (!result)
    {
        return;
    }

    QSet<QString> seen;
    for (const QString &pairKey : pairKeys)
    {
        const QString trimmed = pairKey.trimmed();
        if (trimmed.isEmpty() || seen.contains(trimmed))
        {
            continue;
        }

        seen.insert(trimmed);
        // 手动影像对键可能来自 GUI 状态或 .lis 文件，此时不一定有影像下标。
        // 因此这里仅要求 pairKey 有效。
        PairCandidate candidate;
        candidate.pairKey = trimmed;
        candidate.priorityScore = 1000.0;
        candidate.detail = QStringLiteral("manual pair");
        appendPairSource(&candidate, PairSource::Manual);
        result->candidates.push_back(candidate);
    }
}

void addExhaustiveCandidates(PairSelectionResult *result, const QStringList &images)
{
    const int imageCount = static_cast<int>(images.size());
    for (int i = 0; i < imageCount; ++i)
    {
        for (int j = i + 1; j < imageCount; ++j)
        {
            addOrUpdateCandidate(result, images, i, j, PairSource::Exhaustive, 10.0);
        }
    }
}

void addSequenceCandidates(PairSelectionResult *result, const QStringList &images, int window)
{
    const int safeWindow = std::max(1, window);
    const int imageCount = static_cast<int>(images.size());
    for (int i = 0; i < imageCount; ++i)
    {
        const int last = std::min(imageCount - 1, i + safeWindow);
        for (int j = i + 1; j <= last; ++j)
        {
            const int distance = j - i;
            // 对有序航拍/巡视器序列，相邻影像更可能重叠；
            // 因此距离越近的邻居给略高排序分。
            const double sequenceScore =
                static_cast<double>(safeWindow - distance + 1) / static_cast<double>(safeWindow);
            PairCandidate *candidate =
                addOrUpdateCandidate(result, images, i, j, PairSource::SequenceWindow, 40.0 + 10.0 * sequenceScore);
            if (candidate)
            {
                candidate->sequenceScore = std::max(candidate->sequenceScore, sequenceScore);
            }
        }
    }
}

void addKnownCameraOverlapPairs(PairSelectionResult *result,
                                const QStringList &images,
                                const std::vector<std::array<int, 2>> &pairs)
{
    for (const auto &pair : pairs)
    {
        // 这些影像对已经被上游相机/元数据阶段接受。
        // 即使没有数值化 overlap score，也把它们视为强证据。
        PairCandidate *candidate =
            addOrUpdateCandidate(result, images, pair[0], pair[1], PairSource::CameraOverlap, 100.0);
        if (candidate)
        {
            candidate->overlapScore = std::max(candidate->overlapScore, 1.0);
        }
    }
}

void addOverlapResultPairs(PairSelectionResult *result,
                           const QStringList &images,
                           const OverlapAnalysisResult &overlapResult)
{
    for (const OverlapPairResult &pair : overlapResult.pairs)
    {
        PairCandidate *candidate = addOrUpdateCandidate(result,
                                                        images,
                                                        pair.indexA,
                                                        pair.indexB,
                                                        PairSource::CameraOverlap,
                                                        100.0 + 50.0 * pair.overlapScore);
        if (candidate)
        {
            candidate->overlapScore = std::max(candidate->overlapScore, pair.overlapScore);
        }
    }
}

void addVocabularyPairs(PairSelectionResult *result,
                        const QStringList &images,
                        const VocabularyOverlapResult &vocabularyResult)
{
    // 优先使用 retriever 的 acceptedPairs。
    // 如果只有 candidates，则只保留 accepted=true 的候选，避免把被拒绝的诊断行
    // 静默变成真正要执行的匹配任务。
    const std::vector<VocabularyOverlapPairResult> &pairs =
        vocabularyResult.acceptedPairs.empty() ? vocabularyResult.candidates : vocabularyResult.acceptedPairs;
    for (const VocabularyOverlapPairResult &pair : pairs)
    {
        if (!pair.accepted)
        {
            continue;
        }

        PairCandidate *candidate = addOrUpdateCandidate(result,
                                                        images,
                                                        pair.indexA,
                                                        pair.indexB,
                                                        PairSource::VocabularyOverlap,
                                                        80.0 + 40.0 * pair.bowScore);
        if (candidate)
        {
            candidate->vocabularyScore = std::max(candidate->vocabularyScore, pair.bowScore);
            candidate->sharedWordCount = std::max(candidate->sharedWordCount, pair.sharedWordCount);
            candidate->geometricInliers = std::max(candidate->geometricInliers, pair.geometricInliers);
        }
    }
}

int findComponentRoot(std::vector<int> *parents, int index)
{
    if (!parents || index < 0 || index >= static_cast<int>(parents->size()))
    {
        return index;
    }

    int root = index;
    while ((*parents)[root] != root)
    {
        root = (*parents)[root];
    }

    while ((*parents)[index] != index)
    {
        const int parent = (*parents)[index];
        (*parents)[index] = root;
        index = parent;
    }
    return root;
}

void uniteComponents(std::vector<int> *parents, int lhs, int rhs)
{
    const int rootL = findComponentRoot(parents, lhs);
    const int rootR = findComponentRoot(parents, rhs);
    if (rootL != rootR)
    {
        (*parents)[rootR] = rootL;
    }
}

int addSequenceBridgeCandidates(PairSelectionResult *result, const QStringList &images, int window)
{
    if (!result || result->candidates.empty() || images.size() < 2)
    {
        return 0;
    }

    const int imageCount = static_cast<int>(images.size());
    std::vector<int> parents(static_cast<std::size_t>(imageCount));
    for (int i = 0; i < imageCount; ++i)
    {
        parents[static_cast<std::size_t>(i)] = i;
    }

    for (const PairCandidate &candidate : result->candidates)
    {
        if (candidate.pair.isValid(imageCount))
        {
            uniteComponents(&parents, candidate.pair.indexA, candidate.pair.indexB);
        }
    }

    auto componentCount = [&]() -> int
    {
        QSet<int> roots;
        for (int i = 0; i < imageCount; ++i)
        {
            roots.insert(findComponentRoot(&parents, i));
        }
        return roots.size();
    };

    if (componentCount() <= 1)
    {
        return 0;
    }

    int added = 0;
    const int safeWindow = std::max(1, window);
    for (int distance = 1; distance <= safeWindow && componentCount() > 1; ++distance)
    {
        for (int i = 0; i + distance < imageCount && componentCount() > 1; ++i)
        {
            const int j = i + distance;
            const int rootI = findComponentRoot(&parents, i);
            const int rootJ = findComponentRoot(&parents, j);
            if (rootI == rootJ)
            {
                continue;
            }

            // 通用/词汇预选可能只在局部分量内召回影像对。这里仅补跨分量的
            // 序列桥接边，给后续匹配和 SfM 一次把分量接起来的机会。
            const double sequenceScore =
                static_cast<double>(safeWindow - distance + 1) / static_cast<double>(safeWindow);
            PairCandidate *candidate =
                addOrUpdateCandidate(result, images, i, j, PairSource::SequenceWindow, 40.0 + 10.0 * sequenceScore);
            if (candidate)
            {
                candidate->sequenceScore = std::max(candidate->sequenceScore, sequenceScore);
                candidate->detail = QStringLiteral("sequence bridge for disconnected preselection graph");
            }
            uniteComponents(&parents, i, j);
            ++added;
        }
    }

    return added;
}

void finalizePairSelection(PairSelectionResult *result, int maxPairs)
{
    if (!result)
    {
        return;
    }

    // 人工审查和调试时需要稳定输出：先按 priorityScore 决定计划顺序，
    // 同分时用 pairKey 保证跨运行顺序确定。
    std::sort(result->candidates.begin(), result->candidates.end(),
              [](const PairCandidate &lhs, const PairCandidate &rhs)
    {
        if (lhs.priorityScore != rhs.priorityScore)
        {
            return lhs.priorityScore > rhs.priorityScore;
        }
        return lhs.pairKey < rhs.pairKey;
    });

    if (maxPairs > 0 && static_cast<int>(result->candidates.size()) > maxPairs)
    {
        result->candidates.resize(static_cast<std::size_t>(maxPairs));
    }

    result->allowedPairKeys.clear();
    for (const PairCandidate &candidate : result->candidates)
    {
        if (!candidate.pairKey.isEmpty())
        {
            result->allowedPairKeys.append(candidate.pairKey);
        }
    }
}

bool coversAllImagePairs(const PairSelectionResult &result, const QStringList &images)
{
    if (result.allPairCount <= 0 || result.allowedPairKeys.size() != result.allPairCount)
    {
        return false;
    }

    QSet<QString> allowedKeys;
    allowedKeys.reserve(result.allowedPairKeys.size());
    for (const QString &pairKey : result.allowedPairKeys)
    {
        allowedKeys.insert(pairKey);
    }

    for (int indexA = 0; indexA < images.size(); ++indexA)
    {
        for (int indexB = indexA + 1; indexB < images.size(); ++indexB)
        {
            const QString pairKey = makePairKey(images, indexA, indexB);
            if (pairKey.isEmpty() || !allowedKeys.contains(pairKey))
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace

PairSelectionResult PairSelector::select(const PairSelectionInput &input,
                                         const PairSelectionPolicy &policy,
                                         QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }

    PairSelectionResult result;
    result.imageCount = input.images.size();
    result.allPairCount = allPairCount(result.imageCount);

    if (result.imageCount < 2 && input.manualPairKeys.isEmpty())
    {
        result.detail = QStringLiteral("影像数量不足，未生成候选对");
        return result;
    }

    const bool manualOnly = policy.mode == PairSelectionMode::ManualOnly;
    if (!input.manualPairKeys.isEmpty())
    {
        addManualCandidates(&result, input.manualPairKeys);
        result.restrictPairs = true;
    }

    // 对小数据集，全量匹配仍然是最干净的选择：
    // 在缺少其它证据前，不提前丢弃任何可能的影像对。
    const bool hasExternalPreselection =
        (policy.includeCameraOverlap &&
         (!input.knownCameraOverlapPairs.empty() || input.cameraOverlapResult)) ||
        (policy.includeVocabularyOverlap && input.vocabularyOverlapResult);
    const bool useExhaustive =
        policy.mode == PairSelectionMode::Exhaustive ||
        (policy.mode == PairSelectionMode::Auto &&
         result.imageCount <= std::max(2, policy.exhaustiveMaxImages) &&
         !hasExternalPreselection &&
         !manualOnly);

    if (!manualOnly && useExhaustive)
    {
        addExhaustiveCandidates(&result, input.images);
        result.restrictPairs = false;
    }

    if (!manualOnly && policy.includeCameraOverlap)
    {
        // 相机重叠和词汇召回候选是叠加信号。
        // 它们可以提升已经由全量/手动来源加入的影像对。
        addKnownCameraOverlapPairs(&result, input.images, input.knownCameraOverlapPairs);
        if (input.cameraOverlapResult)
        {
            addOverlapResultPairs(&result, input.images, *input.cameraOverlapResult);
        }
    }

    if (!manualOnly && policy.includeVocabularyOverlap && input.vocabularyOverlapResult)
    {
        addVocabularyPairs(&result, input.images, *input.vocabularyOverlapResult);
    }

    if (!manualOnly &&
        !useExhaustive &&
        (policy.mode == PairSelectionMode::Sequence ||
         (policy.mode == PairSelectionMode::Auto &&
          result.candidates.empty() &&
          !hasExternalPreselection &&
          policy.useSequenceFallback)))
    {
        // 序列窗口 fallback 有意放在最后：
        // 当 overlap 或 retrieval 已提供更强先验时，应优先使用那些结果。
        addSequenceCandidates(&result, input.images, policy.sequenceWindow);
        result.restrictPairs = true;
    }

    if (!manualOnly &&
        !useExhaustive &&
        hasExternalPreselection &&
        policy.useSequenceFallback &&
        !result.candidates.empty())
    {
        const int bridgeCount = addSequenceBridgeCandidates(&result, input.images, policy.sequenceWindow);
        if (bridgeCount > 0)
        {
            result.restrictPairs = true;
        }
    }

    finalizePairSelection(&result, policy.maxPairs);
    // 完整的全量列表用“非限制匹配”表示，
    // 下游无需携带一份冗余的 N^2 白名单。
    if (coversAllImagePairs(result, input.images))
    {
        result.restrictPairs = false;
    }
    else if (!result.allowedPairKeys.isEmpty())
    {
        result.restrictPairs = true;
    }
    else if (hasExternalPreselection)
    {
        result.restrictPairs = true;
    }

    result.detail = QStringLiteral("影像 %1 张，候选对 %2/%3，%4")
                        .arg(result.imageCount)
                        .arg(result.allowedPairKeys.size())
                        .arg(result.allPairCount)
                        .arg(result.restrictPairs ? QStringLiteral("限制匹配对")
                                                  : QStringLiteral("全量匹配"));
    return result;
}

} // namespace matchphotos
} // namespace xjw
