#include "PairSelector.h"

#include <QSet>

#include <algorithm>
#include <utility>

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

            PairCandidate* findCandidate(std::vector<PairCandidate>* candidates, const QString& pairKey)
            {
                if (!candidates || pairKey.isEmpty())
                {
                    return nullptr;
                }

                const auto found =
                    std::find_if(candidates->begin(),
                                 candidates->end(),
                                 [&](const PairCandidate& candidate) { return candidate.pairKey == pairKey; });
                return found == candidates->end() ? nullptr : &*found;
            }

            PairCandidate* addCandidate(PairSelectionResult* result,
                                        const QStringList& images,
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
                PairCandidate* candidate = findCandidate(&result->candidates, pairKey);
                if (!candidate)
                {
                    PairCandidate created;
                    created.pair = pair.normalized();
                    created.pairKey = pairKey;
                    result->candidates.push_back(std::move(created));
                    candidate = &result->candidates.back();
                }
                else if (!candidate->pair.isValid(images.size()))
                {
                    candidate->pair = pair.normalized();
                }

                appendPairSource(candidate, source);
                candidate->priorityScore += std::max(0.0, priorityScore);
                return candidate;
            }

            void addManualCandidates(PairSelectionResult* result, const QStringList& pairKeys)
            {
                QSet<QString> seen;
                for (const QString& pairKey : pairKeys)
                {
                    const QString trimmed = pairKey.trimmed();
                    if (trimmed.isEmpty() || seen.contains(trimmed))
                    {
                        continue;
                    }

                    seen.insert(trimmed);
                    PairCandidate candidate;
                    candidate.pairKey = trimmed;
                    candidate.priorityScore = 1000.0;
                    candidate.detail = QStringLiteral("manual pair");
                    appendPairSource(&candidate, PairSource::Manual);
                    result->candidates.push_back(std::move(candidate));
                }
            }

            void addExhaustiveCandidates(PairSelectionResult* result, const QStringList& images)
            {
                for (int first = 0; first < images.size(); ++first)
                {
                    for (int second = first + 1; second < images.size(); ++second)
                    {
                        addCandidate(result, images, first, second, PairSource::Exhaustive, 10.0);
                    }
                }
            }

            void addSequenceCandidates(PairSelectionResult* result,
                                       const QStringList& images,
                                       int window,
                                       bool closeSequenceLoop)
            {
                if (images.size() < 2)
                {
                    return;
                }

                const int imageCount = static_cast<int>(images.size());
                const int safeWindow = std::min(std::max(1, window), imageCount - 1);
                for (int first = 0; first < imageCount; ++first)
                {
                    for (int second = first + 1; second < imageCount; ++second)
                    {
                        const int linearDistance = second - first;
                        const int distance =
                            closeSequenceLoop ? std::min(linearDistance, imageCount - linearDistance) : linearDistance;
                        if (distance > safeWindow)
                        {
                            continue;
                        }

                        const double sequenceScore =
                            static_cast<double>(safeWindow - distance + 1) / static_cast<double>(safeWindow);
                        PairCandidate* candidate = addCandidate(
                            result, images, first, second, PairSource::SequenceWindow, 40.0 + 10.0 * sequenceScore);
                        if (candidate)
                        {
                            candidate->sequenceScore = sequenceScore;
                        }
                    }
                }
            }

            void finalize(PairSelectionResult* result, int maxPairs)
            {
                std::sort(result->candidates.begin(),
                          result->candidates.end(),
                          [](const PairCandidate& left, const PairCandidate& right)
                          {
                              if (left.priorityScore != right.priorityScore)
                              {
                                  return left.priorityScore > right.priorityScore;
                              }
                              return left.pairKey < right.pairKey;
                          });

                if (maxPairs > 0 && static_cast<int>(result->candidates.size()) > maxPairs)
                {
                    result->candidates.resize(static_cast<std::size_t>(maxPairs));
                }

                for (const PairCandidate& candidate : result->candidates)
                {
                    if (!candidate.pairKey.isEmpty())
                    {
                        result->allowedPairKeys.append(candidate.pairKey);
                    }
                }
            }

        } // namespace

        PairSelectionResult
        PairSelector::select(const PairSelectionInput& input, const PairSelectionPolicy& policy, QString* errorMessage)
        {
            if (errorMessage)
            {
                errorMessage->clear();
            }

            PairSelectionResult result;
            result.imageCount = input.images.size();
            result.allPairCount = allPairCount(result.imageCount);
            addManualCandidates(&result, input.manualPairKeys);

            const bool manualOnly = policy.mode == PairSelectionMode::ManualOnly;
            const bool useExhaustive = policy.mode == PairSelectionMode::Exhaustive ||
                                       (policy.mode == PairSelectionMode::Auto &&
                                        result.imageCount <= std::max(2, policy.exhaustiveMaxImages));
            if (!manualOnly && useExhaustive)
            {
                addExhaustiveCandidates(&result, input.images);
            }
            else if (!manualOnly && (policy.mode == PairSelectionMode::Sequence ||
                                     (policy.mode == PairSelectionMode::Auto && policy.useSequenceFallback)))
            {
                addSequenceCandidates(&result, input.images, policy.sequenceWindow, policy.closeSequenceLoop);
            }

            finalize(&result, policy.maxPairs);
            result.restrictPairs = manualOnly || policy.maxPairs > 0 ||
                                   static_cast<int>(result.allowedPairKeys.size()) != result.allPairCount;
            result.detail = QStringLiteral("影像 %1 张，候选对 %2/%3，%4")
                                .arg(result.imageCount)
                                .arg(result.allowedPairKeys.size())
                                .arg(result.allPairCount)
                                .arg(result.restrictPairs ? QStringLiteral("限制匹配对") : QStringLiteral("全量匹配"));
            return result;
        }

    } // namespace matchphotos
} // namespace xjw
