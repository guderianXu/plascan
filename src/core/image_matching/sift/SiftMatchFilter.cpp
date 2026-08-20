#include "SiftMatchFilter.h"

#include "AutoSiftAlgorithm.h"

#include <algorithm>
#include <cmath>

namespace xjw::image_matching
{
    namespace
    {

        float descriptorConfidence(const SiftNearestMatch& match)
        {
            if (!std::isfinite(match.similarity) || !std::isfinite(match.ambiguity))
            {
                return 0.0f;
            }
            return std::clamp(match.similarity, 0.0f, 1.0f);
        }

        float effectiveRatioLimit(const std::vector<SiftNearestMatch>& forward,
                                  const std::vector<SiftNearestMatch>& reverse,
                                  const SiftMatchFilterOptions& options)
        {
            const float configuredMaximum = std::clamp(options.maximumRatio, 0.0f, 1.0f);
            if (!options.adaptiveRatio)
            {
                return configuredMaximum;
            }

            std::vector<float> mutualRatios;
            mutualRatios.reserve(forward.size());
            const float confidenceThreshold = std::clamp(options.confidenceThreshold, 0.0f, 1.0f);
            for (int index0 = 0; index0 < static_cast<int>(forward.size()); ++index0)
            {
                const SiftNearestMatch& candidate = forward[static_cast<std::size_t>(index0)];
                const int index1 = candidate.index;
                if (index1 < 0 || index1 >= static_cast<int>(reverse.size()))
                {
                    continue;
                }
                const SiftNearestMatch& reverseCandidate = reverse[static_cast<std::size_t>(index1)];
                if (reverseCandidate.index != index0 ||
                    std::min(descriptorConfidence(candidate), descriptorConfidence(reverseCandidate)) <
                        confidenceThreshold)
                {
                    continue;
                }
                const float ratio = std::max(candidate.ambiguity, reverseCandidate.ambiguity);
                if (std::isfinite(ratio))
                {
                    mutualRatios.push_back(std::clamp(ratio, 0.0f, 1.0f));
                }
            }

            // 小影像/低纹理像对本来候选就少，保留用户设置的宽松上限。候选充足时
            // 取 75% 分位并留 0.02 余量，稳定地删除分布尾部的重复纹理歧义项。
            if (static_cast<int>(mutualRatios.size()) < std::max(8, options.sparseCandidateCount))
            {
                return configuredMaximum;
            }
            const std::size_t quartileIndex =
                std::min(mutualRatios.size() - 1, (mutualRatios.size() * 3) / 4);
            std::nth_element(mutualRatios.begin(),
                             mutualRatios.begin() + static_cast<std::ptrdiff_t>(quartileIndex),
                             mutualRatios.end());
            const float minimum = std::clamp(options.minimumAdaptiveRatio, 0.0f, configuredMaximum);
            return std::clamp(mutualRatios[quartileIndex] + 0.02f, minimum, configuredMaximum);
        }

    } // namespace

    MatchResult filterSiftMutualMatches(const std::vector<SiftNearestMatch>& forward,
                                        const std::vector<SiftNearestMatch>& reverse,
                                        const SiftMatchFilterOptions& options)
    {
        MatchResult result;
        result.sourceAlgorithm = kAutoSiftAlgorithmId;
        result.matches0.assign(forward.size(), -1);
        result.matches1.assign(reverse.size(), -1);
        result.matchingScores0.assign(forward.size(), 0.0f);
        result.matchingScores1.assign(reverse.size(), 0.0f);

        const float threshold = std::clamp(options.confidenceThreshold, 0.0f, 1.0f);
        const float ambiguityLimit = effectiveRatioLimit(forward, reverse, options);
        for (int index0 = 0; index0 < static_cast<int>(forward.size()); ++index0)
        {
            const SiftNearestMatch& candidate = forward[static_cast<std::size_t>(index0)];
            const int index1 = candidate.index;
            if (index1 < 0 || index1 >= static_cast<int>(reverse.size()) ||
                reverse[static_cast<std::size_t>(index1)].index != index0)
            {
                continue;
            }

            const SiftNearestMatch& reverseCandidate = reverse[static_cast<std::size_t>(index1)];
            if (candidate.ambiguity > ambiguityLimit || reverseCandidate.ambiguity > ambiguityLimit)
            {
                continue;
            }

            const float confidence = std::min(descriptorConfidence(candidate), descriptorConfidence(reverseCandidate));
            if (confidence < threshold)
            {
                continue;
            }

            result.matches0[static_cast<std::size_t>(index0)] = index1;
            result.matches1[static_cast<std::size_t>(index1)] = index0;
            result.matchingScores0[static_cast<std::size_t>(index0)] = confidence;
            result.matchingScores1[static_cast<std::size_t>(index1)] = confidence;
        }
        result.buildCvMatchesFromIndices();
        return result;
    }

} // namespace xjw::image_matching
