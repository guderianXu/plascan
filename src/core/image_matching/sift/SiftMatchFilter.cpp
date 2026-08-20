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

    } // namespace

    MatchResult filterSiftMutualMatches(const std::vector<SiftNearestMatch>& forward,
                                        const std::vector<SiftNearestMatch>& reverse,
                                        float confidenceThreshold,
                                        float maximumAmbiguity)
    {
        MatchResult result;
        result.sourceAlgorithm = kAutoSiftAlgorithmId;
        result.matches0.assign(forward.size(), -1);
        result.matches1.assign(reverse.size(), -1);
        result.matchingScores0.assign(forward.size(), 0.0f);
        result.matchingScores1.assign(reverse.size(), 0.0f);

        const float threshold = std::clamp(confidenceThreshold, 0.0f, 1.0f);
        const float ambiguityLimit = std::clamp(maximumAmbiguity, 0.0f, 1.0f);
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
