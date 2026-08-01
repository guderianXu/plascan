#include "MatchResult.h"

#include <algorithm>

namespace xjw::image_matching
{

bool MatchResult::empty() const
{
    return numMatches <= 0;
}

void MatchResult::buildCvMatchesFromIndices()
{
    cvMatches.clear();
    const int count = static_cast<int>(matches0.size());
    cvMatches.reserve(static_cast<std::size_t>(count));
    for (int index0 = 0; index0 < count; ++index0)
    {
        const int index1 = matches0[static_cast<std::size_t>(index0)];
        if (index1 < 0)
        {
            continue;
        }
        const float score = index0 < static_cast<int>(matchingScores0.size())
            ? matchingScores0[static_cast<std::size_t>(index0)]
            : 1.0f;
        cvMatches.emplace_back(index0, index1, std::max(0.0f, 1.0f - score));
    }
    numMatches = static_cast<int>(cvMatches.size());
}

} // namespace xjw::image_matching
