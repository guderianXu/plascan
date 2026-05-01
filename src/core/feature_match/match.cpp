#include "match.h"

#include <algorithm>
#include <cmath>

namespace xjw::feature_match
{

// ---------------------------------------------------------------------------
// MatchResult — 辅助方法
// ---------------------------------------------------------------------------

void MatchResult::buildCvMatchesFromIndices()
{
    cvMatches.clear();

    const int n = static_cast<int>(matches0.size());
    for (int i = 0; i < n; ++i)
    {
        if (matches0[i] < 0)
        {
            continue;
        }

        cv::DMatch dm;
        dm.queryIdx = i;
        dm.trainIdx = matches0[i];
        dm.distance = (i < static_cast<int>(matchingScores0.size()) && matchingScores0[i] > 0.0f)
                          ? (1.0f / matchingScores0[i] - 1.0f) // 逆映射 score -> distance
                          : 1.0f;
        cvMatches.push_back(dm);
    }

    numMatches = static_cast<int>(cvMatches.size());
}

void MatchResult::buildIndicesFromCvMatches(int numKeypoints0, int numKeypoints1)
{
    matches0.assign(numKeypoints0, -1);
    matches1.assign(numKeypoints1, -1);
    matchingScores0.assign(numKeypoints0, 0.0f);
    matchingScores1.assign(numKeypoints1, 0.0f);

    for (const cv::DMatch &dm : cvMatches)
    {
        const int q = dm.queryIdx;
        const int t = dm.trainIdx;
        if (q < 0 || q >= numKeypoints0 || t < 0 || t >= numKeypoints1)
        {
            continue;
        }

        const float score = 1.0f / (1.0f + std::max(0.0f, dm.distance));
        matches0[q] = t;
        matches1[t] = q;
        matchingScores0[q] = score;
        matchingScores1[t] = score;
    }

    numMatches = static_cast<int>(cvMatches.size());
}

// ---------------------------------------------------------------------------
// MatchResult — 工厂方法
// ---------------------------------------------------------------------------

MatchResult MatchResult::fromCvMatches(const std::vector<cv::DMatch> &matches,
                                       int numKeypoints0,
                                       int numKeypoints1,
                                       const std::string &algoName)
{
    MatchResult result;
    result.cvMatches = matches;
    result.numMatches = static_cast<int>(matches.size());
    result.sourceAlgorithm = algoName;
    result.buildIndicesFromCvMatches(numKeypoints0, numKeypoints1);
    return result;
}

MatchResult MatchResult::fromIndices(const std::vector<int> &matches0,
                                     const std::vector<int> &matches1,
                                     const std::vector<float> &scores0,
                                     const std::vector<float> &scores1,
                                     const std::string &algoName)
{
    MatchResult result;
    result.matches0 = matches0;
    result.matches1 = matches1;
    result.matchingScores0 = scores0;
    result.matchingScores1 = scores1;
    result.sourceAlgorithm = algoName;
    result.buildCvMatchesFromIndices();
    return result;
}

} // namespace xjw::feature_match
