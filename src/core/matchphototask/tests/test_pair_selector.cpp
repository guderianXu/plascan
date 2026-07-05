#include "PairSelector.h"

#include <gtest/gtest.h>

#include <QDir>

#include <algorithm>

namespace
{

QStringList makeImages(int count)
{
    QStringList images;
    const QDir dir(QDir::tempPath());
    for (int i = 0; i < count; ++i)
    {
        images.append(dir.filePath(QStringLiteral("plascan_matchphotos_%1.png").arg(i)));
    }
    return images;
}

bool hasSource(const xjw::matchphotos::PairCandidate &candidate,
               xjw::matchphotos::PairSource source)
{
    return std::find(candidate.sources.begin(), candidate.sources.end(), source) != candidate.sources.end();
}

const xjw::matchphotos::PairCandidate *findPair(
    const std::vector<xjw::matchphotos::PairCandidate> &candidates,
    int indexA,
    int indexB)
{
    for (const xjw::matchphotos::PairCandidate &candidate : candidates)
    {
        if (candidate.pair.indexA == indexA && candidate.pair.indexB == indexB)
        {
            return &candidate;
        }
    }
    return nullptr;
}

} // namespace

TEST(MatchPhotosPairSelectorTest, SmallSetUsesExhaustivePairs)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(4);

    const xjw::matchphotos::PairSelectionPolicy policy =
        xjw::matchphotos::makePairSelectionPolicy(xjw::matchphotos::PairSelectionPreset::Auto);
    const xjw::matchphotos::PairSelectionResult result =
        xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_FALSE(result.restrictPairs);
    EXPECT_EQ(result.allPairCount, 6);
    ASSERT_EQ(result.candidates.size(), 6);
    EXPECT_TRUE(hasSource(result.candidates.front(), xjw::matchphotos::PairSource::Exhaustive));
}

TEST(MatchPhotosPairSelectorTest, LargeSetFallsBackToSequenceWindow)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(6);

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.exhaustiveMaxImages = 3;
    policy.sequenceWindow = 2;

    const xjw::matchphotos::PairSelectionResult result =
        xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_TRUE(result.restrictPairs);
    EXPECT_EQ(result.allPairCount, 15);
    EXPECT_EQ(result.candidates.size(), 9);
    ASSERT_NE(findPair(result.candidates, 0, 2), nullptr);
}

TEST(MatchPhotosPairSelectorTest, MergesOverlapAndVocabularyCandidates)
{
    xjw::OverlapAnalysisResult cameraOverlap;
    xjw::OverlapPairResult cameraPair;
    cameraPair.indexA = 1;
    cameraPair.indexB = 3;
    cameraPair.overlapScore = 0.7;
    cameraOverlap.pairs.push_back(cameraPair);

    xjw::VocabularyOverlapResult vocabularyOverlap;
    xjw::VocabularyOverlapPairResult vocabularyPair;
    vocabularyPair.indexA = 0;
    vocabularyPair.indexB = 3;
    vocabularyPair.bowScore = 0.5;
    vocabularyPair.sharedWordCount = 12;
    vocabularyPair.accepted = true;
    vocabularyOverlap.acceptedPairs.push_back(vocabularyPair);

    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(4);
    input.knownCameraOverlapPairs.push_back({0, 2});
    input.cameraOverlapResult = &cameraOverlap;
    input.vocabularyOverlapResult = &vocabularyOverlap;

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.exhaustiveMaxImages = 1;
    policy.useSequenceFallback = false;

    const xjw::matchphotos::PairSelectionResult result =
        xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_TRUE(result.restrictPairs);
    ASSERT_EQ(result.candidates.size(), 3);
    ASSERT_NE(findPair(result.candidates, 0, 2), nullptr);
    ASSERT_NE(findPair(result.candidates, 1, 3), nullptr);

    const xjw::matchphotos::PairCandidate *vocabulary = findPair(result.candidates, 0, 3);
    ASSERT_NE(vocabulary, nullptr);
    EXPECT_TRUE(hasSource(*vocabulary, xjw::matchphotos::PairSource::VocabularyOverlap));
    EXPECT_DOUBLE_EQ(vocabulary->vocabularyScore, 0.5);
}
