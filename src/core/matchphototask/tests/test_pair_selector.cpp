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
        for (int index = 0; index < count; ++index)
        {
            images.append(dir.filePath(QStringLiteral("plascan_matchphotos_%1.png").arg(index)));
        }
        return images;
    }

    bool hasSource(const xjw::matchphotos::PairCandidate& candidate, xjw::matchphotos::PairSource source)
    {
        return std::find(candidate.sources.begin(), candidate.sources.end(), source) != candidate.sources.end();
    }

    const xjw::matchphotos::PairCandidate*
    findPair(const std::vector<xjw::matchphotos::PairCandidate>& candidates, int indexA, int indexB)
    {
        const auto found = std::find_if(candidates.begin(),
                                        candidates.end(),
                                        [&](const auto& candidate)
                                        { return candidate.pair.indexA == indexA && candidate.pair.indexB == indexB; });
        return found == candidates.end() ? nullptr : &*found;
    }

} // namespace

TEST(MatchPhotosPairSelectorTest, SmallAutoPlanUsesExhaustivePairs)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(4);

    const xjw::matchphotos::PairSelectionPolicy policy =
        xjw::matchphotos::makePairSelectionPolicy(xjw::matchphotos::PairSelectionPreset::Auto);
    const xjw::matchphotos::PairSelectionResult result = xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_FALSE(result.restrictPairs);
    EXPECT_EQ(result.allPairCount, 6);
    ASSERT_EQ(result.candidates.size(), 6u);
    EXPECT_TRUE(hasSource(result.candidates.front(), xjw::matchphotos::PairSource::Exhaustive));
}

TEST(MatchPhotosPairSelectorTest, LargeAutoPlanFallsBackToSequenceWindow)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(6);

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.exhaustiveMaxImages = 3;
    policy.sequenceWindow = 2;
    const xjw::matchphotos::PairSelectionResult result = xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_TRUE(result.restrictPairs);
    EXPECT_EQ(result.allPairCount, 15);
    EXPECT_EQ(result.candidates.size(), 9u);
    EXPECT_NE(findPair(result.candidates, 0, 2), nullptr);
    EXPECT_EQ(findPair(result.candidates, 0, 5), nullptr);
}

TEST(MatchPhotosPairSelectorTest, ClosedSequenceWindowSupportsRingCapture)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(16);

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.mode = xjw::matchphotos::PairSelectionMode::Sequence;
    policy.sequenceWindow = 2;
    policy.closeSequenceLoop = true;
    const xjw::matchphotos::PairSelectionResult result = xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_TRUE(result.restrictPairs);
    const auto* wrappedPair = findPair(result.candidates, 0, 15);
    ASSERT_NE(wrappedPair, nullptr);
    EXPECT_TRUE(hasSource(*wrappedPair, xjw::matchphotos::PairSource::SequenceWindow));
}

TEST(MatchPhotosPairSelectorTest, ManualOnlyKeepsOnlyExplicitPairs)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(4);
    input.manualPairKeys = {xjw::matchphotos::makePairKey(input.images, 0, 3)};

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.mode = xjw::matchphotos::PairSelectionMode::ManualOnly;
    const xjw::matchphotos::PairSelectionResult result = xjw::matchphotos::PairSelector::select(input, policy);

    ASSERT_EQ(result.candidates.size(), 1u);
    EXPECT_TRUE(result.restrictPairs);
    EXPECT_TRUE(hasSource(result.candidates.front(), xjw::matchphotos::PairSource::Manual));
}

TEST(MatchPhotosPairSelectorTest, MaxPairsAppliesAfterStablePrioritySort)
{
    xjw::matchphotos::PairSelectionInput input;
    input.images = makeImages(5);

    xjw::matchphotos::PairSelectionPolicy policy;
    policy.mode = xjw::matchphotos::PairSelectionMode::Exhaustive;
    policy.maxPairs = 3;
    const xjw::matchphotos::PairSelectionResult result = xjw::matchphotos::PairSelector::select(input, policy);

    EXPECT_TRUE(result.restrictPairs);
    EXPECT_EQ(result.candidates.size(), 3u);
    EXPECT_EQ(result.allowedPairKeys.size(), 3);
}
