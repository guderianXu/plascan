#include "tracks/ReferenceTrackBuilder.h"

#include <gtest/gtest.h>

namespace
{

    std::vector<xjw::FeatureKeypoint> keypoints(std::initializer_list<xjw::FeatureKeypoint> values)
    {
        return std::vector<xjw::FeatureKeypoint>(values);
    }

} // namespace

TEST(ReferenceTrackBuilderTest, RemovesEveryObservationFromAnImageThatOccursTwiceInOneTrack)
{
    xjw::ReferenceTrackBuilder builder;
    builder.setImageKeypoints(0, keypoints({{10.0f, 10.0f}, {20.0f, 20.0f}}), 100.0f, 100.0f);
    builder.setImageKeypoints(1, keypoints({{30.0f, 30.0f}}), 100.0f, 100.0f);
    builder.setImageKeypoints(2, keypoints({{40.0f, 40.0f}}), 100.0f, 100.0f);
    builder.addMatchPair(0, 1, {{0, 0}});
    builder.addMatchPair(0, 2, {{1, 0}});
    builder.addMatchPair(1, 2, {{0, 0}});

    const xjw::ReferenceTrackBuildResult result = builder.build();

    ASSERT_EQ(result.generatedTrackCount, 1);
    EXPECT_EQ(result.removedDuplicateObservations, 2);
    ASSERT_EQ(result.tracks.size(), 1u);
    ASSERT_EQ(result.tracks.front().elements.size(), 2u);
    EXPECT_EQ(result.tracks.front().elements[0].imageId, 1u);
    EXPECT_EQ(result.tracks.front().elements[1].imageId, 2u);
}

TEST(ReferenceTrackBuilderTest, MergesAValidSingleEdgeBridgeWithoutPlaScanConfidenceHeuristics)
{
    xjw::ReferenceTrackBuilder builder;
    for (xjw::ImageId imageId = 0; imageId < 4; ++imageId)
    {
        builder.setImageKeypoints(imageId, keypoints({{10.0f + imageId, 20.0f}}), 100.0f, 100.0f);
    }
    builder.addMatchPair(0, 1, {{0, 0}});
    builder.addMatchPair(2, 3, {{0, 0}});
    builder.addMatchPair(1, 2, {{0, 0}});

    const xjw::ReferenceTrackBuildResult result = builder.build();

    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().length(), 4u);
}

TEST(ReferenceTrackBuilderTest, StationaryFilterUsesFourTimesMeanFeatureScale)
{
    xjw::ReferenceTrackBuilder builder;
    builder.setImageKeypoints(0, keypoints({{10.0f, 10.0f, 5.0f}, {10.0f, 50.0f, 1.0f}}), 100.0f, 100.0f);
    builder.setImageKeypoints(1, keypoints({{25.0f, 10.0f, 5.0f}, {30.0f, 50.0f, 1.0f}}), 100.0f, 100.0f);
    builder.addMatchPair(0, 1, {{0, 0}, {1, 1}});

    xjw::ReferenceTrackBuildOptions options;
    options.excludeStationaryTracks = true;
    const xjw::ReferenceTrackBuildResult result = builder.build(options);

    EXPECT_EQ(result.prunedStationaryTracks, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().elements.front().featureIdx, 1u);
}

TEST(ReferenceTrackBuilderTest, SpatialWaterFillPrefersTrackWithLargerInverseScaleWeight)
{
    xjw::ReferenceTrackBuilder builder;
    builder.setImageKeypoints(0, keypoints({{10.0f, 10.0f, 1.0f}, {20.0f, 20.0f, 4.0f}}), 100.0f, 100.0f);
    builder.setImageKeypoints(1, keypoints({{12.0f, 10.0f, 1.0f}, {22.0f, 20.0f, 4.0f}}), 100.0f, 100.0f);
    builder.addMatchPair(0, 1, {{0, 0}, {1, 1}});

    xjw::ReferenceTrackBuildOptions options;
    options.tiePointLimit = 1;
    const xjw::ReferenceTrackBuildResult result = builder.build(options);

    EXPECT_EQ(result.tracksBeforeSpatialSelection, 2);
    EXPECT_EQ(result.prunedBySpatialSelection, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    EXPECT_EQ(result.tracks.front().elements.front().featureIdx, 0u);
}
