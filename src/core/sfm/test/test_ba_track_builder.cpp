#include "project/BaTrackBuilder.h"

#include <gtest/gtest.h>

namespace
{

xjw::Camera makeCamera(const std::array<double, 3> &center)
{
    xjw::Camera camera;
    camera.setIntrinsics(100.0, 100.0, 0.0, 0.0);
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   center);
    return camera;
}

} // namespace

TEST(BaTrackBuilderTest, IndexedTrackTriesLaterObservationPairWhenFirstPairIsDegenerate)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameras = {
        makeCamera({0.0, 0.0, 0.0}),
        makeCamera({0.0, 0.0, 0.0}),
        makeCamera({1.0, 0.0, 0.0}),
    };

    xjw::core::project::ProjectMatchPair firstPair;
    firstPair.cameraIndexA = 0;
    firstPair.cameraIndexB = 1;
    firstPair.indexed = true;
    firstPair.observations.push_back({{0.0, 0.0}, {0.0, 0.0}, 0, 0, 1.0});

    xjw::core::project::ProjectMatchPair secondPair;
    secondPair.cameraIndexA = 1;
    secondPair.cameraIndexB = 2;
    secondPair.indexed = true;
    secondPair.observations.push_back({{0.0, 0.0}, {-10.0, 0.0}, 0, 0, 1.0});
    input.pairs = {firstPair, secondPair};

    xjw::core::project::BaInputBuildResult result;
    xjw::core::project::appendBaTracks(input, &result);

    ASSERT_EQ(result.tracks.size(), 1u);
    ASSERT_EQ(result.tracks.front().observations.size(), 3u);
    EXPECT_NEAR(result.tracks.front().initialPoint[0], 0.0, 1e-9);
    EXPECT_NEAR(result.tracks.front().initialPoint[1], 0.0, 1e-9);
    EXPECT_NEAR(result.tracks.front().initialPoint[2], 10.0, 1e-9);
}
