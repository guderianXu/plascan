#include "project/BaTrackBuilder.h"

#include <gtest/gtest.h>

namespace
{

    xjw::FramePinholeCamera makeCamera(const std::array<double, 3>& center)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(100.0, 100.0, 0.0, 0.0);
        camera.setPose({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}, center);
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

TEST(BaTrackBuilderTest, IndexedTrackUsesReferenceDuplicateObservationCleanup)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameras = {
        makeCamera({0.0, 0.0, 0.0}),
        makeCamera({1.0, 0.0, 0.0}),
        makeCamera({2.0, 0.0, 0.0}),
    };

    xjw::core::project::ProjectMatchPair pair01;
    pair01.cameraIndexA = 0;
    pair01.cameraIndexB = 1;
    pair01.indexed = true;
    pair01.observations.push_back({{0.0, 0.0}, {-10.0, 0.0}, 0, 0, 1.0});

    xjw::core::project::ProjectMatchPair pair02;
    pair02.cameraIndexA = 0;
    pair02.cameraIndexB = 2;
    pair02.indexed = true;
    pair02.observations.push_back({{1.0, 0.0}, {-20.0, 0.0}, 1, 0, 1.0});

    xjw::core::project::ProjectMatchPair pair12;
    pair12.cameraIndexA = 1;
    pair12.cameraIndexB = 2;
    pair12.indexed = true;
    pair12.observations.push_back({{-10.0, 0.0}, {-20.0, 0.0}, 0, 0, 1.0});
    input.pairs = {pair01, pair02, pair12};

    xjw::core::project::BaInputBuildResult result;
    xjw::core::project::appendBaTracks(input, &result);

    ASSERT_EQ(result.multiViewTrackCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    ASSERT_EQ(result.tracks.front().observations.size(), 2u);
    EXPECT_EQ(result.tracks.front().observations[0].cameraIndex, 1);
    EXPECT_EQ(result.tracks.front().observations[1].cameraIndex, 2);
}

TEST(ProjectMatchInputReaderTest, ResolvesRelocatedImageByUniqueFileName)
{
    const QMap<QString, int> cameras{{QStringLiteral("E:/project/demo.files/shared/images/abc/000001.jpg"), 0},
                                     {QStringLiteral("E:/project/demo.files/shared/images/def/000002.jpg"), 1}};
    const QStringList allImages = cameras.keys();

    EXPECT_EQ(xjw::core::project::cameraIndexForRelocatedMatchToken(
                  QStringLiteral("D:/download/Ignatius/000001.jpg"), cameras, allImages),
              0);
    EXPECT_EQ(xjw::core::project::cameraIndexForImageToken(QStringLiteral("D:/download/Ignatius/000001.jpg"), cameras),
              -1);
}

TEST(ProjectMatchInputReaderTest, RejectsFileNameAmbiguousInFullChunk)
{
    const QMap<QString, int> cameras{{QStringLiteral("E:/project/demo.files/shared/images/abc/frame.jpg"), 0}};
    const QStringList allImages{QStringLiteral("E:/project/demo.files/shared/images/abc/frame.jpg"),
                                QStringLiteral("E:/project/demo.files/shared/images/def/frame.jpg")};

    EXPECT_EQ(xjw::core::project::cameraIndexForRelocatedMatchToken(
                  QStringLiteral("D:/download/scene/frame.jpg"), cameras, allImages),
              -1);
}
