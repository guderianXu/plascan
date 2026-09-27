#include "project/BaTrackBuilder.h"
#include "project/MarkerBaAdapter.h"
#include "model/MarkerSet.h"

#include <gtest/gtest.h>
#include <placamera/frame_camera.h>

#include <memory>
#include <string>
#include <utility>

namespace
{

    std::shared_ptr<const placamera::FramePinholeModel> makeNativeCamera(const std::array<double, 3>& center,
                                                                          int index)
    {
        const placamera::FrameId frame("ba-track-world");
        auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId("ba-track-definition-" + std::to_string(index)),
            {100.0, 100.0, 0.0, 0.0, 1.0, 1, 1},
            {},
            placamera::PixelConvention::PixelCenter,
            frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("ba-track-instance-" + std::to_string(index)),
            placamera::ImageId("ba-track-image-" + std::to_string(index)),
            std::move(definition),
            {64, 48},
            placamera::Pose::create(frame,
                                    center,
                                    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

} // namespace

TEST(BaTrackBuilderTest, IndexedTrackTriesLaterObservationPairWhenFirstPairIsDegenerate)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameraInstances = {makeNativeCamera({0.0, 0.0, 0.0}, 0),
                             makeNativeCamera({0.0, 0.0, 0.0}, 1),
                             makeNativeCamera({1.0, 0.0, 0.0}, 2)};
    input.imageIdByIndex = {placamera::ImageId("ba-track-image-0"),
                            placamera::ImageId("ba-track-image-1"),
                            placamera::ImageId("ba-track-image-2")};

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
    ASSERT_TRUE(xjw::core::project::appendBaTracks(input, &result));

    ASSERT_EQ(result.tracks.size(), 1u);
    ASSERT_EQ(result.tracks.front().observations.size(), 3u);
    EXPECT_NEAR(result.tracks.front().initialPoint[0], 0.0, 1e-9);
    EXPECT_NEAR(result.tracks.front().initialPoint[1], 0.0, 1e-9);
    EXPECT_NEAR(result.tracks.front().initialPoint[2], 10.0, 1e-9);
}

TEST(BaTrackBuilderTest, IndexedTrackUsesReferenceDuplicateObservationCleanup)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameraInstances = {makeNativeCamera({0.0, 0.0, 0.0}, 0),
                             makeNativeCamera({1.0, 0.0, 0.0}, 1),
                             makeNativeCamera({2.0, 0.0, 0.0}, 2)};
    input.imageIdByIndex = {placamera::ImageId("ba-track-image-0"),
                            placamera::ImageId("ba-track-image-1"),
                            placamera::ImageId("ba-track-image-2")};

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
    ASSERT_TRUE(xjw::core::project::appendBaTracks(input, &result));

    ASSERT_EQ(result.multiViewTrackCount, 1);
    ASSERT_EQ(result.tracks.size(), 1u);
    ASSERT_EQ(result.tracks.front().observations.size(), 2u);
    EXPECT_EQ(result.tracks.front().observations[0].cameraIndex, 1);
    EXPECT_EQ(result.tracks.front().observations[1].cameraIndex, 2);
}

TEST(BaTrackBuilderTest, RejectsMissingPlaCameraInstanceForCanonicalImageId)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameraInstances = {makeNativeCamera({0.0, 0.0, 0.0}, 0)};
    input.imageIdByIndex = {placamera::ImageId("ba-track-image-0"), placamera::ImageId("ba-track-image-1")};

    xjw::core::project::BaInputBuildResult result;
    EXPECT_FALSE(xjw::core::project::appendBaTracks(input, &result));
    EXPECT_TRUE(result.tracks.empty());
    EXPECT_TRUE(result.matchDiagnostics.firstCameraError.contains(QStringLiteral("数量不一致")));
}

TEST(BaTrackBuilderTest, RejectsPlaCameraImageIdentityMismatch)
{
    xjw::core::project::ProjectMatchInput input;
    input.cameraInstances = {makeNativeCamera({0.0, 0.0, 0.0}, 0),
                             makeNativeCamera({1.0, 0.0, 0.0}, 1)};
    input.imageIdByIndex = {placamera::ImageId("ba-track-image-0"),
                            placamera::ImageId("wrong-image")};

    xjw::core::project::BaInputBuildResult result;
    EXPECT_FALSE(xjw::core::project::appendBaTracks(input, &result));
    EXPECT_TRUE(result.tracks.empty());
    EXPECT_TRUE(result.matchDiagnostics.firstCameraError.contains(QStringLiteral("ImageId 不一致")));
}

TEST(MarkerBaAdapterTest, RejectsPlaCameraImageIdentityMismatch)
{
    xjw::control_points::MarkerSet markers;
    xjw::core::project::MarkerBaInput input;
    input.markerSet = &markers;
    xjw::core::project::BaInputBuildResult result;
    result.cameraInstances = {makeNativeCamera({0.0, 0.0, 0.0}, 0)};
    result.imageIdByIndex = {placamera::ImageId("wrong-image")};

    xjw::core::project::appendMarkerBaInput(&input, {}, &result);
    EXPECT_TRUE(result.firstControlInputError.contains(QStringLiteral("ImageId 不一致")));
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

TEST(ProjectMatchInputReaderTest, RejectsAmbiguousNormalizedPathAliases)
{
    const QMap<QString, int> cameras{
        {QStringLiteral("E:/project/./frame.jpg"), 0},
        {QStringLiteral("E:/project/sub/../frame.jpg"), 1}};

    EXPECT_EQ(xjw::core::project::cameraIndexForImageToken(
                  QStringLiteral("E:/project/frame.jpg"), cameras),
              -1);
}

TEST(ProjectMatchInputReaderTest, AcceptsEquivalentAliasesOnlyWhenTheyShareAnIndex)
{
    const QMap<QString, int> cameras{
        {QStringLiteral("E:/project/./frame.jpg"), 0},
        {QStringLiteral("E:/project/sub/../frame.jpg"), 0}};

    EXPECT_EQ(xjw::core::project::cameraIndexForImageToken(
                  QStringLiteral("E:/project/frame.jpg"), cameras),
              0);
}

TEST(ProjectMatchInputReaderTest, ExactPathDoesNotHideConflictingAlias)
{
    const QMap<QString, int> cameras{
        {QStringLiteral("E:/project/frame.jpg"), 0},
        {QStringLiteral("E:/project/sub/../frame.jpg"), 1}};

    EXPECT_EQ(xjw::core::project::cameraIndexForImageToken(
                  QStringLiteral("E:/project/frame.jpg"), cameras),
              -1);
}
