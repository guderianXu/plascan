#include "TriangulationService.h"
#include "ImageMatchRepository.h"
#include "placamera_runtime/ProjectCameraStore.h"
#include "io/ImageIO.h"
#include "io/PathIO.h"

#include <plapoint/io/ply_io.h>
#include <opencv2/imgcodecs.hpp>
#include <gtest/gtest.h>
#include <placamera/frame_camera.h>
#include <placamera/frame_numeric_state.h>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QTemporaryDir>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace
{
    QString temporaryTemplate()
    {
        const QString root = QString::fromUtf8(PLASCAN_SFM_COLOR_TEST_TMP_DIR);
        EXPECT_TRUE(QDir().mkpath(root));
        return QDir(root).filePath(QStringLiteral("run-XXXXXX"));
    }

    QJsonObject sceneMetadata(const QString& directory, const QStringList& paths)
    {
        xjw::image_matching::PairMatchData pair;
        pair.image0 = xjw::image_matching::ImageMatchFile::identityForImage(paths[0], 32, 24);
        pair.image1 = xjw::image_matching::ImageMatchFile::identityForImage(paths[1], 32, 24);
        pair.algorithmId = QStringLiteral("sift-lightglue");
        pair.algorithmVersion = 1;
        pair.rawMatchCount = pair.geometryInlierCount = pair.tiePointMatchCount = 2;
        pair.geometryPassed = true;
        pair.geometryModel = xjw::image_matching::GeometryModel::Fundamental;
        for (std::uint32_t index = 0; index < 2; ++index)
        {
            xjw::image_matching::PairCorrespondence correspondence;
            correspondence.observation0.featureId = index;
            correspondence.observation1.featureId = index;
            correspondence.observation0.x = 16.5f + index;
            correspondence.observation1.x = 0.5f + index;
            correspondence.observation0.y = correspondence.observation1.y = 8.5f + index;
            correspondence.confidence = 1.0f;
            correspondence.residualPixels = 0.0f;
            correspondence.flags = xjw::image_matching::MatchRecordFlag::GeometryInlier |
                                   xjw::image_matching::MatchRecordFlag::InTiePointTrack;
            pair.correspondences.push_back(correspondence);
        }
        xjw::image_matching::ImageMatchRepository repository(QDir(directory).filePath(QStringLiteral("matches")));
        const auto written = repository.writePairs({pair}, false);
        EXPECT_TRUE(written.success) << qPrintable(written.errorMessage);
        QJsonArray images;
        QJsonArray records;
        const placamera::FrameId frame("project-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("triangulation-test-definition"),
                                                      {32.0, 32.0, 8.0, 6.0, 1.0, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        placamera::CameraInstanceSet cameras;
        for (int index = 0; index < 2; ++index)
        {
            const QString image_id = QStringLiteral("image-%1").arg(index);
            images.append(QJsonObject{{QStringLiteral("image_uuid"), image_id},
                                      {QStringLiteral("path"), paths[index]},
                                      {QStringLiteral("samples"), 32},
                                      {QStringLiteral("lines"), 24}});
            const auto added =
                cameras.add(std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                    placamera::CameraInstanceId("triangulation-test-" + image_id.toStdString()),
                    placamera::ImageId(image_id.toStdString()),
                    definition,
                    {32, 24},
                    placamera::Pose::create(
                        frame, {2.0 * index, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}))));
            EXPECT_TRUE(added.ok()) << added.message();
            /* Keep the match records separate from the canonical camera graph. */
            records.append(QJsonObject{{QStringLiteral("image"), paths[index]},
                                       {QStringLiteral("output"), repository.shardPath(paths[index])}});
        }
        QJsonObject metadata{{QStringLiteral("images"), images},
                             {QStringLiteral("image_match_results"), records},
                             {QStringLiteral("camera_definitions"), QJsonArray{}},
                             {QStringLiteral("camera_instances"), QJsonArray{}}};
        const auto update = xjw::placamera_runtime::upsertProjectCameras(&metadata, cameras);
        EXPECT_TRUE(update.ok()) << update.errors.join(';').toStdString();
        return metadata;
    }

    xjw::core::project::TriangulationServiceResult runScene(const QString& directory, const QStringList& paths)
    {
        xjw::core::project::TriangulationServiceOptions options;
        options.outputDir = QDir(directory).filePath(QStringLiteral("output"));
        options.minTriAngleDeg = 1.0;
        return xjw::core::project::TriangulationService::run(sceneMetadata(directory, paths), paths, options);
    }

    void expectColors(const xjw::core::project::TriangulationServiceResult& result,
                      const std::array<std::array<std::uint8_t, 3>, 2>& expected)
    {
        ASSERT_TRUE(result.success) << qPrintable(result.errorMessage);
        ASSERT_EQ(result.exportedPointCount, 2);
        const auto cloud = plapoint::io::readPly<float>(xjw::common::io::toNativeNarrowPath(result.sparseCloudPath));
        ASSERT_TRUE(cloud);
        ASSERT_TRUE(cloud->hasColors());
        ASSERT_EQ(cloud->size(), 2);
        for (int row = 0; row < 2; ++row)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                EXPECT_EQ((*cloud->colors())(row, channel), expected[row][channel]);
            }
        }
    }
} // namespace

TEST(TriangulationColorTest, SamplesRoundedPixelsInRgbOrderAndAveragesViews)
{
    QTemporaryDir directory(temporaryTemplate());
    ASSERT_TRUE(directory.isValid());
    const QStringList paths = {directory.filePath(QStringLiteral("月球左.png")),
                               directory.filePath(QStringLiteral("月球右.png"))};
    cv::Mat left(24, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat right = left.clone();
    left.at<cv::Vec3b>(9, 17) = {11, 31, 201};
    right.at<cv::Vec3b>(9, 1) = {21, 51, 101};
    left.at<cv::Vec3b>(10, 18) = {60, 20, 80};
    right.at<cv::Vec3b>(10, 2) = {62, 24, 86};
    ASSERT_TRUE(xjw::common::io::writeImage(paths[0], left));
    ASSERT_TRUE(xjw::common::io::writeImage(paths[1], right));
    const auto result = runScene(directory.path(), paths);
    expectColors(result, {{{151, 41, 16}, {83, 22, 61}}});
    EXPECT_FALSE(result.resultJson.contains(QStringLiteral("color_read_failures")));
    EXPECT_FALSE(result.resultJson.contains(QStringLiteral("uncolored_point_count")));
}

TEST(TriangulationColorTest, ClampsSamplingToActualRasterEdges)
{
    QTemporaryDir directory(temporaryTemplate());
    ASSERT_TRUE(directory.isValid());
    const QStringList paths = {directory.filePath(QStringLiteral("left.png")),
                               directory.filePath(QStringLiteral("right.png"))};
    cv::Mat left(5, 5, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat right = left.clone();
    left.at<cv::Vec3b>(4, 4) = {10, 30, 200};
    right.at<cv::Vec3b>(4, 1) = {20, 50, 100};
    right.at<cv::Vec3b>(4, 2) = {40, 70, 120};
    ASSERT_TRUE(xjw::common::io::writeImage(paths[0], left));
    ASSERT_TRUE(xjw::common::io::writeImage(paths[1], right));
    expectColors(runScene(directory.path(), paths), {{{150, 40, 15}, {160, 50, 25}}});
}

TEST(TriangulationColorTest, ReadsSingleBandSixteenBitTiffAsEightBitColor)
{
    QTemporaryDir directory(temporaryTemplate());
    ASSERT_TRUE(directory.isValid());
    const QStringList paths = {directory.filePath(QStringLiteral("left.tif")),
                               directory.filePath(QStringLiteral("right.tif"))};
    ASSERT_TRUE(xjw::common::io::writeImage(paths[0], cv::Mat(24, 32, CV_16UC1, cv::Scalar(32768))));
    ASSERT_TRUE(xjw::common::io::writeImage(paths[1], cv::Mat(24, 32, CV_16UC1, cv::Scalar(16384))));
    expectColors(runScene(directory.path(), paths), {{{96, 96, 96}, {96, 96, 96}}});
}

TEST(TriangulationColorTest, UsesRemainingViewWhenOneImageIsUnreadableAndReportsFailureOnce)
{
    QTemporaryDir directory(temporaryTemplate());
    ASSERT_TRUE(directory.isValid());
    const QStringList paths = {directory.filePath(QStringLiteral("missing.png")),
                               directory.filePath(QStringLiteral("right.png"))};
    ASSERT_TRUE(xjw::common::io::writeImage(paths[1], cv::Mat(24, 32, CV_8UC3, cv::Scalar(20, 40, 80))));
    const auto result = runScene(directory.path(), paths);
    expectColors(result, {{{80, 40, 20}, {80, 40, 20}}});
    const auto failures = result.resultJson.value(QStringLiteral("color_read_failures")).toArray();
    ASSERT_EQ(failures.size(), 1);
    EXPECT_EQ(failures[0].toObject().value(QStringLiteral("image_path")).toString(), paths[0]);
    EXPECT_FALSE(failures[0].toObject().value(QStringLiteral("error")).toString().isEmpty());
    EXPECT_FALSE(result.resultJson.contains(QStringLiteral("uncolored_point_count")));
}

TEST(TriangulationColorTest, KeepsNeutralGrayAndGeometryWhenAllImagesAreUnreadable)
{
    QTemporaryDir directory(temporaryTemplate());
    ASSERT_TRUE(directory.isValid());
    const QStringList paths = {directory.filePath(QStringLiteral("missing.png")),
                               directory.filePath(QStringLiteral("corrupt.png"))};
    QFile corrupt(paths[1]);
    ASSERT_TRUE(corrupt.open(QIODevice::WriteOnly));
    ASSERT_EQ(corrupt.write("invalid-png"), 11);
    corrupt.close();
    const auto result = runScene(directory.path(), paths);
    expectColors(result, {{{128, 128, 128}, {128, 128, 128}}});
    EXPECT_EQ(result.resultJson.value(QStringLiteral("color_read_failures")).toArray().size(), 2);
    EXPECT_EQ(result.resultJson.value(QStringLiteral("uncolored_point_count")).toInt(), 2);
}

TEST(TriangulationGeometryTest, RejectsDifferentWorldFrames)
{
    using placamera::BrownConradyDistortion;
    using placamera::FrameIntrinsics;
    using placamera::FramePinholeDefinition;
    using placamera::FramePinholeModel;
    using placamera::PixelConvention;
    using placoordinate::CoordinateFrameId;

    const auto makeState = [](const char* image, const char* frame, double centerX)
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 32.0;
        intrinsics.focalY = 32.0;
        intrinsics.principalX = 8.0;
        intrinsics.principalY = 6.0;
        const auto definition =
            FramePinholeDefinition::create(placamera::CameraDefinitionId(std::string("definition-") + image),
                                           intrinsics,
                                           BrownConradyDistortion{},
                                           PixelConvention::PixelCenter,
                                           CoordinateFrameId(frame));
        const auto pose = placamera::Pose::create(
            CoordinateFrameId(frame), {{centerX, 0.0, 0.0}}, {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}});
        const FramePinholeModel instance =
            FramePinholeModel::create(placamera::CameraInstanceId(std::string("instance-") + image),
                                      placamera::ImageId(image),
                                      definition,
                                      placamera::ImageSize{32, 24},
                                      pose);
        return placamera::FramePinholeNumericState::fromModel(instance);
    };

    const placamera::FramePinholeNumericState left = makeState("left", "world-a", 0.0);
    const placamera::FramePinholeNumericState right = makeState("right", "world-b", 2.0);
    const auto intersection = placamera::FramePinholeNumericState::triangulatePair(
        left, {8.0, 6.0}, right, {1.6, 6.0});

    EXPECT_FALSE(intersection.ok());
    EXPECT_EQ(intersection.errorCode(), placamera::CameraErrorCode::FrameMismatch);

    const placamera::FramePinholeNumericState sameFrameRight =
        makeState("right-same", "world-a", 2.0);
    const auto sameFrameIntersection = placamera::FramePinholeNumericState::triangulatePair(
        left, {8.0, 6.0}, sameFrameRight, {1.6, 6.0});
    ASSERT_TRUE(sameFrameIntersection.ok()) << sameFrameIntersection.message();
    EXPECT_EQ(sameFrameIntersection.value().point.frame.value(), "world-a");
}
