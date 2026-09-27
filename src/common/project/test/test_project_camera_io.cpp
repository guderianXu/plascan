#include "project/ProjectCameraIO.h"

#include <placamera/tsai.h>
#include "placamera_runtime/ProjectCameraStore.h"
#include "project/ProjectFramePinholeMetadataIO.h"

#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/rpc_camera.h>

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using namespace xjw::common::project;

    placamera::FramePinholeModel makeCamera()
    {
        const placamera::FrameId frame("survey-world");
        auto definition = placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("report-definition"),
                                                                    {7000.0, 7100.0, 40.0, -20.0, 0.005, -1, 1},
                                                                    {0.01, -0.001, 0.0001, 0.0002, -0.0003},
                                                                    placamera::PixelConvention::PixelCenter,
                                                                    frame,
                                                                    true);
        return placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("report-instance"),
            placamera::ImageId("report-image"),
            std::move(definition),
            {640, 480},
            placamera::Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    placamera::LineScanModel makeLineScanCamera()
    {
        placamera::LineScanDetectorGeometry detector;
        detector.detectorSampleOrigin = 512.0;
        detector.focalToPixelSamples = {0.0, 0.0, 1.0};
        detector.focalToPixelLines = {0.0, 1.0, 0.0};
        placamera::LineScanOptics optics;
        optics.focalLengthMillimeters = 700.0;
        optics.distortionK1 = 0.001;
        optics.distortionModel = placamera::LineScanDistortionModel::LroNacFocalPlane;
        optics.detectorGeometry = detector;
        auto definition = placamera::LineScanDefinition::create(
            placamera::CameraDefinitionId("line-definition"), placamera::FrameId("MOON_ME"), optics);

        const placamera::RotationMatrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        placamera::FrameComposedTrajectory trajectory;
        trajectory.inertialStates = {
            {placamera::TimeReference::create(placamera::TimeScale::Tdb, 10.0), {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0}},
            {placamera::TimeReference::create(placamera::TimeScale::Tdb, 20.0), {11.0, 2.0, 3.0}, {1.0, 0.0, 0.0}}};
        trajectory.inertialToWorld.constantRotation = identity;
        trajectory.inertialToSensor.constantRotation = identity;
        const std::vector<placamera::QuaternionTrajectorySample> attitudes{
            {placamera::TimeReference::create(placamera::TimeScale::Tdb, 10.0), {1.0, 0.0, 0.0, 0.0}},
            {placamera::TimeReference::create(placamera::TimeScale::Tdb, 20.0), {1.0, 0.0, 0.0, 0.0}}};
        trajectory.inertialToWorld.samples = attitudes;
        trajectory.inertialToSensor.samples = attitudes;

        placamera::LineTiming timing;
        timing.timeScale = placamera::TimeScale::Tdb;
        timing.segments = {{0.5, 10.0, 0.01}, {400.5, 14.0, 0.02}};
        placamera::LineScanTrajectoryBias bias;
        bias.translationMeters = {0.1, 0.2, 0.3};
        bias.timeOffsetSeconds = 0.005;
        return placamera::LineScanModel::create(
            placamera::CameraInstanceId("line-instance"),
            placamera::ImageId("line-image"),
            std::move(definition),
            placamera::ImageSize{1024, 512},
            placamera::LineScanTrajectory::createFrameComposed(std::move(trajectory)),
            std::move(timing),
            bias,
            placamera::TimeReference::create(placamera::TimeScale::Tdb, 15.0));
    }

    TEST(ProjectCameraIOTest, SerializesPlaCameraReportFields)
    {
        const QJsonObject json = serializeFramePinholeModel(makeCamera());
        EXPECT_EQ(json.value(QStringLiteral("model")).toString(), QStringLiteral("frame_pinhole"));
        EXPECT_DOUBLE_EQ(json.value(QStringLiteral("fu")).toDouble(), 35.0);
        EXPECT_DOUBLE_EQ(json.value(QStringLiteral("fv")).toDouble(), 35.5);
        EXPECT_EQ(json.value(QStringLiteral("C")).toArray().size(), 3);
        EXPECT_EQ(json.value(QStringLiteral("R")).toArray().size(), 9);
    }

    TEST(ProjectCameraIOTest, PreservesWorldFrameImageSizeAndAxisConventions)
    {
        const QJsonObject json = serializeFramePinholeModel(makeCamera());
        EXPECT_EQ(json.value(QStringLiteral("world_frame")).toString(), QStringLiteral("survey-world"));
        EXPECT_EQ(json.value(QStringLiteral("image_width")).toInt(), 640);
        EXPECT_EQ(json.value(QStringLiteral("image_height")).toInt(), 480);
        EXPECT_EQ(json.value(QStringLiteral("u_direction")).toInt(), -1);
        EXPECT_EQ(json.value(QStringLiteral("v_direction")).toInt(), 1);
        EXPECT_TRUE(json.value(QStringLiteral("depth_axis_flipped")).toBool());
    }

    TEST(ProjectCameraIOTest, NativeReportDecodesWithoutChangingGeometry)
    {
        const QJsonObject json = serializeFramePinholeModel(makeCamera());
        QString error;
        const auto decoded =
            decodeFramePinholeMetadata(json, placamera::CameraDefinitionId("decoded-report-definition"), &error);
        ASSERT_TRUE(decoded.has_value()) << error.toStdString();
        EXPECT_EQ(decoded->definition->groundFrame().value(), "survey-world");
        EXPECT_DOUBLE_EQ(decoded->definition->intrinsics().focalX, 7000.0);
        EXPECT_DOUBLE_EQ(decoded->definition->intrinsics().focalY, 7100.0);
        EXPECT_DOUBLE_EQ(decoded->pose.center[0], 1.0);
        EXPECT_TRUE(decoded->definition->depthAxisFlipped());
    }

    TEST(ProjectCameraIOTest, RoundTripsTypedLineScanThroughCanonicalProjectRecords)
    {
        const QString imagePath = QStringLiteral("images/line-image.tif");
        QJsonObject files{{QStringLiteral("images"),
                           QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("line-image")},
                                                  {QStringLiteral("path"), imagePath},
                                                  {QStringLiteral("samples"), 1024},
                                                  {QStringLiteral("lines"), 512}}}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(std::make_shared<const placamera::LineScanModel>(makeLineScanCamera())).ok());
        const auto update = xjw::placamera_runtime::insertProjectCameras(&files, cameras);
        ASSERT_TRUE(update.ok()) << update.errors.join('\n').toStdString();
        const QJsonObject definition = files.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject();
        EXPECT_EQ(definition.value(QStringLiteral("schema_version")).toInt(),
                  placamera::LineScanDefinition::ParameterSchemaVersion);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(files);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto lookup = loaded.instances.forImage(placamera::ImageId("line-image"));
        ASSERT_TRUE(lookup.ok()) << lookup.message().c_str();
        const auto* line = dynamic_cast<const placamera::LineScanModel*>(lookup.value().get());
        ASSERT_NE(line, nullptr);
        ASSERT_NE(line->trajectory().frameComposed(), nullptr);
        EXPECT_EQ(line->trajectory().frameComposed()->inertialStates.size(), 2U);
        EXPECT_EQ(line->trajectory().frameComposed()->inertialToWorld.samples.size(), 2U);
        EXPECT_EQ(line->lineTiming().segments.size(), 2U);
        EXPECT_TRUE(line->lineScanDefinition().optics().detectorGeometry.has_value());
        EXPECT_DOUBLE_EQ(line->trajectoryBias().translationMeters[0], 0.1);
        EXPECT_DOUBLE_EQ(line->trajectoryBias().timeOffsetSeconds, 0.005);
        ASSERT_TRUE(line->captureTime().has_value());
        EXPECT_DOUBLE_EQ(line->captureTime()->seconds, 15.0);

        const auto written = xjw::placamera_runtime::writeProjectCameras(&files, loaded.instances);
        ASSERT_TRUE(written.ok()) << written.errors.join('\n').toStdString();
        const auto reloaded = xjw::placamera_runtime::loadProjectCameras(files);
        ASSERT_TRUE(reloaded.ok()) << reloaded.errors.join('\n').toStdString();
        const auto restored = reloaded.instances.forImage(placamera::ImageId("line-image"));
        ASSERT_TRUE(restored.ok());
        const auto* restored_line = dynamic_cast<const placamera::LineScanModel*>(restored.value().get());
        ASSERT_NE(restored_line, nullptr);
        ASSERT_NE(restored_line->trajectory().frameComposed(), nullptr);
        EXPECT_EQ(restored_line->trajectory().frameComposed()->inertialStates.size(), 2U);
        EXPECT_EQ(restored_line->trajectory().frameComposed()->inertialToSensor.samples.size(), 2U);
        ASSERT_TRUE(restored_line->captureTime().has_value());
        EXPECT_DOUBLE_EQ(restored_line->captureTime()->seconds, 15.0);
        EXPECT_DOUBLE_EQ(restored_line->trajectoryBias().translationMeters[0], 0.1);
        EXPECT_DOUBLE_EQ(restored_line->trajectoryBias().timeOffsetSeconds, 0.005);
    }

    TEST(ProjectCameraIOTest, RejectsInvalidPlaCameraLineScanRecord)
    {
        const QString imagePath = QStringLiteral("images/line-image.tif");
        QJsonObject files{{QStringLiteral("images"),
                           QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("line-image")},
                                                  {QStringLiteral("path"), imagePath},
                                                  {QStringLiteral("samples"), 1024},
                                                  {QStringLiteral("lines"), 512}}}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(std::make_shared<const placamera::LineScanModel>(makeLineScanCamera())).ok());
        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&files, cameras);
        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        QJsonArray records = files.value(QStringLiteral("camera_instances")).toArray();
        QJsonObject record = records.at(0).toObject();
        QJsonObject state = record.value(QStringLiteral("state")).toObject();
        QJsonObject trajectory = state.value(QStringLiteral("trajectory")).toObject();
        trajectory.insert(QStringLiteral("representation"), QStringLiteral("unsupported_trajectory"));
        state.insert(QStringLiteral("trajectory"), trajectory);
        record.insert(QStringLiteral("state"), state);
        records.replace(0, record);
        files.insert(QStringLiteral("camera_instances"), records);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(files);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraIOTest, ImportsTsaiThroughPlaCameraGeometry)
    {
        const QString path = QString::fromUtf8(TEST_DATA_DIR "/tsai/1.tsai");
        const QString image_path = QStringLiteral("images/tsai-image.tif");
        QJsonObject files{{QStringLiteral("images"),
                           QJsonArray{QJsonObject{{QStringLiteral("image_uuid"), QStringLiteral("tsai-image")},
                                                  {QStringLiteral("path"), image_path},
                                                  {QStringLiteral("samples"), 4000},
                                                  {QStringLiteral("lines"), 3000}}}},
                          {QStringLiteral("camera_definitions"), QJsonArray{}},
                          {QStringLiteral("camera_instances"), QJsonArray{}}};
        const auto parsed = placamera::loadTsaiFramePinhole(
            path.toStdString(), placamera::CameraDefinitionId("tsai-definition"), placamera::FrameId("project-world"));
        ASSERT_TRUE(parsed) << parsed.message();
        const auto bound = placamera::bindFramePinhole(parsed.value(),
                                                       {placamera::CameraInstanceId("tsai-instance"),
                                                        placamera::ImageId("tsai-image"),
                                                        placamera::ImageSize{4000, 3000}});
        ASSERT_TRUE(bound) << bound.message();
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(bound.value()).ok());
        const auto update = xjw::placamera_runtime::insertProjectCameras(&files, cameras);
        ASSERT_TRUE(update.ok()) << update.errors.join('\n').toStdString();
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(files);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto camera = loaded.instances.forImage(placamera::ImageId("tsai-image"));
        ASSERT_TRUE(camera.ok()) << camera.message();
        EXPECT_NE(dynamic_cast<const placamera::FramePinholeModel*>(camera.value().get()), nullptr);
    }

} // namespace
