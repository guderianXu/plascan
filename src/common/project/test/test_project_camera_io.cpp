#include "ProjectCameraIO.h"

#include "camera/models/CameraModelFactories.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/models/rpc/RpcInstance.h"
#include "camera/project/CameraProjectRecords.h"
#include "camera/project/CameraProjectRuntime.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using namespace xjw::common::project;

    xjw::camera_models::frame_pinhole::FramePinholeNumericState makeCamera()
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        camera.setIntrinsicsMillimeters(35.0, 35.5, 0.2, -0.1, 0.005);
        camera.setAxisDirections(-1, 1);
        camera.setDepthAxisFlipped(true);
        camera.setDistortion(0.01, -0.001, 0.0001, 0.0002, -0.0003);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{1.0, 2.0, 3.0}});
        return camera;
    }

    xjw::camera_models::rpc::RpcInstance makeRpcCamera()
    {
        xjw::camera_models::rpc::RpcDefinition::Parameters parameters;
        parameters.lineOffset = 3000.0;
        parameters.sampleOffset = 4000.0;
        parameters.latitudeOffset = 34.0;
        parameters.longitudeOffset = 108.0;
        parameters.heightOffset = 1200.0;
        parameters.lineScale = 3000.0;
        parameters.sampleScale = 4000.0;
        parameters.latitudeScale = 0.1;
        parameters.longitudeScale = 0.1;
        parameters.heightScale = 1000.0;
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        parameters.errorBiasMeters = 3.5;

        auto definition =
            xjw::camera_models::rpc::RpcDefinition::create(xjw::camera_core::CameraDefinitionId("rpc-definition"),
                                                           xjw::coordinate_system::CoordinateFrameId("EPSG:4978"),
                                                           parameters);
        xjw::camera_models::rpc::ImageCorrection correction;
        correction.sampleOffsetPixels = 1.25;
        correction.sampleLinePixels = -0.5;
        correction.lineOffsetPixels = -2.0;
        correction.lineSamplePixels = 0.75;
        return xjw::camera_models::rpc::RpcInstance::create(xjw::camera_core::CameraInstanceId("rpc-instance"),
                                                            xjw::camera_core::ImageId("rpc-image"),
                                                            std::move(definition),
                                                            xjw::camera_core::ImageSize{8000, 6000},
                                                            correction);
    }

    xjw::camera_models::linescan::LineScanInstance makeLineScanCamera()
    {
        using namespace xjw::camera_models::linescan;
        LineScanDetectorGeometry detector;
        detector.detectorSampleOrigin = 512.0;
        detector.focalToPixelSamples = {0.0, 0.0, 1.0};
        detector.focalToPixelLines = {0.0, 1.0, 0.0};
        LineScanOptics optics;
        optics.focalLengthMillimeters = 700.0;
        optics.distortionK1 = 0.001;
        optics.distortionModel = LineScanDistortionModel::LroNacFocalPlane;
        optics.detectorGeometry = detector;
        auto definition = LineScanDefinition::create(xjw::camera_core::CameraDefinitionId("line-definition"),
                                                     xjw::coordinate_system::CoordinateFrameId("MOON_ME"),
                                                     optics);

        const xjw::camera_core::Rotation identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        FrameComposedTrajectory trajectory;
        trajectory.inertialStates = {{xjw::coordinate_system::TimeReference::create(xjw::coordinate_system::TimeScale::Tdb, 10.0),
                                      {1.0, 2.0, 3.0},
                                      {1.0, 0.0, 0.0}},
                                     {xjw::coordinate_system::TimeReference::create(xjw::coordinate_system::TimeScale::Tdb, 20.0),
                                      {11.0, 2.0, 3.0},
                                      {1.0, 0.0, 0.0}}};
        trajectory.inertialToWorld.constantRotation = identity;
        trajectory.inertialToSensor.constantRotation = identity;
        const std::vector<QuaternionTrajectorySample> attitudes{
            {xjw::coordinate_system::TimeReference::create(xjw::coordinate_system::TimeScale::Tdb, 10.0), {1.0, 0.0, 0.0, 0.0}},
            {xjw::coordinate_system::TimeReference::create(xjw::coordinate_system::TimeScale::Tdb, 20.0), {1.0, 0.0, 0.0, 0.0}}};
        trajectory.inertialToWorld.samples = attitudes;
        trajectory.inertialToSensor.samples = attitudes;

        LineTiming timing;
        timing.timeScale = xjw::coordinate_system::TimeScale::Tdb;
        timing.segments = {{0.5, 10.0, 0.01}, {400.5, 14.0, 0.02}};
        return LineScanInstance::create(
            xjw::camera_core::CameraInstanceId("line-instance"),
            xjw::camera_core::ImageId("line-image"),
            std::move(definition),
            xjw::camera_core::ImageSize{1024, 512},
            LineScanTrajectory::createFrameComposed(std::move(trajectory)),
            std::move(timing),
            xjw::coordinate_system::TimeReference::create(xjw::coordinate_system::TimeScale::Tdb, 15.0));
    }

    TEST(ProjectCameraIOTest, RoundTripsCameraJson)
    {
        const xjw::camera_models::frame_pinhole::FramePinholeNumericState source = makeCamera();
        const QJsonObject json = serializeFramePinholeNumericState(source);

        xjw::camera_models::frame_pinhole::FramePinholeNumericState restored;
        ASSERT_TRUE(decodeFramePinholeNumericState(json, &restored));
        EXPECT_TRUE(restored.isValid());
        EXPECT_DOUBLE_EQ(restored.focalXMillimeters(), source.focalXMillimeters());
        EXPECT_DOUBLE_EQ(restored.focalYMillimeters(), source.focalYMillimeters());
        EXPECT_EQ(restored.uAxisSign(), source.uAxisSign());
        EXPECT_EQ(restored.depthAxisFlipped(), source.depthAxisFlipped());
        EXPECT_EQ(restored.cameraCenter(), source.cameraCenter());
    }

    TEST(ProjectCameraIOTest, ReadsFrameCameraFromModelJson)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState camera;
        EXPECT_TRUE(decodeFramePinholeNumericState(serializeFramePinholeNumericState(makeCamera()), &camera));
        EXPECT_TRUE(camera.isValid());
    }

    TEST(ProjectCameraIOTest, DecodesValidSerializedNumericState)
    {
        const QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        ASSERT_TRUE(decodeFramePinholeNumericState(json, &state));
        EXPECT_TRUE(state.validateNumericalState());
    }

    TEST(ProjectCameraIOTest, SerializesBoundNumericStateWorldFrameForProjectWriteback)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState source;
        source.setIntrinsics(800.0, 805.0, 320.0, 240.0);
        source.setImageSize(xjw::camera_core::ImageSize{640, 480});
        source.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{1.0, 2.0, 3.0}});
        ASSERT_TRUE(source.bindIdentity(xjw::camera_core::CameraInstanceId("instance-1"),
                                        xjw::camera_core::ImageId("image-1"),
                                        xjw::coordinate_system::CoordinateFrameId("survey-world")));

        const QJsonObject json = serializeFramePinholeNumericState(source);
        EXPECT_EQ(json.value(QStringLiteral("world_frame")).toString(), QStringLiteral("survey-world"));

        xjw::camera_models::frame_pinhole::FramePinholeNumericState restored;
        ASSERT_TRUE(decodeFramePinholeNumericState(json, &restored));
        EXPECT_FALSE(restored.hasBoundIdentity());
        ASSERT_TRUE(restored.imageSize().has_value());
        EXPECT_EQ(restored.imageSize()->samples, 640);
        EXPECT_EQ(restored.imageSize()->lines, 480);
    }

    TEST(ProjectCameraIOTest, RejectsMalformedOptionalNumericStateImageSize)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        json.insert(QStringLiteral("image_width"), QStringLiteral("640"));
        json.insert(QStringLiteral("image_height"), 480);

        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, RejectsZeroFocalLength)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        json[QStringLiteral("fu")] = 0.0;
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, RejectsMissingRequiredNumericField)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        json.remove(QStringLiteral("fu"));
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, RejectsNonCanonicalFrameRecordIdentifiersAndUnits)
    {
        const QJsonObject canonical = serializeFramePinholeNumericState(makeCamera());
        for (const auto& replacement :
             {std::pair{QStringLiteral("model"), QJsonValue(QStringLiteral("tsai"))},
              std::pair{QStringLiteral("intrinsics_unit"), QJsonValue(QStringLiteral("px"))},
              std::pair{QStringLiteral("camera_center_unit"), QJsonValue(QStringLiteral("mm"))},
              std::pair{QStringLiteral("pixel_convention"), QJsonValue(QStringLiteral("pixel_center"))}})
        {
            QJsonObject changed = canonical;
            changed.insert(replacement.first, replacement.second);
            xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
            EXPECT_FALSE(decodeFramePinholeNumericState(changed, &state));
        }
    }

    TEST(ProjectCameraIOTest, RejectsNegativePixelPitch)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        json[QStringLiteral("pitch")] = -0.005;
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, RejectsDegenerateRotation)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        QJsonArray rotation;
        for (const double value : {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0})
        {
            rotation.append(value);
        }
        json[QStringLiteral("R")] = rotation;
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, RejectsNonFiniteRotationElement)
    {
        QJsonObject json = serializeFramePinholeNumericState(makeCamera());
        QJsonArray rotation;
        for (const double value : {1.0, 0.0, 0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 1.0})
        {
            rotation.append(value);
        }
        json[QStringLiteral("R")] = rotation;
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;

        EXPECT_FALSE(decodeFramePinholeNumericState(json, &state));
    }

    TEST(ProjectCameraIOTest, ReturnsNullForInvalidNumericState)
    {
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        state.setIntrinsics(100.0, 100.0, 320.0, 240.0);
        state.setImageSize(xjw::camera_core::ImageSize{640, 480});
        state.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}});

        EXPECT_EQ(state.toInstance(), nullptr);
    }

    TEST(ProjectCameraIOTest, SerializesTypedRpcInstanceForCanonicalWriteback)
    {
        const xjw::camera_models::rpc::RpcInstance source = makeRpcCamera();
        const QJsonObject json = serializeRpcInstance(source);
        EXPECT_EQ(json.value(QStringLiteral("model")).toString(), QStringLiteral("rpc00b"));
        EXPECT_EQ(json.value(QStringLiteral("world_frame")).toString(), QStringLiteral("EPSG:4978"));
        EXPECT_EQ(json.value(QStringLiteral("image_samples")).toInt(), 8000);
        EXPECT_EQ(json.value(QStringLiteral("image_lines")).toInt(), 6000);
        EXPECT_EQ(json.value(QStringLiteral("line_num_coeff")).toArray().size(), 20);
        EXPECT_DOUBLE_EQ(json.value(QStringLiteral("err_bias_m")).toDouble(), 3.5);
        const QJsonObject correction = json.value(QStringLiteral("image_correction")).toObject();
        EXPECT_DOUBLE_EQ(correction.value(QStringLiteral("sample_offset_px")).toDouble(), 1.25);
        EXPECT_DOUBLE_EQ(correction.value(QStringLiteral("sample_line_px")).toDouble(), -0.5);
        EXPECT_DOUBLE_EQ(correction.value(QStringLiteral("line_offset_px")).toDouble(), -2.0);
        EXPECT_DOUBLE_EQ(correction.value(QStringLiteral("line_sample_px")).toDouble(), 0.75);
    }

    TEST(ProjectCameraIOTest, SerializesExactTypedLineScanStateForCanonicalWriteback)
    {
        const QJsonObject json = serializeLineScanInstance(makeLineScanCamera());
        EXPECT_EQ(json.value(QStringLiteral("model")).toString(), QStringLiteral("planetary_linescan"));
        EXPECT_EQ(json.value(QStringLiteral("world_frame")).toString(), QStringLiteral("MOON_ME"));
        EXPECT_EQ(json.value(QStringLiteral("image_samples")).toInt(), 1024);
        const QJsonObject optics = json.value(QStringLiteral("optics")).toObject();
        EXPECT_EQ(optics.value(QStringLiteral("distortion_model")).toString(), QStringLiteral("lro_nac_focal_plane"));
        EXPECT_EQ(optics.value(QStringLiteral("sample_geometry")).toObject().value(QStringLiteral("type")).toString(),
                  QStringLiteral("detector_affine"));
        const QJsonObject trajectory = json.value(QStringLiteral("trajectory")).toObject();
        EXPECT_EQ(trajectory.value(QStringLiteral("representation")).toString(), QStringLiteral("frame_composed"));
        EXPECT_EQ(trajectory.value(QStringLiteral("inertial_states")).toArray().size(), 2);
        EXPECT_EQ(
            json.value(QStringLiteral("line_timing")).toObject().value(QStringLiteral("segments")).toArray().size(), 2);
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
        const QJsonObject metadata = serializeLineScanInstance(makeLineScanCamera());
        const xjw::camera_project::CameraProjectUpdateResult update =
            xjw::camera_project::CameraProjectRecords::upsertByImagePath(
                &files, QMap<QString, QJsonObject>{{imagePath, metadata}});
        ASSERT_TRUE(update.ok()) << update.errors.join('\n').toStdString();
        const QJsonObject definition = files.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject();
        EXPECT_EQ(definition.value(QStringLiteral("schema_version")).toInt(), 2);

        const xjw::camera_project::CameraProjectRuntimeResult loaded = xjw::camera_project::CameraProjectRuntime::load(
            files, xjw::camera_models::makeBuiltinCameraModelRegistry());
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto lookup = loaded.instances.forImage(xjw::camera_core::ImageId("line-image"));
        ASSERT_TRUE(lookup.ok()) << lookup.error;
        const auto* line = dynamic_cast<const xjw::camera_models::linescan::LineScanInstance*>(lookup.instance.get());
        ASSERT_NE(line, nullptr);
        ASSERT_NE(line->trajectory().frameComposed(), nullptr);
        EXPECT_EQ(line->lineTiming().segments.size(), 2U);
        EXPECT_TRUE(line->lineScanDefinition().optics().detectorGeometry.has_value());
    }

    TEST(ProjectCameraIOTest, ReportsInvalidTsaiPath)
    {
        QJsonObject metadata;
        QString error;

        EXPECT_FALSE(parseTsaiCamera(QStringLiteral("missing-camera.tsai"), &metadata, &error));
        EXPECT_TRUE(metadata.isEmpty());
        EXPECT_TRUE(error.contains(QStringLiteral("missing-camera.tsai")));
    }

} // namespace
