#include "placamera_runtime/ProjectCameraStore.h"

#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/rpc_camera.h>
#include <placamera/state_codec.h>

#include <QJsonArray>
#include <QJsonObject>

#include <gtest/gtest.h>

#include <memory>

namespace
{

    QJsonObject image(const QString& id)
    {
        return {{QStringLiteral("image_uuid"), id},
                {QStringLiteral("path"), QStringLiteral("images/") + id + QStringLiteral(".tif")}};
    }

    QJsonArray rpcCoefficients(bool denominator)
    {
        QJsonArray coefficients;
        for (int index = 0; index < 20; ++index)
        {
            coefficients.append(denominator && index == 0 ? 1.0 : 0.0);
        }
        return coefficients;
    }

    QJsonObject frameDefinition()
    {
        return {{QStringLiteral("id"), QStringLiteral("frame-definition")},
                {QStringLiteral("model_type"), QStringLiteral("frame_pinhole")},
                {QStringLiteral("schema_version"), 1},
                {QStringLiteral("frame"), QStringLiteral("EPSG:4978")},
                {QStringLiteral("parameters"),
                 QJsonObject{{QStringLiteral("intrinsics"),
                              QJsonObject{{QStringLiteral("fx_px"), 100.0},
                                          {QStringLiteral("fy_px"), 100.0},
                                          {QStringLiteral("cx_px"), 50.0},
                                          {QStringLiteral("cy_px"), 40.0},
                                          {QStringLiteral("pixel_pitch_mm"), 0.01},
                                          {QStringLiteral("u_axis_sign"), 1},
                                          {QStringLiteral("v_axis_sign"), 1}}},
                             {QStringLiteral("distortion"),
                              QJsonObject{{QStringLiteral("k1"), 0.0},
                                          {QStringLiteral("k2"), 0.0},
                                          {QStringLiteral("k3"), 0.0},
                                          {QStringLiteral("p1"), 0.0},
                                          {QStringLiteral("p2"), 0.0}}},
                             {QStringLiteral("pixel_convention"), QStringLiteral("center")},
                             {QStringLiteral("depth_axis_flipped"), false}}}};
    }

    QJsonObject frameInstance()
    {
        QJsonObject pose{
            {QStringLiteral("frame"), QStringLiteral("EPSG:4978")},
            {QStringLiteral("center_m"), QJsonArray{0.0, 0.0, 0.0}},
            {QStringLiteral("camera_to_world_rotation"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}};
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("frame-instance")},
                           {QStringLiteral("image_uuid"), QStringLiteral("frame-image")},
                           {QStringLiteral("definition_id"), QStringLiteral("frame-definition")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), 100}, {QStringLiteral("lines"), 80}}},
                           {QStringLiteral("pose"), pose}};
    }

    QJsonObject rpcDefinition()
    {
        return {{QStringLiteral("id"), QStringLiteral("rpc-definition")},
                {QStringLiteral("model_type"), QStringLiteral("rpc00b")},
                {QStringLiteral("schema_version"), 1},
                {QStringLiteral("frame"), QStringLiteral("MOON_ME")},
                {QStringLiteral("parameters"),
                 QJsonObject{{QStringLiteral("rpc_spec"), QStringLiteral("RPC00B")},
                             {QStringLiteral("line_offset"), 40.0},
                             {QStringLiteral("sample_offset"), 50.0},
                             {QStringLiteral("latitude_offset"), 0.0},
                             {QStringLiteral("longitude_offset"), 0.0},
                             {QStringLiteral("height_offset"), 0.0},
                             {QStringLiteral("line_scale"), 40.0},
                             {QStringLiteral("sample_scale"), 50.0},
                             {QStringLiteral("latitude_scale"), 1.0},
                             {QStringLiteral("longitude_scale"), 1.0},
                             {QStringLiteral("height_scale"), 1000.0},
                             {QStringLiteral("line_numerator"), rpcCoefficients(false)},
                             {QStringLiteral("line_denominator"), rpcCoefficients(true)},
                             {QStringLiteral("sample_numerator"), rpcCoefficients(false)},
                             {QStringLiteral("sample_denominator"), rpcCoefficients(true)}}}};
    }

    QJsonObject rpcInstance()
    {
        return {{QStringLiteral("id"), QStringLiteral("rpc-instance")},
                {QStringLiteral("image_uuid"), QStringLiteral("rpc-image")},
                {QStringLiteral("definition_id"), QStringLiteral("rpc-definition")},
                {QStringLiteral("schema_version"), 1},
                {QStringLiteral("image_size"),
                 QJsonObject{{QStringLiteral("samples"), 100}, {QStringLiteral("lines"), 80}}}};
    }

    QJsonObject lineDefinition()
    {
        QJsonObject optics{{QStringLiteral("focal_length_mm"), 10.0},
                           {QStringLiteral("sample_pitch_mm"), 0.01},
                           {QStringLiteral("principal_sample"), 5.0},
                           {QStringLiteral("distortion_k1"), 0.0},
                           {QStringLiteral("distortion_model"), QStringLiteral("radial_normalized")},
                           {QStringLiteral("detector_geometry"), QJsonValue()}};
        QJsonObject parameters{{QStringLiteral("optics"), optics},
                               {QStringLiteral("pixel_convention"), QStringLiteral("pixel_center")}};
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("line-definition")},
                           {QStringLiteral("model_type"), QStringLiteral("planetary_linescan")},
                           {QStringLiteral("schema_version"), 2},
                           {QStringLiteral("frame"), QStringLiteral("local")},
                           {QStringLiteral("parameters"), parameters}};
    }

    QJsonObject lineInstance()
    {
        QJsonObject sample0{
            {QStringLiteral("time_seconds"), 0.0},
            {QStringLiteral("center_m"), QJsonArray{0.0, -1.0, 0.0}},
            {QStringLiteral("camera_to_world_rotation"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}};
        QJsonObject sample1{
            {QStringLiteral("time_seconds"), 1.0},
            {QStringLiteral("center_m"), QJsonArray{0.0, 1.0, 0.0}},
            {QStringLiteral("camera_to_world_rotation"), QJsonArray{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}};
        QJsonObject trajectory{{QStringLiteral("time_scale"), QStringLiteral("tdb")},
                               {QStringLiteral("representation"), QStringLiteral("direct_pose_samples")},
                               {QStringLiteral("samples"), QJsonArray{sample0, sample1}}};
        QJsonObject timing{{QStringLiteral("time_scale"), QStringLiteral("tdb")},
                           {QStringLiteral("segments"),
                            QJsonArray{QJsonObject{{QStringLiteral("start_line"), 0.5},
                                                   {QStringLiteral("start_time_seconds"), 0.0},
                                                   {QStringLiteral("seconds_per_line"), 0.1}}}}};
        QJsonObject state{{QStringLiteral("trajectory"), trajectory}, {QStringLiteral("line_timing"), timing}};
        return QJsonObject{{QStringLiteral("id"), QStringLiteral("line-instance")},
                           {QStringLiteral("image_uuid"), QStringLiteral("line-image")},
                           {QStringLiteral("definition_id"), QStringLiteral("line-definition")},
                           {QStringLiteral("schema_version"), 1},
                           {QStringLiteral("image_size"),
                            QJsonObject{{QStringLiteral("samples"), 1000}, {QStringLiteral("lines"), 11}}},
                           {QStringLiteral("state"), state}};
    }

    QJsonObject validProject()
    {
        return {
            {QStringLiteral("images"),
             QJsonArray{image(QStringLiteral("frame-image")),
                        image(QStringLiteral("rpc-image")),
                        image(QStringLiteral("line-image"))}},
            {QStringLiteral("camera_definitions"), QJsonArray{frameDefinition(), rpcDefinition(), lineDefinition()}},
            {QStringLiteral("camera_instances"), QJsonArray{frameInstance(), rpcInstance(), lineInstance()}}};
    }

    std::shared_ptr<const placamera::FramePinholeModel>
    nativeFrame(const std::string& image_id,
                const std::string& instance_id,
                std::optional<placamera::TimeReference> capture_time = std::nullopt)
    {
        const placamera::FrameId frame("project-world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("native-definition"),
                                                      {120.0, 120.0, 50.0, 40.0, 1.0, 1, 1},
                                                      {},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(instance_id),
            placamera::ImageId(image_id),
            definition,
            {100, 80},
            placamera::Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
            capture_time));
    }

    std::shared_ptr<const placamera::CentralCameraModel> nativeRichCentralCamera()
    {
        placamera::FrameCalibration calibration;
        calibration.f = 1000.0;
        calibration.cx = 640.0;
        calibration.cy = 512.0;
        calibration.b1 = 2.5;
        calibration.b2 = 0.75;
        calibration.k1 = 0.01;
        calibration.k2 = -0.002;
        calibration.k3 = 0.0003;
        calibration.k4 = -0.00004;
        calibration.p1 = 0.0005;
        calibration.p2 = -0.0006;
        calibration.p3 = 0.00007;
        calibration.p4 = -0.000008;
        calibration.principalPointDecomposition = placamera::PrincipalPointDecomposition{600.0, 500.0, 40.0, 12.0};

        placamera::SensorMountState mount;
        mount.masterSensorId = placamera::CameraDefinitionId("master-sensor");
        mount.translation = {0.1, 0.2, 0.3};
        mount.fixedTranslation = false;
        const placamera::FrameId frame("project-world");
        const auto definition =
            placamera::CentralCameraDefinition::create(placamera::CameraDefinitionId("rich-central-definition"),
                                                       calibration,
                                                       placamera::PixelConvention::PixelCenter,
                                                       frame,
                                                       false,
                                                       0.005,
                                                       1,
                                                       1,
                                                       placamera::FrameProjectionModel::EquisolidFisheye,
                                                       mount);

        placamera::CameraAcquisitionState acquisition;
        acquisition.role = placamera::CameraRole::Keyframe;
        acquisition.captureGroupId = placamera::CaptureGroupId("capture-group");
        acquisition.masterCameraId = placamera::CameraInstanceId("master-camera");
        acquisition.layerIndex = 3;
        acquisition.rollingShutterMode = placamera::RollingShutterMode::Full;
        acquisition.rollingShutter.translation = {0.01, -0.02, 0.03};
        acquisition.rollingShutter.rotationVector = {0.001, -0.002, 0.003};
        acquisition.rollingShutterInitialized = true;

        return std::make_shared<const placamera::CentralCameraModel>(placamera::CentralCameraModel::create(
            placamera::CameraInstanceId("rich-central-instance"),
            placamera::ImageId("rich-central-image"),
            definition,
            placamera::ImageSize{1280, 1024},
            placamera::Pose::create(frame, {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
            placamera::TimeReference::create(placoordinate::TimeScale::Utc, 123.0),
            acquisition));
    }

    TEST(ProjectCameraStoreTest, InsertsNativeFrameCamerasForUnboundImages)
    {
        QJsonObject project{
            {QStringLiteral("images"), QJsonArray{image(QStringLiteral("first")), image(QStringLiteral("second"))}}};
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeFrame("first", "first-instance")).ok());
        ASSERT_TRUE(cameras.add(nativeFrame("second", "second-instance")).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);

        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        EXPECT_EQ(inserted.insertedCount, 2);
        EXPECT_EQ(project.value(QStringLiteral("camera_definitions")).toArray().size(), 1);
        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        EXPECT_EQ(restored.instances.size(), 2U);
        const auto first = restored.instances.forImage(placamera::ImageId("first"));
        ASSERT_TRUE(first.ok());
        const auto* frame = dynamic_cast<const placamera::FramePinholeModel*>(first.value().get());
        ASSERT_NE(frame, nullptr);
        EXPECT_DOUBLE_EQ(frame->pose().center[0], 1.0);
        EXPECT_DOUBLE_EQ(frame->pinholeDefinition().intrinsics().focalX, 120.0);
    }

    TEST(ProjectCameraStoreTest, PreservesNativeFrameCaptureTimeOnInsertAndUpdate)
    {
        QJsonObject project{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("first"))}}};
        placamera::CameraInstanceSet inserted_models;
        ASSERT_TRUE(inserted_models
                        .add(nativeFrame("first",
                                         "first-instance",
                                         placamera::TimeReference::create(placoordinate::TimeScale::Utc, 123.5)))
                        .ok());
        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, inserted_models);
        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        auto first = restored.instances.forImage(placamera::ImageId("first"));
        ASSERT_TRUE(first.ok());
        ASSERT_TRUE(first.value()->captureTime());
        EXPECT_DOUBLE_EQ(first.value()->captureTime()->seconds, 123.5);

        placamera::CameraInstanceSet updated_models;
        ASSERT_TRUE(updated_models
                        .add(nativeFrame("first",
                                         "first-instance",
                                         placamera::TimeReference::create(placoordinate::TimeScale::Utc, 456.75)))
                        .ok());
        const auto updated = xjw::placamera_runtime::writeProjectCameras(&project, updated_models);
        ASSERT_TRUE(updated.ok()) << updated.errors.join('\n').toStdString();
        restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        first = restored.instances.forImage(placamera::ImageId("first"));
        ASSERT_TRUE(first.ok());
        ASSERT_TRUE(first.value()->captureTime());
        EXPECT_DOUBLE_EQ(first.value()->captureTime()->seconds, 456.75);
    }

    TEST(ProjectCameraStoreTest, PreservesCompleteCentralCameraState)
    {
        QJsonObject project{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("rich-central-image"))}}};
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeRichCentralCamera()).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);

        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        const QJsonObject definition = project.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject();
        EXPECT_EQ(definition.value(QStringLiteral("model_type")), QStringLiteral("frame_equisolid_fisheye"));
        EXPECT_EQ(definition.value(QStringLiteral("schema_version")).toInt(),
                  placamera::CentralCameraDefinition::ParameterSchemaVersion);
        EXPECT_DOUBLE_EQ(definition.value(QStringLiteral("parameters"))
                             .toObject()
                             .value(QStringLiteral("intrinsics"))
                             .toObject()
                             .value(QStringLiteral("skew"))
                             .toDouble(),
                         0.75);
        const QJsonObject instance = project.value(QStringLiteral("camera_instances")).toArray().at(0).toObject();
        EXPECT_EQ(instance.value(QStringLiteral("schema_version")).toInt(),
                  placamera::CentralCameraModel::InstanceSchemaVersion);
        EXPECT_EQ(
            instance.value(QStringLiteral("acquisition")).toObject().value(QStringLiteral("rolling_shutter_mode")),
            QStringLiteral("full"));

        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        const auto selected = restored.instances.forImage(placamera::ImageId("rich-central-image"));
        ASSERT_TRUE(selected.ok());
        const auto* camera = dynamic_cast<const placamera::CentralCameraModel*>(selected.value().get());
        ASSERT_NE(camera, nullptr);
        EXPECT_EQ(camera->pinholeDefinition().projectionModel(), placamera::FrameProjectionModel::EquisolidFisheye);
        const placamera::FrameCalibration calibration = camera->pinholeDefinition().calibration();
        EXPECT_DOUBLE_EQ(calibration.f, 1000.0);
        EXPECT_DOUBLE_EQ(calibration.b1, 2.5);
        EXPECT_DOUBLE_EQ(calibration.b2, 0.75);
        EXPECT_DOUBLE_EQ(calibration.k4, -0.00004);
        EXPECT_DOUBLE_EQ(calibration.p4, -0.000008);
        ASSERT_TRUE(calibration.principalPointDecomposition);
        EXPECT_EQ(*calibration.principalPointDecomposition,
                  (placamera::PrincipalPointDecomposition{600.0, 500.0, 40.0, 12.0}));
        ASSERT_TRUE(camera->pinholeDefinition().sensorMount().masterSensorId);
        EXPECT_EQ(camera->pinholeDefinition().sensorMount().masterSensorId->value(), "master-sensor");
        EXPECT_EQ(camera->acquisition().role, placamera::CameraRole::Keyframe);
        ASSERT_TRUE(camera->acquisition().captureGroupId);
        EXPECT_EQ(camera->acquisition().captureGroupId->value(), "capture-group");
        EXPECT_EQ(camera->acquisition().layerIndex, 3U);
        EXPECT_EQ(camera->acquisition().rollingShutterMode, placamera::RollingShutterMode::Full);
        EXPECT_DOUBLE_EQ(camera->acquisition().rollingShutter.translation[1], -0.02);
        EXPECT_TRUE(camera->acquisition().rollingShutterInitialized);
    }

    TEST(ProjectCameraStoreTest, LoadsPreviousCentralCameraSchemaWithoutPrincipalPointDecomposition)
    {
        QJsonObject project{{QStringLiteral("images"), QJsonArray{image(QStringLiteral("rich-central-image"))}}};
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeRichCentralCamera()).ok());
        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);
        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();

        QJsonArray definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        QJsonObject definition = definitions.at(0).toObject();
        definition.insert(QStringLiteral("schema_version"), 3);
        QJsonObject parameters = definition.value(QStringLiteral("parameters")).toObject();
        parameters.remove(QStringLiteral("principal_point_decomposition"));
        definition.insert(QStringLiteral("parameters"), parameters);
        definitions.replace(0, definition);
        project.insert(QStringLiteral("camera_definitions"), definitions);

        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        const auto selected = restored.instances.forImage(placamera::ImageId("rich-central-image"));
        ASSERT_TRUE(selected.ok());
        const auto* camera = dynamic_cast<const placamera::CentralCameraModel*>(selected.value().get());
        ASSERT_NE(camera, nullptr);
        EXPECT_EQ(camera->pinholeDefinition().projectionModel(), placamera::FrameProjectionModel::EquisolidFisheye);
        const placamera::FrameCalibration calibration = camera->pinholeDefinition().calibration();
        EXPECT_DOUBLE_EQ(calibration.cx, 640.0);
        EXPECT_DOUBLE_EQ(calibration.cy, 512.0);
        EXPECT_FALSE(calibration.principalPointDecomposition);
    }

    TEST(ProjectCameraStoreTest, RejectsOccupiedImageWithoutPartialInsertion)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("new-image")));
        project.insert(QStringLiteral("images"), images);
        const QJsonObject before = project;
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeFrame("new-image", "new-instance")).ok());
        ASSERT_TRUE(cameras.add(nativeFrame("frame-image", "occupied-instance")).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);

        EXPECT_FALSE(inserted.ok());
        EXPECT_EQ(inserted.insertedCount, 0);
        EXPECT_EQ(project, before);
    }

    TEST(ProjectCameraStoreTest, InsertsNativeRpcCameraWithoutFlattenedMetadata)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("new-rpc")));
        project.insert(QStringLiteral("images"), images);
        const auto current = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(current.ok()) << current.errors.join('\n').toStdString();
        const auto existing = current.instances.forImage(placamera::ImageId("rpc-image"));
        ASSERT_TRUE(existing.ok());
        const auto* existingRpc = dynamic_cast<const placamera::RpcModel*>(existing.value().get());
        ASSERT_NE(existingRpc, nullptr);
        const auto definition = placamera::RpcDefinition::create(placamera::CameraDefinitionId("new-rpc-definition"),
                                                                 existingRpc->groundFrame(),
                                                                 existingRpc->rpcDefinition().parameters(),
                                                                 placamera::ReferenceEllipsoid{7000000.0, 300.0});
        auto imported = std::make_shared<const placamera::RpcModel>(
            placamera::RpcModel::create(placamera::CameraInstanceId("new-rpc-instance"),
                                        placamera::ImageId("new-rpc"),
                                        definition,
                                        placamera::ImageSize{100, 80},
                                        placamera::RpcImageCorrection{1.25, 0.0, -0.5, -2.0, 0.75, 0.0},
                                        placamera::TimeReference::create(placoordinate::TimeScale::Tdb, 789.25)));
        ASSERT_TRUE(placamera::encodeCameraInstanceJson(*imported).ok());
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(imported).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);

        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        EXPECT_EQ(inserted.insertedCount, 1);
        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        const auto reloaded = restored.instances.forImage(placamera::ImageId("new-rpc"));
        ASSERT_TRUE(reloaded.ok());
        const auto* rpc = dynamic_cast<const placamera::RpcModel*>(reloaded.value().get());
        ASSERT_NE(rpc, nullptr);
        EXPECT_DOUBLE_EQ(rpc->rpcDefinition().ellipsoid().semiMajorAxisMeters, 7000000.0);
        EXPECT_DOUBLE_EQ(rpc->rpcDefinition().ellipsoid().inverseFlattening, 300.0);
        const QJsonObject new_definition = project.value(QStringLiteral("camera_definitions"))
                                               .toArray()
                                               .last()
                                               .toObject()
                                               .value(QStringLiteral("parameters"))
                                               .toObject();
        EXPECT_FALSE(new_definition.contains(QStringLiteral("ground_crs")));
        EXPECT_FALSE(new_definition.contains(QStringLiteral("height_datum")));
        EXPECT_DOUBLE_EQ(rpc->imageCorrection().sampleOffsetPixels, 1.25);
        EXPECT_DOUBLE_EQ(rpc->imageCorrection().sampleLinePixels, -0.5);
        EXPECT_DOUBLE_EQ(rpc->imageCorrection().lineOffsetPixels, -2.0);
        EXPECT_DOUBLE_EQ(rpc->imageCorrection().lineSamplePixels, 0.75);
        ASSERT_TRUE(rpc->captureTime());
        EXPECT_DOUBLE_EQ(rpc->captureTime()->seconds, 789.25);
    }

    TEST(ProjectCameraStoreTest, PreservesGroundDomainRpcCorrectionWithoutReinterpretingIt)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("ground-rpc")));
        project.insert(QStringLiteral("images"), images);
        const auto current = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(current.ok()) << current.errors.join('\n').toStdString();
        const auto existing = current.instances.forImage(placamera::ImageId("rpc-image"));
        ASSERT_TRUE(existing.ok());
        const auto* base = dynamic_cast<const placamera::RpcModel*>(existing.value().get());
        ASSERT_NE(base, nullptr);
        const auto definition = placamera::RpcDefinition::create(placamera::CameraDefinitionId("ground-rpc-definition"),
                                                                 base->groundFrame(),
                                                                 base->rpcDefinition().parameters());
        placamera::RpcGroundCorrection correction;
        correction.sampleOffsetPixels = 1.0;
        correction.lineOffsetPixels = -2.0;
        correction.sampleLongitudePixelsPerDegree = 3.0;
        correction.sampleLatitudePixelsPerDegree = 4.0;
        correction.sampleHeightPixelsPerMeter = 5.0;
        correction.lineLongitudePixelsPerDegree = 6.0;
        correction.lineLatitudePixelsPerDegree = 7.0;
        correction.lineHeightPixelsPerMeter = 8.0;
        auto model = std::make_shared<const placamera::RpcModel>(
            placamera::RpcModel::createWithCorrection(placamera::CameraInstanceId("ground-rpc-instance"),
                                                      placamera::ImageId("ground-rpc"),
                                                      definition,
                                                      placamera::ImageSize{100, 80},
                                                      placamera::RpcCorrection::groundCoordinates(correction)));
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(model).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, cameras);
        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        const QJsonObject stored = project.value(QStringLiteral("camera_instances"))
                                       .toArray()
                                       .last()
                                       .toObject()
                                       .value(QStringLiteral("state"))
                                       .toObject()
                                       .value(QStringLiteral("image_correction"))
                                       .toObject();
        EXPECT_EQ(stored.value(QStringLiteral("model")), QStringLiteral("affine_ground_coordinates_v1"));
        EXPECT_DOUBLE_EQ(stored.value(QStringLiteral("line_height_px_per_m")).toDouble(), 8.0);

        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        const auto selected = restored.instances.forImage(placamera::ImageId("ground-rpc"));
        ASSERT_TRUE(selected.ok());
        const auto* rpc = dynamic_cast<const placamera::RpcModel*>(selected.value().get());
        ASSERT_NE(rpc, nullptr);
        EXPECT_EQ(rpc->correctionDomain(), placamera::RpcCorrectionDomain::GroundCoordinates);
        ASSERT_NE(rpc->groundCorrection(), nullptr);
        EXPECT_DOUBLE_EQ(rpc->groundCorrection()->sampleLatitudePixelsPerDegree, 4.0);
        EXPECT_THROW(static_cast<void>(rpc->imageCorrection()), placamera::CameraValidationError);
    }

    TEST(ProjectCameraStoreTest, InsertsAndUpdatesNativeLineScanCamera)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("new-line")));
        project.insert(QStringLiteral("images"), images);
        const auto current = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(current.ok()) << current.errors.join('\n').toStdString();
        const auto existing = current.instances.forImage(placamera::ImageId("line-image"));
        ASSERT_TRUE(existing.ok());
        const auto* original = dynamic_cast<const placamera::LineScanModel*>(existing.value().get());
        ASSERT_NE(original, nullptr);
        auto complete_optics = original->lineScanDefinition().optics();
        placamera::MetashapeCalibration complete_calibration;
        complete_calibration.f = 980.0;
        complete_calibration.cx = 512.25;
        complete_calibration.cy = 384.75;
        complete_calibration.b1 = 3.0;
        complete_calibration.b2 = -0.4;
        complete_calibration.k1 = 0.01;
        complete_calibration.k4 = -0.00003;
        complete_calibration.p1 = 0.0005;
        complete_calibration.p4 = -0.000005;
        complete_calibration.principalPointDecomposition =
            placamera::PrincipalPointDecomposition{400.0, 300.0, 112.25, 84.75};
        complete_optics.completeCalibration = complete_calibration;
        const auto definition =
            placamera::LineScanDefinition::create(placamera::CameraDefinitionId("new-line-definition"),
                                                  original->groundFrame(),
                                                  complete_optics,
                                                  original->lineScanDefinition().pixelConvention());
        auto samples = original->trajectory().samples();
        samples.front().constraints.positionFixed = false;
        samples.front().constraints.positionSigmaMeters = placamera::Vector3{{1.0, 2.0, 3.0}};
        const auto model = std::make_shared<const placamera::LineScanModel>(
            placamera::LineScanModel::create(placamera::CameraInstanceId("new-line-instance"),
                                             placamera::ImageId("new-line"),
                                             definition,
                                             original->imageSize(),
                                             placamera::LineScanTrajectory::create(std::move(samples)),
                                             original->lineTiming(),
                                             original->trajectoryBias(),
                                             placamera::TimeReference::create(placoordinate::TimeScale::Tdb, 0.25),
                                             placamera::LineScanTimeOffsetPrior{0.0, 0.02}));
        placamera::CameraInstanceSet inserts;
        ASSERT_TRUE(inserts.add(model).ok());

        const auto inserted = xjw::placamera_runtime::insertProjectCameras(&project, inserts);
        ASSERT_TRUE(inserted.ok()) << inserted.errors.join('\n').toStdString();
        EXPECT_EQ(inserted.insertedCount, 1);
        auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        auto selected = restored.instances.forImage(placamera::ImageId("new-line"));
        ASSERT_TRUE(selected.ok());
        const auto* line = dynamic_cast<const placamera::LineScanModel*>(selected.value().get());
        ASSERT_NE(line, nullptr);
        ASSERT_TRUE(line->captureTime());
        EXPECT_DOUBLE_EQ(line->captureTime()->seconds, 0.25);
        EXPECT_EQ(line->trajectory().samples().size(), original->trajectory().samples().size());
        EXPECT_FALSE(line->trajectory().samples().front().constraints.positionFixed);
        ASSERT_TRUE(line->trajectory().samples().front().constraints.positionSigmaMeters);
        EXPECT_EQ(*line->trajectory().samples().front().constraints.positionSigmaMeters,
                  (placamera::Vector3{1.0, 2.0, 3.0}));
        ASSERT_TRUE(line->timeOffsetPrior());
        EXPECT_DOUBLE_EQ(line->timeOffsetPrior()->sigmaSeconds, 0.02);
        ASSERT_TRUE(line->lineScanDefinition().optics().completeCalibration);
        EXPECT_EQ(*line->lineScanDefinition().optics().completeCalibration, complete_calibration);

        auto optics = line->lineScanDefinition().optics();
        optics.focalLengthMillimeters = 12.0;
        const auto refined_definition =
            placamera::LineScanDefinition::create(placamera::CameraDefinitionId("refined-line-definition"),
                                                  line->groundFrame(),
                                                  optics,
                                                  line->lineScanDefinition().pixelConvention());
        placamera::CameraInstanceSet updates;
        ASSERT_TRUE(updates
                        .add(std::make_shared<const placamera::LineScanModel>(
                            placamera::LineScanModel::create(line->instanceId(),
                                                             line->imageId(),
                                                             refined_definition,
                                                             line->imageSize(),
                                                             line->trajectory(),
                                                             line->lineTiming(),
                                                             line->trajectoryBias(),
                                                             line->captureTime(),
                                                             line->timeOffsetPrior())))
                        .ok());
        const auto written = xjw::placamera_runtime::writeProjectCameras(&project, updates);
        ASSERT_TRUE(written.ok()) << written.errors.join('\n').toStdString();
        restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        selected = restored.instances.forImage(placamera::ImageId("new-line"));
        ASSERT_TRUE(selected.ok());
        line = dynamic_cast<const placamera::LineScanModel*>(selected.value().get());
        ASSERT_NE(line, nullptr);
        EXPECT_DOUBLE_EQ(line->lineScanDefinition().optics().focalLengthMillimeters, 12.0);
        ASSERT_TRUE(line->lineScanDefinition().optics().completeCalibration);
        EXPECT_EQ(*line->lineScanDefinition().optics().completeCalibration, complete_calibration);
        ASSERT_TRUE(line->timeOffsetPrior());
        EXPECT_DOUBLE_EQ(line->timeOffsetPrior()->sigmaSeconds, 0.02);
    }

    TEST(ProjectCameraStoreTest, UpsertsNewAndExistingNativeCamerasAtomically)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("new-image")));
        project.insert(QStringLiteral("images"), images);
        const auto current = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(current.ok()) << current.errors.join('\n').toStdString();
        const auto existing = current.instances.forImage(placamera::ImageId("frame-image"));
        ASSERT_TRUE(existing.ok());
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(existing.value()).ok());
        ASSERT_TRUE(cameras.add(nativeFrame("new-image", "new-instance")).ok());
        const QMap<QString, QJsonObject> annotations{
            {QStringLiteral("new-image"),
             QJsonObject{{QStringLiteral("source"), QStringLiteral("initialization")},
                         {QStringLiteral("metadata"),
                          QJsonObject{{QStringLiteral("focal_source"), QStringLiteral("exif_35mm")}}}}}};

        const auto written = xjw::placamera_runtime::upsertProjectCameras(&project, cameras, annotations);

        ASSERT_TRUE(written.ok()) << written.errors.join('\n').toStdString();
        EXPECT_EQ(written.insertedCount, 1);
        EXPECT_EQ(written.updatedCount, 1);
        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        EXPECT_EQ(restored.instances.size(), 4U);
        const QJsonObject new_record = project.value(QStringLiteral("camera_instances")).toArray().last().toObject();
        EXPECT_EQ(new_record.value(QStringLiteral("state")).toObject().value(QStringLiteral("source")),
                  QStringLiteral("initialization"));
    }

    TEST(ProjectCameraStoreTest, ReplacesSelectedBindingsAndClearsUnregisteredImages)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("new-image")));
        project.insert(QStringLiteral("images"), images);
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeFrame("new-image", "new-instance")).ok());

        const auto replaced = xjw::placamera_runtime::replaceProjectCameras(
            &project, {placamera::ImageId("rpc-image"), placamera::ImageId("new-image")}, cameras);

        ASSERT_TRUE(replaced.ok()) << replaced.errors.join('\n').toStdString();
        EXPECT_EQ(replaced.clearedCount, 1);
        EXPECT_EQ(replaced.insertedCount, 1);
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        EXPECT_EQ(loaded.instances.size(), 3U);
        EXPECT_FALSE(loaded.instances.forImage(placamera::ImageId("rpc-image")).ok());
        EXPECT_TRUE(loaded.instances.forImage(placamera::ImageId("new-image")).ok());
        EXPECT_TRUE(loaded.instances.forImage(placamera::ImageId("frame-image")).ok());
    }

    TEST(ProjectCameraStoreTest, RejectsStaleReplacementWithoutClearingOtherImages)
    {
        QJsonObject project = validProject();
        const QJsonObject before = project;
        placamera::CameraInstanceSet cameras;
        ASSERT_TRUE(cameras.add(nativeFrame("frame-image", "different-instance")).ok());

        const auto replaced = xjw::placamera_runtime::replaceProjectCameras(
            &project, {placamera::ImageId("frame-image"), placamera::ImageId("rpc-image")}, cameras);

        EXPECT_FALSE(replaced.ok());
        EXPECT_EQ(replaced.clearedCount, 0);
        EXPECT_EQ(project, before);
    }

    TEST(ProjectCameraStoreTest, RejectsAmbiguousTargetImageIdentity)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("frame-image")));
        project.insert(QStringLiteral("images"), images);
        const QJsonObject before = project;

        const auto replaced =
            xjw::placamera_runtime::replaceProjectCameras(&project, {placamera::ImageId("frame-image")}, {});

        EXPECT_FALSE(replaced.ok());
        EXPECT_EQ(replaced.clearedCount, 0);
        EXPECT_EQ(project, before);
    }

    TEST(ProjectCameraStoreTest, LoadsFrameRpcAndLineScanDirectlyIntoPlaCameraSet)
    {
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(validProject());

        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        ASSERT_EQ(loaded.instances.size(), 3U);
        EXPECT_NE(dynamic_cast<const placamera::FramePinholeModel*>(
                      loaded.instances.forImage(placamera::ImageId("frame-image")).value().get()),
                  nullptr);
        EXPECT_NE(dynamic_cast<const placamera::RpcModel*>(
                      loaded.instances.forImage(placamera::ImageId("rpc-image")).value().get()),
                  nullptr);
        EXPECT_NE(dynamic_cast<const placamera::LineScanModel*>(
                      loaded.instances.forImage(placamera::ImageId("line-image")).value().get()),
                  nullptr);
    }

    TEST(ProjectCameraStoreTest, RejectsMissingCollectionsAndDuplicateImageIds)
    {
        QJsonObject missing = validProject();
        missing.remove(QStringLiteral("camera_definitions"));
        const auto missing_result = xjw::placamera_runtime::loadProjectCameras(missing);
        EXPECT_FALSE(missing_result.ok());
        EXPECT_TRUE(missing_result.instances.empty());

        QJsonObject duplicate = validProject();
        QJsonArray images = duplicate.value(QStringLiteral("images")).toArray();
        images.append(image(QStringLiteral("frame-image")));
        duplicate.insert(QStringLiteral("images"), images);
        const auto duplicate_result = xjw::placamera_runtime::loadProjectCameras(duplicate);
        EXPECT_FALSE(duplicate_result.ok());
        EXPECT_TRUE(duplicate_result.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsIncorrectProjectCameraSchemaVersions)
    {
        QJsonObject project = validProject();
        QJsonArray definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        QJsonObject line = definitions.at(2).toObject();
        line.insert(QStringLiteral("schema_version"), 1);
        definitions.replace(2, line);
        project.insert(QStringLiteral("camera_definitions"), definitions);
        const auto definition_result = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(definition_result.ok());
        EXPECT_TRUE(definition_result.instances.empty());

        project = validProject();
        QJsonArray instances = project.value(QStringLiteral("camera_instances")).toArray();
        QJsonObject frame = instances.at(0).toObject();
        frame.insert(QStringLiteral("schema_version"), 3);
        instances.replace(0, frame);
        project.insert(QStringLiteral("camera_instances"), instances);
        const auto instance_result = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(instance_result.ok());
        EXPECT_TRUE(instance_result.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsInvalidFrameConventionAndRpcSpec)
    {
        QJsonObject project = validProject();
        QJsonArray definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        QJsonObject frame = definitions.at(0).toObject();
        QJsonObject parameters = frame.value(QStringLiteral("parameters")).toObject();
        parameters.insert(QStringLiteral("pixel_convention"), QStringLiteral("centre"));
        frame.insert(QStringLiteral("parameters"), parameters);
        definitions.replace(0, frame);
        project.insert(QStringLiteral("camera_definitions"), definitions);
        auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());

        project = validProject();
        definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        QJsonObject rpc = definitions.at(1).toObject();
        parameters = rpc.value(QStringLiteral("parameters")).toObject();
        parameters.insert(QStringLiteral("rpc_spec"), QStringLiteral("RPC00A"));
        rpc.insert(QStringLiteral("parameters"), parameters);
        definitions.replace(1, rpc);
        project.insert(QStringLiteral("camera_definitions"), definitions);
        loaded = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsMismatchedFramePoseWithoutPublishingPartialSet)
    {
        QJsonObject project = validProject();
        QJsonArray instances = project.value(QStringLiteral("camera_instances")).toArray();
        QJsonObject frame = instances.at(0).toObject();
        QJsonObject pose = frame.value(QStringLiteral("pose")).toObject();
        pose.insert(QStringLiteral("frame"), QStringLiteral("other-world"));
        frame.insert(QStringLiteral("pose"), pose);
        instances.replace(0, frame);
        project.insert(QStringLiteral("camera_instances"), instances);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsInconsistentRpcCrsAndEllipsoidMetadata)
    {
        QJsonObject project = validProject();
        QJsonArray definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        QJsonObject rpc = definitions.at(1).toObject();
        QJsonObject parameters = rpc.value(QStringLiteral("parameters")).toObject();
        parameters.insert(QStringLiteral("ground_crs"), QStringLiteral("EPSG:4979"));
        parameters.insert(QStringLiteral("height_datum"), QStringLiteral("WGS84_ellipsoidal"));
        rpc.insert(QStringLiteral("parameters"), parameters);
        definitions.replace(1, rpc);
        project.insert(QStringLiteral("camera_definitions"), definitions);
        auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());

        project = validProject();
        definitions = project.value(QStringLiteral("camera_definitions")).toArray();
        rpc = definitions.at(1).toObject();
        parameters = rpc.value(QStringLiteral("parameters")).toObject();
        parameters.insert(QStringLiteral("semi_major_axis_m"), QStringLiteral("invalid"));
        rpc.insert(QStringLiteral("parameters"), parameters);
        definitions.replace(1, rpc);
        project.insert(QStringLiteral("camera_definitions"), definitions);
        loaded = xjw::placamera_runtime::loadProjectCameras(project);
        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsDuplicateIdentityWithoutPublishingPartialSet)
    {
        QJsonObject project = validProject();
        QJsonArray instances = project.value(QStringLiteral("camera_instances")).toArray();
        instances.append(frameInstance());
        project.insert(QStringLiteral("camera_instances"), instances);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);

        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraStoreTest, WritesUpdatedPlaCameraStateAtomically)
    {
        QJsonObject project = validProject();
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();

        const QJsonObject before = project;
        const auto write = xjw::placamera_runtime::writeProjectCameras(&project, loaded.instances);

        ASSERT_TRUE(write.ok()) << write.errors.join('\n').toStdString();
        EXPECT_EQ(write.updatedCount, 3);
        EXPECT_NE(project, before);
        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        EXPECT_EQ(restored.instances.size(), 3U);
    }

    TEST(ProjectCameraStoreTest, WritesRefinedFrameCalibrationAsNewDefinition)
    {
        QJsonObject project = validProject();
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto selected = loaded.instances.forImage(placamera::ImageId("frame-image"));
        ASSERT_TRUE(selected.ok());
        const auto* original = dynamic_cast<const placamera::FramePinholeModel*>(selected.value().get());
        ASSERT_NE(original, nullptr);

        auto intrinsics = original->pinholeDefinition().intrinsics();
        intrinsics.focalX = 150.0;
        intrinsics.focalY = 150.0;
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("refined-frame-definition"),
                                                      intrinsics,
                                                      original->pinholeDefinition().distortion(),
                                                      original->pinholeDefinition().pixelConvention(),
                                                      original->groundFrame(),
                                                      original->pinholeDefinition().depthAxisFlipped());
        const auto pose =
            placamera::Pose::create(original->groundFrame(), {5.0, 6.0, 7.0}, original->pose().cameraToWorldRotation);
        placamera::CameraInstanceSet updates;
        ASSERT_TRUE(updates
                        .add(std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                            original->instanceId(), original->imageId(), definition, original->imageSize(), pose)))
                        .ok());

        const auto written = xjw::placamera_runtime::writeProjectCameras(&project, updates);
        ASSERT_TRUE(written.ok()) << written.errors.join('\n').toStdString();
        EXPECT_EQ(written.updatedCount, 1);
        EXPECT_EQ(project.value(QStringLiteral("camera_definitions")).toArray().size(), 4);
        EXPECT_EQ(project.value(QStringLiteral("camera_instances"))
                      .toArray()
                      .at(0)
                      .toObject()
                      .value(QStringLiteral("definition_id"))
                      .toString(),
                  QStringLiteral("refined-frame-definition"));
        EXPECT_EQ(project.value(QStringLiteral("camera_definitions")).toArray().at(0).toObject(), frameDefinition());

        const auto restored = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(restored.ok()) << restored.errors.join('\n').toStdString();
        const auto updated = restored.instances.forImage(placamera::ImageId("frame-image"));
        ASSERT_TRUE(updated.ok());
        const auto* refined = dynamic_cast<const placamera::FramePinholeModel*>(updated.value().get());
        ASSERT_NE(refined, nullptr);
        EXPECT_DOUBLE_EQ(refined->pinholeDefinition().intrinsics().focalX, 150.0);
        EXPECT_DOUBLE_EQ(refined->pose().center[0], 5.0);
    }

    TEST(ProjectCameraStoreTest, RejectsChangedCalibrationUnderExistingDefinitionIdAtomically)
    {
        QJsonObject project = validProject();
        const QJsonObject before = project;
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto selected = loaded.instances.forImage(placamera::ImageId("frame-image"));
        ASSERT_TRUE(selected.ok());
        const auto* original = dynamic_cast<const placamera::FramePinholeModel*>(selected.value().get());
        ASSERT_NE(original, nullptr);

        auto intrinsics = original->pinholeDefinition().intrinsics();
        intrinsics.focalX = 150.0;
        const auto definition =
            placamera::FramePinholeDefinition::create(original->definitionId(),
                                                      intrinsics,
                                                      original->pinholeDefinition().distortion(),
                                                      original->pinholeDefinition().pixelConvention(),
                                                      original->groundFrame(),
                                                      original->pinholeDefinition().depthAxisFlipped());
        placamera::CameraInstanceSet updates;
        ASSERT_TRUE(
            updates
                .add(std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
                    original->instanceId(), original->imageId(), definition, original->imageSize(), original->pose())))
                .ok());

        const auto written = xjw::placamera_runtime::writeProjectCameras(&project, updates);
        EXPECT_FALSE(written.ok());
        EXPECT_EQ(project, before);
    }

    TEST(ProjectCameraStoreTest, RejectsImageSizeChangeOnWriteback)
    {
        QJsonObject project = validProject();
        const QJsonObject before = project;
        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto selected = loaded.instances.forImage(placamera::ImageId("frame-image"));
        ASSERT_TRUE(selected.ok());
        const auto* original = dynamic_cast<const placamera::FramePinholeModel*>(selected.value().get());
        ASSERT_NE(original, nullptr);

        placamera::CameraInstanceSet updates;
        ASSERT_TRUE(
            updates.add(std::make_shared<const placamera::FramePinholeModel>(original->withImageSize({101, 80}))).ok());
        const auto written = xjw::placamera_runtime::writeProjectCameras(&project, updates);
        EXPECT_FALSE(written.ok());
        EXPECT_EQ(project, before);
    }

    TEST(ProjectCameraStoreTest, RetainsLineScanBiasAndProjectMetadataAcrossWriteback)
    {
        QJsonObject project = validProject();
        QJsonArray records = project.value(QStringLiteral("camera_instances")).toArray();
        QJsonObject line = records.at(2).toObject();
        QJsonObject state = line.value(QStringLiteral("state")).toObject();
        state.insert(QStringLiteral("bias"),
                     QJsonObject{{QStringLiteral("translation_m"), QJsonArray{1.0, 2.0, 3.0}},
                                 {QStringLiteral("rotation_rad"), QJsonArray{0.1, 0.2, 0.3}},
                                 {QStringLiteral("time_offset_seconds"), 0.25}});
        state.insert(QStringLiteral("source_file"), QStringLiteral("images/line-image.isd"));
        state.insert(QStringLiteral("metadata"),
                     QJsonObject{{QStringLiteral("spacecraft"), QStringLiteral("orbiter")}});
        line.insert(QStringLiteral("state"), state);
        records.replace(2, line);
        project.insert(QStringLiteral("camera_instances"), records);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(loaded.ok()) << loaded.errors.join('\n').toStdString();
        const auto model = loaded.instances.forImage(placamera::ImageId("line-image"));
        ASSERT_TRUE(model.ok());
        const auto* line_model = dynamic_cast<const placamera::LineScanModel*>(model.value().get());
        ASSERT_NE(line_model, nullptr);
        EXPECT_DOUBLE_EQ(line_model->trajectoryBias().translationMeters[0], 1.0);
        EXPECT_DOUBLE_EQ(line_model->trajectoryBias().timeOffsetSeconds, 0.25);

        const auto written = xjw::placamera_runtime::writeProjectCameras(&project, loaded.instances);
        ASSERT_TRUE(written.ok()) << written.errors.join('\n').toStdString();
        const QJsonObject written_state = project.value(QStringLiteral("camera_instances"))
                                              .toArray()
                                              .at(2)
                                              .toObject()
                                              .value(QStringLiteral("state"))
                                              .toObject();
        EXPECT_EQ(written_state.value(QStringLiteral("source_file")), QStringLiteral("images/line-image.isd"));
        EXPECT_EQ(written_state.value(QStringLiteral("metadata")).toObject().value(QStringLiteral("spacecraft")),
                  QStringLiteral("orbiter"));
        EXPECT_EQ(written_state.value(QStringLiteral("bias"))
                      .toObject()
                      .value(QStringLiteral("translation_m"))
                      .toArray()
                      .at(0)
                      .toDouble(),
                  1.0);

        const auto reloaded = xjw::placamera_runtime::loadProjectCameras(project);
        ASSERT_TRUE(reloaded.ok()) << reloaded.errors.join('\n').toStdString();
        const auto restored = reloaded.instances.forImage(placamera::ImageId("line-image"));
        ASSERT_TRUE(restored.ok());
        const auto* restored_line = dynamic_cast<const placamera::LineScanModel*>(restored.value().get());
        ASSERT_NE(restored_line, nullptr);
        EXPECT_DOUBLE_EQ(restored_line->trajectoryBias().translationMeters[0], 1.0);
        EXPECT_DOUBLE_EQ(restored_line->trajectoryBias().timeOffsetSeconds, 0.25);
    }

    TEST(ProjectCameraStoreTest, RejectsLegacyEmbeddedCameraFields)
    {
        QJsonObject project = validProject();
        QJsonArray images = project.value(QStringLiteral("images")).toArray();
        QJsonObject first = images.at(0).toObject();
        first.insert(QStringLiteral("camera"), QJsonObject{});
        images.replace(0, first);
        project.insert(QStringLiteral("images"), images);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);

        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

    TEST(ProjectCameraStoreTest, RejectsUnknownLineScanTrajectoryRepresentation)
    {
        QJsonObject project = validProject();
        QJsonArray records = project.value(QStringLiteral("camera_instances")).toArray();
        QJsonObject line = records.at(2).toObject();
        QJsonObject state = line.value(QStringLiteral("state")).toObject();
        QJsonObject trajectory = state.value(QStringLiteral("trajectory")).toObject();
        trajectory.insert(QStringLiteral("representation"), QStringLiteral("unrecognized"));
        state.insert(QStringLiteral("trajectory"), trajectory);
        line.insert(QStringLiteral("state"), state);
        records.replace(2, line);
        project.insert(QStringLiteral("camera_instances"), records);

        const auto loaded = xjw::placamera_runtime::loadProjectCameras(project);

        EXPECT_FALSE(loaded.ok());
        EXPECT_TRUE(loaded.instances.empty());
    }

} // namespace
