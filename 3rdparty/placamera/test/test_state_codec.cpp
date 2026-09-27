#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/rpc_camera.h>
#include <placamera/state_codec.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace
{

    using namespace placamera;

    FramePinholeModel makeFrameModel()
    {
        FrameCalibration calibration;
        calibration.f = 810.0;
        calibration.cx = 500.0;
        calibration.cy = 400.0;
        calibration.b1 = -10.0;
        calibration.b2 = 0.25;
        calibration.k1 = 0.01;
        calibration.k4 = -0.00003;
        calibration.p2 = -0.002;
        calibration.p3 = 0.00007;
        calibration.p4 = -0.000006;
        calibration.principalPointDecomposition = PrincipalPointDecomposition{400.0, 300.0, 100.0, 100.0};
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("frame-definition"),
                                                               calibration,
                                                               PixelConvention::PixelCenter,
                                                               FrameId("world"),
                                                               false,
                                                               0.01);
        return FramePinholeModel::create(
            CameraInstanceId("frame-instance"),
            ImageId("frame-image"),
            definition,
            ImageSize{1000, 800},
            Pose::create(FrameId("world"), {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
            TimeReference::create(TimeScale::Utc, 42.0));
    }

    RpcModel makeRpcModel()
    {
        RpcParameters parameters;
        parameters.lineOffset = 500.0;
        parameters.sampleOffset = 500.0;
        parameters.latitudeOffset = 20.0;
        parameters.longitudeOffset = 110.0;
        parameters.heightOffset = 1000.0;
        parameters.lineScale = 1000.0;
        parameters.sampleScale = 1000.0;
        parameters.latitudeScale = 0.1;
        parameters.longitudeScale = 0.1;
        parameters.heightScale = 1000.0;
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleNumerator[3] = 0.25;
        parameters.sampleDenominator[0] = 1.0;
        parameters.errorBiasMeters = 3.0;
        const auto definition =
            RpcDefinition::create(CameraDefinitionId("rpc-definition"), FrameId("wgs84-ecef"), parameters);
        RpcImageCorrection correction;
        correction.sampleOffsetPixels = 2.0;
        correction.lineOffsetPixels = -1.0;
        return RpcModel::create(
            CameraInstanceId("rpc-instance"), ImageId("rpc-image"), definition, ImageSize{1000, 1000}, correction);
    }

    LineScanModel makeLineScanModel()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 10.0;
        optics.samplePitchMillimeters = 0.01;
        optics.principalSample = 5.0;
        optics.distortionK1 = 0.001;
        MetashapeCalibration calibration;
        calibration.f = 980.0;
        calibration.cx = 512.25;
        calibration.cy = 384.75;
        calibration.b1 = 3.0;
        calibration.b2 = -0.4;
        calibration.k1 = 0.01;
        calibration.k2 = -0.001;
        calibration.k3 = 0.0002;
        calibration.k4 = -0.00003;
        calibration.p1 = 0.0005;
        calibration.p2 = -0.0004;
        calibration.p3 = 0.00007;
        calibration.p4 = -0.000005;
        calibration.principalPointDecomposition = PrincipalPointDecomposition{400.0, 300.0, 112.25, 84.75};
        optics.completeCalibration = calibration;
        const auto definition =
            LineScanDefinition::create(CameraDefinitionId("line-definition"), FrameId("moon-fixed"), optics);
        const RotationMatrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        TrajectoryKnotConstraints first_constraints;
        first_constraints.positionFixed = false;
        first_constraints.positionSigmaMeters = Vector3{{1.0, 2.0, 3.0}};
        auto trajectory = LineScanTrajectory::create(
            {TrajectorySample{
                 TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity, first_constraints},
             TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity}});
        LineTiming timing;
        timing.timeScale = TimeScale::Tdb;
        timing.segments = {{0.5, 0.0, 0.1}, {5.5, 0.5, 0.2}};
        LineScanTrajectoryBias bias;
        bias.translationMeters = {1.0, 2.0, 3.0};
        bias.timeOffsetSeconds = 0.25;
        return LineScanModel::create(CameraInstanceId("line-instance"),
                                     ImageId("line-image"),
                                     definition,
                                     ImageSize{1000, 11},
                                     std::move(trajectory),
                                     std::move(timing),
                                     bias,
                                     std::nullopt,
                                     LineScanTimeOffsetPrior{0.2, 0.05});
    }

    RasterModelPtr roundTrip(const RasterModel& source)
    {
        const auto encoded_definition = encodeCameraDefinitionJson(source.definition());
        const auto encoded_instance = encodeCameraInstanceJson(source);
        EXPECT_TRUE(encoded_definition) << encoded_definition.message();
        EXPECT_TRUE(encoded_instance) << encoded_instance.message();
        if (!encoded_definition || !encoded_instance)
        {
            return nullptr;
        }

        const auto definition_state = decodeCameraDefinitionJson(encoded_definition.value());
        const auto instance_state = decodeCameraInstanceJson(encoded_instance.value());
        EXPECT_TRUE(definition_state) << definition_state.message();
        EXPECT_TRUE(instance_state) << instance_state.message();
        if (!definition_state || !instance_state)
        {
            return nullptr;
        }

        ModelRegistry registry;
        EXPECT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition = registry.createDefinition(definition_state.value());
        EXPECT_TRUE(definition) << definition.message();
        if (!definition)
        {
            return nullptr;
        }
        auto instance = registry.createInstance(instance_state.value(), definition.value());
        EXPECT_TRUE(instance) << instance.message();
        if (!instance)
        {
            return nullptr;
        }
        return instance.value();
    }

    TEST(CameraStateCodecTest, RoundTripsFrameDefinitionAndInstanceDeterministically)
    {
        const FramePinholeModel source = makeFrameModel();
        const auto first = encodeCameraDefinitionJson(source.definition());
        const auto second = encodeCameraDefinitionJson(source.definition());
        ASSERT_TRUE(first);
        ASSERT_TRUE(second);
        EXPECT_EQ(first.value(), second.value());

        const auto restored_base = roundTrip(source);
        ASSERT_NE(restored_base, nullptr);
        const auto* restored = dynamic_cast<const FramePinholeModel*>(restored_base.get());
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->instanceId(), source.instanceId());
        EXPECT_EQ(restored->pose().center, source.pose().center);
        EXPECT_EQ(restored->pinholeDefinition().intrinsics().focalX, source.pinholeDefinition().intrinsics().focalX);
        EXPECT_EQ(restored->pinholeDefinition().intrinsics().skew, source.pinholeDefinition().intrinsics().skew);
        EXPECT_EQ(restored->pinholeDefinition().distortion().radialK4,
                  source.pinholeDefinition().distortion().radialK4);
        EXPECT_EQ(restored->pinholeDefinition().distortion().tangentialP4,
                  source.pinholeDefinition().distortion().tangentialP4);
        EXPECT_EQ(restored->pinholeDefinition().distortion().tangentialConvention,
                  BrownTangentialConvention::Metashape);
        EXPECT_EQ(restored->pinholeDefinition().principalPointDecomposition(),
                  source.pinholeDefinition().principalPointDecomposition());
        EXPECT_EQ(restored->captureTime(), source.captureTime());
    }

    TEST(CameraStateCodecTest, MigratesLegacyFiveCoefficientFrameDefinition)
    {
        const std::string schema_one =
            R"json({"kind":"placamera.definition","envelope_schema":1,"model_type":"frame_pinhole","definition_id":"legacy-frame","ground_frame":"world","parameter_schema":1,"parameters":{"intrinsics":{"focal_x":800.0,"focal_y":810.0,"principal_x":500.0,"principal_y":400.0,"pixel_pitch":0.01,"u_axis_sign":1,"v_axis_sign":1},"distortion":{"radial_k1":0.01,"radial_k2":-0.002,"radial_k3":0.0001,"tangential_p1":0.003,"tangential_p2":-0.004},"pixel_convention":"pixel_center","depth_axis_flipped":false}})json";
        const auto state = decodeCameraDefinitionJson(schema_one);
        ASSERT_TRUE(state) << state.message();
        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition = registry.createDefinition(state.value());
        ASSERT_TRUE(definition) << definition.message();
        const auto* frame = dynamic_cast<const FramePinholeDefinition*>(definition.value().get());
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->parameterSchemaVersion(), FramePinholeDefinition::ParameterSchemaVersion);
        EXPECT_EQ(frame->distortion().tangentialConvention, BrownTangentialConvention::OpenCv);
        EXPECT_DOUBLE_EQ(frame->distortion().radialK4, 0.0);
        EXPECT_DOUBLE_EQ(frame->intrinsics().skew, 0.0);

        const auto migrated = encodeCameraDefinitionJson(*definition.value());
        ASSERT_TRUE(migrated) << migrated.message();
        EXPECT_NE(migrated.value().find("\"parameter_schema\":4"), std::string::npos);
        EXPECT_NE(migrated.value().find("\"tangential_convention\":\"opencv\""), std::string::npos);
        EXPECT_NE(migrated.value().find("\"projection_model\":\"frame_pinhole\""), std::string::npos);
    }

    TEST(CameraStateCodecTest, MigratesSchemaTwoFrameDefinitionWithExplicitDefaults)
    {
        const std::string schema_two =
            R"json({"kind":"placamera.definition","envelope_schema":1,"model_type":"frame_pinhole","definition_id":"schema-two-frame","ground_frame":"world","parameter_schema":2,"parameters":{"intrinsics":{"focal_x":802.0,"focal_y":800.0,"principal_x":500.0,"principal_y":400.0,"pixel_pitch":0.01,"u_axis_sign":1,"v_axis_sign":1,"skew":0.25},"distortion":{"radial_k1":0.01,"radial_k2":-0.002,"radial_k3":0.0001,"radial_k4":-0.00001,"tangential_p1":0.003,"tangential_p2":-0.004,"tangential_p3":0.0002,"tangential_p4":-0.00003,"tangential_convention":"metashape"},"pixel_convention":"pixel_center","depth_axis_flipped":false}})json";
        const auto state = decodeCameraDefinitionJson(schema_two);
        ASSERT_TRUE(state) << state.message();
        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition = registry.createDefinition(state.value());
        ASSERT_TRUE(definition) << definition.message();
        const auto* frame = dynamic_cast<const FramePinholeDefinition*>(definition.value().get());
        ASSERT_NE(frame, nullptr);
        EXPECT_EQ(frame->projectionModel(), FrameProjectionModel::Perspective);
        EXPECT_FALSE(frame->sensorMount().masterSensorId);
        EXPECT_DOUBLE_EQ(frame->intrinsics().skew, 0.25);
        EXPECT_DOUBLE_EQ(frame->distortion().tangentialP4, -0.00003);
    }

    TEST(CameraStateCodecTest, RoundTripsProjectionTopologyAndRollingShutter)
    {
        FrameCalibration calibration;
        calibration.f = 500.0;
        calibration.cx = 320.0;
        calibration.cy = 240.0;
        SensorMountState mount;
        mount.masterSensorId = CameraDefinitionId("master-sensor");
        mount.translation = {0.1, -0.2, 0.3};
        mount.fixedTranslation = false;
        mount.fixedRotation = true;
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("equisolid-definition"),
                                                               calibration,
                                                               PixelConvention::PixelCenter,
                                                               FrameId("world"),
                                                               false,
                                                               1.0,
                                                               1,
                                                               1,
                                                               FrameProjectionModel::EquisolidFisheye,
                                                               mount);
        CameraAcquisitionState acquisition;
        acquisition.role = CameraRole::Keyframe;
        acquisition.captureGroupId = CaptureGroupId("capture-group");
        acquisition.masterCameraId = CameraInstanceId("master-camera");
        acquisition.layerIndex = 3;
        acquisition.rollingShutterMode = RollingShutterMode::Full;
        acquisition.rollingShutter.translation = {0.04, -0.02, 0.01};
        acquisition.rollingShutter.rotationVector = {0.001, -0.002, 0.003};
        acquisition.rollingShutterInitialized = true;
        const auto source = FramePinholeModel::create(
            CameraInstanceId("equisolid-instance"),
            ImageId("equisolid-image"),
            definition,
            {640, 480},
            Pose::create(FrameId("world"), {1.0, 2.0, 3.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}),
            TimeReference::create(TimeScale::Utc, 12.0),
            acquisition);

        const auto restored_base = roundTrip(source);
        ASSERT_NE(restored_base, nullptr);
        const auto* restored = dynamic_cast<const FramePinholeModel*>(restored_base.get());
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->modelType(), "frame_equisolid_fisheye");
        EXPECT_EQ(restored->pinholeDefinition().projectionModel(), FrameProjectionModel::EquisolidFisheye);
        ASSERT_TRUE(restored->pinholeDefinition().sensorMount().masterSensorId);
        EXPECT_EQ(*restored->pinholeDefinition().sensorMount().masterSensorId, CameraDefinitionId("master-sensor"));
        EXPECT_EQ(restored->pinholeDefinition().sensorMount().translation, mount.translation);
        EXPECT_EQ(restored->acquisition().role, CameraRole::Keyframe);
        ASSERT_TRUE(restored->acquisition().captureGroupId);
        EXPECT_EQ(*restored->acquisition().captureGroupId, CaptureGroupId("capture-group"));
        ASSERT_TRUE(restored->acquisition().masterCameraId);
        EXPECT_EQ(*restored->acquisition().masterCameraId, CameraInstanceId("master-camera"));
        EXPECT_EQ(restored->acquisition().layerIndex, 3U);
        EXPECT_EQ(restored->acquisition().rollingShutterMode, RollingShutterMode::Full);
        EXPECT_EQ(restored->acquisition().rollingShutter.translation, acquisition.rollingShutter.translation);
        EXPECT_EQ(restored->acquisition().rollingShutter.rotationVector, acquisition.rollingShutter.rotationVector);
        EXPECT_TRUE(restored->acquisition().rollingShutterInitialized);
    }

    TEST(CameraStateCodecTest, MigratesPoseOnlyFrameInstanceToDefaultAcquisition)
    {
        const FramePinholeModel source = makeFrameModel();
        const auto encoded_definition = encodeCameraDefinitionJson(source.definition());
        ASSERT_TRUE(encoded_definition);
        const auto definition_state = decodeCameraDefinitionJson(encoded_definition.value());
        ASSERT_TRUE(definition_state);
        const std::string legacy_instance =
            R"json({"kind":"placamera.instance","envelope_schema":1,"model_type":"frame_pinhole","instance_id":"legacy-instance","image_id":"legacy-image","definition_id":"frame-definition","instance_schema":1,"image_size":{"samples":1000,"lines":800},"capture_time":null,"state":{"pose":{"center":[1.0,2.0,3.0],"camera_to_world_rotation":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]}}})json";
        const auto instance_state = decodeCameraInstanceJson(legacy_instance);
        ASSERT_TRUE(instance_state) << instance_state.message();

        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition = registry.createDefinition(definition_state.value());
        ASSERT_TRUE(definition) << definition.message();
        const auto restored_base = registry.createInstance(instance_state.value(), definition.value());
        ASSERT_TRUE(restored_base) << restored_base.message();
        const auto* restored = dynamic_cast<const FramePinholeModel*>(restored_base.value().get());
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->acquisition().role, CameraRole::Regular);
        EXPECT_EQ(restored->acquisition().rollingShutterMode, RollingShutterMode::Disabled);
        EXPECT_FALSE(restored->acquisition().captureGroupId);
        EXPECT_FALSE(restored->acquisition().masterCameraId);
    }

    TEST(CameraStateCodecTest, RoundTripsRpcAndLineScanModels)
    {
        const RpcModel rpc = makeRpcModel();
        const auto restored_rpc_base = roundTrip(rpc);
        ASSERT_NE(restored_rpc_base, nullptr);
        const auto* restored_rpc = dynamic_cast<const RpcModel*>(restored_rpc_base.get());
        ASSERT_NE(restored_rpc, nullptr);
        EXPECT_DOUBLE_EQ(restored_rpc->imageCorrection().sampleOffsetPixels, 2.0);
        EXPECT_EQ(restored_rpc->rpcDefinition().parameters().lineNumerator,
                  rpc.rpcDefinition().parameters().lineNumerator);

        const LineScanModel line_scan = makeLineScanModel();
        const auto restored_line_base = roundTrip(line_scan);
        ASSERT_NE(restored_line_base, nullptr);
        const auto* restored_line = dynamic_cast<const LineScanModel*>(restored_line_base.get());
        ASSERT_NE(restored_line, nullptr);
        EXPECT_EQ(restored_line->trajectory().samples().size(), 2U);
        EXPECT_EQ(restored_line->lineTiming().segments.size(), 2U);
        EXPECT_DOUBLE_EQ(restored_line->trajectoryBias().timeOffsetSeconds, 0.25);
        EXPECT_FALSE(restored_line->trajectory().samples()[0].constraints.positionFixed);
        ASSERT_TRUE(restored_line->trajectory().samples()[0].constraints.positionSigmaMeters.has_value());
        EXPECT_EQ(*restored_line->trajectory().samples()[0].constraints.positionSigmaMeters, (Vector3{1.0, 2.0, 3.0}));
        ASSERT_TRUE(restored_line->timeOffsetPrior().has_value());
        EXPECT_DOUBLE_EQ(restored_line->timeOffsetPrior()->meanSeconds, 0.2);
        ASSERT_TRUE(restored_line->lineScanDefinition().optics().completeCalibration);
        EXPECT_EQ(restored_line->lineScanDefinition().optics().completeCalibration,
                  line_scan.lineScanDefinition().optics().completeCalibration);
    }

    TEST(CameraStateCodecTest, RoundTripsGroundRpcCorrectionWithoutChangingDomains)
    {
        const RpcModel base = makeRpcModel();
        RpcGroundCorrection correction;
        correction.sampleOffsetPixels = 1.0;
        correction.lineOffsetPixels = -2.0;
        correction.sampleLongitudePixelsPerDegree = 3.0;
        correction.sampleLatitudePixelsPerDegree = 4.0;
        correction.sampleHeightPixelsPerMeter = 5.0;
        correction.lineLongitudePixelsPerDegree = 6.0;
        correction.lineLatitudePixelsPerDegree = 7.0;
        correction.lineHeightPixelsPerMeter = 8.0;
        const RpcModel model =
            RpcModel::createWithCorrection(CameraInstanceId("ground-rpc"),
                                           base.imageId(),
                                           std::make_shared<const RpcDefinition>(base.rpcDefinition()),
                                           base.imageSize(),
                                           RpcCorrection::groundCoordinates(correction));
        const auto encoded = encodeCameraInstanceJson(model);
        ASSERT_TRUE(encoded) << encoded.message();
        EXPECT_NE(encoded.value().find("\"instance_schema\":2"), std::string::npos);
        EXPECT_NE(encoded.value().find("\"correction_domain\":\"ground_coordinates\""), std::string::npos);

        const auto restored_base = roundTrip(model);
        const auto* restored = dynamic_cast<const RpcModel*>(restored_base.get());
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->correctionDomain(), RpcCorrectionDomain::GroundCoordinates);
        ASSERT_NE(restored->groundCorrection(), nullptr);
        EXPECT_DOUBLE_EQ(restored->groundCorrection()->lineHeightPixelsPerMeter, 8.0);
    }

    TEST(CameraStateCodecTest, MigratesRpcSchemaOneAndRejectsDomainPayloadMismatch)
    {
        const RpcModel model = makeRpcModel();
        const auto definition_json = encodeCameraDefinitionJson(model.definition());
        ASSERT_TRUE(definition_json);
        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition_state = decodeCameraDefinitionJson(definition_json.value());
        ASSERT_TRUE(definition_state);
        const auto definition = registry.createDefinition(definition_state.value());
        ASSERT_TRUE(definition);

        const std::string schema_one =
            R"json({"kind":"placamera.instance","envelope_schema":1,"model_type":"rpc00b","instance_id":"legacy-rpc","image_id":"rpc-image","definition_id":"rpc-definition","instance_schema":1,"image_size":{"samples":1000,"lines":1000},"capture_time":null,"state":{"image_correction":{"sample_offset":1.0,"sample_sample":2.0,"sample_line":3.0,"line_offset":4.0,"line_sample":5.0,"line_line":6.0}}})json";
        const auto legacy_state = decodeCameraInstanceJson(schema_one);
        ASSERT_TRUE(legacy_state) << legacy_state.message();
        const auto legacy = registry.createInstance(legacy_state.value(), definition.value());
        ASSERT_TRUE(legacy) << legacy.message();
        const auto* rpc = dynamic_cast<const RpcModel*>(legacy.value().get());
        ASSERT_NE(rpc, nullptr);
        EXPECT_EQ(rpc->correctionDomain(), RpcCorrectionDomain::NormalizedImage);
        const auto migrated = encodeCameraInstanceJson(*rpc);
        ASSERT_TRUE(migrated);
        EXPECT_NE(migrated.value().find("\"instance_schema\":2"), std::string::npos);

        const std::string mismatched =
            R"json({"kind":"placamera.instance","envelope_schema":1,"model_type":"rpc00b","instance_id":"bad-rpc","image_id":"rpc-image","definition_id":"rpc-definition","instance_schema":2,"image_size":{"samples":1000,"lines":1000},"capture_time":null,"state":{"correction_domain":"ground_coordinates","normalized_image_correction":{"sample_offset":0.0,"sample_sample":0.0,"sample_line":0.0,"line_offset":0.0,"line_sample":0.0,"line_line":0.0}}})json";
        const auto mismatched_state = decodeCameraInstanceJson(mismatched);
        ASSERT_TRUE(mismatched_state);
        EXPECT_FALSE(registry.createInstance(mismatched_state.value(), definition.value()));
    }

    TEST(CameraStateCodecTest, MigratesLineScanSchemaOneToCurrentDefinition)
    {
        const std::string schema_one =
            R"json({"kind":"placamera.definition","envelope_schema":1,"model_type":"planetary_linescan","definition_id":"legacy-line","ground_frame":"moon-fixed","parameter_schema":1,"parameters":{"optics":{"focal_length_mm":10.0,"sample_pitch_mm":0.01,"principal_sample":5.0,"distortion_k1":0.0},"pixel_convention":"pixel_center"}})json";
        const auto state = decodeCameraDefinitionJson(schema_one);
        ASSERT_TRUE(state) << state.message();
        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition = registry.createDefinition(state.value());
        ASSERT_TRUE(definition) << definition.message();
        EXPECT_EQ(definition.value()->parameterSchemaVersion(), LineScanDefinition::ParameterSchemaVersion);

        const auto migrated = encodeCameraDefinitionJson(*definition.value());
        ASSERT_TRUE(migrated) << migrated.message();
        EXPECT_NE(migrated.value().find("\"parameter_schema\":3"), std::string::npos);
        EXPECT_NE(migrated.value().find("\"distortion_model\":\"radial_normalized\""), std::string::npos);
        const auto* line = dynamic_cast<const LineScanDefinition*>(definition.value().get());
        ASSERT_NE(line, nullptr);
        EXPECT_FALSE(line->optics().completeCalibration);
    }

    TEST(CameraStateCodecTest, MigratesLineScanInstanceSchemaOneWithFixedKnots)
    {
        const LineScanModel current = makeLineScanModel();
        const auto definition_json = encodeCameraDefinitionJson(current.definition());
        ASSERT_TRUE(definition_json);
        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto definition_state = decodeCameraDefinitionJson(definition_json.value());
        ASSERT_TRUE(definition_state);
        const auto definition = registry.createDefinition(definition_state.value());
        ASSERT_TRUE(definition);

        const std::string schema_one =
            R"json({"kind":"placamera.instance","envelope_schema":1,"model_type":"planetary_linescan","instance_id":"legacy-line","image_id":"line-image","definition_id":"line-definition","instance_schema":1,"image_size":{"samples":1000,"lines":11},"capture_time":null,"state":{"trajectory":{"kind":"samples","samples":[{"time":{"scale":"tdb","seconds":0.0},"center":[0.0,-1.0,0.0],"camera_to_world_rotation":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]},{"time":{"scale":"tdb","seconds":1.0},"center":[0.0,1.0,0.0],"camera_to_world_rotation":[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]}]},"timing":{"line_zero":0.5,"start_time_seconds":0.0,"seconds_per_line":0.1,"time_scale":"tdb","segments":[]},"bias":{"translation_m":[0.0,0.0,0.0],"rotation_rad":[0.0,0.0,0.0],"time_offset_seconds":0.0}}})json";
        const auto state = decodeCameraInstanceJson(schema_one);
        ASSERT_TRUE(state) << state.message();
        const auto restored = registry.createInstance(state.value(), definition.value());
        ASSERT_TRUE(restored) << restored.message();
        const auto* line = dynamic_cast<const LineScanModel*>(restored.value().get());
        ASSERT_NE(line, nullptr);
        EXPECT_TRUE(line->trajectory().samples()[0].constraints.positionFixed);
        EXPECT_TRUE(line->trajectory().samples()[0].constraints.rotationFixed);
        EXPECT_FALSE(line->timeOffsetPrior());

        const auto migrated = encodeCameraInstanceJson(*line);
        ASSERT_TRUE(migrated);
        EXPECT_NE(migrated.value().find("\"instance_schema\":2"), std::string::npos);
        EXPECT_NE(migrated.value().find("\"position_fixed\":true"), std::string::npos);
        EXPECT_NE(migrated.value().find("\"time_offset_prior\":null"), std::string::npos);
    }

    TEST(CameraStateCodecTest, RejectsUnknownEnvelopeFieldsAndDefinitionMismatch)
    {
        const std::string unknown =
            R"json({"kind":"placamera.definition","envelope_schema":1,"model_type":"frame_pinhole","definition_id":"d","ground_frame":"world","parameter_schema":1,"parameters":{},"unknown":true})json";
        EXPECT_FALSE(decodeCameraDefinitionJson(unknown));

        const FramePinholeModel frame = makeFrameModel();
        const RpcModel rpc = makeRpcModel();
        const auto frame_instance_json = encodeCameraInstanceJson(frame);
        const auto rpc_definition_json = encodeCameraDefinitionJson(rpc.definition());
        ASSERT_TRUE(frame_instance_json);
        ASSERT_TRUE(rpc_definition_json);
        const auto frame_state = decodeCameraInstanceJson(frame_instance_json.value());
        const auto rpc_state = decodeCameraDefinitionJson(rpc_definition_json.value());
        ASSERT_TRUE(frame_state);
        ASSERT_TRUE(rpc_state);

        ModelRegistry registry;
        ASSERT_TRUE(registerBuiltinJsonModelFactories(registry));
        const auto rpc_definition = registry.createDefinition(rpc_state.value());
        ASSERT_TRUE(rpc_definition);
        EXPECT_FALSE(registry.createInstance(frame_state.value(), rpc_definition.value()));
    }

} // namespace
