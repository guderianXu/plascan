#include <placamera/frame_camera.h>
#include <placamera/linescan_camera.h>
#include <placamera/rpc_camera.h>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace
{

    using namespace placamera;

    FramePinholeModel makeFrameModel()
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        intrinsics.principalX = 50.0;
        intrinsics.principalY = 50.0;
        const auto definition = FramePinholeDefinition::create(
            CameraDefinitionId("frame-definition"), intrinsics, {}, PixelConvention::PixelCenter, FrameId("world"));
        return FramePinholeModel::create(
            CameraInstanceId("frame-instance"),
            ImageId("frame-image"),
            definition,
            ImageSize{100, 100},
            Pose::create(FrameId("world"), {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    }

    RpcModel makeRpcModel()
    {
        RpcParameters parameters;
        parameters.lineScale = 1.0;
        parameters.sampleScale = 1.0;
        parameters.latitudeScale = 1.0;
        parameters.longitudeScale = 1.0;
        parameters.heightScale = 1.0;
        parameters.lineNumerator[2] = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleNumerator[1] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        const auto definition =
            RpcDefinition::create(CameraDefinitionId("rpc-definition"), FrameId("ecef"), parameters);
        return RpcModel::create(
            CameraInstanceId("rpc-instance"), ImageId("rpc-image"), definition, ImageSize{100, 100});
    }

    LineScanModel makeLineScanModel()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 10.0;
        optics.samplePitchMillimeters = 0.01;
        const auto definition =
            LineScanDefinition::create(CameraDefinitionId("line-definition"), FrameId("world"), optics);
        const RotationMatrix identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        auto trajectory =
            LineScanTrajectory::create({{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, 0.0, 0.0}, identity},
                                        {TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity}});
        return LineScanModel::create(CameraInstanceId("line-instance"),
                                     ImageId("line-image"),
                                     definition,
                                     ImageSize{100, 11},
                                     std::move(trajectory),
                                     LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}});
    }

    TEST(OptimizationContractTest, AppliesFramePoseAndCalibrationUpdatesWithIdentityRules)
    {
        const FramePinholeModel model = makeFrameModel();
        const OptimizationLayout layout = model.optimizationLayout();
        EXPECT_TRUE(layout.isValid());
        EXPECT_EQ(layout.parameterCount(), 15U);
        EXPECT_TRUE(model.capabilities().contains(CapabilityKind::Optimization));

        std::vector<double> pose_delta(15, 0.0);
        pose_delta[3] = 1.0;
        auto pose_update =
            model.withOptimizationUpdate({CameraInstanceId("frame-pose-updated"), std::nullopt, pose_delta});
        ASSERT_TRUE(pose_update) << pose_update.message();
        const auto* pose_model = dynamic_cast<const FramePinholeModel*>(pose_update.value().get());
        ASSERT_NE(pose_model, nullptr);
        EXPECT_DOUBLE_EQ(pose_model->pose().center[0], 1.0);

        std::vector<double> calibration_delta(15, 0.0);
        calibration_delta[6] = 10.0;
        const auto missing_definition =
            model.withOptimizationUpdate({CameraInstanceId("frame-invalid"), std::nullopt, calibration_delta});
        EXPECT_FALSE(missing_definition);

        auto calibration_update = model.withOptimizationUpdate(
            {CameraInstanceId("frame-calibrated"), CameraDefinitionId("frame-definition-2"), calibration_delta});
        ASSERT_TRUE(calibration_update) << calibration_update.message();
        const auto* calibrated = dynamic_cast<const FramePinholeModel*>(calibration_update.value().get());
        ASSERT_NE(calibrated, nullptr);
        EXPECT_DOUBLE_EQ(calibrated->pinholeDefinition().intrinsics().focalX, 110.0);
        EXPECT_EQ(calibrated->definitionId(), CameraDefinitionId("frame-definition-2"));
    }

    TEST(OptimizationContractTest, AppliesRpcAndLineScanInstanceUpdates)
    {
        const RpcModel rpc = makeRpcModel();
        EXPECT_TRUE(rpc.optimizationLayout().isValid());
        EXPECT_EQ(rpc.optimizationLayout().parameterCount(), 6U);
        auto rpc_update =
            rpc.withOptimizationUpdate({CameraInstanceId("rpc-updated"), std::nullopt, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}});
        ASSERT_TRUE(rpc_update) << rpc_update.message();
        const auto* updated_rpc = dynamic_cast<const RpcModel*>(rpc_update.value().get());
        ASSERT_NE(updated_rpc, nullptr);
        EXPECT_DOUBLE_EQ(updated_rpc->imageCorrection().sampleOffsetPixels, 1.0);
        EXPECT_DOUBLE_EQ(updated_rpc->imageCorrection().lineLinePixels, 6.0);
        EXPECT_TRUE(updated_rpc->capabilities().contains(CapabilityKind::Optimization));

        const LineScanModel line_scan = makeLineScanModel();
        EXPECT_TRUE(line_scan.optimizationLayout().isValid());
        EXPECT_EQ(line_scan.optimizationLayout().parameterCount(), 7U);
        auto line_update = line_scan.withOptimizationUpdate(
            {CameraInstanceId("line-updated"), std::nullopt, {1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.5}});
        ASSERT_TRUE(line_update) << line_update.message();
        const auto* updated_line = dynamic_cast<const LineScanModel*>(line_update.value().get());
        ASSERT_NE(updated_line, nullptr);
        EXPECT_DOUBLE_EQ(updated_line->trajectoryBias().translationMeters[1], 2.0);
        EXPECT_DOUBLE_EQ(updated_line->trajectoryBias().timeOffsetSeconds, 0.5);
    }

    TEST(OptimizationContractTest, SelectsExtendedFrameCalibrationParameters)
    {
        const FramePinholeModel model = makeFrameModel();
        auto selection = FrameOptimizationSelection::none();
        selection.set(FrameOptimizationParameter::F)
            .set(FrameOptimizationParameter::B1)
            .set(FrameOptimizationParameter::B2)
            .set(FrameOptimizationParameter::K4)
            .set(FrameOptimizationParameter::P3)
            .set(FrameOptimizationParameter::P4);
        const OptimizationLayout layout = model.optimizationLayout(selection);
        ASSERT_TRUE(layout.isValid());
        ASSERT_EQ(layout.parameterCount(), 6U);
        EXPECT_EQ(layout.blocks[0].name, "calibration.f");
        EXPECT_EQ(layout.blocks[5].name, "calibration.p4");

        const std::vector<double> delta{10.0, 2.0, -0.5, 0.01, 0.02, -0.03};
        const auto updated = model.withOptimizationUpdate(
            {CameraInstanceId("extended-updated"), CameraDefinitionId("extended-definition"), delta}, selection);
        ASSERT_TRUE(updated) << updated.message();
        const auto* frame = dynamic_cast<const FramePinholeModel*>(updated.value().get());
        ASSERT_NE(frame, nullptr);
        const FrameCalibration calibration = frame->pinholeDefinition().calibration();
        EXPECT_DOUBLE_EQ(calibration.f, 110.0);
        EXPECT_DOUBLE_EQ(calibration.b1, 2.0);
        EXPECT_DOUBLE_EQ(calibration.b2, -0.5);
        EXPECT_DOUBLE_EQ(calibration.k4, 0.01);
        EXPECT_DOUBLE_EQ(calibration.p3, 0.02);
        EXPECT_DOUBLE_EQ(calibration.p4, -0.03);
        EXPECT_FALSE(model.withOptimizationUpdate(
            {CameraInstanceId("wrong-size"), CameraDefinitionId("wrong-definition"), {0.0}}, selection));
    }

    TEST(OptimizationContractTest, RejectsWrongSizedOrNonFiniteUpdates)
    {
        const FramePinholeModel model = makeFrameModel();
        EXPECT_FALSE(model.withOptimizationUpdate({CameraInstanceId("short"), std::nullopt, {0.0}}));
        std::vector<double> invalid(15, 0.0);
        invalid[0] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_FALSE(model.withOptimizationUpdate({CameraInstanceId("nan"), std::nullopt, invalid}));
    }

    TEST(OptimizationContractTest, SelectsRollingShutterParametersByModeAndInitialization)
    {
        const FramePinholeModel base = makeFrameModel();
        const auto definition = FramePinholeDefinition::create(CameraDefinitionId("rolling-definition"),
                                                               base.pinholeDefinition().intrinsics(),
                                                               base.pinholeDefinition().distortion(),
                                                               base.pinholeDefinition().pixelConvention(),
                                                               base.groundFrame());
        CameraAcquisitionState regularized;
        regularized.rollingShutterMode = RollingShutterMode::Regularized;
        const FramePinholeModel regularized_model = FramePinholeModel::create(CameraInstanceId("regularized"),
                                                                              base.imageId(),
                                                                              definition,
                                                                              base.imageSize(),
                                                                              base.pose(),
                                                                              base.captureTime(),
                                                                              regularized);
        const OptimizationLayout regularized_layout = regularized_model.optimizationLayout();
        EXPECT_EQ(regularized_layout.parameterCount(), 21U);
        EXPECT_EQ(regularized_layout.blocks.back().name, "rolling_shutter.translation");
        EXPECT_EQ(regularized_layout.blocks.back().size, 2U);

        CameraAcquisitionState full;
        full.rollingShutterMode = RollingShutterMode::Full;
        const FramePinholeModel full_model = FramePinholeModel::create(CameraInstanceId("full"),
                                                                       base.imageId(),
                                                                       definition,
                                                                       base.imageSize(),
                                                                       base.pose(),
                                                                       base.captureTime(),
                                                                       full);
        EXPECT_EQ(full_model.optimizationLayout().parameterCount(), 25U);

        full.rollingShutterInitialized = true;
        const FramePinholeModel initialized_model = FramePinholeModel::create(CameraInstanceId("initialized"),
                                                                              base.imageId(),
                                                                              definition,
                                                                              base.imageSize(),
                                                                              base.pose(),
                                                                              base.captureTime(),
                                                                              full);
        EXPECT_EQ(initialized_model.optimizationLayout().parameterCount(), 25U);

        FrameCalibration spherical_calibration;
        spherical_calibration.f = 100.0;
        spherical_calibration.cx = 50.0;
        spherical_calibration.cy = 50.0;
        const auto spherical_definition = FramePinholeDefinition::create(CameraDefinitionId("spherical-definition"),
                                                                         spherical_calibration,
                                                                         PixelConvention::PixelCenter,
                                                                         base.groundFrame(),
                                                                         false,
                                                                         1.0,
                                                                         1,
                                                                         1,
                                                                         FrameProjectionModel::Spherical);
        const auto spherical_model = FramePinholeModel::create(
            CameraInstanceId("spherical"), base.imageId(), spherical_definition, base.imageSize(), base.pose());
        EXPECT_EQ(spherical_model.optimizationLayout().parameterCount(), 6U);
    }

} // namespace
