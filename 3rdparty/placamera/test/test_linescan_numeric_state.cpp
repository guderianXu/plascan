#include <placamera/linescan_numeric_state.h>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

namespace
{
    using namespace placamera;

    LineScanModel makeSolverModel()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 10.0;
        optics.samplePitchMillimeters = 0.01;
        optics.principalSample = 5.0;
        const auto definition =
            LineScanDefinition::create(CameraDefinitionId("line-definition"), FrameId("body-fixed"), optics);

        const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        TrajectoryKnotConstraints fixed;
        TrajectoryKnotConstraints free_middle;
        free_middle.positionFixed = false;
        free_middle.rotationFixed = false;
        free_middle.positionSigmaMeters = Vector3{{2.0, 2.0, 2.0}};
        free_middle.rotationSigmaRadians = Vector3{{0.01, 0.01, 0.01}};
        TrajectoryKnotConstraints free_position;
        free_position.positionFixed = false;

        auto trajectory = LineScanTrajectory::create(
            {TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity, fixed},
             TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.5), {0.0, 0.0, 0.0}, identity, free_middle},
             TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity, free_position}});
        return LineScanModel::create(CameraInstanceId("line-instance"),
                                     ImageId("line-image"),
                                     definition,
                                     ImageSize{1000, 11},
                                     std::move(trajectory),
                                     LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}},
                                     {},
                                     std::nullopt,
                                     LineScanTimeOffsetPrior{0.1, 0.05});
    }

    LineScanModel makeCompleteCalibrationModel()
    {
        LineScanOptics optics;
        optics.focalLengthMillimeters = 35.0;
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
            LineScanDefinition::create(CameraDefinitionId("complete-line-definition"), FrameId("body-fixed"), optics);

        const RotationMatrix identity{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
        auto trajectory = LineScanTrajectory::create(
            {TrajectorySample{TimeReference::create(TimeScale::Tdb, 0.0), {0.0, -1.0, 0.0}, identity},
             TrajectorySample{TimeReference::create(TimeScale::Tdb, 1.0), {0.0, 1.0, 0.0}, identity}});
        return LineScanModel::create(CameraInstanceId("complete-line-instance"),
                                     ImageId("complete-line-image"),
                                     definition,
                                     ImageSize{1200, 11},
                                     std::move(trajectory),
                                     LineTiming{0.5, 0.0, 0.1, TimeScale::Tdb, {}});
    }

    TEST(LineScanNumericStateTest, BuildsExplicitFreeKnotAndDetectorLayout)
    {
        LineScanOptimizationSelection selection;
        selection.positionSecondDifferenceWeight = 4.0;
        selection.rotationSecondDifferenceWeight = 9.0;
        selection.detector.focalLength = true;
        selection.detector.principalSample = true;

        const auto result = LineScanNumericState::fromModel(makeSolverModel(), selection);
        ASSERT_TRUE(result) << result.message();
        const auto& layout = result.value().optimizationLayout();
        ASSERT_EQ(layout.blocks.size(), 6U);
        EXPECT_EQ(layout.blocks[0].name, "trajectory.knots.1.position");
        EXPECT_EQ(layout.blocks[1].name, "trajectory.knots.1.rotation");
        EXPECT_EQ(layout.blocks[2].name, "trajectory.knots.2.position");
        EXPECT_EQ(layout.blocks[3].name, "trajectory.time");
        EXPECT_EQ(layout.blocks[4].name, "detector.focal_length");
        EXPECT_EQ(layout.blocks[5].name, "detector.principal_sample");
        EXPECT_EQ(layout.parameterCount(), 12U);
        EXPECT_EQ(result.value().regularizationResidualCount(), 13U);
    }

    TEST(LineScanNumericStateTest, AppliesKnotPriorsSmoothnessTimeAndDetectorUpdates)
    {
        LineScanOptimizationSelection selection;
        selection.positionSecondDifferenceWeight = 4.0;
        selection.rotationSecondDifferenceWeight = 9.0;
        selection.detector.focalLength = true;
        selection.detector.principalSample = true;
        auto state_result = LineScanNumericState::fromModel(makeSolverModel(), selection);
        ASSERT_TRUE(state_result) << state_result.message();
        auto state = state_result.takeValue();

        const std::array<double, 12> delta{{1.0, 2.0, 3.0, 0.01, 0.02, 0.03, 2.0, 4.0, 6.0, 0.25, 1.0, 2.0}};
        const auto update = state.applyOptimizationDelta(delta);
        ASSERT_TRUE(update) << update.message();
        EXPECT_EQ(state.trajectorySamples()[0].center, (Vector3{0.0, -1.0, 0.0}));
        EXPECT_EQ(state.trajectorySamples()[1].center, (Vector3{1.0, 2.0, 3.0}));
        EXPECT_EQ(state.trajectorySamples()[2].center, (Vector3{2.0, 5.0, 6.0}));
        EXPECT_DOUBLE_EQ(state.trajectoryBias().timeOffsetSeconds, 0.25);
        EXPECT_DOUBLE_EQ(state.optics().focalLengthMillimeters, 11.0);
        EXPECT_DOUBLE_EQ(state.optics().principalSample, 7.0);
        EXPECT_TRUE(state.definitionDirty());

        std::array<double, 13> residuals{};
        const auto written = state.writeRegularizationResiduals(residuals);
        ASSERT_TRUE(written) << written.message();
        EXPECT_EQ(
            (std::array<double, 6>{residuals[0], residuals[1], residuals[2], residuals[3], residuals[4], residuals[5]}),
            (std::array<double, 6>{0.5, 1.0, 1.5, 1.0, 2.0, 3.0}));
        EXPECT_NEAR(residuals[6], 3.0, 1.0e-12);
        EXPECT_NEAR(residuals[7], 0.0, 1.0e-12);
        EXPECT_NEAR(residuals[8], 0.0, 1.0e-12);
        EXPECT_NEAR(residuals[9], 0.0, 1.0e-12);
        EXPECT_NEAR(residuals[10], -0.06, 1.0e-12);
        EXPECT_NEAR(residuals[11], -0.12, 1.0e-12);
        EXPECT_NEAR(residuals[12], -0.18, 1.0e-12);

        EXPECT_FALSE(state.toModel(CameraInstanceId("updated-without-definition")));
        const auto updated = state.toModel(CameraInstanceId("updated"), CameraDefinitionId("updated-definition"));
        ASSERT_TRUE(updated) << updated.message();
        EXPECT_EQ(updated.value()->trajectory().samples()[1].center, (Vector3{1.0, 2.0, 3.0}));
        ASSERT_TRUE(updated.value()->timeOffsetPrior().has_value());
        EXPECT_DOUBLE_EQ(updated.value()->timeOffsetPrior()->sigmaSeconds, 0.05);
    }

    TEST(LineScanNumericStateTest, ProjectsDirectlyFromSolverOwnedArrays)
    {
        const LineScanModel model = makeSolverModel();
        auto state_result = LineScanNumericState::fromModel(model);
        ASSERT_TRUE(state_result) << state_result.message();
        auto state = state_result.takeValue();
        const GroundCoordinate ground{FrameId("body-fixed"), {2.0, 0.0, 10.0}};

        const auto model_projection = model.projectAtLine(ground, 5.5);
        const auto numeric_projection = state.projectAtLine(ground, 5.5);
        ASSERT_TRUE(model_projection) << model_projection.message();
        ASSERT_TRUE(numeric_projection) << numeric_projection.message();
        EXPECT_NEAR(numeric_projection.value().projection.image.sample,
                    model_projection.value().projection.image.sample,
                    1.0e-12);
        EXPECT_NEAR(
            numeric_projection.value().lineResidualPixels, model_projection.value().lineResidualPixels, 1.0e-12);

        std::vector<double> delta(state.optimizationLayout().parameterCount(), 0.0);
        ASSERT_EQ(delta.size(), 10U);
        delta[0] = 1.0;
        ASSERT_TRUE(state.applyOptimizationDelta(delta));
        const auto moved = state.projectAtLine(ground, 5.5);
        ASSERT_TRUE(moved) << moved.message();
        EXPECT_NE(moved.value().projection.image.sample, numeric_projection.value().projection.image.sample);
    }

    TEST(LineScanNumericStateTest, RejectsDetectorMaskWithoutDetectorGeometry)
    {
        LineScanOptimizationSelection selection;
        selection.detector.detectorSampleOrigin = true;
        const auto state = LineScanNumericState::fromModel(makeSolverModel(), selection);
        EXPECT_FALSE(state);
        EXPECT_EQ(state.errorCode(), CameraErrorCode::InvalidArgument);
    }

    TEST(LineScanNumericStateTest, OptimizesEveryCompleteCalibrationParameterAndPreservesPrincipalDecomposition)
    {
        LineScanOptimizationSelection selection;
        selection.knotPositions = false;
        selection.knotRotations = false;
        selection.timeOffset = false;
        selection.calibration = {true, true, true, true, true, true, true, true, true, true, true, true, true};

        auto state_result = LineScanNumericState::fromModel(makeCompleteCalibrationModel(), selection);
        ASSERT_TRUE(state_result) << state_result.message();
        auto state = state_result.takeValue();
        const auto& layout = state.optimizationLayout();
        ASSERT_EQ(layout.blocks.size(), 13U);
        EXPECT_EQ(layout.parameterCount(), 13U);
        const std::array<const char*, 13> expected_names{{"calibration.f",
                                                          "calibration.cx",
                                                          "calibration.cy",
                                                          "calibration.b1",
                                                          "calibration.b2",
                                                          "calibration.k1",
                                                          "calibration.k2",
                                                          "calibration.k3",
                                                          "calibration.k4",
                                                          "calibration.p1",
                                                          "calibration.p2",
                                                          "calibration.p3",
                                                          "calibration.p4"}};
        for (std::size_t index = 0; index < expected_names.size(); ++index)
        {
            EXPECT_EQ(layout.blocks[index].name, expected_names[index]);
            EXPECT_EQ(layout.blocks[index].offset, index);
            EXPECT_EQ(layout.blocks[index].size, 1U);
            EXPECT_TRUE(layout.blocks[index].affectsDefinition);
        }

        const std::array<double, 13> delta{
            {2.0, 1.25, -0.75, 0.5, -0.25, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008}};
        const auto update = state.applyOptimizationDelta(delta);
        ASSERT_TRUE(update) << update.message();
        ASSERT_TRUE(state.optics().completeCalibration);
        const MetashapeCalibration& calibration = *state.optics().completeCalibration;
        EXPECT_DOUBLE_EQ(calibration.f, 982.0);
        EXPECT_DOUBLE_EQ(calibration.cx, 513.5);
        EXPECT_DOUBLE_EQ(calibration.cy, 384.0);
        EXPECT_DOUBLE_EQ(calibration.b1, 3.5);
        EXPECT_DOUBLE_EQ(calibration.b2, -0.65);
        EXPECT_DOUBLE_EQ(calibration.k1, 0.011);
        EXPECT_DOUBLE_EQ(calibration.k2, 0.001);
        EXPECT_DOUBLE_EQ(calibration.k3, 0.0032);
        EXPECT_DOUBLE_EQ(calibration.k4, 0.00397);
        EXPECT_DOUBLE_EQ(calibration.p1, 0.0055);
        EXPECT_DOUBLE_EQ(calibration.p2, 0.0056);
        EXPECT_DOUBLE_EQ(calibration.p3, 0.00707);
        EXPECT_DOUBLE_EQ(calibration.p4, 0.007995);
        ASSERT_TRUE(calibration.principalPointDecomposition);
        EXPECT_DOUBLE_EQ(calibration.principalPointDecomposition->imageCenterX, 400.0);
        EXPECT_DOUBLE_EQ(calibration.principalPointDecomposition->imageCenterY, 300.0);
        EXPECT_DOUBLE_EQ(calibration.principalPointDecomposition->cxOffset, 113.5);
        EXPECT_DOUBLE_EQ(calibration.principalPointDecomposition->cyOffset, 84.0);
        EXPECT_TRUE(state.definitionDirty());

        EXPECT_FALSE(state.toModel(CameraInstanceId("complete-updated-without-definition")));
        const auto committed =
            state.toModel(CameraInstanceId("complete-updated"), CameraDefinitionId("complete-updated-definition"));
        ASSERT_TRUE(committed) << committed.message();
        ASSERT_TRUE(committed.value()->lineScanDefinition().optics().completeCalibration);
        EXPECT_EQ(*committed.value()->lineScanDefinition().optics().completeCalibration, calibration);
    }

    TEST(LineScanNumericStateTest, RejectsCompleteCalibrationMaskWithoutCompleteCalibration)
    {
        LineScanOptimizationSelection selection;
        selection.calibration.k4 = true;
        const auto state = LineScanNumericState::fromModel(makeSolverModel(), selection);
        EXPECT_FALSE(state);
        EXPECT_EQ(state.errorCode(), CameraErrorCode::InvalidArgument);
    }

} // namespace
