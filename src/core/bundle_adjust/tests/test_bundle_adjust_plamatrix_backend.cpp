#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "BundleAdjustSolver.h"
#include "BundleAdjustPlaMatrixAssembly.h"
#include "BundleAdjustPlaMatrixProjection.h"
#include "FramePinholeCamera.h"

namespace
{

    xjw::FramePinholeCamera makeCamera(double center_x, double center_y)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(920.0, 910.0, 512.0, 384.0);
        camera.setDistortion(-0.015, 0.001, -0.0001, 0.0004, -0.0003);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{center_x, center_y, 0.0}});
        return camera;
    }

    std::array<double, 2> project(const xjw::FramePinholeCamera& camera, const std::array<double, 3>& point)
    {
        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        EXPECT_TRUE(camera.projectWorldPoint(world, pixel));
        return {{pixel[0], pixel[1]}};
    }

    double numericPointDerivative(const xjw::FramePinholeCamera& camera,
                                  std::array<double, 3> point,
                                  int parameter,
                                  int pixel_axis)
    {
        constexpr double epsilon = 1e-6;
        point[static_cast<std::size_t>(parameter)] += epsilon;
        const auto plus = project(camera, point);
        point[static_cast<std::size_t>(parameter)] -= 2.0 * epsilon;
        const auto minus = project(camera, point);
        return (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
               (2.0 * epsilon);
    }

    double numericCameraDerivative(const xjw::FramePinholeCamera& camera,
                                   const std::array<double, 3>& point,
                                   int parameter,
                                   int pixel_axis)
    {
        constexpr double epsilon = 1e-7;
        double delta[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        delta[parameter] = epsilon;
        xjw::FramePinholeCamera plus_camera = camera;
        plus_camera.applyDeltaPose(delta);
        delta[parameter] = -epsilon;
        xjw::FramePinholeCamera minus_camera = camera;
        minus_camera.applyDeltaPose(delta);
        const auto plus = project(plus_camera, point);
        const auto minus = project(minus_camera, point);
        return (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
               (2.0 * epsilon);
    }

    xjw::FramePinholeCamera applyReferencePoseDelta(xjw::FramePinholeCamera camera, const double delta[6])
    {
        auto rotation = camera.cameraToWorldRotation();
        for (int row = 0; row < 3; ++row)
        {
            rotation[static_cast<std::size_t>(row * 3 + 1)] *= -1.0;
            rotation[static_cast<std::size_t>(row * 3 + 2)] *= -1.0;
        }
        const double sx = std::sin(delta[0]);
        const double cx = std::cos(delta[0]);
        const double sy = std::sin(delta[1]);
        const double cy = std::cos(delta[1]);
        const double sz = std::sin(delta[2]);
        const double cz = std::cos(delta[2]);
        const std::array<double, 9> rx{{1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx}};
        const std::array<double, 9> ry{{cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy}};
        const std::array<double, 9> rz{{cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0}};
        const auto multiply = [](const std::array<double, 9>& left, const std::array<double, 9>& right)
        {
            std::array<double, 9> product{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        product[static_cast<std::size_t>(row * 3 + column)] +=
                            left[static_cast<std::size_t>(row * 3 + inner)] *
                            right[static_cast<std::size_t>(inner * 3 + column)];
                    }
                }
            }
            return product;
        };
        rotation = multiply(multiply(multiply(rotation, rx), ry), rz);
        for (int row = 0; row < 3; ++row)
        {
            rotation[static_cast<std::size_t>(row * 3 + 1)] *= -1.0;
            rotation[static_cast<std::size_t>(row * 3 + 2)] *= -1.0;
        }
        auto center = camera.cameraCenter();
        center[0] += delta[3];
        center[1] += delta[4];
        center[2] += delta[5];
        camera.setPose(rotation, center);
        return camera;
    }

    double numericReferenceCameraDerivative(const xjw::FramePinholeCamera& camera,
                                            const std::array<double, 3>& point,
                                            int parameter,
                                            int pixel_axis)
    {
        constexpr double epsilon = 1.0e-7;
        double delta[6]{};
        delta[parameter] = epsilon;
        const auto plus = project(applyReferencePoseDelta(camera, delta), point);
        delta[parameter] = -epsilon;
        const auto minus = project(applyReferencePoseDelta(camera, delta), point);
        return (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
               (2.0 * epsilon);
    }

    xjw::BATrack makeTrack(const std::vector<xjw::FramePinholeCamera>& truth_cameras,
                           const std::array<double, 3>& truth,
                           const std::array<double, 3>& initial)
    {
        xjw::BATrack track;
        track.initialPoint = initial;
        for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
        {
            const auto pixel = project(truth_cameras[camera_index], truth);
            track.observations.push_back(
                {static_cast<int>(camera_index), pixel[0], pixel[1], camera_index % 2 == 0 ? 1.0 : 0.8});
        }
        return track;
    }

    double pointDistance(const std::array<double, 3>& left, const std::array<double, 3>& right)
    {
        const double dx = left[0] - right[0];
        const double dy = left[1] - right[1];
        const double dz = left[2] - right[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

} // namespace

TEST(BundleAdjustPlaMatrixProjectionTest, AnalyticJacobiansMatchFiniteDifference)
{
    xjw::FramePinholeCamera camera = makeCamera(-1.5, 0.8);
    const double pose_delta[6] = {0.01, -0.02, 0.015, 0.1, -0.05, 0.02};
    camera.applyDeltaPose(pose_delta);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    const auto observed = project(camera, point);
    const xjw::BAObservation observation{0, observed[0] + 0.4, observed[1] - 0.2, 0.75};

    xjw::detail::plamatrix_ba::ObservationLinearization linearization;
    ASSERT_TRUE(xjw::detail::plamatrix_ba::linearizeObservation(camera, point, observation, 3.0, &linearization));

    for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
    {
        for (int parameter = 0; parameter < 3; ++parameter)
        {
            EXPECT_NEAR(linearization.pointJacobian[pixel_axis * 3 + parameter],
                        numericPointDerivative(camera, point, parameter, pixel_axis),
                        2e-4);
        }
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            EXPECT_NEAR(linearization.cameraJacobian[pixel_axis * 6 + parameter],
                        numericCameraDerivative(camera, point, parameter, pixel_axis),
                        2e-3);
        }
    }
}

TEST(BundleAdjustPlaMatrixProjectionTest, ReferenceBaWhitensResidualByFeatureScale)
{
    const xjw::FramePinholeCamera camera = makeCamera(-1.5, 0.8);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    const auto observed = project(camera, point);
    xjw::BAObservation fine{0, observed[0] + 0.4, observed[1] - 0.2, 0.75, 1.0};
    xjw::BAObservation coarse = fine;
    coarse.measurementScale = 2.0;

    xjw::detail::plamatrix_ba::ObservationLinearization fine_linearization;
    xjw::detail::plamatrix_ba::ObservationLinearization coarse_linearization;
    ASSERT_TRUE(
        xjw::detail::plamatrix_ba::linearizeObservation(camera, point, fine, 0.0, &fine_linearization, true, false));
    ASSERT_TRUE(xjw::detail::plamatrix_ba::linearizeObservation(
        camera, point, coarse, 0.0, &coarse_linearization, true, false));

    EXPECT_NEAR(coarse_linearization.normalWeight, fine_linearization.normalWeight / 4.0, 1.0e-14);
    EXPECT_NEAR(coarse_linearization.robustCost, fine_linearization.robustCost / 4.0, 1.0e-14);
}

TEST(BundleAdjustPlaMatrixProjectionTest, ReferenceRotationJacobianMatchesRightEulerFiniteDifference)
{
    xjw::FramePinholeCamera camera = makeCamera(-1.5, 0.8);
    const double initial_delta[6] = {0.01, -0.02, 0.015, 0.1, -0.05, 0.02};
    camera.applyDeltaPose(initial_delta);
    const std::array<double, 3> point{{0.7, -0.4, 12.0}};
    const auto observed = project(camera, point);
    const xjw::BAObservation observation{0, observed[0] + 0.4, observed[1] - 0.2, 1.0, 1.0};
    xjw::detail::plamatrix_ba::ObservationLinearization linearization;
    ASSERT_TRUE(
        xjw::detail::plamatrix_ba::linearizeObservation(camera, point, observation, 0.0, &linearization, true, true));

    for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
    {
        for (int parameter = 0; parameter < 6; ++parameter)
        {
            EXPECT_NEAR(linearization.cameraJacobian[pixel_axis * 6 + parameter],
                        numericReferenceCameraDerivative(camera, point, parameter, pixel_axis),
                        2.0e-3);
        }
    }
}

TEST(BundleAdjustPlaMatrixBackendTest, ReferenceGaugeUsesTwoDimensionalSphereTangent)
{
    xjw::BAOptions options;
    options.referenceGaugeAnchorCameraIndex = 0;
    options.referenceGaugeScaleCameraIndex = 1;
    const std::vector<xjw::FramePinholeCamera> cameras{
        makeCamera(-1.0, 0.25),
        makeCamera(2.0, 1.5),
    };
    options.referenceGaugeBaseline = pointDistance(cameras[0].cameraCenter(), cameras[1].cameraCenter());

    xjw::detail::plamatrix_ba::ActiveProblem active;
    active.cameraBlock = {-1, 0};
    xjw::detail::plamatrix_ba::OptimizationState without_radial;
    without_radial.cameras = cameras;
    xjw::detail::plamatrix_ba::OptimizationState with_radial = without_radial;
    std::vector<double> tangent_step(9, 0.0);
    tangent_step[3] = 0.2;
    tangent_step[4] = -0.1;
    std::vector<double> tangent_and_radial_step = tangent_step;
    tangent_and_radial_step[5] = 100.0;

    xjw::detail::plamatrix_ba::applyStep(active, tangent_step, {}, &without_radial, options, 1.0);
    xjw::detail::plamatrix_ba::applyStep(active, tangent_and_radial_step, {}, &with_radial, options, 1.0);

    EXPECT_NE(pointDistance(cameras[1].cameraCenter(), without_radial.cameras[1].cameraCenter()), 0.0);
    EXPECT_NEAR(pointDistance(without_radial.cameras[0].cameraCenter(), without_radial.cameras[1].cameraCenter()),
                options.referenceGaugeBaseline,
                1.0e-12);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        EXPECT_DOUBLE_EQ(with_radial.cameras[1].cameraCenter()[axis], without_radial.cameras[1].cameraCenter()[axis]);
    }
}

TEST(BundleAdjustPlaMatrixBackendTest, SolvesFixedIntrinsicsJointProblem)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCpu))
    {
        GTEST_SKIP() << "PlaMatrix backend is unavailable in this build";
    }
    ASSERT_TRUE(xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCpu));

    const std::vector<xjw::FramePinholeCamera> truth_cameras{
        makeCamera(-4.0, 0.0),
        makeCamera(4.0, 0.0),
        makeCamera(0.0, -3.5),
        makeCamera(0.0, 3.5),
    };
    std::vector<xjw::FramePinholeCamera> initial_cameras = truth_cameras;
    const double camera_delta_2[6] = {0.008, -0.012, 0.006, 0.22, -0.14, 0.08};
    const double camera_delta_3[6] = {-0.006, 0.009, -0.005, -0.18, 0.11, -0.06};
    initial_cameras[2].applyDeltaPose(camera_delta_2);
    initial_cameras[3].applyDeltaPose(camera_delta_3);

    std::vector<xjw::BATrack> tracks;
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 5; ++col)
        {
            const std::array<double, 3> truth{{-1.6 + 0.8 * col, -1.2 + 0.8 * row, 18.0 + 0.35 * ((row + col) % 3)}};
            const std::array<double, 3> initial{{truth[0] + 0.12 * ((col % 3) - 1),
                                                 truth[1] + 0.09 * ((row % 3) - 1),
                                                 truth[2] + 0.25 * (((row + col) % 3) - 1)}};
            tracks.push_back(makeTrack(truth_cameras, truth, initial));
        }
    }
    tracks.back().observations.back().u += 8.0;

    xjw::BAOptions base_options;
    base_options.refineCameraPose = true;
    base_options.fixedCameraIndices = {0, 1};
    base_options.gaugePolicy = xjw::BAGaugePolicy::RequireExplicitGauge;
    base_options.enablePointFilter = false;
    base_options.huberDelta = 3.0;
    base_options.maxIterations = 40;
    base_options.stepTolerance = 1e-9;
    base_options.maxDenseSchurCameras = 1;
    base_options.allowBackendFallback = false;
    xjw::BAOptions plamatrix_options = base_options;
    plamatrix_options.backend = xjw::BABackend::PlaMatrixCpu;
    plamatrix_options.plaMatrixPreconditionerClusterSize = 2;
    const xjw::BAResult plamatrix_result =
        xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, plamatrix_options);

    ASSERT_TRUE(plamatrix_result.solutionUsable) << plamatrix_result.backendMessage;
    EXPECT_EQ(plamatrix_result.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_GT(plamatrix_result.plaMatrixAcceptedSteps, 0);
    EXPECT_GT(plamatrix_result.plaMatrixLinearizations, 0);
    EXPECT_GT(plamatrix_result.plaMatrixAssemblySeconds, 0.0);
    EXPECT_GT(plamatrix_result.plaMatrixObjectiveSeconds, 0.0);
    EXPECT_GT(plamatrix_result.plaMatrixTrialStateSeconds, 0.0);
    EXPECT_EQ(plamatrix_result.plaMatrixLinearSolverName, "sparse_cholesky_native_cpu");
    EXPECT_DOUBLE_EQ(plamatrix_result.plaMatrixLinearToleranceMinimum, 1.0e-12);
    EXPECT_DOUBLE_EQ(plamatrix_result.plaMatrixLinearToleranceMaximum, 1.0e-12);
    EXPECT_GE(plamatrix_result.plaMatrixObjectiveEvaluations, plamatrix_result.plaMatrixLinearizations);
    EXPECT_GT(plamatrix_result.plaMatrixObjectiveEvaluations, plamatrix_result.plaMatrixLinearizations);
    EXPECT_LT(plamatrix_result.plaMatrixFinalCost, plamatrix_result.plaMatrixInitialCost);
    EXPECT_LT(plamatrix_result.meanRmsAfter, plamatrix_result.meanRmsBefore);

    xjw::BAOptions reference_options = plamatrix_options;
    reference_options.useReferenceOnlineSchur = true;
    const xjw::BAResult reference_result =
        xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, reference_options);
    ASSERT_TRUE(reference_result.solutionUsable) << reference_result.backendMessage;
    EXPECT_TRUE(reference_result.plaMatrixReferenceOnlineSchurUsed);
    EXPECT_GT(reference_result.plaMatrixAcceptedSteps, 0);
    EXPECT_LT(reference_result.plaMatrixFinalCost, reference_result.plaMatrixInitialCost);
    EXPECT_LT(reference_result.meanRmsAfter, reference_result.meanRmsBefore);
    EXPECT_NEAR(reference_result.meanRmsAfter, plamatrix_result.meanRmsAfter, 1.0e-5);
    EXPECT_NEAR(reference_result.plaMatrixFinalCost, plamatrix_result.plaMatrixFinalCost, 5.0e-4);
    for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
    {
        EXPECT_LT(pointDistance(reference_result.refinedCameras[camera_index].cameraCenter(),
                                plamatrix_result.refinedCameras[camera_index].cameraCenter()),
                  2.0e-4);
    }
    for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
    {
        EXPECT_LT(pointDistance(reference_result.points[track_index].point, plamatrix_result.points[track_index].point),
                  3.0e-4);
    }

    const std::array<xjw::BABackend, 2> accelerated_backends{{
        xjw::BABackend::PlaMatrixCuda,
        xjw::BABackend::PlaMatrixOpenCl,
    }};
    for (const xjw::BABackend backend : accelerated_backends)
    {
        if (!xjw::BundleAdjust::isBackendAvailable(backend))
        {
            continue;
        }
        xjw::BAOptions accelerated_options = base_options;
        accelerated_options.backend = backend;
        const xjw::BAResult accelerated_result =
            xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, accelerated_options);

        ASSERT_TRUE(accelerated_result.solutionUsable)
            << xjw::BundleAdjust::backendName(backend) << ": " << accelerated_result.backendMessage;
        EXPECT_EQ(accelerated_result.usedBackend, backend);
        EXPECT_TRUE(accelerated_result.usedGpu);
        EXPECT_FALSE(accelerated_result.backendFallback);
        EXPECT_FALSE(accelerated_result.plaMatrixDeviceName.empty());
        EXPECT_GT(accelerated_result.plaMatrixLinearIterations, 0);
        EXPECT_GE(accelerated_result.plaMatrixSchurPatternBuilds, 1);
        EXPECT_GT(accelerated_result.plaMatrixSchurPatternReuses, 0);
        EXPECT_TRUE(accelerated_result.plaMatrixSchurAssemblyOnDevice);
        EXPECT_GT(accelerated_result.plaMatrixLinearSolveSeconds, 0.0);
        EXPECT_DOUBLE_EQ(accelerated_result.plaMatrixLinearToleranceMinimum, 1.0e-12);
        EXPECT_DOUBLE_EQ(accelerated_result.plaMatrixLinearToleranceMaximum, 1.0e-12);
        EXPECT_NE(accelerated_result.backendMessage.find("Armijo"), std::string::npos);
        EXPECT_NEAR(accelerated_result.meanRmsAfter, plamatrix_result.meanRmsAfter, 1e-5);
        EXPECT_NEAR(accelerated_result.plaMatrixFinalCost, plamatrix_result.plaMatrixFinalCost, 5e-4);
        for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
        {
            EXPECT_LT(pointDistance(accelerated_result.refinedCameras[camera_index].cameraCenter(),
                                    plamatrix_result.refinedCameras[camera_index].cameraCenter()),
                      2e-4);
        }
        for (std::size_t track_index = 0; track_index < tracks.size(); ++track_index)
        {
            EXPECT_LT(
                pointDistance(accelerated_result.points[track_index].point, plamatrix_result.points[track_index].point),
                3e-4);
        }
    }
}

TEST(BundleAdjustPlaMatrixBackendTest, PointOnlyStationaryProblemIsUsable)
{
    const std::vector<xjw::FramePinholeCamera> cameras{makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};
    const xjw::BATrack track = makeTrack(cameras, truth, truth);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.allowBackendFallback = false;

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Success);
    EXPECT_NEAR(result.plaMatrixFinalCost, 0.0, 1e-14);
    ASSERT_EQ(result.points.size(), 1u);
    EXPECT_LT(pointDistance(result.points[0].point, truth), 1e-12);
}

TEST(BundleAdjustPlaMatrixBackendTest, CancellationDoesNotPublishIntermediateState)
{
    const std::vector<xjw::FramePinholeCamera> cameras{makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};
    const std::array<double, 3> initial{{0.8, -0.5, 13.5}};
    const xjw::BATrack track = makeTrack(cameras, truth, initial);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.enablePointFilter = false;
    options.allowBackendFallback = false;
    options.progressCallback = [](int, int, double, int) { return false; };

    const xjw::BAResult result = xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::Cancelled);
    EXPECT_FALSE(result.solutionUsable);
    ASSERT_EQ(result.points.size(), 1u);
    EXPECT_EQ(result.points[0].point, initial);
    EXPECT_EQ(result.refinedCameras[0].cameraCenter(), cameras[0].cameraCenter());
    EXPECT_EQ(result.refinedCameras[1].cameraCenter(), cameras[1].cameraCenter());
}

TEST(BundleAdjustPlaMatrixBackendTest, SharedIntrinsicsAreAcceptedWithoutFallback)
{
    const std::vector<xjw::FramePinholeCamera> cameras{makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};

    xjw::BAOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.enablePointFilter = false;
    options.allowBackendFallback = false;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {makeTrack(cameras, truth, truth)}, options);

    EXPECT_EQ(result.requestedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_FALSE(result.backendFallback);
    EXPECT_EQ(result.refinedCalibrationGroupCount, 1);
}

TEST(BundleAdjustPlaMatrixBackendTest, SharedIntrinsicsRemainOnPlaMatrixWhenFallbackIsAllowed)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCpu))
    {
        GTEST_SKIP() << "PlaMatrix backend is unavailable in this build";
    }
    const std::vector<xjw::FramePinholeCamera> cameras{makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};

    xjw::BAOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.refineCameraPose = false;
    options.refineSharedFocalLength = true;
    options.enablePointFilter = false;
    options.allowBackendFallback = true;

    const xjw::BAResult result =
        xjw::BundleAdjust::optimizePoints(cameras, {makeTrack(cameras, truth, truth)}, options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.requestedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(result.usedBackend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_FALSE(result.backendFallback);
    EXPECT_EQ(result.refinedCalibrationGroupCount, 1);
}

TEST(BundleAdjustPlaMatrixBackendTest, AcceleratedBackendsRejectInvalidDeviceWithoutCpuFallback)
{
    const std::vector<xjw::FramePinholeCamera> cameras{makeCamera(-2.0, 0.0), makeCamera(2.0, 0.0)};
    const std::array<double, 3> truth{{0.2, -0.1, 12.0}};
    for (const xjw::BABackend backend : {xjw::BABackend::PlaMatrixCuda, xjw::BABackend::PlaMatrixOpenCl})
    {
        xjw::BAOptions options;
        options.backend = backend;
        options.plaMatrixDevice = 9999;
        options.refineCameraPose = false;
        options.enablePointFilter = false;
        options.allowBackendFallback = false;

        const xjw::BAResult result =
            xjw::BundleAdjust::optimizePoints(cameras, {makeTrack(cameras, truth, truth)}, options);

        EXPECT_EQ(result.requestedBackend, backend);
        EXPECT_EQ(result.usedBackend, backend);
        EXPECT_EQ(result.solveStatus, xjw::BASolveStatus::BackendUnavailable);
        EXPECT_FALSE(result.solutionUsable);
        EXPECT_FALSE(result.usedGpu);
        EXPECT_FALSE(result.backendFallback);
    }
}
