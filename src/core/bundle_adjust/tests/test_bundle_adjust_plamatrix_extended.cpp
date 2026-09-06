#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "BundleAdjustSolver.h"
#include "BundleAdjustPlaMatrixProblem.h"
#include "BundleAdjustPlaMatrixProjection.h"
#include "FramePinholeCamera.h"

namespace
{

    xjw::FramePinholeCamera makeExtendedCamera()
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(930.0, 905.0, 510.0, 386.0);
        camera.setDistortion(-0.02, 0.0015, -0.0002, 0.0005, -0.0004);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{-1.2, 0.7, 0.1}});
        return camera;
    }

    xjw::FramePinholeCamera cameraWithSharedIntrinsics(const xjw::FramePinholeCamera& pose_camera,
                                                       const xjw::FramePinholeCamera& reference_camera,
                                                       const std::array<double, 9>& parameters)
    {
        xjw::FramePinholeCamera camera = pose_camera;
        const double focal_x = parameters[0];
        const double focal_y = focal_x * std::exp(parameters[1]);
        camera.setIntrinsics(focal_x,
                             focal_y,
                             reference_camera.principalX() + parameters[2],
                             reference_camera.principalY() + parameters[3]);
        xjw::FramePinholeCamera::Distortion distortion;
        distortion.radialK1 = parameters[4];
        distortion.radialK2 = parameters[5];
        distortion.radialK3 = parameters[6];
        distortion.tangentialP1 = parameters[7];
        distortion.tangentialP2 = parameters[8];
        camera.setDistortion(distortion);
        return camera;
    }

    std::array<double, 2> projectShared(const xjw::FramePinholeCamera& pose_camera,
                                        const xjw::FramePinholeCamera& reference_camera,
                                        const std::array<double, 9>& parameters,
                                        const std::array<double, 3>& point)
    {
        const auto camera = cameraWithSharedIntrinsics(pose_camera, reference_camera, parameters);
        const double world[3] = {point[0], point[1], point[2]};
        double pixel[2] = {0.0, 0.0};
        EXPECT_TRUE(camera.projectWorldPoint(world, pixel));
        return {{pixel[0], pixel[1]}};
    }

    xjw::FramePinholeCamera makeCalibrationCamera(double center_x, double center_y, bool truth)
    {
        xjw::FramePinholeCamera camera;
        camera.setIntrinsics(
            truth ? 1040.0 : 900.0, truth ? 998.4 : 900.0, truth ? 516.0 : 512.0, truth ? 381.5 : 384.0);
        camera.setDistortion(truth ? -0.035 : 0.0,
                             truth ? 0.004 : 0.0,
                             truth ? -0.0004 : 0.0,
                             truth ? 0.0008 : 0.0,
                             truth ? -0.0006 : 0.0);
        camera.setPose({{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}}, {{center_x, center_y, 0.0}});
        return camera;
    }

    std::vector<xjw::BATrack> makeCalibrationTracks(const std::vector<xjw::FramePinholeCamera>& truth_cameras)
    {
        std::vector<xjw::BATrack> tracks;
        for (int row = -4; row <= 4; ++row)
        {
            for (int column = -5; column <= 5; ++column)
            {
                xjw::BATrack track;
                track.initialPoint = {{0.45 * column, 0.38 * row, 9.0 + 0.35 * ((row + column + 20) % 4)}};
                track.controlPointConstraints.push_back(
                    {track.initialPoint, 0.01, 1.0, static_cast<int>(tracks.size())});
                for (std::size_t camera_index = 0; camera_index < truth_cameras.size(); ++camera_index)
                {
                    const double world[3] = {track.initialPoint[0], track.initialPoint[1], track.initialPoint[2]};
                    double pixel[2] = {0.0, 0.0};
                    EXPECT_TRUE(truth_cameras[camera_index].projectWorldPoint(world, pixel));
                    track.observations.push_back({static_cast<int>(camera_index), pixel[0], pixel[1], 1.0});
                }
                tracks.push_back(std::move(track));
            }
        }
        return tracks;
    }

    xjw::BAOptions makeFullBrownOptions()
    {
        xjw::BAOptions options;
        options.refineCameraPose = false;
        options.refineSharedFocalLength = true;
        options.refineSharedFocalAspectRatio = true;
        options.refineSharedPrincipalPoint = true;
        options.refineSharedRadialDistortion = true;
        options.refineSharedHighOrderDistortion = true;
        options.enableControlPointConstraints = true;
        options.controlPointWeight = 10000.0;
        options.controlPointHuberDeltaMeters = 0.0;
        options.sharedFocalPriorSigma = 5.0;
        options.sharedFocalAspectPriorSigma = 5.0;
        options.sharedPrincipalPointPriorSigmaFraction = 1.0;
        options.sharedRadialK1PriorSigma = 5.0;
        options.sharedRadialK2PriorSigma = 5.0;
        options.sharedRadialK3PriorSigma = 5.0;
        options.sharedTangentialP1PriorSigma = 5.0;
        options.sharedTangentialP2PriorSigma = 5.0;
        options.huberDelta = 0.0;
        options.enablePointFilter = false;
        options.maxIterations = 60;
        options.damping = 1e-4;
        options.stepTolerance = 1e-10;
        options.allowBackendFallback = false;
        return options;
    }

} // namespace

TEST(BundleAdjustPlaMatrixExtendedProjectionTest, SharedBrownIntrinsicJacobiansMatchFiniteDifference)
{
    const xjw::FramePinholeCamera camera = makeExtendedCamera();
    xjw::FramePinholeCamera reference = camera;
    reference.setIntrinsics(950.0, 920.0, 512.0, 384.0);
    const std::array<double, 9> parameters{
        {940.0, std::log(0.985), 1.5, -2.0, -0.018, 0.0012, -0.00015, 0.00045, -0.00035}};
    const std::array<double, 3> point{{0.8, -0.6, 11.0}};
    const auto pixel = projectShared(camera, reference, parameters, point);
    const xjw::BAObservation observation{0, pixel[0] + 0.3, pixel[1] - 0.2, 1.0};
    xjw::BAIntrinsicParameterMask active{};
    active.fill(true);

    xjw::detail::plamatrix_ba::ObservationLinearization linearization;
    ASSERT_TRUE(xjw::detail::plamatrix_ba::linearizeObservationWithSharedIntrinsics(
        camera, reference, parameters, active, point, observation, 3.0, &linearization));

    for (int parameter = 0; parameter < 9; ++parameter)
    {
        constexpr double epsilon = 1e-7;
        auto plus_parameters = parameters;
        auto minus_parameters = parameters;
        plus_parameters[static_cast<std::size_t>(parameter)] += epsilon;
        minus_parameters[static_cast<std::size_t>(parameter)] -= epsilon;
        const auto plus = projectShared(camera, reference, plus_parameters, point);
        const auto minus = projectShared(camera, reference, minus_parameters, point);
        for (int pixel_axis = 0; pixel_axis < 2; ++pixel_axis)
        {
            const double numeric =
                (plus[static_cast<std::size_t>(pixel_axis)] - minus[static_cast<std::size_t>(pixel_axis)]) /
                (2.0 * epsilon);
            EXPECT_NEAR(
                linearization.intrinsicJacobian[static_cast<std::size_t>(pixel_axis * 9 + parameter)], numeric, 3e-4);
        }
    }
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, FullSharedBrownModelMatchesAcrossCpuCudaAndOpenCl)
{
    if (!xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCpu))
    {
        GTEST_SKIP() << "PlaMatrix backend is unavailable";
    }
    const std::vector<xjw::FramePinholeCamera> truth_cameras{makeCalibrationCamera(-3.0, 0.0, true),
                                                             makeCalibrationCamera(3.0, 0.0, true),
                                                             makeCalibrationCamera(0.0, -2.5, true),
                                                             makeCalibrationCamera(0.0, 2.5, true)};
    const std::vector<xjw::FramePinholeCamera> initial_cameras{makeCalibrationCamera(-3.0, 0.0, false),
                                                               makeCalibrationCamera(3.0, 0.0, false),
                                                               makeCalibrationCamera(0.0, -2.5, false),
                                                               makeCalibrationCamera(0.0, 2.5, false)};
    const auto tracks = makeCalibrationTracks(truth_cameras);

    auto options = makeFullBrownOptions();
    auto plamatrix_options = options;
    plamatrix_options.backend = xjw::BABackend::PlaMatrixCpu;
    const auto plamatrix = xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, plamatrix_options);
    ASSERT_TRUE(plamatrix.solutionUsable) << plamatrix.backendMessage;

    EXPECT_EQ(plamatrix.refinedCalibrationGroupCount, 1);

    std::vector<xjw::BABackend> backends;
    for (const auto backend : {xjw::BABackend::PlaMatrixCuda, xjw::BABackend::PlaMatrixOpenCl})
    {
        if (xjw::BundleAdjust::isBackendAvailable(backend))
        {
            backends.push_back(backend);
        }
    }
    for (const auto backend : backends)
    {
        auto backend_options = options;
        backend_options.backend = backend;
        const auto result = xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, backend_options);
        ASSERT_TRUE(result.solutionUsable) << xjw::BundleAdjust::backendName(backend) << ": " << result.backendMessage;
        EXPECT_EQ(result.usedBackend, backend);
        EXPECT_FALSE(result.backendFallback);
        EXPECT_EQ(result.refinedCalibrationGroupCount, 1);
        const auto& actual = result.refinedCameras.front();
        const auto& expected = plamatrix.refinedCameras.front();
        EXPECT_NEAR(actual.focalX(), expected.focalX(), 2.0);
        EXPECT_NEAR(actual.focalY(), expected.focalY(), 2.0);
        EXPECT_NEAR(actual.principalX(), expected.principalX(), 0.5);
        EXPECT_NEAR(actual.principalY(), expected.principalY(), 0.5);
        EXPECT_NEAR(actual.distortion().radialK1, expected.distortion().radialK1, 2e-3);
        EXPECT_NEAR(actual.distortion().radialK2, expected.distortion().radialK2, 3e-3);
        EXPECT_NEAR(actual.distortion().radialK3, expected.distortion().radialK3, 3e-3);
        EXPECT_NEAR(actual.distortion().tangentialP1, expected.distortion().tangentialP1, 5e-4);
        EXPECT_NEAR(actual.distortion().tangentialP2, expected.distortion().tangentialP2, 5e-4);
        EXPECT_NEAR(result.meanRmsAfter, plamatrix.meanRmsAfter, 2e-3);
    }
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, FullBrownRefinementUsesSingleReferenceStage)
{
    const std::vector<xjw::FramePinholeCamera> truth_cameras{makeCalibrationCamera(-3.0, 0.0, true),
                                                             makeCalibrationCamera(3.0, 0.0, true),
                                                             makeCalibrationCamera(0.0, -2.5, true),
                                                             makeCalibrationCamera(0.0, 2.5, true)};
    const std::vector<xjw::FramePinholeCamera> initial_cameras{makeCalibrationCamera(-3.0, 0.0, false),
                                                               makeCalibrationCamera(3.0, 0.0, false),
                                                               makeCalibrationCamera(0.0, -2.5, false),
                                                               makeCalibrationCamera(0.0, 2.5, false)};
    auto options = makeFullBrownOptions();
    options.backend = xjw::BABackend::PlaMatrixCpu;
    const auto result =
        xjw::BundleAdjust::optimizePoints(initial_cameras, makeCalibrationTracks(truth_cameras), options);

    ASSERT_TRUE(result.solutionUsable) << result.backendMessage;
    EXPECT_EQ(result.selfCalibrationStagesRun, 1);
    EXPECT_LT(result.meanRmsAfter, 2e-3);
    EXPECT_NEAR(result.refinedCameras.front().focalX(), 1040.0, 2.0);
    EXPECT_NEAR(result.refinedCameras.front().distortion().radialK3, -0.0004, 3e-3);
    EXPECT_NEAR(result.refinedCameras.front().distortion().tangentialP1, 0.0008, 5e-4);
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, ReferenceOnlineSchurMatchesFullBrownLegacyPath)
{
    const std::vector<xjw::FramePinholeCamera> truth_cameras{makeCalibrationCamera(-3.0, 0.0, true),
                                                             makeCalibrationCamera(3.0, 0.0, true),
                                                             makeCalibrationCamera(0.0, -2.5, true),
                                                             makeCalibrationCamera(0.0, 2.5, true)};
    const std::vector<xjw::FramePinholeCamera> initial_cameras{makeCalibrationCamera(-3.0, 0.0, false),
                                                               makeCalibrationCamera(3.0, 0.0, false),
                                                               makeCalibrationCamera(0.0, -2.5, false),
                                                               makeCalibrationCamera(0.0, 2.5, false)};
    auto tracks = makeCalibrationTracks(truth_cameras);
    for (auto& track : tracks)
    {
        track.controlPointConstraints.clear();
    }

    auto legacy_options = makeFullBrownOptions();
    legacy_options.backend = xjw::BABackend::PlaMatrixCpu;
    legacy_options.enableControlPointConstraints = false;
    legacy_options.maxIterations = 20;
    const auto legacy = xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, legacy_options);

    auto reference_options = legacy_options;
    reference_options.useReferenceOnlineSchur = true;
    const auto reference = xjw::BundleAdjust::optimizePoints(initial_cameras, tracks, reference_options);

    ASSERT_TRUE(legacy.solutionUsable) << legacy.backendMessage;
    ASSERT_TRUE(reference.solutionUsable) << reference.backendMessage;
    EXPECT_TRUE(reference.plaMatrixReferenceOnlineSchurUsed);
    EXPECT_NEAR(reference.plaMatrixFinalCost, legacy.plaMatrixFinalCost, 1.0e-4);
    EXPECT_NEAR(reference.meanRmsAfter, legacy.meanRmsAfter, 1.0e-5);
    EXPECT_NEAR(reference.refinedCameras.front().focalX(), legacy.refinedCameras.front().focalX(), 1.0e-3);
    EXPECT_NEAR(reference.refinedCameras.front().distortion().radialK1,
                legacy.refinedCameras.front().distortion().radialK1,
                1.0e-5);
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, ReferenceDriverReleasesRequestedIntrinsicsTogether)
{
    xjw::BAIntrinsicParameterMask enabled{};
    enabled.fill(true);
    xjw::BAOptions options;
    for (const int iteration : {0, 1, 5, 9})
    {
        EXPECT_EQ(xjw::detail::plamatrix_ba::activeIntrinsicParameters(options, enabled, iteration), enabled);
    }
    EXPECT_EQ(xjw::detail::plamatrix_ba::intrinsicStageCount(options, enabled), 1);
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, ReferenceCalibrationTransitionUsesRecoveredSensorWeights)
{
    xjw::FramePinholeCamera camera = makeExtendedCamera();
    camera.setImageSize(xjw::CameraImageSize{1020, 772});
    xjw::BAOptions options;
    options.backend = xjw::BABackend::PlaMatrixCpu;
    options.useReferenceCalibrationTransitionPrior = true;
    options.refineSharedFocalLength = true;
    options.refineSharedRadialDistortion = true;
    options.refineSharedHighOrderDistortion = true;
    options.referencePreviousIntrinsicParameterMask.fill(false);
    options.referencePreviousIntrinsicParameterMask[static_cast<std::size_t>(xjw::BAIntrinsicParameter::RadialK1)] =
        true;

    xjw::detail::plamatrix_ba::ActiveProblem active;
    active.intrinsicBlockCount = 1;
    active.calibrationGroupByCamera = {0};
    const auto groups = xjw::detail::plamatrix_ba::initializeIntrinsicGroups({camera}, options, active);

    ASSERT_EQ(groups.size(), 1U);
    const auto& group = groups.front();
    ASSERT_TRUE(group.usesReferenceTransitionPrior);
    const double radius = std::sqrt(std::pow(510.0 / 930.0, 2.0) + std::pow(386.0 / 905.0, 2.0));
    EXPECT_NEAR(group.transitionWeight[0], radius / 0.01, 1.0e-10);
    EXPECT_NEAR(group.transitionWeight[4], std::pow(radius, 3.0) * 930.0 / 1.0e6, 1.0e-12);
    EXPECT_NEAR(group.transitionWeight[5], std::pow(radius, 5.0) * 930.0 / 0.01, 1.0e-8);
    EXPECT_DOUBLE_EQ(group.transitionWeight[1], 0.0);
}

TEST(BundleAdjustPlaMatrixExtendedBackendTest, FocalStateAndStepUsePhysicalPixelUnits)
{
    const xjw::FramePinholeCamera camera = makeExtendedCamera();
    xjw::BAOptions options;
    options.refineSharedFocalLength = true;

    xjw::detail::plamatrix_ba::ActiveProblem active;
    active.intrinsicBlockCount = 1;
    active.calibrationGroupByCamera = {0};
    auto groups = xjw::detail::plamatrix_ba::initializeIntrinsicGroups({camera}, options, active);

    ASSERT_EQ(groups.size(), 1U);
    EXPECT_DOUBLE_EQ(groups.front().parameters[0], 930.0);
    EXPECT_DOUBLE_EQ(groups.front().prior[0], 930.0);
    EXPECT_DOUBLE_EQ(groups.front().lower[0], 930.0 * options.minSharedFocalScale);
    EXPECT_DOUBLE_EQ(groups.front().upper[0], 930.0 * options.maxSharedFocalScale);
    EXPECT_NEAR(groups.front().inverseSigma[0], 1.0 / (930.0 * options.sharedFocalPriorSigma), 1.0e-15);

    std::vector<double> primary_step(9, 0.0);
    primary_step[0] = 12.5;
    xjw::detail::plamatrix_ba::applyIntrinsicStep(active, primary_step, &groups);
    EXPECT_DOUBLE_EQ(groups.front().parameters[0], 942.5);
}
