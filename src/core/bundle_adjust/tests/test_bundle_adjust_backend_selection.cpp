#include <gtest/gtest.h>

#include "BundleAdjustSolver.h"
#include "FramePinholeCamera.h"

#include <array>
#include <string>
#include <vector>

TEST(BundleAdjustBackendSelectionTest, PointOnlyProblemUsesLegacyCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;

    xjw::BAProblemStats stats;
    stats.cameraCount = 10;
    stats.trackCount = 500;
    stats.observationCount = 3000;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    EXPECT_EQ(selected, xjw::BABackend::LegacyCpu);
}
TEST(BundleAdjustBackendSelectionTest, SmallJointProblemUsesPlaMatrixCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixGpuCameras = 50;
    options.minPlaMatrixGpuObservations = 500000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 16;
    stats.trackCount = 4021;
    stats.observationCount = 12000;

    const xjw::BABackendDecision decision =
        xjw::BundleAdjust::decideBackendForProblem(stats, options);

    EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "joint_problem_uses_plamatrix_cpu");
}

TEST(BundleAdjustBackendSelectionTest, DefaultGpuThresholdRequiresBothMeasuredLimits)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;

    EXPECT_EQ(options.minPlaMatrixGpuCameras, 24);
    EXPECT_EQ(options.minPlaMatrixGpuObservations, 30000);

    xjw::BAProblemStats stats;
    stats.cameraCount = options.minPlaMatrixGpuCameras - 1;
    stats.observationCount = options.minPlaMatrixGpuObservations;
    EXPECT_EQ(xjw::BundleAdjust::selectBackendForProblem(stats, options),
              xjw::BABackend::PlaMatrixCpu);

    stats.cameraCount = options.minPlaMatrixGpuCameras;
    stats.observationCount = options.minPlaMatrixGpuObservations - 1;
    EXPECT_EQ(xjw::BundleAdjust::selectBackendForProblem(stats, options),
              xjw::BABackend::PlaMatrixCpu);
}

TEST(BundleAdjustBackendSelectionTest, LargeJointProblemUsesAvailablePlaMatrixGpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixGpuCameras = 50;
    options.minPlaMatrixGpuObservations = 100000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 50000;
    stats.observationCount = 300000;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::PlaMatrixCuda);
    }
    else if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixOpenCl))
    {
        EXPECT_EQ(selected, xjw::BABackend::PlaMatrixOpenCl);
    }
    else
    {
        EXPECT_EQ(selected, xjw::BABackend::PlaMatrixCpu);
    }
}

TEST(BundleAdjustBackendSelectionTest, SharedFocalUsesPlaMatrixEvenForSmallProblem)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.refineSharedFocalLength = true;
    options.minPlaMatrixGpuCameras = 1000;
    options.minPlaMatrixGpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 8;
    stats.trackCount = 300;
    stats.observationCount = 1600;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    EXPECT_EQ(selected, xjw::BABackend::PlaMatrixCpu);
}

TEST(BundleAdjustBackendSelectionTest, SoftConstraintUsesPlaMatrixWhenPoseIsFixed)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;
    options.enableControlPointConstraints = true;
    options.minPlaMatrixGpuCameras = 1000;
    options.minPlaMatrixGpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 3;
    stats.trackCount = 20;
    stats.observationCount = 60;

    const xjw::BABackendDecision decision =
        xjw::BundleAdjust::decideBackendForProblem(stats, options);

    EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "constraint_problem_uses_plamatrix_cpu");
}

TEST(BundleAdjustBackendSelectionTest, LargeSharedRadialProblemUsesAvailablePlaMatrixGpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.refineSharedFocalLength = true;
    options.refineSharedRadialDistortion = true;
    options.minPlaMatrixGpuCameras = 50;
    options.minPlaMatrixGpuObservations = 100000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 442;
    stats.trackCount = 360000;
    stats.observationCount = 800000;

    const xjw::BABackendDecision decision =
        xjw::BundleAdjust::decideBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCuda))
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCuda);
        EXPECT_EQ(decision.reason, "large_joint_problem_uses_plamatrix_cuda");
    }
    else if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixOpenCl))
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixOpenCl);
        EXPECT_EQ(decision.reason, "large_joint_problem_uses_plamatrix_opencl");
    }
    else
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCpu);
        EXPECT_EQ(decision.reason, "joint_problem_uses_plamatrix_cpu");
    }
}
