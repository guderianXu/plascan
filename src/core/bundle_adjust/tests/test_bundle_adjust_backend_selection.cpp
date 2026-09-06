#include <gtest/gtest.h>

#include "BundleAdjustSolver.h"
#include "FramePinholeCamera.h"

#include <array>
#include <string>
#include <vector>

TEST(BundleAdjustBackendSelectionTest, PointOnlyProblemUsesReferenceCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = false;

    xjw::BAProblemStats stats;
    stats.cameraCount = 10;
    stats.trackCount = 500;
    stats.observationCount = 3000;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    EXPECT_EQ(selected, xjw::BABackend::PlaMatrixCpu);
}

TEST(BundleAdjustBackendSelectionTest, ExplicitBackendBypassesAutoScalePolicy)
{
    xjw::BAProblemStats stats;
    stats.cameraCount = 1;
    stats.observationCount = 2;

    for (const xjw::BABackend backend :
         {xjw::BABackend::PlaMatrixCpu, xjw::BABackend::PlaMatrixCuda, xjw::BABackend::PlaMatrixOpenCl})
    {
        xjw::BAOptions options;
        options.backend = backend;
        const xjw::BABackendDecision decision = xjw::BundleAdjust::decideBackendForProblem(stats, options);
        EXPECT_EQ(decision.backend, backend);
        EXPECT_EQ(decision.reason, "explicit_backend");
    }
}

TEST(BundleAdjustBackendSelectionTest, SmallJointProblemUsesPlaMatrixCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixCudaCameras = 50;
    options.minPlaMatrixCudaObservations = 500000;
    options.minPlaMatrixOpenClCameras = 50;
    options.minPlaMatrixOpenClObservations = 500000;
    options.minPlaMatrixDenseCameras = 50;
    options.minPlaMatrixCudaDenseObservations = 500000;
    options.minPlaMatrixOpenClDenseObservations = 500000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 16;
    stats.trackCount = 4021;
    stats.observationCount = 12000;

    const xjw::BABackendDecision decision = xjw::BundleAdjust::decideBackendForProblem(stats, options);

    EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCpu);
    EXPECT_EQ(decision.reason, "joint_problem_uses_plamatrix_cpu");
}

TEST(BundleAdjustBackendSelectionTest, DefaultPolicyMatchesMeasuredBackendCrossovers)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;

    EXPECT_EQ(options.minPlaMatrixCudaCameras, 128);
    EXPECT_EQ(options.minPlaMatrixCudaObservations, 30000);
    EXPECT_EQ(options.minPlaMatrixOpenClCameras, 160);
    EXPECT_EQ(options.minPlaMatrixOpenClObservations, 50000);
    EXPECT_EQ(options.minPlaMatrixDenseCameras, 120);
    EXPECT_EQ(options.minPlaMatrixCudaDenseObservations, 150000);
    EXPECT_EQ(options.minPlaMatrixOpenClDenseObservations, 200000);

    xjw::BAProblemStats stats;
    stats.cameraCount = 96;
    stats.observationCount = 30720;
    EXPECT_FALSE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixCuda, stats, options));
    EXPECT_FALSE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixOpenCl, stats, options));

    stats.cameraCount = 128;
    stats.observationCount = 40960;
    EXPECT_TRUE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixCuda, stats, options));
    EXPECT_FALSE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixOpenCl, stats, options));

    stats.cameraCount = 160;
    stats.observationCount = 51200;
    EXPECT_TRUE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixCuda, stats, options));
    EXPECT_TRUE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixOpenCl, stats, options));
}

TEST(BundleAdjustBackendSelectionTest, DenseObservationTierCoversSouthBuildingScale)
{
    xjw::BAOptions options;
    xjw::BAProblemStats stats;
    stats.cameraCount = 123;
    stats.trackCount = 33007;
    stats.observationCount = 223593;

    EXPECT_TRUE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixCuda, stats, options));
    EXPECT_TRUE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixOpenCl, stats, options));

    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    const xjw::BABackendDecision decision = xjw::BundleAdjust::decideBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixCuda))
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCuda);
        EXPECT_EQ(decision.reason, "dense_joint_problem_uses_plamatrix_cuda");
    }
    else if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::PlaMatrixOpenCl))
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixOpenCl);
        EXPECT_EQ(decision.reason, "dense_joint_problem_uses_plamatrix_opencl");
    }
    else
    {
        EXPECT_EQ(decision.backend, xjw::BABackend::PlaMatrixCpu);
    }

    stats.cameraCount = options.minPlaMatrixDenseCameras - 1;
    EXPECT_FALSE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixCuda, stats, options));
    EXPECT_FALSE(xjw::BundleAdjust::autoBackendMeetsScaleThreshold(xjw::BABackend::PlaMatrixOpenCl, stats, options));
}

TEST(BundleAdjustBackendSelectionTest, LargeJointProblemUsesAvailablePlaMatrixGpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixCudaCameras = 50;
    options.minPlaMatrixCudaObservations = 100000;
    options.minPlaMatrixOpenClCameras = 50;
    options.minPlaMatrixOpenClObservations = 100000;

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
    options.minPlaMatrixCudaCameras = 1000;
    options.minPlaMatrixCudaObservations = 1000000;
    options.minPlaMatrixOpenClCameras = 1000;
    options.minPlaMatrixOpenClObservations = 1000000;
    options.minPlaMatrixDenseCameras = 1000;

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
    options.minPlaMatrixCudaCameras = 1000;
    options.minPlaMatrixCudaObservations = 1000000;
    options.minPlaMatrixOpenClCameras = 1000;
    options.minPlaMatrixOpenClObservations = 1000000;
    options.minPlaMatrixDenseCameras = 1000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 3;
    stats.trackCount = 20;
    stats.observationCount = 60;

    const xjw::BABackendDecision decision = xjw::BundleAdjust::decideBackendForProblem(stats, options);

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
    options.minPlaMatrixCudaCameras = 50;
    options.minPlaMatrixCudaObservations = 100000;
    options.minPlaMatrixOpenClCameras = 50;
    options.minPlaMatrixOpenClObservations = 100000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 442;
    stats.trackCount = 360000;
    stats.observationCount = 800000;

    const xjw::BABackendDecision decision = xjw::BundleAdjust::decideBackendForProblem(stats, options);
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
