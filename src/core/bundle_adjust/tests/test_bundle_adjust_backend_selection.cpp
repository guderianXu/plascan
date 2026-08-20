#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "BundleAdjustCeresPlanning.h"
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

TEST(BundleAdjustBackendSelectionTest, NativeCudaCapabilityPreventsPoseRefinementSelection)
{
    EXPECT_STREQ(xjw::BundleAdjust::backendName(xjw::BABackend::NativeCuda), "native_cuda");
    const xjw::BABackendCapabilities capabilities =
        xjw::BundleAdjust::backendCapabilities(xjw::BABackend::NativeCuda);
    EXPECT_TRUE(capabilities.optimizesPoints);
    EXPECT_FALSE(capabilities.refinesCameraPose);
    EXPECT_FALSE(capabilities.refinesSharedFocalLength);
    EXPECT_FALSE(capabilities.supportsSoftConstraints);

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minPlaMatrixGpuCameras = 1000;
    options.minPlaMatrixGpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 4;
    stats.trackCount = 10;
    stats.observationCount = 30;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    EXPECT_NE(selected, xjw::BABackend::NativeCuda);
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

TEST(BundleAdjustBackendSelectionTest, ProductionBuildRejectsExplicitCeresWithoutLegacyFallback)
{
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        GTEST_SKIP() << "Ceres reference backend is enabled in this build";
    }

    std::vector<xjw::FramePinholeCamera> cameras(2);
    cameras[0].setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    cameras[0].setPose({{1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0}},
                       {{-1.0, 0.0, 0.0}});
    cameras[1].setIntrinsics(1000.0, 1000.0, 512.0, 384.0);
    cameras[1].setPose({{1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0}},
                       {{1.0, 0.0, 0.0}});

    xjw::BATrack track;
    track.initialPoint = {{0.0, 0.0, 10.0}};
    track.observations = {
        xjw::BAObservation{0, 612.0, 384.0, 1.0},
        xjw::BAObservation{1, 412.0, 384.0, 1.0},
    };

    xjw::BAOptions options;
    options.backend = xjw::BABackend::CeresCpu;
    options.allowBackendFallback = true;
    options.refineCameraPose = false;

    const xjw::BAResult cpu_result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(cpu_result.solveStatus, xjw::BASolveStatus::BackendUnavailable);
    EXPECT_EQ(cpu_result.usedBackend, xjw::BABackend::CeresCpu);
    EXPECT_FALSE(cpu_result.backendFallback);
    EXPECT_NE(cpu_result.backendMessage.find("PLASCAN_ENABLE_CERES_REFERENCE=ON"),
              std::string::npos);

    options.backend = xjw::BABackend::CeresCuda;
    const xjw::BAResult cuda_result =
        xjw::BundleAdjust::optimizePoints(cameras, {track}, options);

    EXPECT_EQ(cuda_result.solveStatus, xjw::BASolveStatus::BackendUnavailable);
    EXPECT_EQ(cuda_result.usedBackend, xjw::BABackend::CeresCuda);
    EXPECT_FALSE(cuda_result.backendFallback);
    EXPECT_NE(cuda_result.backendMessage.find("PLASCAN_ENABLE_CERES_REFERENCE=ON"),
              std::string::npos);
}

TEST(BundleAdjustCeresPlanningTest, AutoUsesSolverMatchingCameraScale)
{
    xjw::BAOptions options;
    options.ceresLinearSolver = xjw::BACeresLinearSolver::Auto;
    options.maxDenseSchurCameras = 50;
    options.maxSparseSchurCameras = 500;

    const auto small = xjw::detail::planCeresSolver(
        options, 40, 1, 1000, 5000, false, 0);
    const auto medium = xjw::detail::planCeresSolver(
        options, 200, 1, 10000, 50000, false, 0);
    const auto large = xjw::detail::planCeresSolver(
        options, 1000, 1, 100000, 500000, false, 0);

    EXPECT_EQ(small.solver, xjw::detail::BACeresSolverKind::DenseSchur);
    EXPECT_EQ(medium.solver, xjw::detail::BACeresSolverKind::SparseSchur);
    EXPECT_EQ(large.solver, xjw::detail::BACeresSolverKind::IterativeSchur);
}

TEST(BundleAdjustCeresPlanningTest, PointOnlyProblemUsesDenseQr)
{
    xjw::BAOptions options;
    const auto plan = xjw::detail::planCeresSolver(
        options, 0, 0, 10000, 50000, false, 0);

    EXPECT_EQ(plan.solver, xjw::detail::BACeresSolverKind::DenseQr);
    EXPECT_FALSE(plan.useCuda);
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

TEST(BundleAdjustCeresPlanningTest, MissingSparseLibraryUsesIterativeSchur)
{
    xjw::BAOptions options;
    options.ceresLinearSolver = xjw::BACeresLinearSolver::Auto;
    options.maxDenseSchurCameras = 50;
    options.maxSparseSchurCameras = 500;

    const auto plan = xjw::detail::planCeresSolver(
        options, 200, 1, 10000, 50000, false, 0, false);

    EXPECT_EQ(plan.solver, xjw::detail::BACeresSolverKind::IterativeSchur);
    EXPECT_FALSE(plan.useCuda);
}

TEST(BundleAdjustCeresPlanningTest, CudaFallsBackBeforeExceedingMemoryBudget)
{
    xjw::BAOptions options;
    options.maxCeresCudaMemoryFraction = 0.5;
    options.maxDenseSchurCameras = 20;
    options.maxSparseSchurCameras = 200;

    const auto plan = xjw::detail::planCeresSolver(
        options,
        1000,
        1,
        100000,
        1000000,
        true,
        256ULL * 1024ULL * 1024ULL);

    EXPECT_FALSE(plan.useCuda);
    EXPECT_EQ(plan.solver, xjw::detail::BACeresSolverKind::IterativeSchur);
    EXPECT_GT(plan.estimatedCudaBytes, 128ULL * 1024ULL * 1024ULL);
    EXPECT_NE(plan.reason.find("显存"), std::string::npos);
}
