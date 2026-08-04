#include <gtest/gtest.h>

#include "BundleAdjust.h"
#include "BundleAdjustCeresPlanning.h"

TEST(BundleAdjustBackendSelectionTest, SmallProblemUsesLegacyCpu)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.minCeresCudaCameras = 50;
    options.minCeresCudaObservations = 100000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 10;
    stats.trackCount = 500;
    stats.observationCount = 3000;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    EXPECT_EQ(selected, xjw::BABackend::LegacyCpu);
}

TEST(BundleAdjustBackendSelectionTest, SmallProblemExplainsWhyCudaIsNotUsed)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.minCeresCudaCameras = 50;
    options.minCeresCudaObservations = 500000;
    options.minCeresCpuObservations = 50000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 16;
    stats.trackCount = 4021;
    stats.observationCount = 12000;

    const xjw::BABackendDecision decision =
        xjw::BundleAdjust::decideBackendForProblem(stats, options);

    EXPECT_EQ(decision.backend, xjw::BABackend::LegacyCpu);
    EXPECT_EQ(decision.reason, "below_accelerated_problem_size");
}

TEST(BundleAdjustBackendSelectionTest, LargeProblemCanSelectCudaWhenAvailable)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.minCeresCudaCameras = 50;
    options.minCeresCudaObservations = 100000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 120;
    stats.trackCount = 50000;
    stats.observationCount = 300000;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);

    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::CeresCuda);
    }
    else
    {
        EXPECT_NE(selected, xjw::BABackend::CeresCuda);
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
    options.minCeresCudaCameras = 1000;
    options.minCeresCudaObservations = 1000000;
    options.minCeresCpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 4;
    stats.trackCount = 10;
    stats.observationCount = 30;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    EXPECT_NE(selected, xjw::BABackend::NativeCuda);
}

TEST(BundleAdjustBackendSelectionTest, SharedFocalUsesJointCeresEvenForSmallProblem)
{
    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.refineSharedFocalLength = true;
    options.minCeresCudaCameras = 1000;
    options.minCeresCudaObservations = 1000000;
    options.minCeresCpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 8;
    stats.trackCount = 300;
    stats.observationCount = 1600;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::CeresCpu))
    {
        EXPECT_EQ(selected, xjw::BABackend::CeresCpu);
    }
    else
    {
        EXPECT_EQ(selected, xjw::BABackend::LegacyCpu);
    }
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
