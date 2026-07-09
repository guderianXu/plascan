#include <gtest/gtest.h>

#include "BundleAdjust.h"

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

TEST(BundleAdjustBackendSelectionTest, NativeCudaBackendNameAndAutoThresholdsAreStable)
{
    EXPECT_STREQ(xjw::BundleAdjust::backendName(xjw::BABackend::NativeCuda), "native_cuda");

    xjw::BAOptions options;
    options.backend = xjw::BABackend::Auto;
    options.refineCameraPose = true;
    options.minNativeCudaCameras = 3;
    options.minNativeCudaObservations = 20;
    options.minCeresCudaCameras = 1000;
    options.minCeresCudaObservations = 1000000;
    options.minCeresCpuObservations = 1000000;

    xjw::BAProblemStats stats;
    stats.cameraCount = 4;
    stats.trackCount = 10;
    stats.observationCount = 30;

    const xjw::BABackend selected = xjw::BundleAdjust::selectBackendForProblem(stats, options);
    if (xjw::BundleAdjust::isBackendAvailable(xjw::BABackend::NativeCuda))
    {
        EXPECT_EQ(selected, xjw::BABackend::NativeCuda);
    }
    else
    {
        EXPECT_NE(selected, xjw::BABackend::NativeCuda);
    }
}
