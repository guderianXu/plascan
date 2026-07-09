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

