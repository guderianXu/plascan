#include "PointCloudAlignment.h"

#include <gtest/gtest.h>

#include <cmath>

using xjw::qc::Point3D;
using xjw::qc::PointCloudAlignment;

TEST(PointCloudAlignment, RecoversScaleAndTranslationForPairedClouds)
{
    const std::vector<Point3D> source = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };

    std::vector<Point3D> reference;
    for (const Point3D &point : source)
    {
        reference.push_back({2.0 * point.x + 10.0,
                             2.0 * point.y - 4.0,
                             2.0 * point.z + 1.5});
    }

    const auto result = PointCloudAlignment::alignPairedSimilarity(source, reference);

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_NEAR(result.transform.scale, 2.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.x, 10.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.y, -4.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.z, 1.5, 1e-9);
    EXPECT_NEAR(result.before.rmse, std::sqrt(125.0), 1e-9);
    EXPECT_NEAR(result.after.rmse, 0.0, 1e-9);
    EXPECT_EQ(result.pairCount, 4);
}

TEST(PointCloudAlignment, RejectsMismatchedClouds)
{
    const auto result = PointCloudAlignment::alignPairedSimilarity(
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.isEmpty());
}

TEST(PointCloudAlignment, AlignsUnpairedCloudsWithNearestNeighborTranslation)
{
    const std::vector<Point3D> source = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}
    };
    const std::vector<Point3D> reference = {
        {10.0, -4.0, 1.5},
        {11.0, -4.0, 1.5},
        {10.0, -2.0, 1.5},
        {10.0, -4.0, 4.5},
        {30.0, 20.0, 10.0},
        {-15.0, 8.0, 2.0}
    };

    const auto result = PointCloudAlignment::alignNearestNeighborTranslation(source, reference);

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_EQ(result.method, QStringLiteral("nearest_neighbor_translation"));
    EXPECT_EQ(result.pairCount, 4);
    EXPECT_NEAR(result.transform.scale, 1.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.x, 10.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.y, -4.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.z, 1.5, 1e-9);
    EXPECT_GT(result.before.rmse, 1.0);
    EXPECT_NEAR(result.after.rmse, 0.0, 1e-9);
}
