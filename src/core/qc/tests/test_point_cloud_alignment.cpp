#include "PointCloudAlignment.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using xjw::qc::Point3D;
using xjw::qc::PointCloudAlignment;

namespace
{

Point3D rotateZAndTranslate(const Point3D &point, double radians, const Point3D &translation)
{
    const double c = std::cos(radians);
    const double s = std::sin(radians);
    return {
        c * point.x - s * point.y + translation.x,
        s * point.x + c * point.y + translation.y,
        point.z + translation.z
    };
}

double pointDistance(const Point3D &a, const Point3D &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

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
    EXPECT_EQ(result.method, QStringLiteral("nearest_neighbor_icp"));
    EXPECT_EQ(result.pairCount, 4);
    EXPECT_NEAR(result.transform.scale, 1.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.x, 10.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.y, -4.0, 1e-9);
    EXPECT_NEAR(result.transform.translation.z, 1.5, 1e-9);
    EXPECT_GT(result.before.rmse, 1.0);
    EXPECT_NEAR(result.after.rmse, 0.0, 1e-9);
}

TEST(PointCloudAlignment, RecoversRotationScaleAndTranslationForPairedClouds)
{
    const std::vector<Point3D> source = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0},
        {1.0, 2.0, 3.0}
    };
    const double angle = 0.7;
    const double scale = 2.5;
    const Point3D translation{4.0, -3.0, 1.25};
    std::vector<Point3D> reference;
    for (const Point3D &point : source)
    {
        const Point3D rotated = rotateZAndTranslate(point, angle, {});
        reference.push_back({scale * rotated.x + translation.x,
                             scale * rotated.y + translation.y,
                             scale * rotated.z + translation.z});
    }

    const auto result = PointCloudAlignment::alignPairedSimilarity(source, reference);

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_NEAR(result.transform.scale, scale, 1.0e-9);
    EXPECT_LT(result.after.rmse, 1.0e-9);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        EXPECT_LT(pointDistance(PointCloudAlignment::apply(result.transform, source[index]),
                                reference[index]),
                  1.0e-9);
    }
}

TEST(PointCloudAlignment, AlignsUnpairedCloudsWithPlaPointIcpRotation)
{
    std::vector<Point3D> source;
    source.reserve(30);
    for (int i = 0; i < 30; ++i)
    {
        const double x = static_cast<double>(i % 6) - 2.5 + 0.03 * static_cast<double>(i / 6);
        const double y = static_cast<double>(i / 6) - 2.0 + 0.05 * static_cast<double>((i * 3) % 5);
        const double z = 0.08 * static_cast<double>((i * i) % 7) - 0.2;
        source.push_back({x, y, z});
    }

    const double angle = 0.16;
    const Point3D translation{0.12, -0.08, 0.04};
    std::vector<Point3D> reference;
    reference.reserve(source.size());
    for (const Point3D &point : source)
    {
        reference.push_back(rotateZAndTranslate(point, angle, translation));
    }

    const auto result = PointCloudAlignment::alignNearestNeighborTranslation(source, reference, 40);

    ASSERT_TRUE(result.success) << result.error.toStdString();
    EXPECT_EQ(result.method, QStringLiteral("nearest_neighbor_icp"));
    EXPECT_LT(result.after.rmse, result.before.rmse * 0.05);
    EXPECT_LT(result.after.rmse, 1.0e-5);

    for (std::size_t i = 0; i < source.size(); ++i)
    {
        const Point3D aligned = PointCloudAlignment::apply(result.transform, source[i]);
        EXPECT_LT(pointDistance(aligned, reference[i]), 1.0e-4);
    }
}
