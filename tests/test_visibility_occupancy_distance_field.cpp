#include "VisibilityOccupancyDistanceField.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

std::size_t gridIndex(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(dimensions[1]) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(dimensions[0]) +
           static_cast<std::size_t>(x);
}

std::vector<std::uint8_t> makeAxisSlab(
    const std::array<int, 3> &dimensions,
    int axis)
{
    std::vector<std::uint8_t> occupied(
        static_cast<std::size_t>(dimensions[0]) *
            static_cast<std::size_t>(dimensions[1]) *
            static_cast<std::size_t>(dimensions[2]),
        0);
    for (int z = 1; z + 1 < dimensions[2]; ++z)
    {
        for (int y = 1; y + 1 < dimensions[1]; ++y)
        {
            for (int x = 1; x + 1 < dimensions[0]; ++x)
            {
                const std::array<int, 3> coordinate{x, y, z};
                if (coordinate[axis] >= 3 &&
                    coordinate[axis] + 1 < dimensions[axis])
                {
                    occupied[gridIndex(dimensions, x, y, z)] = 1;
                }
            }
        }
    }
    return occupied;
}

} // namespace

TEST(VisibilityOccupancyDistanceFieldTest, UsesAnisotropicWorldSpacing)
{
    const std::array<int, 3> dimensions{25, 25, 25};
    const std::array<float, 3> bounds_min{0.0f, 0.0f, 0.0f};
    const std::array<float, 3> bounds_max{24.0f, 48.0f, 72.0f};
    const std::array<float, 3> spacing{1.0f, 2.0f, 3.0f};
    for (int axis = 0; axis < 3; ++axis)
    {
        const auto result =
            xjw::mesh::VisibilityOccupancyDistanceField::build(
                dimensions,
                bounds_min,
                bounds_max,
                makeAxisSlab(dimensions, axis));
        ASSERT_TRUE(result.ok) << result.error;
        std::array<int, 3> coordinate{12, 12, 12};
        coordinate[axis] = 3;
        const float first = result.signedWorldDistance[gridIndex(
            dimensions, coordinate[0], coordinate[1], coordinate[2])];
        coordinate[axis] = 4;
        const float second = result.signedWorldDistance[gridIndex(
            dimensions, coordinate[0], coordinate[1], coordinate[2])];
        coordinate[axis] = 5;
        const float third = result.signedWorldDistance[gridIndex(
            dimensions, coordinate[0], coordinate[1], coordinate[2])];
        EXPECT_NEAR(first, -0.5f * spacing[axis], 1e-5f);
        EXPECT_NEAR(second, -1.5f * spacing[axis], 1e-5f);
        EXPECT_NEAR(third, -2.5f * spacing[axis], 1e-5f);
    }
}

TEST(VisibilityOccupancyDistanceFieldTest, PreservesSignAndPositiveOuterShell)
{
    const std::array<int, 3> dimensions{9, 9, 9};
    std::vector<std::uint8_t> occupied(729, 0);
    for (int z = 2; z <= 6; ++z)
    {
        for (int y = 2; y <= 6; ++y)
        {
            for (int x = 2; x <= 6; ++x)
            {
                occupied[gridIndex(dimensions, x, y, z)] = 1;
            }
        }
    }
    occupied[gridIndex(dimensions, 0, 4, 4)] = 1;
    const auto result = xjw::mesh::VisibilityOccupancyDistanceField::build(
        dimensions,
        {-1.0f, -1.0f, -1.0f},
        {1.0f, 1.0f, 1.0f},
        occupied);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_LT(result.signedWorldDistance[gridIndex(dimensions, 4, 4, 4)], 0.0f);
    EXPECT_GT(result.signedWorldDistance[gridIndex(dimensions, 1, 4, 4)], 0.0f);
    for (int z = 0; z < dimensions[2]; ++z)
    {
        for (int y = 0; y < dimensions[1]; ++y)
        {
            EXPECT_GT(result.signedWorldDistance[gridIndex(dimensions, 0, y, z)], 0.0f);
            EXPECT_GT(result.signedWorldDistance[gridIndex(dimensions, 8, y, z)], 0.0f);
        }
    }
}

TEST(VisibilityOccupancyDistanceFieldTest, ProducesOnlyFiniteNonZeroDistances)
{
    const std::array<int, 3> dimensions{7, 6, 5};
    std::vector<std::uint8_t> occupied(210, 0);
    const auto empty_result =
        xjw::mesh::VisibilityOccupancyDistanceField::build(
            dimensions,
            {-2.0f, -3.0f, -4.0f},
            {2.0f, 3.0f, 4.0f},
            occupied);
    ASSERT_TRUE(empty_result.ok) << empty_result.error;
    for (const float distance : empty_result.signedWorldDistance)
    {
        EXPECT_TRUE(std::isfinite(distance));
        EXPECT_GT(distance, 0.0f);
    }

    std::fill(occupied.begin(), occupied.end(), 1);
    const auto full_result =
        xjw::mesh::VisibilityOccupancyDistanceField::build(
            dimensions,
            {-2.0f, -3.0f, -4.0f},
            {2.0f, 3.0f, 4.0f},
            occupied);
    ASSERT_TRUE(full_result.ok) << full_result.error;
    for (const float distance : full_result.signedWorldDistance)
    {
        EXPECT_TRUE(std::isfinite(distance));
        EXPECT_NE(distance, 0.0f);
    }
}

TEST(VisibilityOccupancyDistanceFieldTest, BoxAndSphereDistancesAreMonotone)
{
    const std::array<int, 3> dimensions{17, 17, 17};
    std::vector<std::uint8_t> box(4913, 0);
    std::vector<std::uint8_t> sphere(4913, 0);
    for (int z = 1; z < 16; ++z)
    {
        for (int y = 1; y < 16; ++y)
        {
            for (int x = 1; x < 16; ++x)
            {
                if (x >= 4 && x <= 12 &&
                    y >= 4 && y <= 12 &&
                    z >= 4 && z <= 12)
                {
                    box[gridIndex(dimensions, x, y, z)] = 1;
                }
                const int dx = x - 8;
                const int dy = y - 8;
                const int dz = z - 8;
                if (dx * dx + dy * dy + dz * dz <= 36)
                {
                    sphere[gridIndex(dimensions, x, y, z)] = 1;
                }
            }
        }
    }
    for (const auto *shape : {&box, &sphere})
    {
        const auto result =
            xjw::mesh::VisibilityOccupancyDistanceField::build(
                dimensions,
                {-8.0f, -8.0f, -8.0f},
                {8.0f, 8.0f, 8.0f},
                *shape);
        ASSERT_TRUE(result.ok) << result.error;
        const float near_surface = std::fabs(
            result.signedWorldDistance[gridIndex(dimensions, 12, 8, 8)]);
        const float middle = std::fabs(
            result.signedWorldDistance[gridIndex(dimensions, 10, 8, 8)]);
        const float center = std::fabs(
            result.signedWorldDistance[gridIndex(dimensions, 8, 8, 8)]);
        EXPECT_LT(near_surface, middle);
        EXPECT_LT(middle, center);
    }
}
