#include "VisibilityOccupancyHandleRepair.h"
#include "VisibilityOccupancyWellComposedRepair.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

std::size_t index(
    const std::array<int, 3> &dimensions,
    int x,
    int y,
    int z)
{
    return (static_cast<std::size_t>(z) * dimensions[1] + y) *
        dimensions[0] + x;
}

std::vector<std::uint8_t> emptyGrid(
    const std::array<int, 3> &dimensions)
{
    return std::vector<std::uint8_t>(
        static_cast<std::size_t>(dimensions[0]) * dimensions[1] *
            dimensions[2],
        0);
}

std::uint64_t remainingDefects(
    const xjw::mesh::VisibilityOccupancyWellComposedRepairStatistics &statistics)
{
    return statistics.remainingEdgeCheckerboardCount +
        statistics.remainingVertexOccupiedComponentDefectCount +
        statistics.remainingVertexEmptyComponentDefectCount;
}

xjw::mesh::VisibilityOccupancyWellComposedRepairResult repair(
    const std::array<int, 3> &dimensions,
    const std::vector<std::uint8_t> &occupied,
    const std::vector<std::uint8_t> &protected_empty)
{
    xjw::mesh::VisibilityOccupancyWellComposedRepairOptions options;
    options.maximumPasses = 8;
    options.maximumFilledSampleCount = 64;
    return xjw::mesh::VisibilityOccupancyWellComposedRepair::repair(
        dimensions, occupied, protected_empty, options);
}

} // namespace

TEST(VisibilityOccupancyWellComposedRepairTest,
     RepairsEdgeDiagonalCheckerboard)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 2)] = 1;
    occupied[index(dimensions, 2, 2, 2)] = 1;
    const std::vector<std::uint8_t> protected_empty = emptyGrid(dimensions);
    const int euler_before =
        xjw::mesh::VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            dimensions, occupied);

    const auto result = repair(dimensions, occupied, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_GT(result.statistics.edgeCheckerboardCountBefore, 0U);
    EXPECT_GT(result.statistics.filledSampleCount, 0U);
    EXPECT_EQ(result.statistics.remainingEdgeCheckerboardCount, 0U);
    EXPECT_GE(result.statistics.bodyEulerAfter, euler_before);
}

TEST(VisibilityOccupancyWellComposedRepairTest,
     RefusesProtectedEdgeDiagonalRepair)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 2)] = 1;
    occupied[index(dimensions, 2, 2, 2)] = 1;
    std::vector<std::uint8_t> protected_empty(occupied.size(), 1);
    for (std::size_t sample = 0; sample < occupied.size(); ++sample)
    {
        protected_empty[sample] = occupied[sample] == 0 ? 1 : 0;
    }

    const auto result = repair(dimensions, occupied, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(result.statistics.filledSampleCount, 0U);
    EXPECT_GT(result.statistics.protectedFillRefusalCount, 0U);
    EXPECT_GT(result.statistics.remainingEdgeCheckerboardCount, 0U);
}

TEST(VisibilityOccupancyWellComposedRepairTest,
     ConnectsCornerTouchWithoutLoweringBodyEuler)
{
    const std::array<int, 3> dimensions{5, 5, 5};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    occupied[index(dimensions, 1, 1, 1)] = 1;
    occupied[index(dimensions, 2, 2, 2)] = 1;
    const std::vector<std::uint8_t> protected_empty = emptyGrid(dimensions);
    const int euler_before =
        xjw::mesh::VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            dimensions, occupied);

    const auto result = repair(dimensions, occupied, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_GT(
        result.statistics.vertexOccupiedComponentDefectCountBefore,
        0U);
    EXPECT_GT(result.statistics.filledSampleCount, 0U);
    EXPECT_EQ(remainingDefects(result.statistics), 0U);
    EXPECT_GE(result.statistics.bodyEulerAfter, euler_before);
}

TEST(VisibilityOccupancyWellComposedRepairTest,
     LeavesStraightTunnelUnchanged)
{
    const std::array<int, 3> dimensions{7, 7, 7};
    std::vector<std::uint8_t> occupied = emptyGrid(dimensions);
    for (int z = 1; z <= 5; ++z)
    {
        for (int y = 1; y <= 5; ++y)
        {
            for (int x = 1; x <= 5; ++x)
            {
                occupied[index(dimensions, x, y, z)] =
                    x == 3 && y == 3 ? 0 : 1;
            }
        }
    }
    const std::vector<std::uint8_t> protected_empty = emptyGrid(dimensions);
    const int euler_before =
        xjw::mesh::VisibilityOccupancyHandleRepair::bodyEulerCharacteristic(
            dimensions, occupied);

    const auto result = repair(dimensions, occupied, protected_empty);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.occupied, occupied);
    EXPECT_EQ(result.statistics.filledSampleCount, 0U);
    EXPECT_EQ(result.statistics.acceptedPassCount, 0);
    EXPECT_EQ(remainingDefects(result.statistics), 0U);
    EXPECT_EQ(result.statistics.bodyEulerAfter, euler_before);
}
