#include "BinaryGridMinCutSolver.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace
{

using xjw::mesh::BinaryGridCapacity;
using xjw::mesh::BinaryGridLabel;
using xjw::mesh::BinaryGridMinCutProblem;
using xjw::mesh::BinaryGridMinCutSolver;

BinaryGridMinCutProblem makeProblem(int size_x, int size_y, int size_z)
{
    BinaryGridMinCutProblem problem;
    problem.sizeX = size_x;
    problem.sizeY = size_y;
    problem.sizeZ = size_z;
    const std::size_t node_count = problem.nodeCount();
    problem.sourceCapacities.assign(node_count, 0);
    problem.sinkCapacities.assign(node_count, 0);
    problem.positiveXCapacities.assign(node_count, 0);
    problem.positiveYCapacities.assign(node_count, 0);
    problem.positiveZCapacities.assign(node_count, 0);
    return problem;
}

BinaryGridCapacity energyForMask(
    const BinaryGridMinCutProblem &problem,
    std::uint64_t full_mask)
{
    BinaryGridCapacity energy = 0;
    for (int z = 0; z < problem.sizeZ; ++z)
    {
        for (int y = 0; y < problem.sizeY; ++y)
        {
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                const bool full = (full_mask & (std::uint64_t{1} << index)) != 0;
                energy += full
                    ? problem.sinkCapacities[index]
                    : problem.sourceCapacities[index];
                const auto add_pairwise = [&](std::size_t neighbor, BinaryGridCapacity cost)
                {
                    const bool neighbor_full =
                        (full_mask & (std::uint64_t{1} << neighbor)) != 0;
                    if (full != neighbor_full)
                    {
                        energy += cost;
                    }
                };
                if (x + 1 < problem.sizeX)
                {
                    add_pairwise(
                        problem.index(x + 1, y, z),
                        problem.positiveXCapacities[index]);
                }
                if (y + 1 < problem.sizeY)
                {
                    add_pairwise(
                        problem.index(x, y + 1, z),
                        problem.positiveYCapacities[index]);
                }
                if (z + 1 < problem.sizeZ)
                {
                    add_pairwise(
                        problem.index(x, y, z + 1),
                        problem.positiveZCapacities[index]);
                }
            }
        }
    }
    return energy;
}

std::uint64_t maskFromLabels(const std::vector<BinaryGridLabel> &labels)
{
    std::uint64_t mask = 0;
    for (std::size_t index = 0; index < labels.size(); ++index)
    {
        if (labels[index] == BinaryGridLabel::Full)
        {
            mask |= std::uint64_t{1} << index;
        }
    }
    return mask;
}

} // namespace

TEST(BinaryGridMinCutSolverTest, UnaryTermsChooseLowerCostLabel)
{
    auto problem = makeProblem(2, 1, 1);
    problem.sourceCapacities = {7, 1};
    problem.sinkCapacities = {2, 9};

    const auto result = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(result.solved) << result.error;
    ASSERT_EQ(result.labels.size(), 2U);
    EXPECT_EQ(result.labels[0], BinaryGridLabel::Full);
    EXPECT_EQ(result.labels[1], BinaryGridLabel::Empty);
    EXPECT_EQ(result.statistics.maximumFlow, 3);
    EXPECT_EQ(result.statistics.cutEnergy, 3);
}

TEST(BinaryGridMinCutSolverTest, PairwiseTermSuppressesAnExpensiveLabelTransition)
{
    auto problem = makeProblem(2, 1, 1);
    problem.sourceCapacities = {5, 0};
    problem.sinkCapacities = {0, 5};
    problem.positiveXCapacities[0] = 10;

    const auto result = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(result.solved) << result.error;
    EXPECT_EQ(result.labels[0], result.labels[1]);
    EXPECT_EQ(result.statistics.maximumFlow, 5);
    EXPECT_EQ(result.statistics.cutEnergy, 5);
    EXPECT_EQ(result.statistics.pairwiseEdgeCount, 1U);
}

TEST(BinaryGridMinCutSolverTest, HardUnaryBoundaryAnchorsOppositeSides)
{
    auto problem = makeProblem(3, 1, 1);
    problem.sourceCapacities = {1000000, 0, 0};
    problem.sinkCapacities = {0, 0, 1000000};
    problem.positiveXCapacities[0] = 7;
    problem.positiveXCapacities[1] = 7;

    const auto result = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(result.solved) << result.error;
    EXPECT_EQ(result.labels.front(), BinaryGridLabel::Full);
    EXPECT_EQ(result.labels.back(), BinaryGridLabel::Empty);
    EXPECT_EQ(result.statistics.maximumFlow, 7);
    EXPECT_EQ(result.statistics.cutEnergy, 7);
}

TEST(BinaryGridMinCutSolverTest, MatchesBruteForceEnergyOnSmallThreeDimensionalGrid)
{
    auto problem = makeProblem(2, 2, 2);
    problem.sourceCapacities = {8, 1, 5, 2, 7, 1, 4, 3};
    problem.sinkCapacities = {1, 7, 2, 6, 2, 8, 3, 5};
    problem.positiveXCapacities = {4, 0, 3, 0, 2, 0, 5, 0};
    problem.positiveYCapacities = {2, 1, 0, 0, 4, 3, 0, 0};
    problem.positiveZCapacities = {3, 2, 1, 4, 0, 0, 0, 0};

    BinaryGridCapacity brute_force = std::numeric_limits<BinaryGridCapacity>::max();
    for (std::uint64_t mask = 0; mask < (std::uint64_t{1} << problem.nodeCount()); ++mask)
    {
        brute_force = std::min(brute_force, energyForMask(problem, mask));
    }

    const auto result = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(result.solved) << result.error;
    EXPECT_EQ(result.statistics.maximumFlow, brute_force);
    EXPECT_EQ(result.statistics.cutEnergy, brute_force);
    EXPECT_EQ(
        energyForMask(problem, maskFromLabels(result.labels)),
        brute_force);
}

TEST(BinaryGridMinCutSolverTest, ProducesIdenticalLabelsAndStatisticsRepeatedly)
{
    auto problem = makeProblem(3, 2, 2);
    for (std::size_t index = 0; index < problem.nodeCount(); ++index)
    {
        problem.sourceCapacities[index] = static_cast<BinaryGridCapacity>((index * 7U) % 11U);
        problem.sinkCapacities[index] = static_cast<BinaryGridCapacity>((index * 5U + 3U) % 13U);
    }
    for (int z = 0; z < problem.sizeZ; ++z)
    {
        for (int y = 0; y < problem.sizeY; ++y)
        {
            for (int x = 0; x < problem.sizeX; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                if (x + 1 < problem.sizeX)
                {
                    problem.positiveXCapacities[index] = 3;
                }
                if (y + 1 < problem.sizeY)
                {
                    problem.positiveYCapacities[index] = 4;
                }
                if (z + 1 < problem.sizeZ)
                {
                    problem.positiveZCapacities[index] = 5;
                }
            }
        }
    }

    const auto first = BinaryGridMinCutSolver::solve(problem);
    const auto second = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(first.solved) << first.error;
    ASSERT_TRUE(second.solved) << second.error;
    EXPECT_EQ(first.labels, second.labels);
    EXPECT_EQ(first.statistics.maximumFlow, second.statistics.maximumFlow);
    EXPECT_EQ(first.statistics.cutEnergy, second.statistics.cutEnergy);
    EXPECT_EQ(first.statistics.pushCount, second.statistics.pushCount);
    EXPECT_EQ(first.statistics.relabelCount, second.statistics.relabelCount);
}

TEST(BinaryGridMinCutSolverTest, CancellationReturnsNoPartialLabels)
{
    auto problem = makeProblem(20, 20, 20);
    std::fill(problem.sourceCapacities.begin(), problem.sourceCapacities.end(), 2);
    std::fill(problem.sinkCapacities.begin(), problem.sinkCapacities.end(), 1);
    int cancellation_checks = 0;

    const auto result = BinaryGridMinCutSolver::solve(
        problem,
        [&cancellation_checks]()
        {
            ++cancellation_checks;
            return cancellation_checks >= 2;
        });

    EXPECT_FALSE(result.solved);
    EXPECT_TRUE(result.cancelled);
    EXPECT_TRUE(result.labels.empty());
    EXPECT_GE(cancellation_checks, 2);
}

TEST(BinaryGridMinCutSolverTest, RejectsNonZeroCapacityBeyondGridBoundary)
{
    auto problem = makeProblem(2, 1, 1);
    problem.positiveXCapacities[1] = 3;

    const auto result = BinaryGridMinCutSolver::solve(problem);

    EXPECT_FALSE(result.solved);
    EXPECT_FALSE(result.cancelled);
    EXPECT_TRUE(result.labels.empty());
    EXPECT_NE(result.error.find("boundary"), std::string::npos);
}

TEST(BinaryGridMinCutSolverTest, MediumThreeDimensionalGridHasBoundedSolverWork)
{
    constexpr int kSize = 56;
    constexpr int kInnerRadius = kSize / 4;
    constexpr int kOuterRadius = kSize * 3 / 8;
    auto problem = makeProblem(kSize, kSize, kSize);
    const int center = kSize / 2;
    for (int z = 0; z < kSize; ++z)
    {
        for (int y = 0; y < kSize; ++y)
        {
            for (int x = 0; x < kSize; ++x)
            {
                const std::size_t index = problem.index(x, y, z);
                const int dx = x - center;
                const int dy = y - center;
                const int dz = z - center;
                const int radius_squared = dx * dx + dy * dy + dz * dz;
                if (radius_squared <= kInnerRadius * kInnerRadius)
                {
                    problem.sourceCapacities[index] = 15;
                }
                else if (radius_squared >= kOuterRadius * kOuterRadius)
                {
                    problem.sinkCapacities[index] = 15;
                }
                else
                {
                    problem.sourceCapacities[index] = 4;
                    problem.sinkCapacities[index] = 4;
                }
                if (x + 1 < kSize)
                {
                    problem.positiveXCapacities[index] = 3;
                }
                if (y + 1 < kSize)
                {
                    problem.positiveYCapacities[index] = 3;
                }
                if (z + 1 < kSize)
                {
                    problem.positiveZCapacities[index] = 3;
                }
            }
        }
    }

    const auto result = BinaryGridMinCutSolver::solve(problem);

    ASSERT_TRUE(result.solved) << result.error;
    EXPECT_EQ(result.labels[problem.index(center, center, center)], BinaryGridLabel::Full);
    EXPECT_EQ(result.labels[problem.index(0, 0, 0)], BinaryGridLabel::Empty);
    EXPECT_LT(result.statistics.pushCount, problem.nodeCount() * 100U);
    EXPECT_LT(result.statistics.relabelCount, problem.nodeCount() * 20U);
    EXPECT_EQ(result.statistics.maximumFlow, result.statistics.cutEnergy);
}
