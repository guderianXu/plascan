#include <gtest/gtest.h>

#include "registration/ControlNetworkSolver.h"

#include <array>
#include <algorithm>
#include <cmath>

namespace
{

using xjw::control_points::ControlNetworkPoint;
using xjw::control_points::MarkerRole;

std::array<double, 3> referencePoint(const std::array<double, 3> &local)
{
    // 已知真值：绕 Z 轴旋转 90 度、尺度 2、平移 (10, -5, 3)。
    return {{10.0 - 2.0 * local[1],
             -5.0 + 2.0 * local[0],
             3.0 + 2.0 * local[2]}};
}

ControlNetworkPoint point(const char *id,
                          const std::array<double, 3> &local,
                          MarkerRole role = MarkerRole::ControlPoint)
{
    ControlNetworkPoint value;
    value.markerId = id;
    value.role = role;
    value.estimatedPoint = local;
    value.referencePoint = referencePoint(local);
    value.sigma = {{0.01, 0.01, 0.01}};
    return value;
}

} // namespace

TEST(ControlNetworkSolverTest, RecoversWeightedSimilarityFromNonCollinearControls)
{
    xjw::control_points::ControlNetworkInput input;
    input.points = {
        point("C1", {{0.0, 0.0, 0.0}}),
        point("C2", {{1.0, 0.0, 0.0}}),
        point("C3", {{0.0, 1.0, 0.0}}),
        point("C4", {{0.0, 0.0, 1.0}}),
    };

    const auto result = xjw::control_points::solveControlNetwork(input);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_NEAR(result.transform.scale, 2.0, 1.0e-10);
    const auto mapped = result.transform.apply({{0.25, 0.5, 0.75}});
    const auto expected = referencePoint({{0.25, 0.5, 0.75}});
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(mapped[axis], expected[axis], 1.0e-9);
    }
    EXPECT_EQ(result.controlInlierCount, 4);
}

TEST(ControlNetworkSolverTest, RejectsCollinearControlGeometry)
{
    xjw::control_points::ControlNetworkInput input;
    input.points = {
        point("C1", {{0.0, 0.0, 0.0}}),
        point("C2", {{1.0, 0.0, 0.0}}),
        point("C3", {{2.0, 0.0, 0.0}}),
        point("C4", {{3.0, 0.0, 0.0}}),
    };

    const auto result = xjw::control_points::solveControlNetwork(input);

    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("共线"), std::string::npos);
}

TEST(ControlNetworkSolverTest, RejectsGrossControlOutlierAndExcludesCheckPointFromFit)
{
    xjw::control_points::ControlNetworkInput input;
    input.options.inlierThreshold = 0.08;
    input.points = {
        point("C1", {{0.0, 0.0, 0.0}}),
        point("C2", {{1.0, 0.0, 0.0}}),
        point("C3", {{0.0, 1.0, 0.0}}),
        point("C4", {{0.0, 0.0, 1.0}}),
        point("BAD", {{1.0, 1.0, 1.0}}),
        point("CHK", {{0.25, 0.25, 0.25}}, MarkerRole::CheckPoint),
    };
    input.points[4].referencePoint = {{100.0, -50.0, 20.0}};
    input.points[5].referencePoint[0] += 12.0;

    const auto result = xjw::control_points::solveControlNetwork(input);

    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_NEAR(result.transform.scale, 2.0, 1.0e-8);
    EXPECT_EQ(result.controlInlierCount, 4);
    ASSERT_EQ(result.checkPointResiduals.size(), 1);
    EXPECT_GT(result.checkPointResiduals.front().total, 10.0);
    const auto bad = std::find_if(result.controlResiduals.cbegin(),
                                  result.controlResiduals.cend(),
                                  [](const auto &residual)
                                  {
                                      return residual.markerId == "BAD";
                                  });
    ASSERT_NE(bad, result.controlResiduals.cend());
    EXPECT_FALSE(bad->inlier);
}
