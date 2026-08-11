#include <gtest/gtest.h>

#include <array>

#include "CameraBaseline.h"

namespace xjw
{
namespace
{

FramePinholeCamera makeCamera(const std::array<double, 3> &center, bool depthAxisFlipped = false)
{
    FramePinholeCamera camera;
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   center);
    camera.setDepthAxisFlipped(depthAxisFlipped);
    return camera;
}

} // namespace

TEST(CameraBaselineTest, CalculatesPhysicalCameraCenterDistance)
{
    const FramePinholeCamera first = makeCamera({{0.0, 0.0, 0.0}});
    const FramePinholeCamera second = makeCamera({{3.0, 4.0, 0.0}});

    const CameraBaseline baseline = CameraBaseline::evaluate(first, second);
    EXPECT_TRUE(baseline.isValid());
    EXPECT_DOUBLE_EQ(baseline.length(), 5.0);
    EXPECT_FALSE(baseline.hasPointGeometry());
    EXPECT_FALSE(baseline.triangulationAngleDeg().has_value());
}

TEST(CameraBaselineTest, CalculatesPointGeometryAndDepthToBaselineRatio)
{
    const FramePinholeCamera first = makeCamera({{0.0, 0.0, 0.0}});
    const FramePinholeCamera second = makeCamera({{1.0, 0.0, 0.0}});

    const CameraBaseline baseline = CameraBaseline::evaluate(first, second, {{0.0, 0.0, 10.0}});
    ASSERT_TRUE(baseline.isValid());
    ASSERT_TRUE(baseline.hasPointGeometry());
    ASSERT_TRUE(baseline.triangulationAngleDeg().has_value());
    ASSERT_TRUE(baseline.meanDepthToBaselineRatio().has_value());
    EXPECT_NEAR(*baseline.triangulationAngleDeg(), 5.7105931375, 1e-8);
    EXPECT_TRUE(baseline.isPointInFrontOfBothCameras());
    EXPECT_NEAR(*baseline.meanDepthToBaselineRatio(), 10.0, 1e-12);
}

TEST(CameraBaselineTest, RespectsFlippedPhysicalDepthAxis)
{
    const FramePinholeCamera first = makeCamera({{0.0, 0.0, 0.0}}, true);
    const FramePinholeCamera second = makeCamera({{1.0, 0.0, 0.0}}, true);

    const CameraBaseline baseline = CameraBaseline::evaluate(first, second, {{0.0, 0.0, -10.0}});
    EXPECT_TRUE(baseline.isPointInFrontOfBothCameras());
    ASSERT_TRUE(baseline.meanDepthToBaselineRatio().has_value());
    EXPECT_NEAR(*baseline.meanDepthToBaselineRatio(), 10.0, 1e-12);
}

TEST(CameraBaselineTest, RejectsCoincidentCameraCenters)
{
    const FramePinholeCamera first = makeCamera({{1.0, 2.0, 3.0}});
    const FramePinholeCamera second = makeCamera({{1.0, 2.0, 3.0}});

    const CameraBaseline baseline = CameraBaseline::evaluate(first, second, {{1.0, 2.0, 10.0}});
    EXPECT_FALSE(baseline.isValid());
    EXPECT_DOUBLE_EQ(baseline.length(), 0.0);
    EXPECT_FALSE(baseline.hasPointGeometry());
}

} // namespace xjw
