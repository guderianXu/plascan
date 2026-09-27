#include <gtest/gtest.h>

#include <array>
#include <string>

#include <placamera/camera_baseline.h>

namespace placamera
{
    namespace
    {

        placamera::FramePinholeModel makeCamera(const std::array<double, 3>& center,
                                                bool depthAxisFlipped = false,
                                                const std::string& frameName = "baseline-world")
        {
            const placamera::FrameId frame(frameName);
            const auto definition =
                placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("baseline-definition"),
                                                          placamera::FrameIntrinsics{100.0, 100.0, 32.0, 32.0},
                                                          {},
                                                          placamera::PixelConvention::PixelCenter,
                                                          frame,
                                                          depthAxisFlipped);
            return placamera::FramePinholeModel::create(
                placamera::CameraInstanceId("baseline-instance"),
                placamera::ImageId("baseline-image"),
                definition,
                {64, 64},
                placamera::Pose::create(frame, center, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        }

    } // namespace

    TEST(CameraBaselineTest, CalculatesPhysicalCameraCenterDistance)
    {
        const auto first = makeCamera({{0.0, 0.0, 0.0}});
        const auto second = makeCamera({{3.0, 4.0, 0.0}});

        const placamera::CameraBaseline baseline = placamera::CameraBaseline::evaluate(first, second);
        EXPECT_TRUE(baseline.isValid());
        EXPECT_DOUBLE_EQ(baseline.length(), 5.0);
        EXPECT_FALSE(baseline.hasPointGeometry());
        EXPECT_FALSE(baseline.triangulationAngleDeg().has_value());
    }

    TEST(CameraBaselineTest, CalculatesPointGeometryAndDepthToBaselineRatio)
    {
        const auto first = makeCamera({{0.0, 0.0, 0.0}});
        const auto second = makeCamera({{1.0, 0.0, 0.0}});

        const placamera::CameraBaseline baseline =
            placamera::CameraBaseline::evaluate(first, second, {{0.0, 0.0, 10.0}});
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
        const auto first = makeCamera({{0.0, 0.0, 0.0}}, true);
        const auto second = makeCamera({{1.0, 0.0, 0.0}}, true);

        const placamera::CameraBaseline baseline =
            placamera::CameraBaseline::evaluate(first, second, {{0.0, 0.0, -10.0}});
        EXPECT_TRUE(baseline.isPointInFrontOfBothCameras());
        ASSERT_TRUE(baseline.meanDepthToBaselineRatio().has_value());
        EXPECT_NEAR(*baseline.meanDepthToBaselineRatio(), 10.0, 1e-12);
    }

    TEST(CameraBaselineTest, RejectsCoincidentCameraCenters)
    {
        const auto first = makeCamera({{1.0, 2.0, 3.0}});
        const auto second = makeCamera({{1.0, 2.0, 3.0}});

        const placamera::CameraBaseline baseline =
            placamera::CameraBaseline::evaluate(first, second, {{1.0, 2.0, 10.0}});
        EXPECT_FALSE(baseline.isValid());
        EXPECT_DOUBLE_EQ(baseline.length(), 0.0);
        EXPECT_FALSE(baseline.hasPointGeometry());
    }

    TEST(CameraBaselineTest, RejectsDifferentCoordinateFrames)
    {
        const auto first = makeCamera({{0.0, 0.0, 0.0}}, false, "baseline-world");
        const auto second = makeCamera({{1.0, 0.0, 0.0}}, false, "another-world");

        const placamera::CameraBaseline baseline =
            placamera::CameraBaseline::evaluate(first, second, {{0.0, 0.0, 10.0}});
        EXPECT_FALSE(baseline.isValid());
        EXPECT_FALSE(baseline.hasPointGeometry());
    }

} // namespace placamera
