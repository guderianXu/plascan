#include <placamera/frame_camera.h>
#include <placamera/instance_set.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace
{

    std::shared_ptr<const placamera::FramePinholeModel> makeInstance(const char* image, const char* frameName)
    {
        const placamera::FrameId frame(frameName);
        const auto definition = placamera::FramePinholeDefinition::create(
            placamera::CameraDefinitionId(std::string("definition-") + image),
            placamera::FrameIntrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
            placamera::BrownConradyDistortion{},
            placamera::PixelConvention::PixelCenter,
            frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId(std::string("instance-") + image),
            placamera::ImageId(image),
            definition,
            placamera::ImageSize{640, 480},
            placamera::Pose::create(
                frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

    TEST(CameraInstanceSetTest, EmptySetHasNoCommonFrameFailure)
    {
        const placamera::CameraInstanceSet set;
        const auto result = set.requireCommonGroundFrame();
        EXPECT_TRUE(result.ok());
        EXPECT_FALSE(result.commonGroundFrame.has_value());
    }

    TEST(CameraInstanceSetTest, ReportsCommonFrameForSingleFrameSet)
    {
        placamera::CameraInstanceSet set;
        ASSERT_TRUE(set.add(makeInstance("image-1", "world")).ok());
        ASSERT_TRUE(set.add(makeInstance("image-2", "world")).ok());
        const auto result = set.requireCommonGroundFrame();
        ASSERT_TRUE(result.ok());
        ASSERT_TRUE(result.commonGroundFrame.has_value());
        EXPECT_EQ(result.commonGroundFrame->value(), "world");
    }

    TEST(CameraInstanceSetTest, ReportsEveryImageWithConflictingFrame)
    {
        placamera::CameraInstanceSet set;
        ASSERT_TRUE(set.add(makeInstance("image-1", "world-a")).ok());
        ASSERT_TRUE(set.add(makeInstance("image-2", "world-b")).ok());
        ASSERT_TRUE(set.add(makeInstance("image-3", "world-c")).ok());
        const auto result = set.requireCommonGroundFrame();
        ASSERT_FALSE(result.ok());
        ASSERT_EQ(result.failures.size(), 2U);
        EXPECT_EQ(result.failures.at(0).imageId.value(), "image-2");
        EXPECT_EQ(result.failures.at(1).imageId.value(), "image-3");
        EXPECT_FALSE(result.commonGroundFrame.has_value());
    }

} // namespace
