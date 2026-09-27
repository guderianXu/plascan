#include <placamera/frame_numeric_state.h>

#include <placamera/frame_camera.h>
#include <placamera/instance_set.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace
{

    std::shared_ptr<const placamera::FramePinholeModel> makePinhole(const std::string& imageId)
    {
        const placamera::FrameId frame("world");
        const auto definition =
            placamera::FramePinholeDefinition::create(placamera::CameraDefinitionId("definition-" + imageId),
                                                      placamera::FrameIntrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
                                                      placamera::BrownConradyDistortion{},
                                                      placamera::PixelConvention::PixelCenter,
                                                      frame);
        return std::make_shared<const placamera::FramePinholeModel>(placamera::FramePinholeModel::create(
            placamera::CameraInstanceId("instance-" + imageId),
            placamera::ImageId(imageId),
            definition,
            placamera::ImageSize{100, 100},
            placamera::Pose::create(frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})));
    }

    TEST(CameraCapabilityRequirementsTest, NumericStateAcceptsPlaCameraPinholeModel)
    {
        const auto pinhole = makePinhole("image-pinhole");
        const auto state = placamera::FramePinholeNumericState::fromModel(*pinhole);
        EXPECT_EQ(state.imageId().value(), "image-pinhole");
        EXPECT_TRUE(state.imageSize().isValid());
    }

    TEST(CameraCapabilityRequirementsTest, PlaCameraSetRejectsDuplicateImageAndReportsMissingCapabilities)
    {
        placamera::CameraInstanceSet instances;
        const auto first = makePinhole("image-first");
        const auto second = makePinhole("image-second");
        ASSERT_TRUE(instances.add(first).ok());
        ASSERT_TRUE(instances.add(second).ok());
        const auto duplicate = instances.add(first);
        EXPECT_FALSE(duplicate.ok());
        EXPECT_EQ(duplicate.errorCode(), placamera::CameraErrorCode::InvalidModelState);

        const auto capabilityCheck = instances.requireCapabilities({placamera::CapabilityKind::Trajectory});
        ASSERT_FALSE(capabilityCheck.ok());
        ASSERT_EQ(capabilityCheck.failures.size(), 2U);
        EXPECT_EQ(capabilityCheck.failures.at(0).imageId.value(), "image-first");
        EXPECT_EQ(capabilityCheck.failures.at(0).errorCode, placamera::CameraErrorCode::UnsupportedModel);
        EXPECT_EQ(capabilityCheck.failures.at(1).imageId.value(), "image-second");

        const auto missing = instances.forImage(placamera::ImageId("missing-image"));
        EXPECT_FALSE(missing.ok());
        EXPECT_EQ(missing.errorCode(), placamera::CameraErrorCode::InvalidArgument);
    }

} // namespace
