#include <placamera/frame_camera.h>
#include <placamera/instance_set.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

    using namespace placamera;

    std::shared_ptr<const RasterModel>
    makeModel(std::string instance, std::string image, std::string definition, std::string frame)
    {
        FrameIntrinsics intrinsics;
        intrinsics.focalX = 100.0;
        intrinsics.focalY = 100.0;
        const FrameId ground_frame(frame);
        const auto calibration = FramePinholeDefinition::create(
            CameraDefinitionId(std::move(definition)), intrinsics, {}, PixelConvention::PixelCenter, ground_frame);
        auto model = FramePinholeModel::create(
            CameraInstanceId(std::move(instance)),
            ImageId(std::move(image)),
            calibration,
            ImageSize{100, 100},
            Pose::create(ground_frame, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
        return std::make_shared<FramePinholeModel>(std::move(model));
    }

    TEST(CameraInstanceSetTest, AddsLooksUpAndSelectsByStableImageIdentity)
    {
        CameraInstanceSet instances;
        ASSERT_TRUE(instances.add(makeModel("instance-a", "image-a", "definition-a", "world")));
        ASSERT_TRUE(instances.add(makeModel("instance-b", "image-b", "definition-b", "world")));
        EXPECT_EQ(instances.size(), 2U);

        const auto found = instances.forImage(ImageId("image-b"));
        ASSERT_TRUE(found) << found.message();
        EXPECT_EQ(found.value()->instanceId(), CameraInstanceId("instance-b"));

        const auto selected = instances.select({ImageId("image-b"), ImageId("image-a")});
        ASSERT_TRUE(selected) << selected.message();
        ASSERT_EQ(selected.value().size(), 2U);
        EXPECT_EQ(selected.value().values()[0]->imageId(), ImageId("image-b"));
        EXPECT_EQ(selected.value().values()[1]->imageId(), ImageId("image-a"));
    }

    TEST(CameraInstanceSetTest, RejectsNullAndDuplicateImageOrInstanceIdentity)
    {
        CameraInstanceSet instances;
        EXPECT_FALSE(instances.add({}));
        ASSERT_TRUE(instances.add(makeModel("instance-a", "image-a", "definition-a", "world")));
        EXPECT_FALSE(instances.add(makeModel("instance-b", "image-a", "definition-b", "world")));
        EXPECT_FALSE(instances.add(makeModel("instance-a", "image-b", "definition-c", "world")));

        EXPECT_FALSE(instances.select({ImageId("image-a"), ImageId("image-a")}));
        EXPECT_FALSE(instances.select({ImageId("missing")}));
    }

    TEST(CameraInstanceSetTest, ValidatesCapabilitiesAndCommonGroundFrame)
    {
        CameraInstanceSet same_frame;
        ASSERT_TRUE(same_frame.add(makeModel("instance-a", "image-a", "definition-a", "world")));
        ASSERT_TRUE(same_frame.add(makeModel("instance-b", "image-b", "definition-b", "world")));
        const auto common = same_frame.requireCommonGroundFrame();
        ASSERT_TRUE(common.ok());
        ASSERT_TRUE(common.commonGroundFrame.has_value());
        EXPECT_EQ(*common.commonGroundFrame, FrameId("world"));

        const auto projections = same_frame.requireCapabilities({CapabilityKind::Projection});
        EXPECT_TRUE(projections.ok());
        const auto trajectories = same_frame.requireCapabilities({CapabilityKind::Trajectory});
        EXPECT_FALSE(trajectories.ok());
        EXPECT_EQ(trajectories.failures.size(), 2U);

        CameraInstanceSet mixed;
        ASSERT_TRUE(mixed.add(makeModel("instance-a", "image-a", "definition-a", "world")));
        ASSERT_TRUE(mixed.add(makeModel("instance-b", "image-b", "definition-b", "moon")));
        const auto mismatch = mixed.requireCommonGroundFrame();
        EXPECT_FALSE(mismatch.ok());
        EXPECT_FALSE(mismatch.commonGroundFrame.has_value());
        EXPECT_EQ(mismatch.failures.size(), 1U);
    }

} // namespace
