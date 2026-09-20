#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/core/model/CameraInstanceSet.h"
#include "camera/models/frame_pinhole/FramePinholeDefinition.h"
#include "camera/models/frame_pinhole/FramePinholeInstance.h"
#include "camera/models/frame_pinhole/FramePinholeNumericState.h"
#include "camera/models/linescan/LineScanDefinition.h"
#include "camera/models/linescan/LineScanInstance.h"
#include "camera/models/linescan/LineScanTrajectory.h"
#include "camera/models/rpc/RpcDefinition.h"
#include "camera/models/rpc/RpcInstance.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace
{

    using namespace xjw::camera_core;
    using namespace xjw::coordinate_system;
    using namespace xjw::camera_models;

    std::shared_ptr<const frame_pinhole::FramePinholeInstance> makePinhole()
    {
        const auto definition = frame_pinhole::FramePinholeDefinition::create(
            CameraDefinitionId("pinhole-definition"),
            frame_pinhole::Intrinsics{100.0, 100.0, 50.0, 50.0, 1.0, 1, 1},
            frame_pinhole::Distortion{},
            frame_pinhole::PixelConvention::PixelCenter,
            CoordinateFrameId("world"));
        return std::make_shared<const frame_pinhole::FramePinholeInstance>(frame_pinhole::FramePinholeInstance::create(
            CameraInstanceId("pinhole-instance"),
            ImageId("image-pinhole"),
            definition,
            ImageSize{100, 100},
            Pose::create(
                CoordinateFrameId("world"), {0.0, 0.0, 0.0}, Rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}})));
    }

    std::shared_ptr<const rpc::RpcInstance> makeRpc()
    {
        rpc::RpcDefinition::Parameters parameters;
        parameters.lineScale = 1.0;
        parameters.sampleScale = 1.0;
        parameters.latitudeScale = 1.0;
        parameters.longitudeScale = 1.0;
        parameters.heightScale = 1.0;
        parameters.lineDenominator[0] = 1.0;
        parameters.sampleDenominator[0] = 1.0;
        return std::make_shared<const rpc::RpcInstance>(rpc::RpcInstance::create(
            CameraInstanceId("rpc-instance"),
            ImageId("image-rpc"),
            rpc::RpcDefinition::create(
                CameraDefinitionId("rpc-definition"), CoordinateFrameId("wgs84-geodetic"), parameters),
            ImageSize{100, 100}));
    }

    TEST(CameraCapabilityRequirementsTest, NumericStateAcceptsCapabilityCompletePinholeInstance)
    {
        const auto pinhole = makePinhole();
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        std::string error;

        ASSERT_TRUE(xjw::camera_models::frame_pinhole::FramePinholeNumericState::fromInstance(pinhole, &state, &error))
            << error;
        EXPECT_EQ(state.imageId().value(), "image-pinhole");
        EXPECT_TRUE(state.isValid());
    }

    TEST(CameraCapabilityRequirementsTest, NumericStateRejectsRpcAtPinholeBoundary)
    {
        const auto rpcInstance = makeRpc();
        xjw::camera_models::frame_pinhole::FramePinholeNumericState state;
        std::string error;

        EXPECT_FALSE(
            xjw::camera_models::frame_pinhole::FramePinholeNumericState::fromInstance(rpcInstance, &state, &error));
        EXPECT_NE(error.find("frame-pinhole"), std::string::npos);
        EXPECT_FALSE(state.isValid());
    }

    TEST(CameraCapabilityRequirementsTest, InstanceSetRejectsDuplicateImageAndReportsMixedCapabilities)
    {
        const auto pinhole = makePinhole();
        const auto rpcInstance = makeRpc();
        CameraInstanceSet instances;
        std::string error;
        ASSERT_TRUE(instances.add(pinhole, &error)) << error;
        ASSERT_TRUE(instances.add(rpcInstance, &error)) << error;
        EXPECT_FALSE(instances.add(pinhole, &error));
        EXPECT_NE(error.find("image-pinhole"), std::string::npos);

        const CameraInstanceSetCapabilityResult check =
            instances.requireCapabilities(CapabilitySet{CapabilityKind::Projection, CapabilityKind::StaticPose});
        ASSERT_FALSE(check.ok());
        ASSERT_EQ(check.failures.size(), 1U);
        EXPECT_EQ(check.failures.front().image.value(), "image-rpc");
        EXPECT_NE(check.failures.front().message.find("static_pose"), std::string::npos);

        const CameraInstanceLookupResult missing = instances.forImage(ImageId("missing-image"));
        EXPECT_FALSE(missing.ok());
        EXPECT_NE(missing.error.find("missing-image"), std::string::npos);
    }

} // namespace
