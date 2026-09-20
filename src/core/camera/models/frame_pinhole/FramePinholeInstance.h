#pragma once

#include "FramePinholeDefinition.h"

#include "camera/core/model/CameraInstance.h"
#include "camera/core/capabilities/StaticPoseProvider.h"
#include "camera/core/types/CameraPose.h"

#include <array>
#include <memory>

namespace xjw::camera_models::frame_pinhole
{

    class FramePinholeInstance final : public camera_core::CameraInstance, public camera_core::StaticPoseProvider
    {
    public:
        static FramePinholeInstance create(camera_core::CameraInstanceId instanceId,
                                           camera_core::ImageId imageId,
                                           std::shared_ptr<const FramePinholeDefinition> definition,
                                           camera_core::ImageSize imageSize,
                                           camera_core::Pose pose,
                                           std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);
        static std::unique_ptr<FramePinholeInstance>
        createUnique(camera_core::CameraInstanceId instanceId,
                     camera_core::ImageId imageId,
                     std::shared_ptr<const FramePinholeDefinition> definition,
                     camera_core::ImageSize imageSize,
                     camera_core::Pose pose,
                     std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);

        const FramePinholeDefinition& pinholeDefinition() const noexcept;
        const camera_core::Pose& pose() const noexcept;
        const camera_core::Pose& staticPose() const noexcept override;

        FramePinholeInstance withPose(camera_core::CameraInstanceId instanceId, camera_core::Pose pose) const;

        FramePinholeInstance normalizedForPositiveDepth(camera_core::CameraDefinitionId definitionId,
                                                        camera_core::CameraInstanceId instanceId) const;

    private:
        FramePinholeInstance(camera_core::CameraInstanceId instanceId,
                             camera_core::ImageId imageId,
                             std::shared_ptr<const FramePinholeDefinition> definition,
                             camera_core::ImageSize imageSize,
                             camera_core::Pose pose,
                             std::optional<xjw::coordinate_system::TimeReference> captureTime);

        std::shared_ptr<const FramePinholeDefinition> _pinholeDefinition;
        camera_core::Pose _pose;
    };

} // namespace xjw::camera_models::frame_pinhole
