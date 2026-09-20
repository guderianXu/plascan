#include "FramePinholeInstance.h"

#include "camera/core/types/CameraErrors.h"

#include <utility>

namespace xjw::camera_models::frame_pinhole
{

    FramePinholeInstance FramePinholeInstance::create(camera_core::CameraInstanceId instanceId,
                                                      camera_core::ImageId imageId,
                                                      std::shared_ptr<const FramePinholeDefinition> definition,
                                                      camera_core::ImageSize imageSize,
                                                      camera_core::Pose pose,
                                                      std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        if (!definition)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                     "pinhole instance requires a definition");
        }
        if (pose.frame != definition->worldFrame())
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                     "pinhole pose frame must match the definition world frame");
        }
        return FramePinholeInstance(
            std::move(instanceId), std::move(imageId), std::move(definition), imageSize, std::move(pose), captureTime);
    }

    std::unique_ptr<FramePinholeInstance>
    FramePinholeInstance::createUnique(camera_core::CameraInstanceId instanceId,
                                       camera_core::ImageId imageId,
                                       std::shared_ptr<const FramePinholeDefinition> definition,
                                       camera_core::ImageSize imageSize,
                                       camera_core::Pose pose,
                                       std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        return std::unique_ptr<FramePinholeInstance>(new FramePinholeInstance(create(std::move(instanceId),
                                                                                     std::move(imageId),
                                                                                     std::move(definition),
                                                                                     imageSize,
                                                                                     std::move(pose),
                                                                                     captureTime)));
    }

    const FramePinholeDefinition& FramePinholeInstance::pinholeDefinition() const noexcept
    {
        return *_pinholeDefinition;
    }

    const camera_core::Pose& FramePinholeInstance::pose() const noexcept
    {
        return _pose;
    }

    const camera_core::Pose& FramePinholeInstance::staticPose() const noexcept
    {
        return _pose;
    }

    FramePinholeInstance FramePinholeInstance::withPose(camera_core::CameraInstanceId instanceId,
                                                        camera_core::Pose pose) const
    {
        return create(
            std::move(instanceId), imageId(), _pinholeDefinition, imageSize(), std::move(pose), captureTime());
    }

    FramePinholeInstance
    FramePinholeInstance::normalizedForPositiveDepth(camera_core::CameraDefinitionId definitionId,
                                                     camera_core::CameraInstanceId instanceId) const
    {
        const std::array<double, 3> axisSigns = _pinholeDefinition->positiveDepthAxisSigns();

        const std::array<double, 9>& cameraToWorld = _pose.cameraToWorldRotation;
        const std::array<double, 9> worldToCamera{cameraToWorld[0],
                                                  cameraToWorld[3],
                                                  cameraToWorld[6],
                                                  cameraToWorld[1],
                                                  cameraToWorld[4],
                                                  cameraToWorld[7],
                                                  cameraToWorld[2],
                                                  cameraToWorld[5],
                                                  cameraToWorld[8]};
        std::array<double, 9> normalizedWorldToCamera{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                normalizedWorldToCamera[static_cast<std::size_t>(row * 3 + column)] =
                    axisSigns[row] * worldToCamera[static_cast<std::size_t>(row * 3 + column)];
            }
        }

        std::array<double, 9> normalizedCameraToWorld{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                normalizedCameraToWorld[static_cast<std::size_t>(row * 3 + column)] =
                    normalizedWorldToCamera[static_cast<std::size_t>(column * 3 + row)];
            }
        }

        camera_core::Pose normalizedPose =
            camera_core::Pose::create(_pose.frame, _pose.center, normalizedCameraToWorld);
        return create(std::move(instanceId),
                      imageId(),
                      _pinholeDefinition->normalizedForPositiveDepth(std::move(definitionId)),
                      imageSize(),
                      std::move(normalizedPose),
                      captureTime());
    }

    FramePinholeInstance::FramePinholeInstance(camera_core::CameraInstanceId instanceId,
                                               camera_core::ImageId imageId,
                                               std::shared_ptr<const FramePinholeDefinition> definition,
                                               camera_core::ImageSize imageSize,
                                               camera_core::Pose pose,
                                               std::optional<xjw::coordinate_system::TimeReference> captureTime)
        : camera_core::CameraInstance(std::move(instanceId),
                                      std::move(imageId),
                                      definition,
                                      imageSize,
                                      captureTime,
                                      camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                 camera_core::CapabilityKind::InverseProjection,
                                                                 camera_core::CapabilityKind::Ray,
                                                                 camera_core::CapabilityKind::StaticPose,
                                                                 camera_core::CapabilityKind::ImageCorrection,
                                                                 camera_core::CapabilityKind::Optimization}),
          _pinholeDefinition(std::move(definition)), _pose(std::move(pose))
    {
    }

} // namespace xjw::camera_models::frame_pinhole
