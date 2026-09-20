#include "RpcInstance.h"

#include "camera/core/types/CameraErrors.h"

#include <cmath>
#include <utility>

namespace xjw::camera_models::rpc
{

    RpcInstance RpcInstance::create(camera_core::CameraInstanceId instanceId,
                                    camera_core::ImageId imageId,
                                    std::shared_ptr<const RpcDefinition> definition,
                                    camera_core::ImageSize imageSize,
                                    ImageCorrection correction,
                                    std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        if (!definition)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                     "RPC instance requires a definition");
        }
        for (double value : {correction.sampleOffsetPixels,
                             correction.sampleSamplePixels,
                             correction.sampleLinePixels,
                             correction.lineOffsetPixels,
                             correction.lineSamplePixels,
                             correction.lineLinePixels})
        {
            if (!std::isfinite(value))
            {
                throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidFrame,
                                                         "RPC image correction must contain finite values");
            }
        }
        return RpcInstance(
            std::move(instanceId), std::move(imageId), std::move(definition), imageSize, correction, captureTime);
    }

    std::unique_ptr<RpcInstance> RpcInstance::createUnique(camera_core::CameraInstanceId instanceId,
                                                           camera_core::ImageId imageId,
                                                           std::shared_ptr<const RpcDefinition> definition,
                                                           camera_core::ImageSize imageSize,
                                                           ImageCorrection correction,
                                                           std::optional<xjw::coordinate_system::TimeReference> captureTime)
    {
        return std::unique_ptr<RpcInstance>(new RpcInstance(create(
            std::move(instanceId), std::move(imageId), std::move(definition), imageSize, correction, captureTime)));
    }

    const RpcDefinition& RpcInstance::rpcDefinition() const noexcept
    {
        return *_rpcDefinition;
    }

    const ImageCorrection& RpcInstance::imageCorrection() const noexcept
    {
        return _imageCorrection;
    }

    RpcInstance RpcInstance::withImageCorrection(camera_core::CameraInstanceId instanceId,
                                                 ImageCorrection correction) const
    {
        return create(std::move(instanceId), imageId(), _rpcDefinition, imageSize(), correction, captureTime());
    }

    RpcInstance::RpcInstance(camera_core::CameraInstanceId instanceId,
                             camera_core::ImageId imageId,
                             std::shared_ptr<const RpcDefinition> definition,
                             camera_core::ImageSize imageSize,
                             ImageCorrection correction,
                             std::optional<xjw::coordinate_system::TimeReference> captureTime)
        : camera_core::CameraInstance(std::move(instanceId),
                                      std::move(imageId),
                                      definition,
                                      imageSize,
                                      captureTime,
                                      camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                 camera_core::CapabilityKind::InverseProjection,
                                                                 camera_core::CapabilityKind::Ray,
                                                                 camera_core::CapabilityKind::ImageCorrection}),
          _rpcDefinition(std::move(definition)), _imageCorrection(correction)
    {
    }

} // namespace xjw::camera_models::rpc
