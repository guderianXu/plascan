#include "CameraInstance.h"

#include "../types/CameraErrors.h"

#include <utility>

namespace xjw::camera_core
{

    CameraInstance::CameraInstance(CameraInstanceId id,
                                   ImageId image,
                                   std::shared_ptr<const CameraDefinition> definition,
                                   ImageSize imageSize,
                                   std::optional<xjw::coordinate_system::TimeReference> captureTime,
                                   CapabilitySet capabilities)
        : _instanceId(std::move(id)), _imageId(std::move(image)), _definition(std::move(definition)),
          _imageSize(imageSize), _captureTime(captureTime), _capabilities(std::move(capabilities))
    {
        if (!_definition)
        {
            throw CameraValidationError(CameraErrorCode::InvalidFrame, "camera instance requires a definition");
        }
        if (!_imageSize.isValid())
        {
            throw CameraValidationError(CameraErrorCode::InvalidFrame, "camera instance image size must be positive");
        }
    }

    const CameraInstanceId& CameraInstance::instanceId() const noexcept
    {
        return _instanceId;
    }

    const ImageId& CameraInstance::imageId() const noexcept
    {
        return _imageId;
    }

    const CameraDefinition& CameraInstance::definition() const noexcept
    {
        return *_definition;
    }

    const ImageSize& CameraInstance::imageSize() const noexcept
    {
        return _imageSize;
    }

    const std::optional<xjw::coordinate_system::TimeReference>& CameraInstance::captureTime() const noexcept
    {
        return _captureTime;
    }

    const CapabilitySet& CameraInstance::capabilities() const noexcept
    {
        return _capabilities;
    }

} // namespace xjw::camera_core
