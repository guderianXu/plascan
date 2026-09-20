#pragma once

#include "CameraDefinition.h"
#include "coordinate_system/types/TimeReference.h"

#include <memory>
#include <optional>

namespace xjw::camera_core
{

    struct ImageSize
    {
        int samples = 0;
        int lines = 0;

        bool isValid() const noexcept
        {
            return samples > 0 && lines > 0;
        }
    };

    class CameraInstance
    {
    public:
        CameraInstance(CameraInstanceId id,
                       ImageId image,
                       std::shared_ptr<const CameraDefinition> definition,
                       ImageSize imageSize,
                       std::optional<xjw::coordinate_system::TimeReference> captureTime,
                       CapabilitySet capabilities);
        virtual ~CameraInstance() = default;

        const CameraInstanceId& instanceId() const noexcept;
        const ImageId& imageId() const noexcept;
        const CameraDefinition& definition() const noexcept;
        const ImageSize& imageSize() const noexcept;
        const std::optional<xjw::coordinate_system::TimeReference>& captureTime() const noexcept;
        const CapabilitySet& capabilities() const noexcept;

    private:
        CameraInstanceId _instanceId;
        ImageId _imageId;
        std::shared_ptr<const CameraDefinition> _definition;
        ImageSize _imageSize;
        std::optional<xjw::coordinate_system::TimeReference> _captureTime;
        CapabilitySet _capabilities;
    };

} // namespace xjw::camera_core
