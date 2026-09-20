#pragma once

#include "RpcDefinition.h"

#include "camera/core/model/CameraInstance.h"

#include <memory>
#include <optional>

namespace xjw::camera_models::rpc
{

    struct ImageCorrection
    {
        double sampleOffsetPixels = 0.0;
        double sampleSamplePixels = 0.0;
        double sampleLinePixels = 0.0;
        double lineOffsetPixels = 0.0;
        double lineSamplePixels = 0.0;
        double lineLinePixels = 0.0;
    };

    class RpcInstance final : public camera_core::CameraInstance
    {
    public:
        static RpcInstance create(camera_core::CameraInstanceId instanceId,
                                  camera_core::ImageId imageId,
                                  std::shared_ptr<const RpcDefinition> definition,
                                  camera_core::ImageSize imageSize,
                                  ImageCorrection correction = {},
                                  std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);
        static std::unique_ptr<RpcInstance>
        createUnique(camera_core::CameraInstanceId instanceId,
                     camera_core::ImageId imageId,
                     std::shared_ptr<const RpcDefinition> definition,
                     camera_core::ImageSize imageSize,
                     ImageCorrection correction = {},
                     std::optional<xjw::coordinate_system::TimeReference> captureTime = std::nullopt);

        const RpcDefinition& rpcDefinition() const noexcept;
        const ImageCorrection& imageCorrection() const noexcept;

        RpcInstance withImageCorrection(camera_core::CameraInstanceId instanceId, ImageCorrection correction) const;

    private:
        RpcInstance(camera_core::CameraInstanceId instanceId,
                    camera_core::ImageId imageId,
                    std::shared_ptr<const RpcDefinition> definition,
                    camera_core::ImageSize imageSize,
                    ImageCorrection correction,
                    std::optional<xjw::coordinate_system::TimeReference> captureTime);

        std::shared_ptr<const RpcDefinition> _rpcDefinition;
        ImageCorrection _imageCorrection;
    };

} // namespace xjw::camera_models::rpc
