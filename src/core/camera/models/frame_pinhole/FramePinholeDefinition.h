#pragma once

#include "camera/core/model/CameraDefinition.h"

#include <array>
#include <memory>

namespace xjw::camera_models::frame_pinhole
{

    enum class PixelConvention
    {
        PixelCenter,
        PixelCorner,
    };

    struct Intrinsics
    {
        double focalX = 0.0;
        double focalY = 0.0;
        double principalX = 0.0;
        double principalY = 0.0;
        double pixelPitch = 1.0;
        int uAxisSign = 1;
        int vAxisSign = 1;
    };

    struct Distortion
    {
        double radialK1 = 0.0;
        double radialK2 = 0.0;
        double radialK3 = 0.0;
        double tangentialP1 = 0.0;
        double tangentialP2 = 0.0;
    };

    class FramePinholeDefinition final : public camera_core::CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 1;

        static std::shared_ptr<const FramePinholeDefinition> create(camera_core::CameraDefinitionId definitionId,
                                                                    Intrinsics intrinsics,
                                                                    Distortion distortion,
                                                                    PixelConvention pixelConvention,
                                                                    xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                                    bool depthAxisFlipped = false);
        static std::unique_ptr<FramePinholeDefinition> createUnique(camera_core::CameraDefinitionId definitionId,
                                                                    Intrinsics intrinsics,
                                                                    Distortion distortion,
                                                                    PixelConvention pixelConvention,
                                                                    xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                                    bool depthAxisFlipped = false);

        const Intrinsics& intrinsics() const noexcept;
        const Distortion& distortion() const noexcept;
        PixelConvention pixelConvention() const noexcept;
        bool depthAxisFlipped() const noexcept;

        // Sign matrix used when deriving a positive-depth pose. Its determinant
        // is kept positive so the shared Pose type remains a proper rotation.
        std::array<double, 3> positiveDepthAxisSigns() const noexcept;

        std::shared_ptr<const FramePinholeDefinition>
        scaledIntrinsics(camera_core::CameraDefinitionId definitionId, double scaleX, double scaleY) const;

        std::shared_ptr<const FramePinholeDefinition>
        normalizedForPositiveDepth(camera_core::CameraDefinitionId definitionId) const;

    private:
        FramePinholeDefinition(camera_core::CameraDefinitionId definitionId,
                               Intrinsics intrinsics,
                               Distortion distortion,
                               PixelConvention pixelConvention,
                               xjw::coordinate_system::CoordinateFrameId worldFrame,
                               bool depthAxisFlipped);

        static void validate(const Intrinsics& intrinsics, const Distortion& distortion);

        Intrinsics _intrinsics;
        Distortion _distortion;
        PixelConvention _pixelConvention;
        bool _depthAxisFlipped = false;
    };

} // namespace xjw::camera_models::frame_pinhole
