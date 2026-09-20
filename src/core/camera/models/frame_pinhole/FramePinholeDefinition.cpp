#include "FramePinholeDefinition.h"

#include "camera/core/types/CameraErrors.h"

#include <cmath>
#include <utility>

namespace xjw::camera_models::frame_pinhole
{
    namespace
    {

        bool finite(double value)
        {
            return std::isfinite(value);
        }

    } // namespace

    std::shared_ptr<const FramePinholeDefinition>
    FramePinholeDefinition::create(camera_core::CameraDefinitionId definitionId,
                                   Intrinsics intrinsics,
                                   Distortion distortion,
                                   PixelConvention pixelConvention,
                                   xjw::coordinate_system::CoordinateFrameId worldFrame,
                                   bool depthAxisFlipped)
    {
        validate(intrinsics, distortion);
        return std::shared_ptr<const FramePinholeDefinition>(new FramePinholeDefinition(std::move(definitionId),
                                                                                        std::move(intrinsics),
                                                                                        std::move(distortion),
                                                                                        pixelConvention,
                                                                                        std::move(worldFrame),
                                                                                        depthAxisFlipped));
    }

    std::unique_ptr<FramePinholeDefinition>
    FramePinholeDefinition::createUnique(camera_core::CameraDefinitionId definitionId,
                                         Intrinsics intrinsics,
                                         Distortion distortion,
                                         PixelConvention pixelConvention,
                                         xjw::coordinate_system::CoordinateFrameId worldFrame,
                                         bool depthAxisFlipped)
    {
        validate(intrinsics, distortion);
        return std::unique_ptr<FramePinholeDefinition>(new FramePinholeDefinition(std::move(definitionId),
                                                                                  std::move(intrinsics),
                                                                                  std::move(distortion),
                                                                                  pixelConvention,
                                                                                  std::move(worldFrame),
                                                                                  depthAxisFlipped));
    }

    const Intrinsics& FramePinholeDefinition::intrinsics() const noexcept
    {
        return _intrinsics;
    }

    const Distortion& FramePinholeDefinition::distortion() const noexcept
    {
        return _distortion;
    }

    PixelConvention FramePinholeDefinition::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }

    bool FramePinholeDefinition::depthAxisFlipped() const noexcept
    {
        return _depthAxisFlipped;
    }

    std::array<double, 3> FramePinholeDefinition::positiveDepthAxisSigns() const noexcept
    {
        const double zSign = _depthAxisFlipped ? -1.0 : 1.0;
        std::array<double, 3> signs{zSign * static_cast<double>(_intrinsics.uAxisSign),
                                    zSign * static_cast<double>(_intrinsics.vAxisSign),
                                    zSign};
        if (signs[0] * signs[1] * signs[2] < 0.0)
        {
            // A reflection cannot be represented by camera_core::Pose. Keep
            // the physical depth axis and carry the remaining parity in the
            // derived pixel-axis sign instead.
            signs[0] = -signs[0];
        }
        return signs;
    }

    std::shared_ptr<const FramePinholeDefinition> FramePinholeDefinition::scaledIntrinsics(
        camera_core::CameraDefinitionId definitionId, double scaleX, double scaleY) const
    {
        if (!finite(scaleX) || !finite(scaleY) || scaleX <= 0.0 || scaleY <= 0.0)
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidIntrinsics,
                                                     "pinhole intrinsic scale must be finite and positive");
        }

        Intrinsics scaled = _intrinsics;
        scaled.focalX *= scaleX;
        scaled.focalY *= scaleY;
        scaled.principalX = (scaled.principalX + 0.5) * scaleX - 0.5;
        scaled.principalY = (scaled.principalY + 0.5) * scaleY - 0.5;
        return create(std::move(definitionId), scaled, _distortion, _pixelConvention, worldFrame(), _depthAxisFlipped);
    }

    std::shared_ptr<const FramePinholeDefinition>
    FramePinholeDefinition::normalizedForPositiveDepth(camera_core::CameraDefinitionId definitionId) const
    {
        Intrinsics normalized = _intrinsics;
        Distortion distortion = _distortion;
        const auto signs = positiveDepthAxisSigns();
        const double xRatioSign = signs[0] / signs[2];
        const double yRatioSign = signs[1] / signs[2];
        const int uSign = normalized.uAxisSign < 0 ? -1 : 1;
        const int vSign = normalized.vAxisSign < 0 ? -1 : 1;
        normalized.focalX = std::fabs(normalized.focalX);
        normalized.focalY = std::fabs(normalized.focalY);
        normalized.uAxisSign = uSign * static_cast<int>(xRatioSign);
        normalized.vAxisSign = vSign * static_cast<int>(yRatioSign);
        distortion.tangentialP1 = yRatioSign * distortion.tangentialP1;
        distortion.tangentialP2 = xRatioSign * distortion.tangentialP2;
        return create(std::move(definitionId), normalized, distortion, _pixelConvention, worldFrame(), false);
    }

    FramePinholeDefinition::FramePinholeDefinition(camera_core::CameraDefinitionId definitionId,
                                                   Intrinsics intrinsics,
                                                   Distortion distortion,
                                                   PixelConvention pixelConvention,
                                                   xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                   bool depthAxisFlipped)
        : camera_core::CameraDefinition(std::move(definitionId),
                                        "frame_pinhole",
                                        std::move(worldFrame),
                                        ParameterSchemaVersion,
                                        camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                   camera_core::CapabilityKind::InverseProjection,
                                                                   camera_core::CapabilityKind::Ray,
                                                                   camera_core::CapabilityKind::StaticPose,
                                                                   camera_core::CapabilityKind::ImageCorrection,
                                                                   camera_core::CapabilityKind::Optimization}),
          _intrinsics(std::move(intrinsics)), _distortion(std::move(distortion)), _pixelConvention(pixelConvention),
          _depthAxisFlipped(depthAxisFlipped)
    {
    }

    void FramePinholeDefinition::validate(const Intrinsics& intrinsics, const Distortion& distortion)
    {
        if (!finite(intrinsics.focalX) || !finite(intrinsics.focalY) || !finite(intrinsics.principalX) ||
            !finite(intrinsics.principalY) || !finite(intrinsics.pixelPitch) || intrinsics.focalX <= 0.0 ||
            intrinsics.focalY <= 0.0 || intrinsics.pixelPitch <= 0.0 ||
            (intrinsics.uAxisSign != 1 && intrinsics.uAxisSign != -1) ||
            (intrinsics.vAxisSign != 1 && intrinsics.vAxisSign != -1))
        {
            throw camera_core::CameraValidationError(
                camera_core::CameraErrorCode::InvalidIntrinsics,
                "pinhole intrinsics must be finite, positive, and use signed axes");
        }

        if (!finite(distortion.radialK1) || !finite(distortion.radialK2) || !finite(distortion.radialK3) ||
            !finite(distortion.tangentialP1) || !finite(distortion.tangentialP2))
        {
            throw camera_core::CameraValidationError(camera_core::CameraErrorCode::InvalidDistortion,
                                                     "pinhole distortion coefficients must be finite");
        }
    }

} // namespace xjw::camera_models::frame_pinhole
