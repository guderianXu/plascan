#pragma once

#include "camera/core/model/CameraDefinition.h"

#include <array>
#include <memory>
#include <optional>

namespace xjw::camera_models::linescan
{

    enum class PixelConvention
    {
        PixelCenter,
        ZeroBased,
    };

    enum class LineScanDistortionModel
    {
        RadialNormalized,
        LroNacFocalPlane,
    };

    struct LineScanDetectorGeometry
    {
        double detectorSampleSumming = 1.0;
        double detectorLineSumming = 1.0;
        double detectorSampleOrigin = 0.0;
        double detectorLineOrigin = 0.0;
        double startingDetectorSample = 0.0;
        double startingDetectorLine = 0.0;
        std::array<double, 3> focalToPixelSamples{{0.0, 0.0, 1.0}};
        std::array<double, 3> focalToPixelLines{{0.0, 1.0, 0.0}};
    };

    struct LineScanOptics
    {
        double focalLengthMillimeters = 0.0;
        double samplePitchMillimeters = 0.0;
        double principalSample = 0.0;
        double distortionK1 = 0.0;
        LineScanDistortionModel distortionModel = LineScanDistortionModel::RadialNormalized;
        std::optional<LineScanDetectorGeometry> detectorGeometry;
    };

    class LineScanDefinition final : public camera_core::CameraDefinition
    {
    public:
        static constexpr int ParameterSchemaVersion = 2;

        static std::shared_ptr<const LineScanDefinition>
        create(camera_core::CameraDefinitionId definitionId,
               xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
               LineScanOptics optics,
               PixelConvention pixelConvention = PixelConvention::PixelCenter);
        static std::unique_ptr<LineScanDefinition>
        createUnique(camera_core::CameraDefinitionId definitionId,
                     xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
                     LineScanOptics optics,
                     PixelConvention pixelConvention = PixelConvention::PixelCenter);

        const LineScanOptics& optics() const noexcept;
        PixelConvention pixelConvention() const noexcept;

    private:
        LineScanDefinition(camera_core::CameraDefinitionId definitionId,
                           xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
                           LineScanOptics optics,
                           PixelConvention pixelConvention);

        static void validate(const LineScanOptics& optics);

        LineScanOptics _optics;
        PixelConvention _pixelConvention;
    };

} // namespace xjw::camera_models::linescan
