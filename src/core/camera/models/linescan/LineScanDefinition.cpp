#include "LineScanDefinition.h"

#include "camera/core/types/CameraErrors.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace xjw::camera_models::linescan
{

    std::shared_ptr<const LineScanDefinition> LineScanDefinition::create(camera_core::CameraDefinitionId definitionId,
                                                                         xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
                                                                         LineScanOptics optics,
                                                                         PixelConvention pixelConvention)
    {
        validate(optics);
        return std::shared_ptr<const LineScanDefinition>(
            new LineScanDefinition(std::move(definitionId), std::move(bodyFixedFrame), optics, pixelConvention));
    }

    std::unique_ptr<LineScanDefinition> LineScanDefinition::createUnique(camera_core::CameraDefinitionId definitionId,
                                                                         xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
                                                                         LineScanOptics optics,
                                                                         PixelConvention pixelConvention)
    {
        validate(optics);
        return std::unique_ptr<LineScanDefinition>(
            new LineScanDefinition(std::move(definitionId), std::move(bodyFixedFrame), optics, pixelConvention));
    }

    const LineScanOptics& LineScanDefinition::optics() const noexcept
    {
        return _optics;
    }

    PixelConvention LineScanDefinition::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }

    LineScanDefinition::LineScanDefinition(camera_core::CameraDefinitionId definitionId,
                                           xjw::coordinate_system::CoordinateFrameId bodyFixedFrame,
                                           LineScanOptics optics,
                                           PixelConvention pixelConvention)
        : camera_core::CameraDefinition(std::move(definitionId),
                                        "planetary_linescan",
                                        std::move(bodyFixedFrame),
                                        ParameterSchemaVersion,
                                        camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                   camera_core::CapabilityKind::Ray,
                                                                   camera_core::CapabilityKind::Trajectory,
                                                                   camera_core::CapabilityKind::Optimization}),
          _optics(optics), _pixelConvention(pixelConvention)
    {
    }

    void LineScanDefinition::validate(const LineScanOptics& optics)
    {
        const bool simpleOpticsValid = std::isfinite(optics.samplePitchMillimeters) &&
                                       std::isfinite(optics.principalSample) && optics.samplePitchMillimeters > 0.0;
        bool detectorGeometryValid = true;
        if (optics.detectorGeometry)
        {
            const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
            const std::array<double, 10> scalars{{detector.detectorSampleSumming,
                                                  detector.detectorLineSumming,
                                                  detector.detectorSampleOrigin,
                                                  detector.detectorLineOrigin,
                                                  detector.startingDetectorSample,
                                                  detector.startingDetectorLine,
                                                  detector.focalToPixelSamples[0],
                                                  detector.focalToPixelSamples[1],
                                                  detector.focalToPixelSamples[2],
                                                  detector.focalToPixelLines[0]}};
            detectorGeometryValid =
                std::all_of(scalars.begin(), scalars.end(), [](double value) { return std::isfinite(value); }) &&
                std::isfinite(detector.focalToPixelLines[1]) && std::isfinite(detector.focalToPixelLines[2]) &&
                detector.detectorSampleSumming > 0.0 && detector.detectorLineSumming > 0.0;
            const double determinant = detector.focalToPixelLines[1] * detector.focalToPixelSamples[2] -
                                       detector.focalToPixelLines[2] * detector.focalToPixelSamples[1];
            detectorGeometryValid = detectorGeometryValid && std::abs(determinant) >= 1.0e-15;
        }
        if (!std::isfinite(optics.focalLengthMillimeters) || !std::isfinite(optics.distortionK1) ||
            optics.focalLengthMillimeters <= 0.0 || (!optics.detectorGeometry && !simpleOpticsValid) ||
            !detectorGeometryValid)
        {
            throw camera_core::CameraValidationError(
                camera_core::CameraErrorCode::InvalidIntrinsics,
                "line-scan optics must define finite focal length, distortion and a valid detector mapping");
        }
    }

} // namespace xjw::camera_models::linescan
