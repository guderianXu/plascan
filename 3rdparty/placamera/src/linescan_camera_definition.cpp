#include "placamera/linescan_camera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace placamera
{

    std::shared_ptr<const LineScanDefinition> LineScanDefinition::create(CameraDefinitionId definitionId,
                                                                         FrameId groundFrame,
                                                                         LineScanOptics optics,
                                                                         LineScanPixelConvention pixelConvention)
    {
        validate(optics);
        return std::shared_ptr<const LineScanDefinition>(new LineScanDefinition(
            std::move(definitionId), std::move(groundFrame), std::move(optics), pixelConvention));
    }

    const CameraDefinitionId& LineScanDefinition::definitionId() const noexcept
    {
        return _definitionId;
    }

    std::string_view LineScanDefinition::modelType() const noexcept
    {
        return "planetary_linescan";
    }

    int LineScanDefinition::parameterSchemaVersion() const noexcept
    {
        return ParameterSchemaVersion;
    }

    const FrameId& LineScanDefinition::groundFrame() const noexcept
    {
        return _groundFrame;
    }

    const LineScanOptics& LineScanDefinition::optics() const noexcept
    {
        return _optics;
    }

    LineScanPixelConvention LineScanDefinition::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }

    CapabilitySet LineScanDefinition::capabilities() const noexcept
    {
        return {CapabilityKind::Projection,
                CapabilityKind::ImagingLocus,
                CapabilityKind::Trajectory,
                CapabilityKind::ImageCorrection,
                CapabilityKind::Optimization};
    }

    LineScanDefinition::LineScanDefinition(CameraDefinitionId definitionId,
                                           FrameId groundFrame,
                                           LineScanOptics optics,
                                           LineScanPixelConvention pixelConvention)
        : _definitionId(std::move(definitionId)), _groundFrame(std::move(groundFrame)), _optics(std::move(optics)),
          _pixelConvention(pixelConvention)
    {
    }

    void LineScanDefinition::validate(const LineScanOptics& optics)
    {
        const bool simple_optics_valid = std::isfinite(optics.samplePitchMillimeters) &&
                                         std::isfinite(optics.principalSample) && optics.samplePitchMillimeters > 0.0;
        bool complete_calibration_valid = true;
        if (optics.completeCalibration)
        {
            const MetashapeCalibration& calibration = *optics.completeCalibration;
            const std::array<double, 13> values{{calibration.f,
                                                 calibration.cx,
                                                 calibration.cy,
                                                 calibration.b1,
                                                 calibration.b2,
                                                 calibration.k1,
                                                 calibration.k2,
                                                 calibration.k3,
                                                 calibration.k4,
                                                 calibration.p1,
                                                 calibration.p2,
                                                 calibration.p3,
                                                 calibration.p4}};
            complete_calibration_valid =
                std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }) &&
                calibration.f > 0.0 && calibration.f + calibration.b1 > 0.0;
            if (calibration.principalPointDecomposition)
            {
                const PrincipalPointDecomposition& principal = *calibration.principalPointDecomposition;
                const std::array<double, 4> principal_values{
                    {principal.imageCenterX, principal.imageCenterY, principal.cxOffset, principal.cyOffset}};
                complete_calibration_valid =
                    complete_calibration_valid && std::all_of(principal_values.begin(),
                                                              principal_values.end(),
                                                              [](double value) { return std::isfinite(value); });
            }
        }
        bool detector_geometry_valid = true;
        if (optics.detectorGeometry)
        {
            const LineScanDetectorGeometry& detector = *optics.detectorGeometry;
            const std::array<double, 12> scalars{{detector.detectorSampleSumming,
                                                  detector.detectorLineSumming,
                                                  detector.detectorSampleOrigin,
                                                  detector.detectorLineOrigin,
                                                  detector.startingDetectorSample,
                                                  detector.startingDetectorLine,
                                                  detector.focalToPixelSamples[0],
                                                  detector.focalToPixelSamples[1],
                                                  detector.focalToPixelSamples[2],
                                                  detector.focalToPixelLines[0],
                                                  detector.focalToPixelLines[1],
                                                  detector.focalToPixelLines[2]}};
            detector_geometry_valid =
                std::all_of(scalars.begin(), scalars.end(), [](double value) { return std::isfinite(value); }) &&
                detector.detectorSampleSumming > 0.0 && detector.detectorLineSumming > 0.0;
            const double determinant = detector.focalToPixelLines[1] * detector.focalToPixelSamples[2] -
                                       detector.focalToPixelLines[2] * detector.focalToPixelSamples[1];
            detector_geometry_valid = detector_geometry_valid && std::abs(determinant) >= 1.0e-15;
        }

        const bool pitch_valid_for_complete_detector =
            !optics.completeCalibration || !optics.detectorGeometry || simple_optics_valid;
        const bool legacy_mapping_valid = optics.completeCalibration || optics.detectorGeometry || simple_optics_valid;
        if (!std::isfinite(optics.focalLengthMillimeters) || !std::isfinite(optics.distortionK1) ||
            !std::isfinite(optics.principalSample) || !std::isfinite(optics.samplePitchMillimeters) ||
            optics.focalLengthMillimeters <= 0.0 || !legacy_mapping_valid || !pitch_valid_for_complete_detector ||
            !detector_geometry_valid || !complete_calibration_valid)
        {
            throw CameraValidationError(
                CameraErrorCode::InvalidIntrinsics,
                "line-scan optics must define finite physical optics and a valid complete or legacy detector mapping");
        }
    }

    LineScanTrajectoryBias applyLineScanTrajectoryBiasUpdate(const LineScanTrajectoryBias& current,
                                                             const std::array<double, 7>& delta)
    {
        const auto finite_vector = [](const Vector3& values)
        { return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }); };
        if (!finite_vector(current.translationMeters) || !finite_vector(current.rotationVectorRadians) ||
            !std::isfinite(current.timeOffsetSeconds) ||
            !std::all_of(delta.begin(), delta.end(), [](double value) { return std::isfinite(value); }))
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "line-scan trajectory bias and update must contain finite values");
        }

        LineScanTrajectoryBias updated = current;
        for (int index = 0; index < 3; ++index)
        {
            updated.translationMeters[static_cast<std::size_t>(index)] += delta[static_cast<std::size_t>(index)];
            updated.rotationVectorRadians[static_cast<std::size_t>(index)] +=
                delta[static_cast<std::size_t>(index + 3)];
        }
        updated.timeOffsetSeconds += delta[6];
        return updated;
    }

} // namespace placamera
