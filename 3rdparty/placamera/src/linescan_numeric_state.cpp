#include "placamera/linescan_numeric_state.h"

#include "internal/linescan_numeric_projection.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iterator>
#include <string>
#include <utility>

namespace placamera
{
    namespace
    {

        void addBlock(OptimizationLayout* layout,
                      std::string name,
                      OptimizationParameterKind kind,
                      std::string unit,
                      std::size_t size,
                      bool affectsDefinition)
        {
            const std::size_t offset = layout->parameterCount();
            layout->blocks.push_back({std::move(name), kind, std::move(unit), offset, size, affectsDefinition});
        }

        Result<OptimizationLayout> makeLayout(const std::vector<TrajectorySample>& samples,
                                              const LineScanOptics& optics,
                                              const LineScanOptimizationSelection& selection)
        {
            if (!std::isfinite(selection.positionSecondDifferenceWeight) ||
                !std::isfinite(selection.rotationSecondDifferenceWeight) ||
                selection.positionSecondDifferenceWeight < 0.0 || selection.rotationSecondDifferenceWeight < 0.0)
            {
                return Result<OptimizationLayout>::failure(
                    CameraErrorCode::InvalidArgument, "line-scan smoothness weights must be finite and non-negative");
            }
            if (selection.detector.anyDetectorGeometryParameter() && !optics.detectorGeometry)
            {
                return Result<OptimizationLayout>::failure(
                    CameraErrorCode::InvalidArgument,
                    "line-scan detector-geometry parameters require detector geometry in the definition");
            }
            if (selection.calibration.any() && !optics.completeCalibration)
            {
                return Result<OptimizationLayout>::failure(
                    CameraErrorCode::InvalidArgument,
                    "complete line-scan calibration parameters require complete calibration in the definition");
            }

            OptimizationLayout layout;
            for (std::size_t index = 0; index < samples.size(); ++index)
            {
                const auto& constraints = samples[index].constraints;
                const std::string prefix = "trajectory.knots." + std::to_string(index);
                if (selection.knotPositions && !constraints.positionFixed)
                {
                    addBlock(&layout, prefix + ".position", OptimizationParameterKind::Translation, "m", 3, false);
                }
                if (selection.knotRotations && !constraints.rotationFixed)
                {
                    addBlock(&layout, prefix + ".rotation", OptimizationParameterKind::RotationVector, "rad", 3, false);
                }
            }
            if (selection.globalTranslation)
            {
                addBlock(&layout, "trajectory.translation", OptimizationParameterKind::Translation, "m", 3, false);
            }
            if (selection.globalRotation)
            {
                addBlock(&layout, "trajectory.rotation", OptimizationParameterKind::RotationVector, "rad", 3, false);
            }
            if (selection.timeOffset)
            {
                addBlock(&layout, "trajectory.time", OptimizationParameterKind::TimeOffset, "s", 1, false);
            }

            const auto& calibration = selection.calibration;
            if (calibration.f)
            {
                addBlock(&layout, "calibration.f", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (calibration.cx)
            {
                addBlock(&layout, "calibration.cx", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (calibration.cy)
            {
                addBlock(&layout, "calibration.cy", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (calibration.b1)
            {
                addBlock(&layout, "calibration.b1", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (calibration.b2)
            {
                addBlock(&layout, "calibration.b2", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (calibration.k1)
            {
                addBlock(&layout, "calibration.k1", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.k2)
            {
                addBlock(&layout, "calibration.k2", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.k3)
            {
                addBlock(&layout, "calibration.k3", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.k4)
            {
                addBlock(&layout, "calibration.k4", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.p1)
            {
                addBlock(&layout, "calibration.p1", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.p2)
            {
                addBlock(&layout, "calibration.p2", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.p3)
            {
                addBlock(&layout, "calibration.p3", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }
            if (calibration.p4)
            {
                addBlock(&layout, "calibration.p4", OptimizationParameterKind::Distortion, "normalized", 1, true);
            }

            const auto& mask = selection.detector;
            if (mask.focalLength)
            {
                addBlock(&layout, "detector.focal_length", OptimizationParameterKind::Intrinsics, "mm", 1, true);
            }
            if (mask.distortionK1)
            {
                addBlock(&layout, "detector.distortion_k1", OptimizationParameterKind::Distortion, "1", 1, true);
            }
            if (mask.samplePitch)
            {
                addBlock(&layout, "detector.sample_pitch", OptimizationParameterKind::Intrinsics, "mm", 1, true);
            }
            if (mask.principalSample)
            {
                addBlock(&layout, "detector.principal_sample", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (mask.detectorSampleSumming)
            {
                addBlock(&layout, "detector.sample_summing", OptimizationParameterKind::Intrinsics, "1", 1, true);
            }
            if (mask.detectorLineSumming)
            {
                addBlock(&layout, "detector.line_summing", OptimizationParameterKind::Intrinsics, "1", 1, true);
            }
            if (mask.detectorSampleOrigin)
            {
                addBlock(&layout, "detector.sample_origin", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (mask.detectorLineOrigin)
            {
                addBlock(&layout, "detector.line_origin", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (mask.startingDetectorSample)
            {
                addBlock(&layout, "detector.starting_sample", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (mask.startingDetectorLine)
            {
                addBlock(&layout, "detector.starting_line", OptimizationParameterKind::Intrinsics, "px", 1, true);
            }
            if (mask.focalToPixelSamples)
            {
                addBlock(&layout,
                         "detector.focal_to_pixel_samples",
                         OptimizationParameterKind::Intrinsics,
                         "mixed",
                         3,
                         true);
            }
            if (mask.focalToPixelLines)
            {
                addBlock(
                    &layout, "detector.focal_to_pixel_lines", OptimizationParameterKind::Intrinsics, "mixed", 3, true);
            }
            return Result<OptimizationLayout>::success(std::move(layout));
        }

        RotationMatrix multiplyRotation(const RotationMatrix& first, const RotationMatrix& second) noexcept
        {
            RotationMatrix result{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int inner = 0; inner < 3; ++inner)
                    {
                        result[static_cast<std::size_t>(row * 3 + column)] +=
                            first[static_cast<std::size_t>(row * 3 + inner)] *
                            second[static_cast<std::size_t>(inner * 3 + column)];
                    }
                }
            }
            return result;
        }

        RotationMatrix angleAxisRotation(const Vector3& rotation) noexcept
        {
            const double angle_squared =
                rotation[0] * rotation[0] + rotation[1] * rotation[1] + rotation[2] * rotation[2];
            if (angle_squared < 1.0e-20)
            {
                return {1.0, -rotation[2], rotation[1], rotation[2], 1.0, -rotation[0], -rotation[1], rotation[0], 1.0};
            }
            const double angle = std::sqrt(angle_squared);
            const double sine_over_angle = std::sin(angle) / angle;
            const double one_minus_cosine_over_angle_squared = (1.0 - std::cos(angle)) / angle_squared;
            const double x = rotation[0];
            const double y = rotation[1];
            const double z = rotation[2];
            return {1.0 - one_minus_cosine_over_angle_squared * (y * y + z * z),
                    one_minus_cosine_over_angle_squared * x * y - sine_over_angle * z,
                    one_minus_cosine_over_angle_squared * x * z + sine_over_angle * y,
                    one_minus_cosine_over_angle_squared * x * y + sine_over_angle * z,
                    1.0 - one_minus_cosine_over_angle_squared * (x * x + z * z),
                    one_minus_cosine_over_angle_squared * y * z - sine_over_angle * x,
                    one_minus_cosine_over_angle_squared * x * z - sine_over_angle * y,
                    one_minus_cosine_over_angle_squared * y * z + sine_over_angle * x,
                    1.0 - one_minus_cosine_over_angle_squared * (x * x + y * y)};
        }

        bool validOptics(const LineScanOptics& optics) noexcept
        {
            if (!std::isfinite(optics.focalLengthMillimeters) || !(optics.focalLengthMillimeters > 0.0) ||
                !std::isfinite(optics.distortionK1))
            {
                return false;
            }
            if (!std::isfinite(optics.samplePitchMillimeters) || !std::isfinite(optics.principalSample))
            {
                return false;
            }
            if (optics.completeCalibration)
            {
                const MetashapeCalibration& calibration = *optics.completeCalibration;
                const double values[] = {calibration.f,
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
                                         calibration.p4};
                if (!std::all_of(
                        std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) ||
                    !(calibration.f > 0.0) || !(calibration.f + calibration.b1 > 0.0))
                {
                    return false;
                }
                if (calibration.principalPointDecomposition)
                {
                    const PrincipalPointDecomposition& principal = *calibration.principalPointDecomposition;
                    const double principal_values[] = {
                        principal.imageCenterX, principal.imageCenterY, principal.cxOffset, principal.cyOffset};
                    if (!std::all_of(std::begin(principal_values),
                                     std::end(principal_values),
                                     [](double value) { return std::isfinite(value); }))
                    {
                        return false;
                    }
                }
            }
            if (!optics.detectorGeometry)
            {
                return optics.completeCalibration || optics.samplePitchMillimeters > 0.0;
            }
            if (optics.completeCalibration && !(optics.samplePitchMillimeters > 0.0))
            {
                return false;
            }
            const auto& detector = *optics.detectorGeometry;
            const double determinant = detector.focalToPixelLines[1] * detector.focalToPixelSamples[2] -
                                       detector.focalToPixelLines[2] * detector.focalToPixelSamples[1];
            const double values[] = {detector.detectorSampleSumming,
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
                                     detector.focalToPixelLines[2]};
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   detector.detectorSampleSumming > 0.0 && detector.detectorLineSumming > 0.0 &&
                   std::abs(determinant) >= 1.0e-15;
        }

    } // namespace

    bool LineScanCalibrationOptimizationMask::any() const noexcept
    {
        return f || cx || cy || b1 || b2 || k1 || k2 || k3 || k4 || p1 || p2 || p3 || p4;
    }

    bool LineScanDetectorOptimizationMask::anyDetectorGeometryParameter() const noexcept
    {
        return detectorSampleSumming || detectorLineSumming || detectorSampleOrigin || detectorLineOrigin ||
               startingDetectorSample || startingDetectorLine || focalToPixelSamples || focalToPixelLines;
    }

    bool LineScanDetectorOptimizationMask::any() const noexcept
    {
        return focalLength || distortionK1 || samplePitch || principalSample || anyDetectorGeometryParameter();
    }

    Result<LineScanNumericState> LineScanNumericState::fromModel(const LineScanModel& model,
                                                                 LineScanOptimizationSelection selection)
    {
        if (model.trajectory().frameComposed())
        {
            return Result<LineScanNumericState>::failure(
                CameraErrorCode::UnsupportedModel,
                "per-knot line-scan optimization requires a direct-sample trajectory");
        }
        const auto layout = makeLayout(model.trajectory().samples(), model.lineScanDefinition().optics(), selection);
        if (!layout)
        {
            return Result<LineScanNumericState>::failure(layout.error());
        }
        return Result<LineScanNumericState>::success(LineScanNumericState(model.instanceId(),
                                                                          model.definitionId(),
                                                                          model.imageId(),
                                                                          model.groundFrame(),
                                                                          model.imageSize(),
                                                                          model.captureTime(),
                                                                          model.trajectory().samples(),
                                                                          model.lineTiming(),
                                                                          model.trajectoryBias(),
                                                                          model.timeOffsetPrior(),
                                                                          model.lineScanDefinition().optics(),
                                                                          model.lineScanDefinition().pixelConvention(),
                                                                          std::move(selection),
                                                                          layout.value()));
    }

    const CameraInstanceId& LineScanNumericState::instanceId() const noexcept
    {
        return _instanceId;
    }
    const CameraDefinitionId& LineScanNumericState::definitionId() const noexcept
    {
        return _definitionId;
    }
    const ImageId& LineScanNumericState::imageId() const noexcept
    {
        return _imageId;
    }
    const FrameId& LineScanNumericState::groundFrame() const noexcept
    {
        return _groundFrame;
    }
    const ImageSize& LineScanNumericState::imageSize() const noexcept
    {
        return _imageSize;
    }
    const std::optional<TimeReference>& LineScanNumericState::captureTime() const noexcept
    {
        return _captureTime;
    }
    const std::vector<TrajectorySample>& LineScanNumericState::trajectorySamples() const noexcept
    {
        return _samples;
    }
    const LineTiming& LineScanNumericState::lineTiming() const noexcept
    {
        return _timing;
    }
    const LineScanTrajectoryBias& LineScanNumericState::trajectoryBias() const noexcept
    {
        return _bias;
    }
    const std::optional<LineScanTimeOffsetPrior>& LineScanNumericState::timeOffsetPrior() const noexcept
    {
        return _timeOffsetPrior;
    }
    const LineScanOptics& LineScanNumericState::optics() const noexcept
    {
        return _optics;
    }
    LineScanPixelConvention LineScanNumericState::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }
    const LineScanOptimizationSelection& LineScanNumericState::selection() const noexcept
    {
        return _selection;
    }
    const OptimizationLayout& LineScanNumericState::optimizationLayout() const noexcept
    {
        return _layout;
    }
    bool LineScanNumericState::definitionDirty() const noexcept
    {
        return _definitionDirty;
    }

    Result<void> LineScanNumericState::applyOptimizationDelta(std::span<const double> delta)
    {
        if (delta.size() != _layout.parameterCount())
        {
            return Result<void>::failure(CameraErrorCode::InvalidArgument,
                                         "line-scan numeric update size does not match its cached layout");
        }
        if (!std::all_of(delta.begin(), delta.end(), [](double value) { return std::isfinite(value); }))
        {
            return Result<void>::failure(CameraErrorCode::InvalidArgument,
                                         "line-scan numeric update must contain only finite values");
        }

        LineScanOptics next_optics = _optics;
        LineScanTrajectoryBias next_bias = _bias;
        std::size_t cursor = 0;
        for (const auto& sample : _samples)
        {
            if (_selection.knotPositions && !sample.constraints.positionFixed)
            {
                cursor += 3;
            }
            if (_selection.knotRotations && !sample.constraints.rotationFixed)
            {
                cursor += 3;
            }
        }
        if (_selection.globalTranslation)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                next_bias.translationMeters[axis] += delta[cursor++];
            }
        }
        if (_selection.globalRotation)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                next_bias.rotationVectorRadians[axis] += delta[cursor++];
            }
        }
        if (_selection.timeOffset)
        {
            next_bias.timeOffsetSeconds += delta[cursor++];
        }

        bool definition_changed = false;
        const auto scalar = [&](bool selected, double* value)
        {
            if (selected)
            {
                definition_changed = definition_changed || delta[cursor] != 0.0;
                *value += delta[cursor++];
            }
        };
        if (_selection.calibration.any())
        {
            MetashapeCalibration& calibration = *next_optics.completeCalibration;
            scalar(_selection.calibration.f, &calibration.f);
            const auto principal_scalar = [&](bool selected, bool x_axis)
            {
                if (!selected)
                {
                    return;
                }
                const double value = delta[cursor++];
                definition_changed = definition_changed || value != 0.0;
                if (value == 0.0)
                {
                    return;
                }
                double& absolute = x_axis ? calibration.cx : calibration.cy;
                if (!calibration.principalPointDecomposition)
                {
                    absolute += value;
                    return;
                }
                PrincipalPointDecomposition& principal = *calibration.principalPointDecomposition;
                double& offset = x_axis ? principal.cxOffset : principal.cyOffset;
                offset += value;
                absolute = (x_axis ? principal.imageCenterX : principal.imageCenterY) + offset;
            };
            principal_scalar(_selection.calibration.cx, true);
            principal_scalar(_selection.calibration.cy, false);
            scalar(_selection.calibration.b1, &calibration.b1);
            scalar(_selection.calibration.b2, &calibration.b2);
            scalar(_selection.calibration.k1, &calibration.k1);
            scalar(_selection.calibration.k2, &calibration.k2);
            scalar(_selection.calibration.k3, &calibration.k3);
            scalar(_selection.calibration.k4, &calibration.k4);
            scalar(_selection.calibration.p1, &calibration.p1);
            scalar(_selection.calibration.p2, &calibration.p2);
            scalar(_selection.calibration.p3, &calibration.p3);
            scalar(_selection.calibration.p4, &calibration.p4);
        }
        scalar(_selection.detector.focalLength, &next_optics.focalLengthMillimeters);
        scalar(_selection.detector.distortionK1, &next_optics.distortionK1);
        scalar(_selection.detector.samplePitch, &next_optics.samplePitchMillimeters);
        scalar(_selection.detector.principalSample, &next_optics.principalSample);
        if (_selection.detector.anyDetectorGeometryParameter())
        {
            auto& detector = *next_optics.detectorGeometry;
            scalar(_selection.detector.detectorSampleSumming, &detector.detectorSampleSumming);
            scalar(_selection.detector.detectorLineSumming, &detector.detectorLineSumming);
            scalar(_selection.detector.detectorSampleOrigin, &detector.detectorSampleOrigin);
            scalar(_selection.detector.detectorLineOrigin, &detector.detectorLineOrigin);
            scalar(_selection.detector.startingDetectorSample, &detector.startingDetectorSample);
            scalar(_selection.detector.startingDetectorLine, &detector.startingDetectorLine);
            if (_selection.detector.focalToPixelSamples)
            {
                for (double& value : detector.focalToPixelSamples)
                {
                    scalar(true, &value);
                }
            }
            if (_selection.detector.focalToPixelLines)
            {
                for (double& value : detector.focalToPixelLines)
                {
                    scalar(true, &value);
                }
            }
        }
        if (!validOptics(next_optics))
        {
            return Result<void>::failure(CameraErrorCode::InvalidIntrinsics,
                                         "line-scan numeric update produced invalid detector optics");
        }

        cursor = 0;
        for (std::size_t index = 0; index < _samples.size(); ++index)
        {
            const auto& constraints = _samples[index].constraints;
            if (_selection.knotPositions && !constraints.positionFixed)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    _positionDeltas[index][axis] += delta[cursor++];
                    _samples[index].center[axis] = _nominalSamples[index].center[axis] + _positionDeltas[index][axis];
                }
            }
            if (_selection.knotRotations && !constraints.rotationFixed)
            {
                for (double& value : _rotationDeltas[index])
                {
                    value += delta[cursor++];
                }
                _samples[index].cameraToWorldRotation = multiplyRotation(angleAxisRotation(_rotationDeltas[index]),
                                                                         _nominalSamples[index].cameraToWorldRotation);
            }
        }
        _bias = next_bias;
        _optics = next_optics;
        _definitionDirty = _definitionDirty || definition_changed;
        return Result<void>::success();
    }

    std::size_t LineScanNumericState::regularizationResidualCount() const noexcept
    {
        std::size_t count = 0;
        for (const auto& sample : _samples)
        {
            if (_selection.knotPositions && !sample.constraints.positionFixed && sample.constraints.positionSigmaMeters)
            {
                count += 3;
            }
            if (_selection.knotRotations && !sample.constraints.rotationFixed &&
                sample.constraints.rotationSigmaRadians)
            {
                count += 3;
            }
        }
        if (_selection.timeOffset && _timeOffsetPrior)
        {
            ++count;
        }
        for (std::size_t index = 1; index + 1 < _samples.size(); ++index)
        {
            if (_selection.positionSecondDifferenceWeight > 0.0 && _selection.knotPositions &&
                (!_samples[index - 1].constraints.positionFixed || !_samples[index].constraints.positionFixed ||
                 !_samples[index + 1].constraints.positionFixed))
            {
                count += 3;
            }
            if (_selection.rotationSecondDifferenceWeight > 0.0 && _selection.knotRotations &&
                (!_samples[index - 1].constraints.rotationFixed || !_samples[index].constraints.rotationFixed ||
                 !_samples[index + 1].constraints.rotationFixed))
            {
                count += 3;
            }
        }
        return count;
    }

    Result<void> LineScanNumericState::writeRegularizationResiduals(std::span<double> residuals) const
    {
        if (residuals.size() != regularizationResidualCount())
        {
            return Result<void>::failure(CameraErrorCode::InvalidArgument,
                                         "line-scan regularization buffer size is incorrect");
        }
        std::size_t cursor = 0;
        for (std::size_t index = 0; index < _samples.size(); ++index)
        {
            const auto& constraints = _samples[index].constraints;
            if (_selection.knotPositions && !constraints.positionFixed && constraints.positionSigmaMeters)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    residuals[cursor++] = _positionDeltas[index][axis] / (*constraints.positionSigmaMeters)[axis];
                }
            }
            if (_selection.knotRotations && !constraints.rotationFixed && constraints.rotationSigmaRadians)
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    residuals[cursor++] = _rotationDeltas[index][axis] / (*constraints.rotationSigmaRadians)[axis];
                }
            }
        }
        if (_selection.timeOffset && _timeOffsetPrior)
        {
            residuals[cursor++] =
                (_bias.timeOffsetSeconds - _timeOffsetPrior->meanSeconds) / _timeOffsetPrior->sigmaSeconds;
        }

        const double position_scale = std::sqrt(_selection.positionSecondDifferenceWeight);
        const double rotation_scale = std::sqrt(_selection.rotationSecondDifferenceWeight);
        for (std::size_t index = 1; index + 1 < _samples.size(); ++index)
        {
            if (position_scale > 0.0 && _selection.knotPositions &&
                (!_samples[index - 1].constraints.positionFixed || !_samples[index].constraints.positionFixed ||
                 !_samples[index + 1].constraints.positionFixed))
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    residuals[cursor++] =
                        position_scale * (_positionDeltas[index - 1][axis] - 2.0 * _positionDeltas[index][axis] +
                                          _positionDeltas[index + 1][axis]);
                }
            }
            if (rotation_scale > 0.0 && _selection.knotRotations &&
                (!_samples[index - 1].constraints.rotationFixed || !_samples[index].constraints.rotationFixed ||
                 !_samples[index + 1].constraints.rotationFixed))
            {
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    residuals[cursor++] =
                        rotation_scale * (_rotationDeltas[index - 1][axis] - 2.0 * _rotationDeltas[index][axis] +
                                          _rotationDeltas[index + 1][axis]);
                }
            }
        }
        return Result<void>::success();
    }

    EvaluationResult<TimeReference> LineScanNumericState::timeForLine(double line) const
    {
        if (!std::isfinite(line))
        {
            return EvaluationResult<TimeReference>::failure(CameraErrorCode::InvalidArgument,
                                                            "line coordinate must be finite");
        }
        const double minimum = _pixelConvention == LineScanPixelConvention::PixelCenter ? 0.5 : 0.0;
        if (line < minimum || line > minimum + static_cast<double>(_imageSize.lines - 1))
        {
            return EvaluationResult<TimeReference>::failure(CameraErrorCode::OutsideModelDomain,
                                                            "line coordinate lies outside the image");
        }
        double seconds = 0.0;
        if (_timing.segments.empty())
        {
            seconds = _timing.startTimeSeconds + (line - _timing.lineZero) * _timing.secondsPerLine;
        }
        else
        {
            auto segment = std::upper_bound(_timing.segments.begin(),
                                            _timing.segments.end(),
                                            line,
                                            [](double candidate, const LineRateSegment& value)
                                            { return candidate < value.startLine; });
            if (segment != _timing.segments.begin())
            {
                --segment;
            }
            seconds = segment->startTimeSeconds + (line - segment->startLine) * segment->secondsPerLine;
        }
        if (!std::isfinite(seconds))
        {
            return EvaluationResult<TimeReference>::failure(CameraErrorCode::OutsideModelDomain,
                                                            "line timing produced a non-finite time");
        }
        return EvaluationResult<TimeReference>::success(TimeReference{_timing.timeScale, seconds});
    }

    EvaluationResult<LineScanProjectionDetails> LineScanNumericState::projectAtLine(
        const GroundCoordinate& ground, double line, const EvaluationOptions& options) const
    {
        return internal::projectLineScanNumericAtLine(*this, ground, line, options);
    }

    EvaluationResult<Projection> LineScanNumericState::groundToImage(const GroundCoordinate& ground,
                                                                     const EvaluationOptions& options) const
    {
        return internal::projectLineScanNumeric(*this, ground, options);
    }

    EvaluationResult<ImagingLocus> LineScanNumericState::imageToImagingLocus(const ImageCoordinate& image,
                                                                             const EvaluationOptions& options) const
    {
        return internal::lineScanNumericImagingLocus(*this, image, options);
    }

    Result<CameraModelPtr<LineScanModel>>
    LineScanNumericState::toModel(CameraInstanceId resultInstanceId,
                                  std::optional<CameraDefinitionId> resultDefinitionId) const
    {
        if (_definitionDirty && !resultDefinitionId)
        {
            return Result<CameraModelPtr<LineScanModel>>::failure(
                CameraErrorCode::InvalidArgument,
                "modified line-scan detector optics require an explicit result definition identifier");
        }
        try
        {
            const CameraDefinitionId definition_id = resultDefinitionId ? *resultDefinitionId : _definitionId;
            const auto definition = LineScanDefinition::create(definition_id, _groundFrame, _optics, _pixelConvention);
            auto model = LineScanModel::create(std::move(resultInstanceId),
                                               _imageId,
                                               definition,
                                               _imageSize,
                                               LineScanTrajectory::create(_samples),
                                               _timing,
                                               _bias,
                                               _captureTime,
                                               _timeOffsetPrior);
            return Result<CameraModelPtr<LineScanModel>>::success(
                std::make_shared<const LineScanModel>(std::move(model)));
        }
        catch (const CameraValidationError& error)
        {
            return Result<CameraModelPtr<LineScanModel>>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<CameraModelPtr<LineScanModel>>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

    LineScanNumericState::LineScanNumericState(CameraInstanceId instanceId,
                                               CameraDefinitionId definitionId,
                                               ImageId imageId,
                                               FrameId groundFrame,
                                               ImageSize imageSize,
                                               std::optional<TimeReference> captureTime,
                                               std::vector<TrajectorySample> samples,
                                               LineTiming timing,
                                               LineScanTrajectoryBias bias,
                                               std::optional<LineScanTimeOffsetPrior> timeOffsetPrior,
                                               LineScanOptics optics,
                                               LineScanPixelConvention pixelConvention,
                                               LineScanOptimizationSelection selection,
                                               OptimizationLayout layout)
        : _instanceId(std::move(instanceId)), _definitionId(std::move(definitionId)), _imageId(std::move(imageId)),
          _groundFrame(std::move(groundFrame)), _imageSize(imageSize), _captureTime(captureTime),
          _nominalSamples(samples), _samples(std::move(samples)), _positionDeltas(_samples.size()),
          _rotationDeltas(_samples.size()), _timing(std::move(timing)), _bias(bias), _timeOffsetPrior(timeOffsetPrior),
          _optics(std::move(optics)), _pixelConvention(pixelConvention), _selection(std::move(selection)),
          _layout(std::move(layout))
    {
    }

} // namespace placamera
