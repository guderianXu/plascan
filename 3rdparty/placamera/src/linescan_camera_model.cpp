#include "placamera/linescan_camera.h"

#include "internal/linescan_projection.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <utility>

namespace placamera
{

    namespace
    {

        bool finiteVector(const Vector3& values) noexcept
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

        double minimumLine(LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::PixelCenter ? 0.5 : 0.0;
        }

    } // namespace

    LineScanModel LineScanModel::create(CameraInstanceId instanceId,
                                        ImageId imageId,
                                        std::shared_ptr<const LineScanDefinition> definition,
                                        ImageSize imageSize,
                                        LineScanTrajectory trajectory,
                                        LineTiming timing,
                                        LineScanTrajectoryBias bias,
                                        std::optional<TimeReference> captureTime,
                                        std::optional<LineScanTimeOffsetPrior> timeOffsetPrior)
    {
        if (!definition)
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState, "line-scan model requires a definition");
        }
        if (!imageSize.isValid())
        {
            throw CameraValidationError(CameraErrorCode::InvalidImageSize, "line-scan image size must be positive");
        }

        bool timing_valid = std::isfinite(timing.lineZero) && std::isfinite(timing.startTimeSeconds);
        if (timing.segments.empty())
        {
            timing_valid = timing_valid && std::isfinite(timing.secondsPerLine) && timing.secondsPerLine > 0.0;
        }
        for (std::size_t index = 0; index < timing.segments.size(); ++index)
        {
            const LineRateSegment& segment = timing.segments[index];
            timing_valid = timing_valid && std::isfinite(segment.startLine) &&
                           std::isfinite(segment.startTimeSeconds) && std::isfinite(segment.secondsPerLine) &&
                           segment.secondsPerLine > 0.0 &&
                           (index == 0 || (segment.startLine > timing.segments[index - 1].startLine &&
                                           segment.startTimeSeconds > timing.segments[index - 1].startTimeSeconds));
        }
        if (!timing_valid)
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime,
                                        "line-scan timing must be finite with a positive line period");
        }
        if (trajectory.timeScale() != timing.timeScale)
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime,
                                        "line-scan timing and trajectory must use the same time scale");
        }
        if (!finiteVector(bias.translationMeters) || !finiteVector(bias.rotationVectorRadians) ||
            !std::isfinite(bias.timeOffsetSeconds))
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "line-scan trajectory bias must contain finite values");
        }
        if (captureTime && !std::isfinite(captureTime->seconds))
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime,
                                        "line-scan capture time must contain finite seconds");
        }
        if (timeOffsetPrior &&
            (!std::isfinite(timeOffsetPrior->meanSeconds) || !std::isfinite(timeOffsetPrior->sigmaSeconds) ||
             !(timeOffsetPrior->sigmaSeconds > 0.0)))
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "line-scan time-offset prior must contain a finite mean and positive sigma");
        }

        return LineScanModel(std::move(instanceId),
                             std::move(imageId),
                             std::move(definition),
                             imageSize,
                             std::move(trajectory),
                             std::move(timing),
                             bias,
                             captureTime,
                             timeOffsetPrior);
    }

    const CameraInstanceId& LineScanModel::instanceId() const noexcept
    {
        return _instanceId;
    }

    const CameraDefinition& LineScanModel::definition() const noexcept
    {
        return *_definition;
    }

    const CameraDefinitionId& LineScanModel::definitionId() const noexcept
    {
        return _definition->definitionId();
    }

    const ImageId& LineScanModel::imageId() const noexcept
    {
        return _imageId;
    }

    std::string_view LineScanModel::modelType() const noexcept
    {
        return _definition->modelType();
    }

    int LineScanModel::parameterSchemaVersion() const noexcept
    {
        return _definition->parameterSchemaVersion();
    }

    const FrameId& LineScanModel::groundFrame() const noexcept
    {
        return _definition->groundFrame();
    }

    const ImageSize& LineScanModel::imageSize() const noexcept
    {
        return _imageSize;
    }

    const std::optional<TimeReference>& LineScanModel::captureTime() const noexcept
    {
        return _captureTime;
    }

    CapabilitySet LineScanModel::capabilities() const noexcept
    {
        return _definition->capabilities();
    }

    const LineScanDefinition& LineScanModel::lineScanDefinition() const noexcept
    {
        return *_definition;
    }

    const LineScanTrajectory& LineScanModel::trajectory() const noexcept
    {
        return _trajectory;
    }

    const LineTiming& LineScanModel::lineTiming() const noexcept
    {
        return _timing;
    }

    const LineScanTrajectoryBias& LineScanModel::trajectoryBias() const noexcept
    {
        return _bias;
    }

    const std::optional<LineScanTimeOffsetPrior>& LineScanModel::timeOffsetPrior() const noexcept
    {
        return _timeOffsetPrior;
    }

    EvaluationResult<TimeReference> LineScanModel::timeForLine(double line) const
    {
        if (!std::isfinite(line))
        {
            return EvaluationResult<TimeReference>::failure(CameraErrorCode::InvalidArgument,
                                                            "line coordinate must be finite");
        }
        const double minimum_line = minimumLine(_definition->pixelConvention());
        const double maximum_line = minimum_line + static_cast<double>(_imageSize.lines - 1);
        if (line < minimum_line || line > maximum_line)
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

    EvaluationResult<double> LineScanModel::lineForTime(TimeReference time) const
    {
        if (time.scale != _timing.timeScale || !std::isfinite(time.seconds))
        {
            return EvaluationResult<double>::failure(CameraErrorCode::InvalidTime,
                                                     "time must be finite and use the line timing scale");
        }

        double line = 0.0;
        if (_timing.segments.empty())
        {
            line = _timing.lineZero + (time.seconds - _timing.startTimeSeconds) / _timing.secondsPerLine;
        }
        else
        {
            auto segment = _timing.segments.begin();
            for (auto candidate = _timing.segments.begin(); candidate != _timing.segments.end(); ++candidate)
            {
                if (time.seconds >= candidate->startTimeSeconds)
                {
                    segment = candidate;
                }
            }
            line = segment->startLine + (time.seconds - segment->startTimeSeconds) / segment->secondsPerLine;
        }

        const double minimum_line = minimumLine(_definition->pixelConvention());
        const double maximum_line = minimum_line + static_cast<double>(_imageSize.lines - 1);
        if (!std::isfinite(line) || line < minimum_line || line > maximum_line)
        {
            return EvaluationResult<double>::failure(CameraErrorCode::OutsideModelDomain,
                                                     "time lies outside line-scan image support");
        }
        return EvaluationResult<double>::success(line);
    }

    EvaluationResult<Projection> LineScanModel::groundToImage(const GroundCoordinate& ground,
                                                              const EvaluationOptions& options) const
    {
        const auto details = internal::projectLineScan(*this, ground, options);
        if (!details)
        {
            return EvaluationResult<Projection>::failure(details.errorCode(), details.message());
        }
        return EvaluationResult<Projection>::success(details.value().projection, details.achievedPrecisionPixels());
    }

    EvaluationResult<LineScanProjectionDetails>
    LineScanModel::projectAtLine(const GroundCoordinate& ground, double line, const EvaluationOptions& options) const
    {
        return internal::projectLineScanAtLine(*this, ground, line, options);
    }

    EvaluationResult<LineScanProjectionDetails> LineScanModel::projectAtLine(const GroundCoordinate& ground,
                                                                             double line,
                                                                             const LineScanTrajectoryBias& bias,
                                                                             const EvaluationOptions& options) const
    {
        return internal::projectLineScanAtLine(*this, ground, line, bias, options);
    }

    EvaluationResult<ImagingLocus> LineScanModel::imageToImagingLocus(const ImageCoordinate& image,
                                                                      const EvaluationOptions& options) const
    {
        return internal::lineScanImagingLocus(*this, image, options);
    }

    OptimizationLayout LineScanModel::optimizationLayout() const
    {
        return {{{"trajectory.translation", OptimizationParameterKind::Translation, "m", 0, 3, false},
                 {"trajectory.rotation", OptimizationParameterKind::RotationVector, "rad", 3, 3, false},
                 {"trajectory.time", OptimizationParameterKind::TimeOffset, "s", 6, 1, false}}};
    }

    Result<RasterModelPtr> LineScanModel::withOptimizationUpdate(const OptimizationUpdate& update) const
    {
        if (update.delta.size() != optimizationLayout().parameterCount())
        {
            return Result<RasterModelPtr>::failure(CameraErrorCode::InvalidArgument,
                                                   "line-scan optimization update must contain 7 values");
        }
        for (const double value : update.delta)
        {
            if (!std::isfinite(value))
            {
                return Result<RasterModelPtr>::failure(CameraErrorCode::InvalidArgument,
                                                       "line-scan optimization update must be finite");
            }
        }

        try
        {
            std::array<double, 7> delta{};
            std::copy(update.delta.begin(), update.delta.end(), delta.begin());
            auto model = withTrajectoryBias(update.instanceId, applyLineScanTrajectoryBiasUpdate(_bias, delta));
            return Result<RasterModelPtr>::success(std::make_shared<const LineScanModel>(std::move(model)));
        }
        catch (const CameraValidationError& error)
        {
            return Result<RasterModelPtr>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<RasterModelPtr>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

    LineScanModel LineScanModel::withTrajectoryBias(CameraInstanceId instanceId, LineScanTrajectoryBias bias) const
    {
        return create(std::move(instanceId),
                      _imageId,
                      _definition,
                      _imageSize,
                      _trajectory,
                      _timing,
                      bias,
                      _captureTime,
                      _timeOffsetPrior);
    }

    LineScanModel::LineScanModel(CameraInstanceId instanceId,
                                 ImageId imageId,
                                 std::shared_ptr<const LineScanDefinition> definition,
                                 ImageSize imageSize,
                                 LineScanTrajectory trajectory,
                                 LineTiming timing,
                                 LineScanTrajectoryBias bias,
                                 std::optional<TimeReference> captureTime,
                                 std::optional<LineScanTimeOffsetPrior> timeOffsetPrior)
        : _instanceId(std::move(instanceId)), _imageId(std::move(imageId)), _definition(std::move(definition)),
          _imageSize(imageSize), _trajectory(std::move(trajectory)), _timing(std::move(timing)), _bias(bias),
          _captureTime(captureTime), _timeOffsetPrior(timeOffsetPrior)
    {
    }

} // namespace placamera
