#include "placamera/rpc_camera.h"

#include "internal/rpc_math.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <memory>
#include <utility>

namespace placamera
{

    namespace
    {

        bool validOptions(const EvaluationOptions& options) noexcept
        {
            return std::isfinite(options.desiredPrecisionPixels) && options.desiredPrecisionPixels > 0.0 &&
                   options.maximumIterations > 0;
        }

        bool finiteImage(const ImageCoordinate& image) noexcept
        {
            return std::isfinite(image.sample) && std::isfinite(image.line);
        }

        bool insideImage(const ImageCoordinate& image, const ImageSize& size) noexcept
        {
            return image.sample >= 0.0 && image.line >= 0.0 && image.sample < static_cast<double>(size.samples) &&
                   image.line < static_cast<double>(size.lines);
        }

        bool finiteCorrection(const RpcImageCorrection& correction) noexcept
        {
            const std::array<double, 6> values{{correction.sampleOffsetPixels,
                                                correction.sampleSamplePixels,
                                                correction.sampleLinePixels,
                                                correction.lineOffsetPixels,
                                                correction.lineSamplePixels,
                                                correction.lineLinePixels}};
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

        bool finiteCorrection(const RpcGroundCorrection& correction) noexcept
        {
            const std::array<double, 8> values{{correction.sampleOffsetPixels,
                                                correction.lineOffsetPixels,
                                                correction.sampleLongitudePixelsPerDegree,
                                                correction.sampleLatitudePixelsPerDegree,
                                                correction.sampleHeightPixelsPerMeter,
                                                correction.lineLongitudePixelsPerDegree,
                                                correction.lineLatitudePixelsPerDegree,
                                                correction.lineHeightPixelsPerMeter}};
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }

    } // namespace

    RpcCorrection RpcCorrection::normalizedImage(RpcImageCorrection correction)
    {
        return RpcCorrection(correction);
    }

    RpcCorrection RpcCorrection::groundCoordinates(RpcGroundCorrection correction)
    {
        return RpcCorrection(correction);
    }

    RpcCorrectionDomain RpcCorrection::domain() const noexcept
    {
        return std::holds_alternative<RpcImageCorrection>(_value) ? RpcCorrectionDomain::NormalizedImage
                                                                  : RpcCorrectionDomain::GroundCoordinates;
    }

    const RpcImageCorrection* RpcCorrection::normalizedImageValue() const noexcept
    {
        return std::get_if<RpcImageCorrection>(&_value);
    }

    const RpcGroundCorrection* RpcCorrection::groundCoordinateValue() const noexcept
    {
        return std::get_if<RpcGroundCorrection>(&_value);
    }

    RpcCorrection::RpcCorrection(RpcImageCorrection correction) : _value(correction)
    {
    }

    RpcCorrection::RpcCorrection(RpcGroundCorrection correction) : _value(correction)
    {
    }

    RpcModel RpcModel::create(CameraInstanceId instanceId,
                              ImageId imageId,
                              std::shared_ptr<const RpcDefinition> definition,
                              ImageSize imageSize,
                              RpcImageCorrection correction,
                              std::optional<TimeReference> captureTime)
    {
        return createWithCorrection(std::move(instanceId),
                                    std::move(imageId),
                                    std::move(definition),
                                    imageSize,
                                    RpcCorrection::normalizedImage(correction),
                                    captureTime);
    }

    RpcModel RpcModel::createWithCorrection(CameraInstanceId instanceId,
                                            ImageId imageId,
                                            std::shared_ptr<const RpcDefinition> definition,
                                            ImageSize imageSize,
                                            RpcCorrection correction,
                                            std::optional<TimeReference> captureTime)
    {
        if (!definition)
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState, "RPC model requires a definition");
        }
        if (!imageSize.isValid())
        {
            throw CameraValidationError(CameraErrorCode::InvalidImageSize, "RPC model image size must be positive");
        }
        const bool correction_valid = correction.domain() == RpcCorrectionDomain::NormalizedImage
                                          ? finiteCorrection(*correction.normalizedImageValue())
                                          : finiteCorrection(*correction.groundCoordinateValue());
        if (!correction_valid)
        {
            throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                        "RPC correction must contain finite values in its declared domain");
        }
        if (captureTime && !std::isfinite(captureTime->seconds))
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime, "RPC capture time must contain finite seconds");
        }
        return RpcModel(std::move(instanceId),
                        std::move(imageId),
                        std::move(definition),
                        imageSize,
                        std::move(correction),
                        captureTime);
    }

    const CameraInstanceId& RpcModel::instanceId() const noexcept
    {
        return _instanceId;
    }

    const CameraDefinition& RpcModel::definition() const noexcept
    {
        return *_definition;
    }

    const CameraDefinitionId& RpcModel::definitionId() const noexcept
    {
        return _definition->definitionId();
    }

    const ImageId& RpcModel::imageId() const noexcept
    {
        return _imageId;
    }

    std::string_view RpcModel::modelType() const noexcept
    {
        return _definition->modelType();
    }

    int RpcModel::parameterSchemaVersion() const noexcept
    {
        return _definition->parameterSchemaVersion();
    }

    const FrameId& RpcModel::groundFrame() const noexcept
    {
        return _definition->groundFrame();
    }

    const ImageSize& RpcModel::imageSize() const noexcept
    {
        return _imageSize;
    }

    const std::optional<TimeReference>& RpcModel::captureTime() const noexcept
    {
        return _captureTime;
    }

    CapabilitySet RpcModel::capabilities() const noexcept
    {
        return _definition->capabilities();
    }

    const RpcDefinition& RpcModel::rpcDefinition() const noexcept
    {
        return *_definition;
    }

    RpcCorrectionDomain RpcModel::correctionDomain() const noexcept
    {
        return _correction.domain();
    }

    const RpcCorrection& RpcModel::correction() const noexcept
    {
        return _correction;
    }

    const RpcImageCorrection* RpcModel::normalizedImageCorrection() const noexcept
    {
        return _correction.normalizedImageValue();
    }

    const RpcGroundCorrection* RpcModel::groundCorrection() const noexcept
    {
        return _correction.groundCoordinateValue();
    }

    const RpcImageCorrection& RpcModel::imageCorrection() const
    {
        const RpcImageCorrection* correction = normalizedImageCorrection();
        if (!correction)
        {
            throw CameraValidationError(CameraErrorCode::InvalidArgument,
                                        "RPC model uses a ground-coordinate correction, not an image correction");
        }
        return *correction;
    }

    EvaluationResult<Projection> RpcModel::groundToImage(const GroundCoordinate& ground,
                                                         const EvaluationOptions& options) const
    {
        if (!validOptions(options))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::InvalidArgument,
                                                         "RPC evaluation options must be finite and positive");
        }
        if (ground.frame != groundFrame())
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::FrameMismatch,
                                                         "ground coordinate frame does not match the RPC model");
        }

        const auto geodetic = cartesianToGeodetic(ground.position, _definition->ellipsoid());
        if (!geodetic)
        {
            return EvaluationResult<Projection>::failure(geodetic.errorCode(), geodetic.message());
        }
        return groundToImageGeodetic(geodetic.value(), options);
    }

    EvaluationResult<Projection> RpcModel::groundToImageGeodetic(const GeodeticCoordinate& ground,
                                                                 const EvaluationOptions& options) const
    {
        if (!validOptions(options))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::InvalidArgument,
                                                         "RPC evaluation options must be finite and positive");
        }
        const auto image = internal::projectRpc(*_definition, _correction, ground, true);
        if (!image)
        {
            return EvaluationResult<Projection>::failure(image.errorCode(), image.message());
        }
        if (options.requireInsideImage && !insideImage(image.value(), _imageSize))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                         "RPC projection lies outside the image");
        }
        return EvaluationResult<Projection>::success(Projection{image.value(), std::nullopt, _captureTime},
                                                     image.achievedPrecisionPixels());
    }

    EvaluationResult<GeodeticCoordinate> RpcModel::imageToGroundAtHeight(const ImageCoordinate& image,
                                                                         double ellipsoidalHeightMeters,
                                                                         const EvaluationOptions& options) const
    {
        if (options.requireInsideImage && (!finiteImage(image) || !insideImage(image, _imageSize)))
        {
            return EvaluationResult<GeodeticCoordinate>::failure(CameraErrorCode::OutsideModelDomain,
                                                                 "image coordinate lies outside the RPC image");
        }
        return internal::invertRpcAtHeight(*_definition, _correction, image, ellipsoidalHeightMeters, options);
    }

    EvaluationResult<ImagingLocus> RpcModel::imageToImagingLocus(const ImageCoordinate& image,
                                                                 const EvaluationOptions& options) const
    {
        if (!validOptions(options))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::InvalidArgument,
                                                           "RPC evaluation options must be finite and positive");
        }
        if (options.requireInsideImage && (!finiteImage(image) || !insideImage(image, _imageSize)))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "image coordinate lies outside the RPC image");
        }

        const RpcParameters& parameters = _definition->parameters();
        const double upper_height = parameters.heightOffset + parameters.heightScale;
        const double lower_height = parameters.heightOffset - parameters.heightScale;
        const auto upper_geodetic =
            internal::invertRpcAtHeight(*_definition, _correction, image, upper_height, options);
        const auto lower_geodetic =
            internal::invertRpcAtHeight(*_definition, _correction, image, lower_height, options);
        if (!upper_geodetic || !lower_geodetic)
        {
            return EvaluationResult<ImagingLocus>::failure(
                CameraErrorCode::NonConvergence,
                "RPC imaging locus could not invert the image at its bracketing heights");
        }

        const auto upper = geodeticToCartesian(upper_geodetic.value(), _definition->ellipsoid());
        const auto lower = geodeticToCartesian(lower_geodetic.value(), _definition->ellipsoid());
        if (!upper || !lower)
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "RPC imaging locus could not convert to Cartesian space");
        }

        Vector3 direction{lower.value()[0] - upper.value()[0],
                          lower.value()[1] - upper.value()[1],
                          lower.value()[2] - upper.value()[2]};
        const double norm = std::hypot(direction[0], std::hypot(direction[1], direction[2]));
        if (!std::isfinite(norm) || norm <= 1.0e-9)
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "RPC imaging locus has an invalid direction");
        }
        for (double& component : direction)
        {
            component /= norm;
        }

        ImagingLocus locus{GroundCoordinate{groundFrame(), upper.value()}, direction, _captureTime};
        const double achieved_precision =
            std::max(upper_geodetic.achievedPrecisionPixels(), lower_geodetic.achievedPrecisionPixels());
        return EvaluationResult<ImagingLocus>::success(std::move(locus), achieved_precision);
    }

    OptimizationLayout RpcModel::optimizationLayout() const
    {
        if (_correction.domain() == RpcCorrectionDomain::NormalizedImage)
        {
            return {{{"normalized_image_correction", OptimizationParameterKind::ImageCorrection, "px", 0, 6, false}}};
        }
        return {{{"ground_sample_offset", OptimizationParameterKind::ImageCorrection, "px", 0, 1, false},
                 {"ground_line_offset", OptimizationParameterKind::ImageCorrection, "px", 1, 1, false},
                 {"ground_sample_longitude", OptimizationParameterKind::ImageCorrection, "px/deg", 2, 1, false},
                 {"ground_sample_latitude", OptimizationParameterKind::ImageCorrection, "px/deg", 3, 1, false},
                 {"ground_sample_height", OptimizationParameterKind::ImageCorrection, "px/m", 4, 1, false},
                 {"ground_line_longitude", OptimizationParameterKind::ImageCorrection, "px/deg", 5, 1, false},
                 {"ground_line_latitude", OptimizationParameterKind::ImageCorrection, "px/deg", 6, 1, false},
                 {"ground_line_height", OptimizationParameterKind::ImageCorrection, "px/m", 7, 1, false}}};
    }

    Result<RasterModelPtr> RpcModel::withOptimizationUpdate(const OptimizationUpdate& update) const
    {
        if (update.delta.size() != optimizationLayout().parameterCount())
        {
            return Result<RasterModelPtr>::failure(
                CameraErrorCode::InvalidArgument,
                "RPC optimization update size does not match the correction domain layout");
        }
        for (const double value : update.delta)
        {
            if (!std::isfinite(value))
            {
                return Result<RasterModelPtr>::failure(CameraErrorCode::InvalidArgument,
                                                       "RPC optimization update must be finite");
            }
        }

        try
        {
            RpcCorrection correction = _correction;
            if (const RpcImageCorrection* current = normalizedImageCorrection())
            {
                RpcImageCorrection updated = *current;
                updated.sampleOffsetPixels += update.delta[0];
                updated.sampleSamplePixels += update.delta[1];
                updated.sampleLinePixels += update.delta[2];
                updated.lineOffsetPixels += update.delta[3];
                updated.lineSamplePixels += update.delta[4];
                updated.lineLinePixels += update.delta[5];
                correction = RpcCorrection::normalizedImage(updated);
            }
            else
            {
                RpcGroundCorrection updated = *groundCorrection();
                updated.sampleOffsetPixels += update.delta[0];
                updated.lineOffsetPixels += update.delta[1];
                updated.sampleLongitudePixelsPerDegree += update.delta[2];
                updated.sampleLatitudePixelsPerDegree += update.delta[3];
                updated.sampleHeightPixelsPerMeter += update.delta[4];
                updated.lineLongitudePixelsPerDegree += update.delta[5];
                updated.lineLatitudePixelsPerDegree += update.delta[6];
                updated.lineHeightPixelsPerMeter += update.delta[7];
                correction = RpcCorrection::groundCoordinates(updated);
            }
            auto model = withCorrection(update.instanceId, std::move(correction));
            return Result<RasterModelPtr>::success(std::make_shared<const RpcModel>(std::move(model)));
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

    RpcModel RpcModel::withImageCorrection(CameraInstanceId instanceId, RpcImageCorrection correction) const
    {
        return withCorrection(std::move(instanceId), RpcCorrection::normalizedImage(correction));
    }

    RpcModel RpcModel::withGroundCorrection(CameraInstanceId instanceId, RpcGroundCorrection correction) const
    {
        return withCorrection(std::move(instanceId), RpcCorrection::groundCoordinates(correction));
    }

    RpcModel RpcModel::withCorrection(CameraInstanceId instanceId, RpcCorrection correction) const
    {
        return createWithCorrection(
            std::move(instanceId), _imageId, _definition, _imageSize, std::move(correction), _captureTime);
    }

    RpcModel::RpcModel(CameraInstanceId instanceId,
                       ImageId imageId,
                       std::shared_ptr<const RpcDefinition> definition,
                       ImageSize imageSize,
                       RpcCorrection correction,
                       std::optional<TimeReference> captureTime)
        : _instanceId(std::move(instanceId)), _imageId(std::move(imageId)), _definition(std::move(definition)),
          _imageSize(imageSize), _correction(std::move(correction)), _captureTime(captureTime)
    {
    }

} // namespace placamera
