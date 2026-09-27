#include "placamera/frame_numeric_state.h"

#include "internal/frame_camera_math.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace placamera
{
    namespace
    {

        RotationMatrix multiplyRotation(const RotationMatrix& first, const RotationMatrix& second) noexcept
        {
            RotationMatrix result{};
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    for (int index = 0; index < 3; ++index)
                    {
                        result[static_cast<std::size_t>(row * 3 + column)] +=
                            first[static_cast<std::size_t>(row * 3 + index)] *
                            second[static_cast<std::size_t>(index * 3 + column)];
                    }
                }
            }
            return result;
        }

        RotationMatrix rotationFromDelta(const std::array<double, 6>& delta) noexcept
        {
            const double wx = delta[0];
            const double wy = delta[1];
            const double wz = delta[2];
            const double theta_squared = wx * wx + wy * wy + wz * wz;
            if (theta_squared < 1.0e-20)
            {
                return {1.0, -wz, wy, wz, 1.0, -wx, -wy, wx, 1.0};
            }
            const double theta = std::sqrt(theta_squared);
            const double sine_over_theta = std::sin(theta) / theta;
            const double one_minus_cosine_over_theta_squared = (1.0 - std::cos(theta)) / theta_squared;
            return {1.0 - one_minus_cosine_over_theta_squared * (wy * wy + wz * wz),
                    one_minus_cosine_over_theta_squared * wx * wy - sine_over_theta * wz,
                    one_minus_cosine_over_theta_squared * wx * wz + sine_over_theta * wy,
                    one_minus_cosine_over_theta_squared * wx * wy + sine_over_theta * wz,
                    1.0 - one_minus_cosine_over_theta_squared * (wx * wx + wz * wz),
                    one_minus_cosine_over_theta_squared * wy * wz - sine_over_theta * wx,
                    one_minus_cosine_over_theta_squared * wx * wz - sine_over_theta * wy,
                    one_minus_cosine_over_theta_squared * wy * wz + sine_over_theta * wx,
                    1.0 - one_minus_cosine_over_theta_squared * (wx * wx + wy * wy)};
        }

        double dot(const Vector3& first, const Vector3& second) noexcept
        {
            return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
        }

        Vector3 subtract(const Vector3& first, const Vector3& second) noexcept
        {
            return {{first[0] - second[0], first[1] - second[1], first[2] - second[2]}};
        }

        Vector3 pointOnRay(const ImagingLocus& ray, double distance) noexcept
        {
            return {{ray.origin.position[0] + distance * ray.direction[0],
                     ray.origin.position[1] + distance * ray.direction[1],
                     ray.origin.position[2] + distance * ray.direction[2]}};
        }

    } // namespace

    FramePinholeNumericState FramePinholeNumericState::fromModel(const FramePinholeModel& model)
    {
        return FramePinholeNumericState(model.instanceId(),
                                        model.definitionId(),
                                        model.imageId(),
                                        model.groundFrame(),
                                        model.imageSize(),
                                        model.captureTime(),
                                        model.pinholeDefinition().intrinsics(),
                                        model.pinholeDefinition().distortion(),
                                        model.pinholeDefinition().principalPointDecomposition(),
                                        model.pinholeDefinition().pixelConvention(),
                                        model.pinholeDefinition().depthAxisFlipped(),
                                        model.pinholeDefinition().projectionModel(),
                                        model.pinholeDefinition().sensorMount(),
                                        model.acquisition(),
                                        model.pose());
    }

    const CameraInstanceId& FramePinholeNumericState::instanceId() const noexcept
    {
        return _instanceId;
    }

    const CameraDefinitionId& FramePinholeNumericState::definitionId() const noexcept
    {
        return _definitionId;
    }

    const ImageId& FramePinholeNumericState::imageId() const noexcept
    {
        return _imageId;
    }

    const FrameId& FramePinholeNumericState::groundFrame() const noexcept
    {
        return _groundFrame;
    }

    const ImageSize& FramePinholeNumericState::imageSize() const noexcept
    {
        return _imageSize;
    }

    const std::optional<TimeReference>& FramePinholeNumericState::captureTime() const noexcept
    {
        return _captureTime;
    }

    const FrameIntrinsics& FramePinholeNumericState::intrinsics() const noexcept
    {
        return _intrinsics;
    }

    const BrownConradyDistortion& FramePinholeNumericState::distortion() const noexcept
    {
        return _distortion;
    }

    FrameCalibration FramePinholeNumericState::calibration() const noexcept
    {
        FrameCalibration result;
        result.f = _intrinsics.focalY;
        result.cx = _intrinsics.principalX;
        result.cy = _intrinsics.principalY;
        result.b1 = _intrinsics.focalX - _intrinsics.focalY;
        result.b2 = _intrinsics.skew;
        result.k1 = _distortion.radialK1;
        result.k2 = _distortion.radialK2;
        result.k3 = _distortion.radialK3;
        result.k4 = _distortion.radialK4;
        if (_distortion.tangentialConvention == BrownTangentialConvention::Metashape)
        {
            result.p1 = _distortion.tangentialP1;
            result.p2 = _distortion.tangentialP2;
            result.p3 = _distortion.tangentialP3;
            result.p4 = _distortion.tangentialP4;
        }
        else
        {
            result.p1 = _distortion.tangentialP2;
            result.p2 = _distortion.tangentialP1;
        }
        result.principalPointDecomposition = _principalPointDecomposition;
        return result;
    }

    const std::optional<PrincipalPointDecomposition>&
    FramePinholeNumericState::principalPointDecomposition() const noexcept
    {
        return _principalPointDecomposition;
    }

    PixelConvention FramePinholeNumericState::pixelConvention() const noexcept
    {
        return _pixelConvention;
    }

    bool FramePinholeNumericState::depthAxisFlipped() const noexcept
    {
        return _depthAxisFlipped;
    }

    FrameProjectionModel FramePinholeNumericState::projectionModel() const noexcept
    {
        return _projectionModel;
    }

    const SensorMountState& FramePinholeNumericState::sensorMount() const noexcept
    {
        return _sensorMount;
    }

    const CameraAcquisitionState& FramePinholeNumericState::acquisition() const noexcept
    {
        return _acquisition;
    }

    const Pose& FramePinholeNumericState::pose() const noexcept
    {
        return _pose;
    }

    bool FramePinholeNumericState::definitionDirty() const noexcept
    {
        return _definitionDirty;
    }

    void FramePinholeNumericState::setPose(Pose pose)
    {
        if (pose.frame != _groundFrame)
        {
            throw CameraValidationError(CameraErrorCode::FrameMismatch,
                                        "numeric-state pose frame must match its ground frame");
        }
        _pose = Pose::create(std::move(pose.frame), pose.center, pose.cameraToWorldRotation);
    }

    void FramePinholeNumericState::setIntrinsics(FrameIntrinsics intrinsics)
    {
        static_cast<void>(FramePinholeDefinition::create(_definitionId,
                                                         intrinsics,
                                                         _distortion,
                                                         _pixelConvention,
                                                         _groundFrame,
                                                         _depthAxisFlipped,
                                                         _projectionModel,
                                                         _sensorMount));
        _intrinsics = intrinsics;
        _definitionDirty = true;
    }

    void FramePinholeNumericState::setDistortion(BrownConradyDistortion distortion)
    {
        static_cast<void>(FramePinholeDefinition::create(_definitionId,
                                                         _intrinsics,
                                                         distortion,
                                                         _pixelConvention,
                                                         _groundFrame,
                                                         _depthAxisFlipped,
                                                         _projectionModel,
                                                         _sensorMount));
        _distortion = distortion;
        _definitionDirty = true;
    }

    void FramePinholeNumericState::setCalibration(FrameCalibration calibration)
    {
        const auto definition = FramePinholeDefinition::create(_definitionId,
                                                               calibration,
                                                               _pixelConvention,
                                                               _groundFrame,
                                                               _depthAxisFlipped,
                                                               _intrinsics.pixelPitch,
                                                               _intrinsics.uAxisSign,
                                                               _intrinsics.vAxisSign,
                                                               _projectionModel,
                                                               _sensorMount);
        _intrinsics = definition->intrinsics();
        _distortion = definition->distortion();
        _principalPointDecomposition = definition->principalPointDecomposition();
        _definitionDirty = true;
    }

    void FramePinholeNumericState::applyPoseDelta(const std::array<double, 6>& delta)
    {
        if (!std::all_of(delta.begin(), delta.end(), [](double value) { return std::isfinite(value); }))
        {
            throw CameraValidationError(CameraErrorCode::InvalidPose,
                                        "numeric-state pose delta must contain finite values");
        }
        const RotationMatrix rotation = multiplyRotation(rotationFromDelta(delta), _pose.cameraToWorldRotation);
        const Vector3 center{{_pose.center[0] + delta[3], _pose.center[1] + delta[4], _pose.center[2] + delta[5]}};
        setPose(Pose::create(_groundFrame, center, rotation));
    }

    FramePinholeNumericState FramePinholeNumericState::scaledIntrinsics(double scaleX, double scaleY) const
    {
        const auto definition = _principalPointDecomposition ? FramePinholeDefinition::create(_definitionId,
                                                                                              calibration(),
                                                                                              _pixelConvention,
                                                                                              _groundFrame,
                                                                                              _depthAxisFlipped,
                                                                                              _intrinsics.pixelPitch,
                                                                                              _intrinsics.uAxisSign,
                                                                                              _intrinsics.vAxisSign,
                                                                                              _projectionModel,
                                                                                              _sensorMount)
                                                             : FramePinholeDefinition::create(_definitionId,
                                                                                              _intrinsics,
                                                                                              _distortion,
                                                                                              _pixelConvention,
                                                                                              _groundFrame,
                                                                                              _depthAxisFlipped,
                                                                                              _projectionModel,
                                                                                              _sensorMount);
        const auto scaled_definition = definition->scaledIntrinsics(_definitionId, scaleX, scaleY);
        const double samples = static_cast<double>(_imageSize.samples) * scaleX;
        const double lines = static_cast<double>(_imageSize.lines) * scaleY;
        const double maximum_dimension = static_cast<double>(std::numeric_limits<int>::max());
        if (!std::isfinite(samples) || !std::isfinite(lines) || samples <= 0.0 || lines <= 0.0 ||
            samples > maximum_dimension || lines > maximum_dimension)
        {
            throw CameraValidationError(CameraErrorCode::InvalidImageSize,
                                        "scaled frame image dimensions must be finite and positive");
        }
        const ImageSize scaled_size{static_cast<int>(std::llround(samples)), static_cast<int>(std::llround(lines))};
        if (!scaled_size.isValid())
        {
            throw CameraValidationError(CameraErrorCode::InvalidImageSize,
                                        "scaled frame image dimensions round to zero");
        }

        FramePinholeNumericState result = *this;
        result._intrinsics = scaled_definition->intrinsics();
        result._principalPointDecomposition = scaled_definition->principalPointDecomposition();
        result._definitionDirty = true;
        result._imageSize = scaled_size;
        return result;
    }

    FramePinholeNumericState FramePinholeNumericState::normalizedForPositiveDepth() const
    {
        const auto definition = _principalPointDecomposition ? FramePinholeDefinition::create(_definitionId,
                                                                                              calibration(),
                                                                                              _pixelConvention,
                                                                                              _groundFrame,
                                                                                              _depthAxisFlipped,
                                                                                              _intrinsics.pixelPitch,
                                                                                              _intrinsics.uAxisSign,
                                                                                              _intrinsics.vAxisSign,
                                                                                              _projectionModel,
                                                                                              _sensorMount)
                                                             : FramePinholeDefinition::create(_definitionId,
                                                                                              _intrinsics,
                                                                                              _distortion,
                                                                                              _pixelConvention,
                                                                                              _groundFrame,
                                                                                              _depthAxisFlipped,
                                                                                              _projectionModel,
                                                                                              _sensorMount);
        const auto axis_signs = definition->positiveDepthAxisSigns();
        const auto normalized_definition = definition->normalizedForPositiveDepth(_definitionId);
        const auto& normalized_intrinsics = normalized_definition->intrinsics();
        const auto& normalized_distortion = normalized_definition->distortion();

        FramePinholeNumericState result = *this;
        result._pose = Pose::create(_groundFrame,
                                    _pose.center,
                                    internal::normalizedCameraToWorldRotation(_pose.cameraToWorldRotation, axis_signs));
        result._definitionDirty =
            _definitionDirty || _depthAxisFlipped || _intrinsics.focalX != normalized_intrinsics.focalX ||
            _intrinsics.focalY != normalized_intrinsics.focalY || _intrinsics.skew != normalized_intrinsics.skew ||
            _intrinsics.uAxisSign != normalized_intrinsics.uAxisSign ||
            _intrinsics.vAxisSign != normalized_intrinsics.vAxisSign ||
            _distortion.tangentialP1 != normalized_distortion.tangentialP1 ||
            _distortion.tangentialP2 != normalized_distortion.tangentialP2;
        result._intrinsics = normalized_intrinsics;
        result._distortion = normalized_distortion;
        result._depthAxisFlipped = false;
        return result;
    }

    EvaluationResult<double> FramePinholeNumericState::signedDepth(const GroundCoordinate& ground) const
    {
        return internal::signedFrameDepth(_groundFrame, _depthAxisFlipped, _pose, ground);
    }

    EvaluationResult<Projection> FramePinholeNumericState::groundToImageSigned(const GroundCoordinate& ground,
                                                                               const EvaluationOptions& options) const
    {
        return internal::projectFrameSigned(_groundFrame,
                                            _imageSize,
                                            _captureTime,
                                            _intrinsics,
                                            _distortion,
                                            _projectionModel,
                                            _depthAxisFlipped,
                                            _pose,
                                            _acquisition,
                                            ground,
                                            options);
    }

    EvaluationResult<Projection> FramePinholeNumericState::groundToImage(const GroundCoordinate& ground,
                                                                         const EvaluationOptions& options) const
    {
        return internal::projectFrame(_groundFrame,
                                      _imageSize,
                                      _captureTime,
                                      _intrinsics,
                                      _distortion,
                                      _projectionModel,
                                      _depthAxisFlipped,
                                      _pose,
                                      _acquisition,
                                      ground,
                                      options);
    }

    EvaluationResult<ImagingLocus> FramePinholeNumericState::imageToImagingLocus(const ImageCoordinate& image,
                                                                                 const EvaluationOptions& options) const
    {
        return internal::frameImagingLocus(_groundFrame,
                                           _imageSize,
                                           _captureTime,
                                           _intrinsics,
                                           _distortion,
                                           _principalPointDecomposition,
                                           _projectionModel,
                                           _depthAxisFlipped,
                                           _pose,
                                           _acquisition,
                                           image,
                                           options);
    }

    EvaluationResult<GroundCoordinate> FramePinholeNumericState::imageToGroundAtDepth(
        const ImageCoordinate& image, double positiveDepth, const EvaluationOptions& options) const
    {
        return internal::frameGroundAtDepth(_groundFrame,
                                            _imageSize,
                                            _intrinsics,
                                            _distortion,
                                            _principalPointDecomposition,
                                            _projectionModel,
                                            _depthAxisFlipped,
                                            _pose,
                                            _acquisition,
                                            image,
                                            positiveDepth,
                                            options);
    }

    Result<CameraModelPtr<FramePinholeModel>>
    FramePinholeNumericState::toModel(CameraInstanceId resultInstanceId,
                                      std::optional<CameraDefinitionId> resultDefinitionId) const
    {
        if (_definitionDirty && !resultDefinitionId)
        {
            return Result<CameraModelPtr<FramePinholeModel>>::failure(
                CameraErrorCode::InvalidArgument,
                "modified frame calibration requires an explicit result definition identifier");
        }
        try
        {
            const CameraDefinitionId definition_id = resultDefinitionId ? *resultDefinitionId : _definitionId;
            const auto definition = _principalPointDecomposition
                                        ? FramePinholeDefinition::create(definition_id,
                                                                         calibration(),
                                                                         _pixelConvention,
                                                                         _groundFrame,
                                                                         _depthAxisFlipped,
                                                                         _intrinsics.pixelPitch,
                                                                         _intrinsics.uAxisSign,
                                                                         _intrinsics.vAxisSign,
                                                                         _projectionModel,
                                                                         _sensorMount)
                                        : FramePinholeDefinition::create(definition_id,
                                                                         _intrinsics,
                                                                         _distortion,
                                                                         _pixelConvention,
                                                                         _groundFrame,
                                                                         _depthAxisFlipped,
                                                                         _projectionModel,
                                                                         _sensorMount);
            auto model = FramePinholeModel::create(
                std::move(resultInstanceId), _imageId, definition, _imageSize, _pose, _captureTime, _acquisition);
            return Result<CameraModelPtr<FramePinholeModel>>::success(
                std::make_shared<const FramePinholeModel>(std::move(model)));
        }
        catch (const CameraValidationError& error)
        {
            return Result<CameraModelPtr<FramePinholeModel>>::failure(error.code(), error.what());
        }
        catch (const std::exception& error)
        {
            return Result<CameraModelPtr<FramePinholeModel>>::failure(CameraErrorCode::InvalidModelState, error.what());
        }
    }

    EvaluationResult<FramePairIntersection>
    FramePinholeNumericState::triangulatePair(const FramePinholeNumericState& first,
                                              const ImageCoordinate& firstImage,
                                              const FramePinholeNumericState& second,
                                              const ImageCoordinate& secondImage,
                                              const EvaluationOptions& options)
    {
        if (first.groundFrame() != second.groundFrame())
        {
            return EvaluationResult<FramePairIntersection>::failure(
                CameraErrorCode::FrameMismatch, "frame-pinhole pair must use a common ground frame");
        }
        const auto first_ray = first.imageToImagingLocus(firstImage, options);
        const auto second_ray = second.imageToImagingLocus(secondImage, options);
        if (!first_ray || !second_ray)
        {
            return EvaluationResult<FramePairIntersection>::failure(CameraErrorCode::OutsideModelDomain,
                                                                    "frame-pinhole pair rays could not be evaluated");
        }

        const Vector3 between = subtract(first_ray.value().origin.position, second_ray.value().origin.position);
        const double direction_dot = dot(first_ray.value().direction, second_ray.value().direction);
        const double first_projection = dot(first_ray.value().direction, between);
        const double second_projection = dot(second_ray.value().direction, between);
        const double denominator = 1.0 - direction_dot * direction_dot;
        if (std::abs(denominator) < 1.0e-12)
        {
            return EvaluationResult<FramePairIntersection>::failure(CameraErrorCode::NonConvergence,
                                                                    "frame-pinhole pair rays are nearly parallel");
        }
        const double first_distance = (direction_dot * second_projection - first_projection) / denominator;
        const double second_distance = (second_projection - direction_dot * first_projection) / denominator;
        const Vector3 first_point = pointOnRay(first_ray.value(), first_distance);
        const Vector3 second_point = pointOnRay(second_ray.value(), second_distance);
        const Vector3 midpoint{{0.5 * (first_point[0] + second_point[0]),
                                0.5 * (first_point[1] + second_point[1]),
                                0.5 * (first_point[2] + second_point[2])}};
        if (!(first_distance > 0.0) || !(second_distance > 0.0))
        {
            return EvaluationResult<FramePairIntersection>::failure(CameraErrorCode::OutsideModelDomain,
                                                                    "triangulated frame point lies behind a camera");
        }
        const GroundCoordinate ground{first.groundFrame(), midpoint};
        const auto first_reprojection = first.groundToImage(ground, options);
        const auto second_reprojection = second.groundToImage(ground, options);
        if (!first_reprojection || !second_reprojection)
        {
            return EvaluationResult<FramePairIntersection>::failure(
                CameraErrorCode::OutsideModelDomain, "triangulated frame point could not be reprojected");
        }
        const double first_error = std::hypot(first_reprojection.value().image.sample - firstImage.sample,
                                              first_reprojection.value().image.line - firstImage.line);
        const double second_error = std::hypot(second_reprojection.value().image.sample - secondImage.sample,
                                               second_reprojection.value().image.line - secondImage.line);
        const Vector3 first_to_ground = subtract(midpoint, first_ray.value().origin.position);
        const Vector3 second_to_ground = subtract(midpoint, second_ray.value().origin.position);
        const double first_range = std::sqrt(dot(first_to_ground, first_to_ground));
        const double second_range = std::sqrt(dot(second_to_ground, second_to_ground));
        const Vector3 miss = subtract(first_point, second_point);
        const double miss_distance = std::sqrt(dot(miss, miss));
        if (!(first_range > 0.0) || !(second_range > 0.0) || !std::isfinite(miss_distance))
        {
            return EvaluationResult<FramePairIntersection>::failure(CameraErrorCode::NonConvergence,
                                                                    "triangulated frame geometry is degenerate");
        }
        const double cosine =
            std::clamp(dot(first_to_ground, second_to_ground) / (first_range * second_range), -1.0, 1.0);
        constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;
        const double angle = std::acos(cosine) * radians_to_degrees;
        const double rms = std::sqrt(0.5 * (first_error * first_error + second_error * second_error));
        if (!std::isfinite(angle) || !std::isfinite(rms))
        {
            return EvaluationResult<FramePairIntersection>::failure(CameraErrorCode::NonConvergence,
                                                                    "triangulated frame quality is not finite");
        }
        return EvaluationResult<FramePairIntersection>::success(
            {ground, angle, miss_distance, first_error, second_error, rms}, rms);
    }

    FramePinholeNumericState::FramePinholeNumericState(
        CameraInstanceId instanceId,
        CameraDefinitionId definitionId,
        ImageId imageId,
        FrameId groundFrame,
        ImageSize imageSize,
        std::optional<TimeReference> captureTime,
        FrameIntrinsics intrinsics,
        BrownConradyDistortion distortion,
        std::optional<PrincipalPointDecomposition> principalPointDecomposition,
        PixelConvention pixelConvention,
        bool depthAxisFlipped,
        FrameProjectionModel projectionModel,
        SensorMountState sensorMount,
        CameraAcquisitionState acquisition,
        Pose pose)
        : _instanceId(std::move(instanceId)), _definitionId(std::move(definitionId)), _imageId(std::move(imageId)),
          _groundFrame(std::move(groundFrame)), _imageSize(imageSize), _captureTime(captureTime),
          _intrinsics(intrinsics), _distortion(distortion),
          _principalPointDecomposition(std::move(principalPointDecomposition)), _pixelConvention(pixelConvention),
          _depthAxisFlipped(depthAxisFlipped), _projectionModel(projectionModel), _sensorMount(std::move(sensorMount)),
          _acquisition(std::move(acquisition)), _pose(std::move(pose))
    {
    }

} // namespace placamera
