#include "FramePinholeNumericState.h"

#include "camera/core/capabilities/CapabilityRequirements.h"
#include "camera/core/types/CameraErrors.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::camera_models::frame_pinhole
{
    namespace
    {
        constexpr double kMinimumDepth = 1.0e-9;

        bool finite(double value)
        {
            return std::isfinite(value);
        }

    } // namespace

    bool FramePinholeNumericState::fromInstance(const std::shared_ptr<const camera_core::CameraInstance>& source,
                                                FramePinholeNumericState* state,
                                                std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!source)
        {
            if (error)
            {
                *error = "pinhole numeric state requires a camera instance";
            }
            return false;
        }
        const auto pinhole = std::dynamic_pointer_cast<const FramePinholeInstance>(source);
        if (!pinhole)
        {
            if (error)
            {
                *error = "camera instance is not a frame-pinhole instance";
            }
            return false;
        }
        return fromInstance(*pinhole, state, error);
    }

    bool FramePinholeNumericState::fromInstance(const FramePinholeInstance& source,
                                                FramePinholeNumericState* state,
                                                std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!state)
        {
            if (error)
            {
                *error = "pinhole numeric state output is null";
            }
            return false;
        }
        const auto capabilities =
            camera_core::requireCapabilities(source,
                                             camera_core::CapabilitySet{camera_core::CapabilityKind::Projection,
                                                                        camera_core::CapabilityKind::StaticPose,
                                                                        camera_core::CapabilityKind::Optimization});
        if (!capabilities.ok())
        {
            if (error)
            {
                *error = capabilities.message();
            }
            return false;
        }
        if (!source.imageSize().isValid())
        {
            if (error)
            {
                *error = "pinhole numeric state requires a valid image size";
            }
            return false;
        }

        FramePinholeNumericState result;
        result._instanceId = source.instanceId();
        result._imageId = source.imageId();
        result._worldFrame = source.definition().worldFrame();
        result._captureTime = source.captureTime();
        result._imageSize = source.imageSize();
        result._intrinsics = source.pinholeDefinition().intrinsics();
        result._distortion = source.pinholeDefinition().distortion();
        result._pose.cameraToWorldRotation = source.pose().cameraToWorldRotation;
        result._pose.cameraCenter = source.pose().center;
        result._pose.depthAxisFlipped = source.pinholeDefinition().depthAxisFlipped();
        // Recreate the immutable definition from its validated values; the domain
        // base deliberately does not expose shared ownership.
        result._definition = FramePinholeDefinition::create(source.definition().definitionId(),
                                                            result._intrinsics,
                                                            result._distortion,
                                                            source.pinholeDefinition().pixelConvention(),
                                                            result._worldFrame,
                                                            result._pose.depthAxisFlipped);
        result._isValid = true;
        result._identityBound = true;
        *state = std::move(result);
        return true;
    }

    std::unique_ptr<FramePinholeInstance>
    FramePinholeNumericState::toInstance(std::optional<camera_core::CameraDefinitionId> definitionId) const
    {
        // A numerical state may be useful as a short-lived solver fixture without
        // carrying project identity, but it must never be promoted back into the
        // typed camera graph with the constructor's sentinel IDs.  Promotion is
        // therefore an identity boundary as well as a numerical validation
        // boundary; callers that decode or estimate a state must bind it
        // explicitly first.
        if (!hasBoundIdentity() || !validateNumericalState() || !_imageSize || !_imageSize->isValid())
        {
            return nullptr;
        }
        if (!definitionId)
        {
            definitionId =
                _definition
                    ? std::optional<camera_core::CameraDefinitionId>(_definition->definitionId())
                    : std::optional<camera_core::CameraDefinitionId>(camera_core::CameraDefinitionId("frame-pinhole"));
        }
        const auto definition =
            FramePinholeDefinition::create(*definitionId,
                                           _intrinsics,
                                           _distortion,
                                           _definition ? _definition->pixelConvention() : PixelConvention::PixelCenter,
                                           _worldFrame,
                                           _pose.depthAxisFlipped);
        const camera_core::Pose pose =
            camera_core::Pose::create(_worldFrame, _pose.cameraCenter, _pose.cameraToWorldRotation);
        return FramePinholeInstance::createUnique(_instanceId, _imageId, definition, *_imageSize, pose, _captureTime);
    }

    bool FramePinholeNumericState::validateNumericalState(std::string* error) const noexcept
    {
        const auto setError = [error](const char* message) noexcept
        {
            if (!error)
            {
                return;
            }
            try
            {
                *error = message;
            }
            catch (...)
            {
                // Validation is noexcept; reporting an allocation failure must not
                // turn malformed project metadata into process termination.
            }
        };

        if (error)
        {
            try
            {
                error->clear();
            }
            catch (...)
            {
            }
        }
        if (!_isValid)
        {
            setError("pinhole numeric state is not initialized");
            return false;
        }

        try
        {
            // FramePinholeDefinition owns the canonical intrinsic/distortion
            // invariants.  Reuse its construction path here instead of keeping a
            // second copy of those rules in the solver state.
            static_cast<void>(FramePinholeDefinition::create(
                _definition ? _definition->definitionId() : camera_core::CameraDefinitionId("numeric-state-validation"),
                _intrinsics,
                _distortion,
                _definition ? _definition->pixelConvention() : PixelConvention::PixelCenter,
                _worldFrame,
                _pose.depthAxisFlipped));
            static_cast<void>(camera_core::Pose::create(_worldFrame, _pose.cameraCenter, _pose.cameraToWorldRotation));
        }
        catch (const camera_core::CameraValidationError& exception)
        {
            setError(exception.what());
            return false;
        }
        catch (const std::exception& exception)
        {
            setError(exception.what());
            return false;
        }
        catch (...)
        {
            setError("pinhole numeric state contains invalid intrinsics, distortion, or pose");
            return false;
        }
        return true;
    }

    bool FramePinholeNumericState::isValid() const noexcept
    {
        return _isValid;
    }

    bool FramePinholeNumericState::hasBoundIdentity() const noexcept
    {
        return _identityBound;
    }

    bool FramePinholeNumericState::bindIdentity(camera_core::CameraInstanceId instanceId,
                                                camera_core::ImageId imageId,
                                                xjw::coordinate_system::CoordinateFrameId worldFrame,
                                                std::string* error)
    {
        if (error)
        {
            error->clear();
        }
        if (!_isValid)
        {
            if (error)
            {
                *error = "cannot bind identity to an uninitialized numeric state";
            }
            return false;
        }
        if (hasBoundIdentity())
        {
            if (error)
            {
                *error = "numeric state already has an explicit identity and world frame";
            }
            return false;
        }

        try
        {
            const auto definition = FramePinholeDefinition::create(
                _definition ? _definition->definitionId() : camera_core::CameraDefinitionId("frame-pinhole"),
                _intrinsics,
                _distortion,
                _definition ? _definition->pixelConvention() : PixelConvention::PixelCenter,
                worldFrame,
                _pose.depthAxisFlipped);
            static_cast<void>(camera_core::Pose::create(worldFrame, _pose.cameraCenter, _pose.cameraToWorldRotation));
            _instanceId = std::move(instanceId);
            _imageId = std::move(imageId);
            _worldFrame = std::move(worldFrame);
            _definition = definition;
            _identityBound = true;
            return true;
        }
        catch (const std::exception& exception)
        {
            if (error)
            {
                *error = exception.what();
            }
            return false;
        }
        catch (...)
        {
            if (error)
            {
                *error = "numeric state identity/frame is invalid";
            }
            return false;
        }
    }

    const camera_core::CameraInstanceId& FramePinholeNumericState::instanceId() const noexcept
    {
        return _instanceId;
    }

    const camera_core::ImageId& FramePinholeNumericState::imageId() const noexcept
    {
        return _imageId;
    }

    const xjw::coordinate_system::CoordinateFrameId& FramePinholeNumericState::worldFrame() const noexcept
    {
        return _worldFrame;
    }

    const std::optional<xjw::coordinate_system::TimeReference>& FramePinholeNumericState::captureTime() const noexcept
    {
        return _captureTime;
    }

    std::optional<camera_core::ImageSize> FramePinholeNumericState::imageSize() const noexcept
    {
        return _imageSize;
    }

    void FramePinholeNumericState::setImageSize(camera_core::ImageSize imageSize)
    {
        _imageSize = imageSize;
        _isValid = _isValid || imageSize.isValid();
    }

    FramePinholeNumericState::Intrinsics FramePinholeNumericState::intrinsics() const noexcept
    {
        return _intrinsics;
    }

    FramePinholeNumericState::Distortion FramePinholeNumericState::distortion() const noexcept
    {
        return _distortion;
    }

    FramePinholeNumericState::Pose FramePinholeNumericState::pose() const noexcept
    {
        return _pose;
    }

    std::array<double, 9> FramePinholeNumericState::cameraToWorldRotation() const noexcept
    {
        return _pose.cameraToWorldRotation;
    }

    std::array<double, 9> FramePinholeNumericState::worldToCameraRotation() const noexcept
    {
        return transpose(_pose.cameraToWorldRotation);
    }

    std::array<double, 3> FramePinholeNumericState::cameraCenter() const noexcept
    {
        return _pose.cameraCenter;
    }

    std::array<double, 3> FramePinholeNumericState::worldToCameraTranslation() const noexcept
    {
        const auto rotation = worldToCameraRotation();
        return {-rotation[0] * _pose.cameraCenter[0] - rotation[1] * _pose.cameraCenter[1] -
                    rotation[2] * _pose.cameraCenter[2],
                -rotation[3] * _pose.cameraCenter[0] - rotation[4] * _pose.cameraCenter[1] -
                    rotation[5] * _pose.cameraCenter[2],
                -rotation[6] * _pose.cameraCenter[0] - rotation[7] * _pose.cameraCenter[1] -
                    rotation[8] * _pose.cameraCenter[2]};
    }

    double FramePinholeNumericState::focalX() const noexcept
    {
        return _intrinsics.focalX;
    }
    double FramePinholeNumericState::focalY() const noexcept
    {
        return _intrinsics.focalY;
    }
    double FramePinholeNumericState::principalX() const noexcept
    {
        return _intrinsics.principalX;
    }
    double FramePinholeNumericState::principalY() const noexcept
    {
        return _intrinsics.principalY;
    }
    double FramePinholeNumericState::pixelPitch() const noexcept
    {
        return _intrinsics.pixelPitch;
    }
    double FramePinholeNumericState::focalXMillimeters() const noexcept
    {
        return focalX() * pixelPitch();
    }
    double FramePinholeNumericState::focalYMillimeters() const noexcept
    {
        return focalY() * pixelPitch();
    }
    double FramePinholeNumericState::principalXMillimeters() const noexcept
    {
        return principalX() * pixelPitch();
    }
    double FramePinholeNumericState::principalYMillimeters() const noexcept
    {
        return principalY() * pixelPitch();
    }
    int FramePinholeNumericState::uAxisSign() const noexcept
    {
        return _intrinsics.uAxisSign;
    }
    int FramePinholeNumericState::vAxisSign() const noexcept
    {
        return _intrinsics.vAxisSign;
    }
    bool FramePinholeNumericState::depthAxisFlipped() const noexcept
    {
        return _pose.depthAxisFlipped;
    }

    void FramePinholeNumericState::worldToCamera(const double world[3], double cameraPoint[3]) const
    {
        if (!world || !cameraPoint)
        {
            return;
        }
        const auto point = worldToCamera(_pose, world);
        std::copy(point.begin(), point.end(), cameraPoint);
    }

    bool
    FramePinholeNumericState::projectWorldPointWithDepth(const double world[3], double pixel[2], double& depth) const
    {
        if (!_isValid || !world || !pixel)
        {
            return false;
        }
        double camera[3]{};
        worldToCamera(world, camera);
        depth = positiveDepth(world);
        if (!finite(depth) || depth <= kMinimumDepth || !finite(camera[2]) || std::fabs(camera[2]) <= kMinimumDepth)
        {
            return false;
        }
        const double x = camera[0] / camera[2];
        const double y = camera[1] / camera[2];
        double xd = 0.0;
        double yd = 0.0;
        applyDistortion(x, y, &xd, &yd);
        pixel[0] = static_cast<double>(_intrinsics.uAxisSign) * _intrinsics.focalX * xd + _intrinsics.principalX;
        pixel[1] = static_cast<double>(_intrinsics.vAxisSign) * _intrinsics.focalY * yd + _intrinsics.principalY;
        return finite(pixel[0]) && finite(pixel[1]);
    }

    bool FramePinholeNumericState::projectWorldPoint(const double world[3], double pixel[2]) const
    {
        double depth = 0.0;
        return projectWorldPointWithDepth(world, pixel, depth);
    }

    bool FramePinholeNumericState::projectWorldPointSigned(const double world[3], double pixel[2]) const
    {
        if (!_isValid || !world || !pixel)
        {
            return false;
        }
        double camera[3]{};
        worldToCamera(world, camera);
        if (!finite(camera[2]) || std::fabs(camera[2]) <= kMinimumDepth)
        {
            return false;
        }
        double xd = 0.0;
        double yd = 0.0;
        applyDistortion(camera[0] / camera[2], camera[1] / camera[2], &xd, &yd);
        pixel[0] = static_cast<double>(_intrinsics.uAxisSign) * _intrinsics.focalX * xd + _intrinsics.principalX;
        pixel[1] = static_cast<double>(_intrinsics.vAxisSign) * _intrinsics.focalY * yd + _intrinsics.principalY;
        return finite(pixel[0]) && finite(pixel[1]);
    }

    double FramePinholeNumericState::positiveDepth(const double world[3]) const
    {
        if (!_isValid || !world)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double camera[3]{};
        worldToCamera(world, camera);
        return _pose.depthAxisFlipped ? -camera[2] : camera[2];
    }

    bool FramePinholeNumericState::isPointInFront(const double world[3], double minimumDepth) const
    {
        const double depth = positiveDepth(world);
        return finite(depth) && finite(minimumDepth) && depth > minimumDepth;
    }

    bool FramePinholeNumericState::undistortPixel(const double pixel[2], double norm[2], int maxIter, double tol) const
    {
        if (!_isValid || !pixel || !norm || maxIter < 0 || !finite(tol) || tol <= 0.0 || !finite(pixel[0]) ||
            !finite(pixel[1]))
        {
            return false;
        }
        double x =
            static_cast<double>(_intrinsics.uAxisSign) * (pixel[0] - _intrinsics.principalX) / _intrinsics.focalX;
        double y =
            static_cast<double>(_intrinsics.vAxisSign) * (pixel[1] - _intrinsics.principalY) / _intrinsics.focalY;
        for (int iteration = 0; iteration < maxIter; ++iteration)
        {
            double xd = 0.0;
            double yd = 0.0;
            applyDistortion(x, y, &xd, &yd);
            const double residualU = static_cast<double>(_intrinsics.uAxisSign) * _intrinsics.focalX * xd +
                                     _intrinsics.principalX - pixel[0];
            const double residualV = static_cast<double>(_intrinsics.vAxisSign) * _intrinsics.focalY * yd +
                                     _intrinsics.principalY - pixel[1];
            if (std::fabs(residualU) < tol && std::fabs(residualV) < tol)
            {
                break;
            }
            constexpr double step = 1.0e-7;
            double shiftedX = 0.0;
            double shiftedY = 0.0;
            applyDistortion(x + step, y, &shiftedX, &shiftedY);
            const double j00 = static_cast<double>(_intrinsics.uAxisSign) * _intrinsics.focalX * (shiftedX - xd) / step;
            const double j10 = static_cast<double>(_intrinsics.vAxisSign) * _intrinsics.focalY * (shiftedY - yd) / step;
            applyDistortion(x, y + step, &shiftedX, &shiftedY);
            const double j01 = static_cast<double>(_intrinsics.uAxisSign) * _intrinsics.focalX * (shiftedX - xd) / step;
            const double j11 = static_cast<double>(_intrinsics.vAxisSign) * _intrinsics.focalY * (shiftedY - yd) / step;
            const double determinant = j00 * j11 - j01 * j10;
            if (!finite(determinant) || std::fabs(determinant) < 1.0e-15)
            {
                break;
            }
            x -= (j11 * residualU - j01 * residualV) / determinant;
            y -= (-j10 * residualU + j00 * residualV) / determinant;
            if (!finite(x) || !finite(y))
            {
                return false;
            }
        }
        norm[0] = x;
        norm[1] = y;
        return true;
    }

    bool FramePinholeNumericState::unprojectPixel(const double pixel[2], double depth, double world[3]) const
    {
        if (!_isValid || !pixel || !world || !finite(depth) || depth <= 0.0)
        {
            return false;
        }
        double normalized[2]{};
        if (!undistortPixel(pixel, normalized))
        {
            return false;
        }
        const double cameraZ = _pose.depthAxisFlipped ? -depth : depth;
        const double camera[3] = {normalized[0] * cameraZ, normalized[1] * cameraZ, cameraZ};
        const auto result = cameraToWorld(_pose, camera);
        std::copy(result.begin(), result.end(), world);
        return finite(world[0]) && finite(world[1]) && finite(world[2]);
    }

    bool FramePinholeNumericState::rayForPixel(const std::array<double, 2>& pixel, Ray* ray) const
    {
        if (!ray)
        {
            return false;
        }
        const double pixelValues[2] = {pixel[0], pixel[1]};
        double pointAtUnitDepth[3]{};
        if (!unprojectPixel(pixelValues, 1.0, pointAtUnitDepth))
        {
            return false;
        }
        ray->origin = _pose.cameraCenter;
        for (int axis = 0; axis < 3; ++axis)
        {
            ray->direction[static_cast<std::size_t>(axis)] =
                pointAtUnitDepth[axis] - ray->origin[static_cast<std::size_t>(axis)];
        }
        double normSquared = 0.0;
        for (double value : ray->direction)
        {
            normSquared += value * value;
        }
        const double norm = std::sqrt(normSquared);
        if (!finite(norm) || norm <= kMinimumDepth)
        {
            return false;
        }
        for (double& value : ray->direction)
        {
            value /= norm;
        }
        return true;
    }

    FramePinholeNumericState::PairIntersection
    FramePinholeNumericState::triangulatePair(const FramePinholeNumericState& left,
                                              const std::array<double, 2>& leftPixel,
                                              const FramePinholeNumericState& right,
                                              const std::array<double, 2>& rightPixel)
    {
        PairIntersection result;
        if (left.worldFrame() != right.worldFrame())
        {
            return result;
        }

        Ray leftRay;
        Ray rightRay;
        if (!left.rayForPixel(leftPixel, &leftRay) || !right.rayForPixel(rightPixel, &rightRay))
        {
            return result;
        }
        const std::array<double, 3> w{leftRay.origin[0] - rightRay.origin[0],
                                      leftRay.origin[1] - rightRay.origin[1],
                                      leftRay.origin[2] - rightRay.origin[2]};
        const double a = leftRay.direction[0] * leftRay.direction[0] + leftRay.direction[1] * leftRay.direction[1] +
                         leftRay.direction[2] * leftRay.direction[2];
        const double b = leftRay.direction[0] * rightRay.direction[0] + leftRay.direction[1] * rightRay.direction[1] +
                         leftRay.direction[2] * rightRay.direction[2];
        const double c = rightRay.direction[0] * rightRay.direction[0] + rightRay.direction[1] * rightRay.direction[1] +
                         rightRay.direction[2] * rightRay.direction[2];
        const double d = leftRay.direction[0] * w[0] + leftRay.direction[1] * w[1] + leftRay.direction[2] * w[2];
        const double e = rightRay.direction[0] * w[0] + rightRay.direction[1] * w[1] + rightRay.direction[2] * w[2];
        const double denominator = a * c - b * b;
        if (!finite(denominator) || std::fabs(denominator) <= 1.0e-14)
        {
            return result;
        }
        const double leftParameter = (b * e - c * d) / denominator;
        const double rightParameter = (a * e - b * d) / denominator;
        const std::array<double, 3> leftPoint{leftRay.origin[0] + leftParameter * leftRay.direction[0],
                                              leftRay.origin[1] + leftParameter * leftRay.direction[1],
                                              leftRay.origin[2] + leftParameter * leftRay.direction[2]};
        const std::array<double, 3> rightPoint{rightRay.origin[0] + rightParameter * rightRay.direction[0],
                                               rightRay.origin[1] + rightParameter * rightRay.direction[1],
                                               rightRay.origin[2] + rightParameter * rightRay.direction[2]};
        result.point = {(leftPoint[0] + rightPoint[0]) * 0.5,
                        (leftPoint[1] + rightPoint[1]) * 0.5,
                        (leftPoint[2] + rightPoint[2]) * 0.5};
        double leftProjected[2]{};
        double rightProjected[2]{};
        if (!left.projectWorldPoint(result.point.data(), leftProjected) ||
            !right.projectWorldPoint(result.point.data(), rightProjected))
        {
            return PairIntersection{};
        }
        const double errorLeft = std::hypot(leftProjected[0] - leftPixel[0], leftProjected[1] - leftPixel[1]);
        const double errorRight = std::hypot(rightProjected[0] - rightPixel[0], rightProjected[1] - rightPixel[1]);
        if (!finite(errorLeft) || !finite(errorRight) || !left.isPointInFront(result.point.data()) ||
            !right.isPointInFront(result.point.data()))
        {
            return PairIntersection{};
        }
        result.rmsReprojectionPixels = std::sqrt((errorLeft * errorLeft + errorRight * errorRight) * 0.5);
        result.valid = true;
        return result;
    }

    void FramePinholeNumericState::setPose(const std::array<double, 9>& rotation, const std::array<double, 3>& center)
    {
        _pose.cameraToWorldRotation = rotation;
        _pose.cameraCenter = center;
        _isValid = true;
    }

    void FramePinholeNumericState::setCameraCenter(const std::array<double, 3>& center)
    {
        _pose.cameraCenter = center;
    }

    void FramePinholeNumericState::setIntrinsics(double focalXValue,
                                                 double focalYValue,
                                                 double principalXValue,
                                                 double principalYValue)
    {
        _intrinsics.focalX = focalXValue;
        _intrinsics.focalY = focalYValue;
        _intrinsics.principalX = principalXValue;
        _intrinsics.principalY = principalYValue;
        _isValid = true;
    }

    void FramePinholeNumericState::setIntrinsicsMillimeters(double focalXMillimetersValue,
                                                            double focalYMillimetersValue,
                                                            double principalXMillimetersValue,
                                                            double principalYMillimetersValue,
                                                            double pixelPitchMillimeters)
    {
        _intrinsics.pixelPitch = pixelPitchMillimeters > 0.0 ? pixelPitchMillimeters : 1.0;
        setIntrinsics(focalXMillimetersValue / _intrinsics.pixelPitch,
                      focalYMillimetersValue / _intrinsics.pixelPitch,
                      principalXMillimetersValue / _intrinsics.pixelPitch,
                      principalYMillimetersValue / _intrinsics.pixelPitch);
    }

    void FramePinholeNumericState::setPixelPitch(double pixelPitchValue)
    {
        if (pixelPitchValue > 0.0)
        {
            _intrinsics.pixelPitch = pixelPitchValue;
        }
    }

    void FramePinholeNumericState::setAxisDirections(int uDirection, int vDirection)
    {
        _intrinsics.uAxisSign = uDirection < 0 ? -1 : 1;
        _intrinsics.vAxisSign = vDirection < 0 ? -1 : 1;
        _isValid = true;
    }

    void FramePinholeNumericState::setDepthAxisFlipped(bool flipped)
    {
        _pose.depthAxisFlipped = flipped;
        _isValid = true;
    }

    void FramePinholeNumericState::setDistortion(const Distortion& distortion)
    {
        _distortion = distortion;
        _isValid = true;
    }

    void FramePinholeNumericState::setDistortion(double k1, double k2, double k3, double p1, double p2)
    {
        setDistortion(Distortion{k1, k2, k3, p1, p2});
    }

    void FramePinholeNumericState::applyDeltaPose(const double delta[6])
    {
        if (!delta)
        {
            return;
        }
        const auto deltaRotation = rotationFromDelta(delta);
        _pose.cameraToWorldRotation = multiply(deltaRotation, _pose.cameraToWorldRotation);
        _pose.cameraCenter[0] += delta[3];
        _pose.cameraCenter[1] += delta[4];
        _pose.cameraCenter[2] += delta[5];
    }

    FramePinholeNumericState FramePinholeNumericState::normalizedForPositiveDepth() const
    {
        FramePinholeNumericState result = *this;
        const double zSign = _pose.depthAxisFlipped ? -1.0 : 1.0;
        const double uSign = _intrinsics.uAxisSign < 0 ? -1.0 : 1.0;
        const double vSign = _intrinsics.vAxisSign < 0 ? -1.0 : 1.0;
        std::array<double, 3> signs{zSign * uSign, zSign * vSign, zSign};
        if (signs[0] * signs[1] * signs[2] < 0.0)
        {
            // A reflection cannot be represented by the proper camera pose
            // rotation.  Keep the pose proper and carry the parity in the pixel
            // axis sign, matching FramePinholeDefinition's normalization rule.
            signs[0] = -signs[0];
        }
        const double xRatioSign = signs[0] / signs[2];
        const double yRatioSign = signs[1] / signs[2];
        auto worldToCameraRotationValue = transpose(_pose.cameraToWorldRotation);
        std::array<double, 9> normalizedWorldToCamera{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                normalizedWorldToCamera[static_cast<std::size_t>(row * 3 + column)] =
                    signs[static_cast<std::size_t>(row)] *
                    worldToCameraRotationValue[static_cast<std::size_t>(row * 3 + column)];
            }
        }
        result._pose.cameraToWorldRotation = transpose(normalizedWorldToCamera);
        result._pose.depthAxisFlipped = false;
        result._intrinsics.focalX = std::fabs(result._intrinsics.focalX);
        result._intrinsics.focalY = std::fabs(result._intrinsics.focalY);
        result._intrinsics.uAxisSign = static_cast<int>(uSign * xRatioSign);
        result._intrinsics.vAxisSign = static_cast<int>(vSign * yRatioSign);
        result._distortion.tangentialP1 *= yRatioSign;
        result._distortion.tangentialP2 *= xRatioSign;
        return result;
    }

    FramePinholeNumericState FramePinholeNumericState::scaledIntrinsics(double scaleX, double scaleY) const
    {
        FramePinholeNumericState result = *this;
        if (scaleX <= 0.0 || scaleY <= 0.0 || !finite(scaleX) || !finite(scaleY))
        {
            return result;
        }
        result._intrinsics.focalX *= scaleX;
        result._intrinsics.focalY *= scaleY;
        result._intrinsics.principalX = (result._intrinsics.principalX + 0.5) * scaleX - 0.5;
        result._intrinsics.principalY = (result._intrinsics.principalY + 0.5) * scaleY - 0.5;
        if (result._imageSize)
        {
            result._imageSize->samples = static_cast<int>(std::llround(result._imageSize->samples * scaleX));
            result._imageSize->lines = static_cast<int>(std::llround(result._imageSize->lines * scaleY));
        }
        return result;
    }

    bool FramePinholeNumericState::finiteArray(const double* values, std::size_t count) noexcept
    {
        if (!values)
        {
            return false;
        }
        for (std::size_t index = 0; index < count; ++index)
        {
            if (!finite(values[index]))
            {
                return false;
            }
        }
        return true;
    }

    std::array<double, 9> FramePinholeNumericState::transpose(const std::array<double, 9>& matrix) noexcept
    {
        return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
    }

    std::array<double, 9> FramePinholeNumericState::multiply(const std::array<double, 9>& left,
                                                             const std::array<double, 9>& right) noexcept
    {
        std::array<double, 9> result{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                for (int index = 0; index < 3; ++index)
                {
                    result[static_cast<std::size_t>(row * 3 + column)] +=
                        left[static_cast<std::size_t>(row * 3 + index)] *
                        right[static_cast<std::size_t>(index * 3 + column)];
                }
            }
        }
        return result;
    }

    std::array<double, 9> FramePinholeNumericState::rotationFromDelta(const double delta[6]) noexcept
    {
        const double wx = delta[0];
        const double wy = delta[1];
        const double wz = delta[2];
        const double thetaSquared = wx * wx + wy * wy + wz * wz;
        if (thetaSquared < 1.0e-20)
        {
            return {1.0, -wz, wy, wz, 1.0, -wx, -wy, wx, 1.0};
        }
        const double theta = std::sqrt(thetaSquared);
        const double sineOverTheta = std::sin(theta) / theta;
        const double oneMinusCosineOverThetaSquared = (1.0 - std::cos(theta)) / thetaSquared;
        return {1.0 - oneMinusCosineOverThetaSquared * (wy * wy + wz * wz),
                oneMinusCosineOverThetaSquared * wx * wy - sineOverTheta * wz,
                oneMinusCosineOverThetaSquared * wx * wz + sineOverTheta * wy,
                oneMinusCosineOverThetaSquared * wx * wy + sineOverTheta * wz,
                1.0 - oneMinusCosineOverThetaSquared * (wx * wx + wz * wz),
                oneMinusCosineOverThetaSquared * wy * wz - sineOverTheta * wx,
                oneMinusCosineOverThetaSquared * wx * wz - sineOverTheta * wy,
                oneMinusCosineOverThetaSquared * wy * wz + sineOverTheta * wx,
                1.0 - oneMinusCosineOverThetaSquared * (wx * wx + wy * wy)};
    }

    std::array<double, 3> FramePinholeNumericState::worldToCamera(const Pose& pose, const double world[3]) noexcept
    {
        const auto rotation = transpose(pose.cameraToWorldRotation);
        const double dx = world[0] - pose.cameraCenter[0];
        const double dy = world[1] - pose.cameraCenter[1];
        const double dz = world[2] - pose.cameraCenter[2];
        return {rotation[0] * dx + rotation[1] * dy + rotation[2] * dz,
                rotation[3] * dx + rotation[4] * dy + rotation[5] * dz,
                rotation[6] * dx + rotation[7] * dy + rotation[8] * dz};
    }

    std::array<double, 3> FramePinholeNumericState::cameraToWorld(const Pose& pose, const double camera[3]) noexcept
    {
        return {pose.cameraToWorldRotation[0] * camera[0] + pose.cameraToWorldRotation[1] * camera[1] +
                    pose.cameraToWorldRotation[2] * camera[2] + pose.cameraCenter[0],
                pose.cameraToWorldRotation[3] * camera[0] + pose.cameraToWorldRotation[4] * camera[1] +
                    pose.cameraToWorldRotation[5] * camera[2] + pose.cameraCenter[1],
                pose.cameraToWorldRotation[6] * camera[0] + pose.cameraToWorldRotation[7] * camera[1] +
                    pose.cameraToWorldRotation[8] * camera[2] + pose.cameraCenter[2]};
    }

    void
    FramePinholeNumericState::applyDistortion(double x, double y, double* distortedX, double* distortedY) const noexcept
    {
        const double radiusSquared = x * x + y * y;
        const double radial = 1.0 + _distortion.radialK1 * radiusSquared +
                              _distortion.radialK2 * radiusSquared * radiusSquared +
                              _distortion.radialK3 * radiusSquared * radiusSquared * radiusSquared;
        *distortedX = x * radial + 2.0 * _distortion.tangentialP1 * x * y +
                      _distortion.tangentialP2 * (radiusSquared + 2.0 * x * x);
        *distortedY = y * radial + _distortion.tangentialP1 * (radiusSquared + 2.0 * y * y) +
                      2.0 * _distortion.tangentialP2 * x * y;
    }

    bool makeFramePinholeNumericState(const std::shared_ptr<const camera_core::CameraInstance>& source,
                                      FramePinholeNumericState* state,
                                      std::string* error)
    {
        return FramePinholeNumericState::fromInstance(source, state, error);
    }

} // namespace xjw::camera_models::frame_pinhole
