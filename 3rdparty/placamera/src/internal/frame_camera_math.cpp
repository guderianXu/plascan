#include "frame_camera_math.h"

#include <algorithm>
#include <cmath>

namespace placamera::internal
{
    namespace
    {

        constexpr double MinimumDirectionNorm = 1.0e-28;
        constexpr double Pi = 3.141592653589793238462643383279502884;
        constexpr double HalfPi = Pi * 0.5;

        bool usesBrownDistortion(FrameProjectionModel model) noexcept
        {
            return model == FrameProjectionModel::Perspective || model == FrameProjectionModel::Fisheye ||
                   model == FrameProjectionModel::EquidistantFisheye || model == FrameProjectionModel::EquisolidFisheye;
        }

        bool requiresPositiveDepth(FrameProjectionModel model) noexcept
        {
            return model == FrameProjectionModel::Perspective || model == FrameProjectionModel::Fisheye;
        }

        ImageCoordinate pixelFromPlane(const FrameIntrinsics& intrinsics, double x, double y) noexcept
        {
            return {static_cast<double>(intrinsics.uAxisSign) * (intrinsics.focalX * x + intrinsics.skew * y) +
                        intrinsics.principalX,
                    static_cast<double>(intrinsics.vAxisSign) * intrinsics.focalY * y + intrinsics.principalY};
        }

        NormalizedCoordinate
        planeFromPixel(const FrameIntrinsics& intrinsics,
                       const std::optional<PrincipalPointDecomposition>& principalPointDecomposition,
                       const ImageCoordinate& image) noexcept
        {
            double center_x = intrinsics.principalX;
            double center_y = intrinsics.principalY;
            double cx_offset = 0.0;
            double cy_offset = 0.0;
            if (principalPointDecomposition)
            {
                const PrincipalPointDecomposition& principal = *principalPointDecomposition;
                if (principal.imageCenterX + principal.cxOffset == intrinsics.principalX)
                {
                    center_x = principal.imageCenterX;
                    cx_offset = principal.cxOffset;
                }
                if (principal.imageCenterY + principal.cyOffset == intrinsics.principalY)
                {
                    center_y = principal.imageCenterY;
                    cy_offset = principal.cyOffset;
                }
            }
            const double y =
                static_cast<double>(intrinsics.vAxisSign) * ((image.line - center_y) - cy_offset) / intrinsics.focalY;
            const double x = (static_cast<double>(intrinsics.uAxisSign) * ((image.sample - center_x) - cx_offset) -
                              intrinsics.skew * y) /
                             intrinsics.focalX;
            return {x, y};
        }

        RotationMatrix rodrigues(const Vector3& vector) noexcept
        {
            const double wx = vector[0];
            const double wy = vector[1];
            const double wz = vector[2];
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

        Vector3 multiply(const RotationMatrix& rotation, const Vector3& vector) noexcept
        {
            return {rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
                    rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
                    rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2]};
        }

        Vector3 transposeMultiply(const RotationMatrix& rotation, const Vector3& vector) noexcept
        {
            return {rotation[0] * vector[0] + rotation[3] * vector[1] + rotation[6] * vector[2],
                    rotation[1] * vector[0] + rotation[4] * vector[1] + rotation[7] * vector[2],
                    rotation[2] * vector[0] + rotation[5] * vector[1] + rotation[8] * vector[2]};
        }

        double rollingScan(double line, const ImageSize& imageSize) noexcept
        {
            double scan = line / static_cast<double>(imageSize.lines);
            return scan + scan - 1.0;
        }

        Vector3 applyRollingMotion(const RollingShutterMotion& motion, const Vector3& camera, double scan) noexcept
        {
            const Vector3 translated{camera[0] - motion.translation[0] * scan,
                                     camera[1] - motion.translation[1] * scan,
                                     camera[2] - motion.translation[2] * scan};
            const Vector3 scaled_rotation{
                motion.rotationVector[0] * scan, motion.rotationVector[1] * scan, motion.rotationVector[2] * scan};
            return multiply(rodrigues(scaled_rotation), translated);
        }

        bool cameraPlane(FrameProjectionModel model, const Vector3& camera, NormalizedCoordinate* plane) noexcept
        {
            const double x = camera[0];
            const double y = camera[1];
            const double z = camera[2];
            if (model == FrameProjectionModel::Perspective)
            {
                if (!std::isfinite(z) || std::abs(z) < 1.0e-9)
                {
                    return false;
                }
                *plane = {x / z, y / z};
                return std::isfinite((*plane)[0]) && std::isfinite((*plane)[1]);
            }
            if (model == FrameProjectionModel::Fisheye || model == FrameProjectionModel::EquidistantFisheye ||
                model == FrameProjectionModel::EquisolidFisheye)
            {
                if (model == FrameProjectionModel::Fisheye && z <= 0.0)
                {
                    return false;
                }
                const double radius = std::hypot(x, y);
                const double norm_squared = radius * radius + z * z;
                if (!std::isfinite(norm_squared) || norm_squared <= MinimumDirectionNorm)
                {
                    return false;
                }
                if (radius == 0.0)
                {
                    if (z <= 0.0)
                    {
                        return false;
                    }
                    *plane = {0.0, 0.0};
                    return true;
                }
                const double theta = std::atan2(radius, z);
                const double image_radius =
                    model == FrameProjectionModel::EquisolidFisheye ? 2.0 * std::sin(theta * 0.5) : theta;
                const double scale = image_radius / radius;
                *plane = {x * scale, y * scale};
                return std::isfinite((*plane)[0]) && std::isfinite((*plane)[1]);
            }
            if (model == FrameProjectionModel::Spherical)
            {
                const double horizontal = std::hypot(x, z);
                const double norm_squared = horizontal * horizontal + y * y;
                if (!std::isfinite(norm_squared) || norm_squared <= MinimumDirectionNorm)
                {
                    return false;
                }
                *plane = {std::atan2(x, z), std::atan2(y, horizontal)};
                return true;
            }
            if (model == FrameProjectionModel::Cylindrical)
            {
                const double horizontal = std::hypot(x, z);
                if (!std::isfinite(horizontal) || horizontal <= 0.0)
                {
                    return false;
                }
                *plane = {std::atan2(x, z), y / horizontal};
                return std::isfinite((*plane)[1]);
            }
            return false;
        }

        bool projectCameraPoint(const FrameIntrinsics& intrinsics,
                                const BrownConradyDistortion& distortion,
                                FrameProjectionModel model,
                                const Vector3& camera,
                                ImageCoordinate* image) noexcept
        {
            NormalizedCoordinate plane{};
            if (!cameraPlane(model, camera, &plane))
            {
                return false;
            }
            double x = plane[0];
            double y = plane[1];
            if (usesBrownDistortion(model))
            {
                distort(distortion, plane[0], plane[1], &x, &y);
            }
            *image = pixelFromPlane(intrinsics, x, y);
            return finiteImage(*image);
        }

        EvaluationResult<Vector3>
        cameraBearing(const FrameIntrinsics& intrinsics,
                      const BrownConradyDistortion& distortion,
                      const std::optional<PrincipalPointDecomposition>& principalPointDecomposition,
                      FrameProjectionModel model,
                      bool depthAxisFlipped,
                      const ImageCoordinate& image,
                      const EvaluationOptions& options)
        {
            const auto plane_result =
                undistort(intrinsics, distortion, model, principalPointDecomposition, image, options);
            if (!plane_result)
            {
                return EvaluationResult<Vector3>::failure(plane_result.errorCode(), plane_result.message());
            }
            const double x = plane_result.value()[0];
            const double y = plane_result.value()[1];
            Vector3 bearing{};
            if (model == FrameProjectionModel::Perspective)
            {
                const double z = depthAxisFlipped ? -1.0 : 1.0;
                bearing = {x * z, y * z, z};
            }
            else if (model == FrameProjectionModel::Fisheye || model == FrameProjectionModel::EquidistantFisheye ||
                     model == FrameProjectionModel::EquisolidFisheye)
            {
                const double radius = std::hypot(x, y);
                if ((model == FrameProjectionModel::Fisheye && radius >= HalfPi) ||
                    (model == FrameProjectionModel::EquidistantFisheye && radius >= Pi) ||
                    (model == FrameProjectionModel::EquisolidFisheye && radius >= 2.0))
                {
                    return EvaluationResult<Vector3>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "fisheye pixel lies outside the invertible projection domain");
                }
                double theta = radius;
                if (model == FrameProjectionModel::EquisolidFisheye)
                {
                    theta = 2.0 * std::asin(radius * 0.5);
                }
                if (model == FrameProjectionModel::Fisheye)
                {
                    if (radius == 0.0)
                    {
                        bearing = {0.0, 0.0, 1.0};
                    }
                    else
                    {
                        const double scale = std::tan(theta) / radius;
                        bearing = {x * scale, y * scale, 1.0};
                    }
                }
                else
                {
                    const double scale = radius > 0.0 ? std::sin(theta) / radius : 1.0;
                    bearing = {x * scale, y * scale, std::cos(theta)};
                }
            }
            else if (model == FrameProjectionModel::Spherical)
            {
                if (std::abs(x) > Pi || std::abs(y) > HalfPi || (std::abs(y) == HalfPi && x != 0.0))
                {
                    return EvaluationResult<Vector3>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "spherical pixel lies outside the longitude/latitude projection domain");
                }
                const double cos_latitude = std::cos(y);
                bearing = {std::sin(x) * cos_latitude, std::sin(y), std::cos(x) * cos_latitude};
            }
            else if (model == FrameProjectionModel::Cylindrical)
            {
                if (std::abs(x) > Pi)
                {
                    return EvaluationResult<Vector3>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "cylindrical pixel lies outside the longitude projection domain");
                }
                bearing = {std::sin(x), y, std::cos(x)};
            }
            else
            {
                return EvaluationResult<Vector3>::failure(CameraErrorCode::UnsupportedModel,
                                                          "frame projection model is unsupported");
            }
            const double norm_squared = bearing[0] * bearing[0] + bearing[1] * bearing[1] + bearing[2] * bearing[2];
            if (!std::isfinite(norm_squared) || norm_squared <= MinimumDirectionNorm)
            {
                return EvaluationResult<Vector3>::failure(CameraErrorCode::OutsideModelDomain,
                                                          "camera bearing is outside the projection domain");
            }
            const double inverse_norm = 1.0 / std::sqrt(norm_squared);
            for (double& value : bearing)
            {
                value *= inverse_norm;
            }
            return EvaluationResult<Vector3>::success(bearing, plane_result.achievedPrecisionPixels());
        }

    } // namespace

    bool finiteVector(const Vector3& value) noexcept
    {
        return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
    }

    bool finiteImage(const ImageCoordinate& value) noexcept
    {
        return std::isfinite(value.sample) && std::isfinite(value.line);
    }

    bool insideImage(const ImageCoordinate& value, const ImageSize& size) noexcept
    {
        return value.sample >= 0.0 && value.line >= 0.0 && value.sample < static_cast<double>(size.samples) &&
               value.line < static_cast<double>(size.lines);
    }

    bool validOptions(const EvaluationOptions& options) noexcept
    {
        return std::isfinite(options.desiredPrecisionPixels) && options.desiredPrecisionPixels > 0.0 &&
               options.maximumIterations > 0;
    }

    void distort(
        const BrownConradyDistortion& distortion, double x, double y, double* distortedX, double* distortedY) noexcept
    {
        const double x_squared = x * x;
        const double y_squared = y * y;
        const double radius_squared = x_squared + y_squared;
        const double radius_fourth = radius_squared * radius_squared;
        const double radius_sixth = radius_fourth * radius_squared;
        const double radius_eighth = radius_fourth * radius_fourth;
        const double radial_delta = distortion.radialK1 * radius_squared + distortion.radialK2 * radius_fourth +
                                    distortion.radialK3 * radius_sixth + distortion.radialK4 * radius_eighth;
        if (distortion.tangentialConvention == BrownTangentialConvention::Metashape)
        {
            double diagonal_x = 3.0 * x;
            diagonal_x *= x;
            diagonal_x += y_squared;
            double diagonal_y = 3.0 * y;
            diagonal_y *= y;
            diagonal_y += x_squared;
            double cross_x = distortion.tangentialP2 + distortion.tangentialP2;
            cross_x *= x;
            cross_x *= y;
            double cross_y = distortion.tangentialP1 + distortion.tangentialP1;
            cross_y *= x;
            cross_y *= y;
            const double tangential_scale =
                1.0 + distortion.tangentialP3 * radius_squared + distortion.tangentialP4 * radius_fourth;
            const double tangential_x = (distortion.tangentialP1 * diagonal_x + cross_x) * tangential_scale;
            const double tangential_y = (cross_y + distortion.tangentialP2 * diagonal_y) * tangential_scale;
            *distortedX = x + (x * radial_delta + tangential_x);
            *distortedY = y + (y * radial_delta + tangential_y);
            return;
        }

        const double radial = 1.0 + radial_delta;
        *distortedX = x * radial + 2.0 * distortion.tangentialP1 * x * y +
                      distortion.tangentialP2 * (radius_squared + 2.0 * x_squared);
        *distortedY = y * radial + distortion.tangentialP1 * (radius_squared + 2.0 * y_squared) +
                      2.0 * distortion.tangentialP2 * x * y;
    }

    Vector3 worldToCamera(const Pose& pose, const Vector3& world) noexcept
    {
        const Vector3 offset{world[0] - pose.center[0], world[1] - pose.center[1], world[2] - pose.center[2]};
        return transposeMultiply(pose.cameraToWorldRotation, offset);
    }

    Vector3 cameraToWorld(const Pose& pose, const Vector3& camera) noexcept
    {
        const Vector3 offset = multiply(pose.cameraToWorldRotation, camera);
        return {pose.center[0] + offset[0], pose.center[1] + offset[1], pose.center[2] + offset[2]};
    }

    EvaluationResult<double> signedFrameDepth(const FrameId& groundFrame,
                                              bool depthAxisFlipped,
                                              const Pose& pose,
                                              const GroundCoordinate& ground)
    {
        if (!finiteVector(ground.position))
        {
            return EvaluationResult<double>::failure(CameraErrorCode::InvalidArgument,
                                                     "ground coordinate must be finite");
        }
        if (ground.frame != groundFrame)
        {
            return EvaluationResult<double>::failure(CameraErrorCode::FrameMismatch,
                                                     "ground coordinate frame does not match the camera model");
        }
        const Vector3 camera = worldToCamera(pose, ground.position);
        const double depth = depthAxisFlipped ? -camera[2] : camera[2];
        if (!std::isfinite(depth))
        {
            return EvaluationResult<double>::failure(CameraErrorCode::InvalidArgument,
                                                     "signed camera depth must be finite");
        }
        return EvaluationResult<double>::success(depth);
    }

    RotationMatrix normalizedCameraToWorldRotation(const RotationMatrix& cameraToWorld,
                                                   const std::array<double, 3>& axisSigns) noexcept
    {
        RotationMatrix normalized{};
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                const std::size_t index = static_cast<std::size_t>(row * 3 + column);
                normalized[index] = cameraToWorld[index] * axisSigns[static_cast<std::size_t>(column)];
            }
        }
        return normalized;
    }

    EvaluationResult<NormalizedCoordinate>
    undistort(const FramePinholeDefinition& definition, const ImageCoordinate& image, const EvaluationOptions& options)
    {
        return undistort(definition.intrinsics(),
                         definition.distortion(),
                         definition.projectionModel(),
                         definition.principalPointDecomposition(),
                         image,
                         options);
    }

    EvaluationResult<NormalizedCoordinate>
    undistort(const FrameIntrinsics& intrinsics,
              const BrownConradyDistortion& distortion,
              FrameProjectionModel projectionModel,
              const std::optional<PrincipalPointDecomposition>& principalPointDecomposition,
              const ImageCoordinate& image,
              const EvaluationOptions& options)
    {
        if (!validOptions(options) || !finiteImage(image))
        {
            return EvaluationResult<NormalizedCoordinate>::failure(
                CameraErrorCode::InvalidArgument,
                "image coordinate and evaluation options must be finite and positive");
        }
        const NormalizedCoordinate initial = planeFromPixel(intrinsics, principalPointDecomposition, image);
        if (!usesBrownDistortion(projectionModel))
        {
            return EvaluationResult<NormalizedCoordinate>::success(initial, 0.0);
        }

        double x = initial[0];
        double y = initial[1];
        if (distortion.radialK1 == 0.0 && distortion.radialK2 == 0.0 && distortion.radialK3 == 0.0 &&
            distortion.radialK4 == 0.0 && distortion.tangentialP1 == 0.0 && distortion.tangentialP2 == 0.0 &&
            distortion.tangentialP3 == 0.0 && distortion.tangentialP4 == 0.0)
        {
            return EvaluationResult<NormalizedCoordinate>::success({x, y}, 0.0);
        }

        double achieved_precision = 0.0;
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            double distorted_x = 0.0;
            double distorted_y = 0.0;
            distort(distortion, x, y, &distorted_x, &distorted_y);
            const ImageCoordinate current = pixelFromPlane(intrinsics, distorted_x, distorted_y);
            const double residual_sample = current.sample - image.sample;
            const double residual_line = current.line - image.line;
            achieved_precision = std::max(std::abs(residual_sample), std::abs(residual_line));
            if (achieved_precision <= options.desiredPrecisionPixels)
            {
                return EvaluationResult<NormalizedCoordinate>::success({x, y}, achieved_precision);
            }

            constexpr double step = 1.0e-7;
            double shifted_x = 0.0;
            double shifted_y = 0.0;
            distort(distortion, x + step, y, &shifted_x, &shifted_y);
            const ImageCoordinate shifted_pixel_x = pixelFromPlane(intrinsics, shifted_x, shifted_y);
            const double j00 = (shifted_pixel_x.sample - current.sample) / step;
            const double j10 = (shifted_pixel_x.line - current.line) / step;
            distort(distortion, x, y + step, &shifted_x, &shifted_y);
            const ImageCoordinate shifted_pixel_y = pixelFromPlane(intrinsics, shifted_x, shifted_y);
            const double j01 = (shifted_pixel_y.sample - current.sample) / step;
            const double j11 = (shifted_pixel_y.line - current.line) / step;
            const double determinant = j00 * j11 - j01 * j10;
            if (!std::isfinite(determinant) || std::abs(determinant) < 1.0e-15)
            {
                return EvaluationResult<NormalizedCoordinate>::failure(
                    CameraErrorCode::NonConvergence, "frame distortion inverse has a singular numerical Jacobian");
            }
            x -= (j11 * residual_sample - j01 * residual_line) / determinant;
            y -= (-j10 * residual_sample + j00 * residual_line) / determinant;
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                return EvaluationResult<NormalizedCoordinate>::failure(
                    CameraErrorCode::NonConvergence, "frame distortion inverse produced a non-finite iterate");
            }
        }
        return EvaluationResult<NormalizedCoordinate>::failure(
            CameraErrorCode::NonConvergence, "frame distortion inverse did not reach the requested pixel precision");
    }

    EvaluationResult<Projection> projectFrame(const FrameId& groundFrame,
                                              const ImageSize& imageSize,
                                              const std::optional<TimeReference>& captureTime,
                                              const FrameIntrinsics& intrinsics,
                                              const BrownConradyDistortion& distortion,
                                              FrameProjectionModel projectionModel,
                                              bool depthAxisFlipped,
                                              const Pose& pose,
                                              const CameraAcquisitionState& acquisition,
                                              const GroundCoordinate& ground,
                                              const EvaluationOptions& options)
    {
        EvaluationOptions signed_options = options;
        signed_options.requireInsideImage = false;
        const auto projected = projectFrameSigned(groundFrame,
                                                  imageSize,
                                                  captureTime,
                                                  intrinsics,
                                                  distortion,
                                                  projectionModel,
                                                  depthAxisFlipped,
                                                  pose,
                                                  acquisition,
                                                  ground,
                                                  signed_options);
        if (!projected)
        {
            return projected;
        }
        if (requiresPositiveDepth(projectionModel) && !projected.value().positiveDepth)
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                         "ground coordinate is outside the forward projection domain");
        }
        if (options.requireInsideImage && !insideImage(projected.value().image, imageSize))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                         "projected coordinate lies outside the image");
        }
        return projected;
    }

    EvaluationResult<Projection> projectFrameSigned(const FrameId& groundFrame,
                                                    const ImageSize& imageSize,
                                                    const std::optional<TimeReference>& captureTime,
                                                    const FrameIntrinsics& intrinsics,
                                                    const BrownConradyDistortion& distortion,
                                                    FrameProjectionModel projectionModel,
                                                    bool depthAxisFlipped,
                                                    const Pose& pose,
                                                    const CameraAcquisitionState& acquisition,
                                                    const GroundCoordinate& ground,
                                                    const EvaluationOptions& options)
    {
        if (!validOptions(options) || !finiteVector(ground.position))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::InvalidArgument,
                                                         "ground coordinate and evaluation options must be valid");
        }
        if (ground.frame != groundFrame)
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::FrameMismatch,
                                                         "ground coordinate frame does not match the camera model");
        }

        const Vector3 base_camera = worldToCamera(pose, ground.position);
        Vector3 effective_camera = base_camera;
        ImageCoordinate image{};
        if (!projectCameraPoint(intrinsics, distortion, projectionModel, effective_camera, &image))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                         "ground coordinate lies outside the projection domain");
        }
        if (acquisition.rollingShutterMode != RollingShutterMode::Disabled && !acquisition.rollingShutter.isIdentity())
        {
            for (std::size_t iteration = 0; iteration < 5; ++iteration)
            {
                effective_camera =
                    applyRollingMotion(acquisition.rollingShutter, base_camera, rollingScan(image.line, imageSize));
                ImageCoordinate next{};
                if (!projectCameraPoint(intrinsics, distortion, projectionModel, effective_camera, &next))
                {
                    return EvaluationResult<Projection>::failure(
                        CameraErrorCode::OutsideModelDomain,
                        "rolling-shutter motion moved the coordinate outside the projection domain");
                }
                const double change_squared = (next.sample - image.sample) * (next.sample - image.sample) +
                                              (next.line - image.line) * (next.line - image.line);
                image = next;
                if (iteration == 4 && change_squared > 1.0)
                {
                    return EvaluationResult<Projection>::failure(
                        CameraErrorCode::NonConvergence,
                        "rolling-shutter projection did not stabilize within five iterations");
                }
            }
        }
        if (options.requireInsideImage && !insideImage(image, imageSize))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                         "projected coordinate lies outside the image");
        }
        const double forward_depth = depthAxisFlipped ? -effective_camera[2] : effective_camera[2];
        return EvaluationResult<Projection>::success(
            Projection{
                image, forward_depth > 1.0e-9 ? std::optional<double>(forward_depth) : std::nullopt, captureTime},
            0.0);
    }

    EvaluationResult<ImagingLocus>
    frameImagingLocus(const FrameId& groundFrame,
                      const ImageSize& imageSize,
                      const std::optional<TimeReference>& captureTime,
                      const FrameIntrinsics& intrinsics,
                      const BrownConradyDistortion& distortion,
                      const std::optional<PrincipalPointDecomposition>& principalPointDecomposition,
                      FrameProjectionModel projectionModel,
                      bool depthAxisFlipped,
                      const Pose& pose,
                      const CameraAcquisitionState& acquisition,
                      const ImageCoordinate& image,
                      const EvaluationOptions& options)
    {
        if (options.requireInsideImage && (!finiteImage(image) || !insideImage(image, imageSize)))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "image coordinate lies outside the image");
        }
        const auto adjusted_bearing = cameraBearing(
            intrinsics, distortion, principalPointDecomposition, projectionModel, depthAxisFlipped, image, options);
        if (!adjusted_bearing)
        {
            return EvaluationResult<ImagingLocus>::failure(adjusted_bearing.errorCode(), adjusted_bearing.message());
        }

        Vector3 local_origin{};
        Vector3 local_direction = adjusted_bearing.value();
        if (acquisition.rollingShutterMode != RollingShutterMode::Disabled && !acquisition.rollingShutter.isIdentity())
        {
            const double scan = rollingScan(image.line, imageSize);
            local_origin = {acquisition.rollingShutter.translation[0] * scan,
                            acquisition.rollingShutter.translation[1] * scan,
                            acquisition.rollingShutter.translation[2] * scan};
            const Vector3 scaled_rotation{acquisition.rollingShutter.rotationVector[0] * scan,
                                          acquisition.rollingShutter.rotationVector[1] * scan,
                                          acquisition.rollingShutter.rotationVector[2] * scan};
            local_direction = transposeMultiply(rodrigues(scaled_rotation), local_direction);
        }
        const Vector3 world_origin = cameraToWorld(pose, local_origin);
        Vector3 world_direction = multiply(pose.cameraToWorldRotation, local_direction);
        const double norm_squared = world_direction[0] * world_direction[0] + world_direction[1] * world_direction[1] +
                                    world_direction[2] * world_direction[2];
        if (!std::isfinite(norm_squared) || norm_squared <= MinimumDirectionNorm)
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "camera ray has invalid direction");
        }
        const double inverse_norm = 1.0 / std::sqrt(norm_squared);
        for (double& value : world_direction)
        {
            value *= inverse_norm;
        }
        ImagingLocus locus{GroundCoordinate{groundFrame, world_origin}, world_direction, captureTime};
        return EvaluationResult<ImagingLocus>::success(std::move(locus), adjusted_bearing.achievedPrecisionPixels());
    }

    EvaluationResult<GroundCoordinate>
    frameGroundAtDepth(const FrameId& groundFrame,
                       const ImageSize& imageSize,
                       const FrameIntrinsics& intrinsics,
                       const BrownConradyDistortion& distortion,
                       const std::optional<PrincipalPointDecomposition>& principalPointDecomposition,
                       FrameProjectionModel projectionModel,
                       bool depthAxisFlipped,
                       const Pose& pose,
                       const CameraAcquisitionState& acquisition,
                       const ImageCoordinate& image,
                       double positiveDepth,
                       const EvaluationOptions& options)
    {
        if (!std::isfinite(positiveDepth) || positiveDepth <= 0.0)
        {
            return EvaluationResult<GroundCoordinate>::failure(CameraErrorCode::InvalidArgument,
                                                               "positive depth must be finite and greater than zero");
        }
        const auto locus = frameImagingLocus(groundFrame,
                                             imageSize,
                                             std::nullopt,
                                             intrinsics,
                                             distortion,
                                             principalPointDecomposition,
                                             projectionModel,
                                             depthAxisFlipped,
                                             pose,
                                             acquisition,
                                             image,
                                             options);
        if (!locus)
        {
            return EvaluationResult<GroundCoordinate>::failure(locus.errorCode(), locus.message());
        }
        const Vector3 local_origin = worldToCamera(pose, locus.value().origin.position);
        const Vector3 local_direction = transposeMultiply(pose.cameraToWorldRotation, locus.value().direction);
        const double target_z = depthAxisFlipped ? -positiveDepth : positiveDepth;
        if (!std::isfinite(local_direction[2]) || std::abs(local_direction[2]) < 1.0e-15)
        {
            return EvaluationResult<GroundCoordinate>::failure(CameraErrorCode::OutsideModelDomain,
                                                               "camera ray is parallel to the requested depth plane");
        }
        const double distance = (target_z - local_origin[2]) / local_direction[2];
        if (!std::isfinite(distance) || distance <= 0.0)
        {
            return EvaluationResult<GroundCoordinate>::failure(CameraErrorCode::OutsideModelDomain,
                                                               "requested positive depth lies behind the camera ray");
        }
        GroundCoordinate ground{groundFrame,
                                {locus.value().origin.position[0] + distance * locus.value().direction[0],
                                 locus.value().origin.position[1] + distance * locus.value().direction[1],
                                 locus.value().origin.position[2] + distance * locus.value().direction[2]}};
        return EvaluationResult<GroundCoordinate>::success(std::move(ground), locus.achievedPrecisionPixels());
    }

} // namespace placamera::internal
