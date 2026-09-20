#include "LineScanProjection.h"

#include "LineScanOptics.h"
#include "camera/core/types/CameraErrors.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace xjw::camera_models::linescan
{
    namespace
    {

        using Rotation = camera_core::Rotation;
        using Vector3 = std::array<double, 3>;

        Rotation multiply(const Rotation& first, const Rotation& second)
        {
            Rotation result{};
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

        Vector3 multiply(const Rotation& rotation, const Vector3& vector)
        {
            return {rotation[0] * vector[0] + rotation[1] * vector[1] + rotation[2] * vector[2],
                    rotation[3] * vector[0] + rotation[4] * vector[1] + rotation[5] * vector[2],
                    rotation[6] * vector[0] + rotation[7] * vector[1] + rotation[8] * vector[2]};
        }

        Vector3 transposeMultiply(const Rotation& rotation, const Vector3& vector)
        {
            return {rotation[0] * vector[0] + rotation[3] * vector[1] + rotation[6] * vector[2],
                    rotation[1] * vector[0] + rotation[4] * vector[1] + rotation[7] * vector[2],
                    rotation[2] * vector[0] + rotation[5] * vector[1] + rotation[8] * vector[2]};
        }

        bool normalize(Vector3* vector)
        {
            const double norm = std::sqrt((*vector)[0] * (*vector)[0] + (*vector)[1] * (*vector)[1] +
                                          (*vector)[2] * (*vector)[2]);
            if (!std::isfinite(norm) || !(norm > 0.0))
            {
                return false;
            }
            for (double& value : *vector)
            {
                value /= norm;
            }
            return true;
        }

        Rotation angleAxisRotation(const Vector3& rotationVector)
        {
            const double angle = std::sqrt(rotationVector[0] * rotationVector[0] +
                                           rotationVector[1] * rotationVector[1] +
                                           rotationVector[2] * rotationVector[2]);
            if (!(angle > 0.0))
            {
                return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            }
            const double x = rotationVector[0] / angle;
            const double y = rotationVector[1] / angle;
            const double z = rotationVector[2] / angle;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            const double oneMinusCosine = 1.0 - cosine;
            return {cosine + x * x * oneMinusCosine,
                    x * y * oneMinusCosine - z * sine,
                    x * z * oneMinusCosine + y * sine,
                    y * x * oneMinusCosine + z * sine,
                    cosine + y * y * oneMinusCosine,
                    y * z * oneMinusCosine - x * sine,
                    z * x * oneMinusCosine - y * sine,
                    z * y * oneMinusCosine + x * sine,
                    cosine + z * z * oneMinusCosine};
        }

        bool finiteBias(const LineScanTrajectoryBias& bias)
        {
            return std::all_of(bias.translationMeters.begin(), bias.translationMeters.end(), [](double value)
                               { return std::isfinite(value); }) &&
                   std::all_of(bias.rotationVectorRadians.begin(), bias.rotationVectorRadians.end(), [](double value)
                               { return std::isfinite(value); }) &&
                   std::isfinite(bias.timeOffsetSeconds);
        }

        bool poseForLine(const LineScanInstance& instance,
                         double line,
                         const LineScanTrajectoryBias& bias,
                         std::optional<camera_core::Pose>* pose,
                         double* timeSeconds)
        {
            double nominalTime = 0.0;
            if (!pose || !timeSeconds || !finiteBias(bias) || !instance.timeForLine(line, &nominalTime))
            {
                return false;
            }
            *timeSeconds = nominalTime + bias.timeOffsetSeconds;
            try
            {
                camera_core::Pose nominal = instance.trajectory().poseAt(
                    xjw::coordinate_system::TimeReference::create(instance.lineTiming().timeScale, *timeSeconds),
                    instance.definition().worldFrame());
                for (int axis = 0; axis < 3; ++axis)
                {
                    nominal.center[static_cast<std::size_t>(axis)] +=
                        bias.translationMeters[static_cast<std::size_t>(axis)];
                }
                pose->emplace(camera_core::Pose::create(nominal.frame,
                                                        nominal.center,
                                                        multiply(angleAxisRotation(bias.rotationVectorRadians),
                                                                 nominal.cameraToWorldRotation)));
            }
            catch (const camera_core::CameraValidationError&)
            {
                return false;
            }
            return true;
        }

    } // namespace

    bool LineScanProjection::ray(const LineScanInstance& instance, double sample, double line, LineScanRay* result)
    {
        return ray(instance, sample, line, LineScanTrajectoryBias{}, result);
    }

    bool LineScanProjection::ray(const LineScanInstance& instance,
                                 double sample,
                                 double line,
                                 const LineScanTrajectoryBias& bias,
                                 LineScanRay* result)
    {
        if (!result)
        {
            return false;
        }
        std::optional<camera_core::Pose> pose;
        double timeSeconds = 0.0;
        FocalPlaneCoordinate focal;
        if (!poseForLine(instance, line, bias, &pose, &timeSeconds) ||
            !LineScanOpticsTransform::pixelToUndistortedFocal(instance.lineScanDefinition(), sample, &focal))
        {
            return false;
        }
        Vector3 sensorDirection{focal.xMillimeters,
                                focal.yMillimeters,
                                instance.lineScanDefinition().optics().focalLengthMillimeters};
        if (!normalize(&sensorDirection))
        {
            return false;
        }
        result->origin = pose->center;
        result->direction = multiply(pose->cameraToWorldRotation, sensorDirection);
        result->timeSeconds = timeSeconds;
        return normalize(&result->direction);
    }

    bool LineScanProjection::projectAtLine(const LineScanInstance& instance,
                                           const Vector3& world,
                                           double line,
                                           LineScanProjectionResult* result)
    {
        return projectAtLine(instance, world, line, LineScanTrajectoryBias{}, result);
    }

    bool LineScanProjection::projectAtLine(const LineScanInstance& instance,
                                           const Vector3& world,
                                           double line,
                                           const LineScanTrajectoryBias& bias,
                                           LineScanProjectionResult* result)
    {
        if (!result || !std::all_of(world.begin(), world.end(), [](double value) { return std::isfinite(value); }))
        {
            return false;
        }
        std::optional<camera_core::Pose> pose;
        double timeSeconds = 0.0;
        if (!poseForLine(instance, line, bias, &pose, &timeSeconds))
        {
            return false;
        }
        const Vector3 offset{world[0] - pose->center[0],
                             world[1] - pose->center[1],
                             world[2] - pose->center[2]};
        const Vector3 sensor = transposeMultiply(pose->cameraToWorldRotation, offset);
        if (!std::isfinite(sensor[2]) || !(sensor[2] > 1.0e-9))
        {
            return false;
        }
        const double focalLength = instance.lineScanDefinition().optics().focalLengthMillimeters;
        const FocalPlaneCoordinate focal{focalLength * sensor[0] / sensor[2],
                                         focalLength * sensor[1] / sensor[2]};
        double sample = 0.0;
        double lineResidual = 0.0;
        if (!LineScanOpticsTransform::undistortedFocalToPixel(
                instance.lineScanDefinition(), focal, &sample, &lineResidual))
        {
            return false;
        }
        *result = {sample,
                   line,
                   lineResidual,
                   sensor[2],
                   timeSeconds,
                   focal.xMillimeters,
                   focal.yMillimeters};
        return true;
    }

    bool LineScanProjection::project(const LineScanInstance& instance,
                                     const Vector3& world,
                                     LineScanProjectionResult* result,
                                     int maximumIterations)
    {
        LineScanProjectionOptions options;
        options.maximumIterations = maximumIterations;
        return project(instance, world, options, LineScanTrajectoryBias{}, result);
    }

    bool LineScanProjection::project(const LineScanInstance& instance,
                                     const Vector3& world,
                                     const LineScanProjectionOptions& options,
                                     const LineScanTrajectoryBias& bias,
                                     LineScanProjectionResult* result)
    {
        if (!result || options.maximumIterations < 1 || !(options.desiredLinePrecisionPixels > 0.0) ||
            !std::isfinite(options.desiredLinePrecisionPixels))
        {
            return false;
        }
        const bool pixelCenter = instance.lineScanDefinition().pixelConvention() == PixelConvention::PixelCenter;
        const double minimumLine = pixelCenter ? 0.5 : 0.0;
        const double maximumLine = minimumLine + static_cast<double>(instance.imageSize().lines - 1);
        double line0 = 0.5 * (minimumLine + maximumLine);
        double line1 = std::min(maximumLine, line0 + 0.1);
        LineScanProjectionResult projection0;
        LineScanProjectionResult projection1;
        if (!projectAtLine(instance, world, line0, bias, &projection0) ||
            !projectAtLine(instance, world, line1, bias, &projection1))
        {
            return false;
        }
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            if (std::abs(projection1.lineResidualPixels) <= options.desiredLinePrecisionPixels)
            {
                if (options.requireInsideImage)
                {
                    const double minimumSample = pixelCenter ? 0.5 : 0.0;
                    const double maximumSample = minimumSample + static_cast<double>(instance.imageSize().samples - 1);
                    if (projection1.sample < minimumSample || projection1.sample > maximumSample)
                    {
                        return false;
                    }
                }
                *result = projection1;
                return true;
            }
            const double denominator = projection1.lineResidualPixels - projection0.lineResidualPixels;
            if (std::abs(denominator) < 1.0e-15)
            {
                return false;
            }
            const double nextLine = std::clamp(
                line1 - projection1.lineResidualPixels * (line1 - line0) / denominator, minimumLine, maximumLine);
            line0 = line1;
            projection0 = projection1;
            line1 = nextLine;
            if (!projectAtLine(instance, world, line1, bias, &projection1))
            {
                return false;
            }
        }
        return false;
    }

} // namespace xjw::camera_models::linescan
