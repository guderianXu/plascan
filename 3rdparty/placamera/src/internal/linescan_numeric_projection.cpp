#include "linescan_numeric_projection.h"

#include "linescan_projection.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace placamera::internal
{
    namespace
    {

        using Quaternion = std::array<double, 4>;

        bool validOptions(const EvaluationOptions& options) noexcept
        {
            return std::isfinite(options.desiredPrecisionPixels) && options.desiredPrecisionPixels > 0.0 &&
                   options.maximumIterations > 0;
        }

        double minimumCoordinate(LineScanPixelConvention convention) noexcept
        {
            return convention == LineScanPixelConvention::PixelCenter ? 0.5 : 0.0;
        }

        RotationMatrix multiply(const RotationMatrix& first, const RotationMatrix& second) noexcept
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

        bool normalize(Vector3* vector) noexcept
        {
            const double norm = std::hypot((*vector)[0], std::hypot((*vector)[1], (*vector)[2]));
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

        Quaternion normalize(Quaternion value) noexcept
        {
            const double norm =
                std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2] + value[3] * value[3]);
            for (double& component : value)
            {
                component /= norm;
            }
            return value;
        }

        Quaternion matrixToQuaternion(const RotationMatrix& matrix) noexcept
        {
            const double trace = matrix[0] + matrix[4] + matrix[8];
            Quaternion result{};
            if (trace > 0.0)
            {
                const double scale = 0.5 / std::sqrt(trace + 1.0);
                result = {0.25 / scale,
                          (matrix[7] - matrix[5]) * scale,
                          (matrix[2] - matrix[6]) * scale,
                          (matrix[3] - matrix[1]) * scale};
            }
            else if (matrix[0] > matrix[4] && matrix[0] > matrix[8])
            {
                const double scale = 2.0 * std::sqrt(1.0 + matrix[0] - matrix[4] - matrix[8]);
                result = {(matrix[7] - matrix[5]) / scale,
                          0.25 * scale,
                          (matrix[1] + matrix[3]) / scale,
                          (matrix[2] + matrix[6]) / scale};
            }
            else if (matrix[4] > matrix[8])
            {
                const double scale = 2.0 * std::sqrt(1.0 + matrix[4] - matrix[0] - matrix[8]);
                result = {(matrix[2] - matrix[6]) / scale,
                          (matrix[1] + matrix[3]) / scale,
                          0.25 * scale,
                          (matrix[5] + matrix[7]) / scale};
            }
            else
            {
                const double scale = 2.0 * std::sqrt(1.0 + matrix[8] - matrix[0] - matrix[4]);
                result = {(matrix[3] - matrix[1]) / scale,
                          (matrix[2] + matrix[6]) / scale,
                          (matrix[5] + matrix[7]) / scale,
                          0.25 * scale};
            }
            return normalize(result);
        }

        RotationMatrix quaternionToMatrix(const Quaternion& input) noexcept
        {
            const Quaternion q = normalize(input);
            const double w = q[0];
            const double x = q[1];
            const double y = q[2];
            const double z = q[3];
            return {1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y)};
        }

        Quaternion slerp(Quaternion first, Quaternion second, double fraction) noexcept
        {
            first = normalize(first);
            second = normalize(second);
            double dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2] + first[3] * second[3];
            if (dot < 0.0)
            {
                dot = -dot;
                for (double& component : second)
                {
                    component = -component;
                }
            }
            dot = std::clamp(dot, -1.0, 1.0);
            if (dot > 0.9995)
            {
                Quaternion result{};
                for (std::size_t index = 0; index < result.size(); ++index)
                {
                    result[index] = first[index] + fraction * (second[index] - first[index]);
                }
                return normalize(result);
            }
            const double angle = std::acos(dot);
            const double sine = std::sin(angle);
            const double first_weight = std::sin((1.0 - fraction) * angle) / sine;
            const double second_weight = std::sin(fraction * angle) / sine;
            Quaternion result{};
            for (std::size_t index = 0; index < result.size(); ++index)
            {
                result[index] = first_weight * first[index] + second_weight * second[index];
            }
            return result;
        }

        bool poseAt(const LineScanNumericState& state,
                    double seconds,
                    Vector3* center,
                    RotationMatrix* cameraToWorld) noexcept
        {
            const auto& samples = state.trajectorySamples();
            if (seconds < samples.front().time.seconds || seconds > samples.back().time.seconds)
            {
                return false;
            }
            const auto upper = std::upper_bound(samples.begin(),
                                                samples.end(),
                                                seconds,
                                                [](double value, const TrajectorySample& sample)
                                                { return value < sample.time.seconds; });
            const std::size_t first_index =
                upper == samples.begin() ? 0 : static_cast<std::size_t>(std::distance(samples.begin(), upper) - 1);
            if (first_index >= samples.size() - 1)
            {
                *center = samples.back().center;
                *cameraToWorld = samples.back().cameraToWorldRotation;
            }
            else
            {
                const auto& first = samples[first_index];
                const auto& second = samples[first_index + 1];
                const double fraction =
                    std::clamp((seconds - first.time.seconds) / (second.time.seconds - first.time.seconds), 0.0, 1.0);
                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    (*center)[axis] = first.center[axis] + fraction * (second.center[axis] - first.center[axis]);
                }
                *cameraToWorld = quaternionToMatrix(slerp(matrixToQuaternion(first.cameraToWorldRotation),
                                                          matrixToQuaternion(second.cameraToWorldRotation),
                                                          fraction));
            }
            const auto& bias = state.trajectoryBias();
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                (*center)[axis] += bias.translationMeters[axis];
            }
            *cameraToWorld = multiply(angleAxisRotation(bias.rotationVectorRadians), *cameraToWorld);
            return true;
        }

        bool insideImage(const LineScanNumericState& state, const ImageCoordinate& image) noexcept
        {
            const double minimum = minimumCoordinate(state.pixelConvention());
            return image.sample >= minimum && image.line >= minimum &&
                   image.sample <= minimum + static_cast<double>(state.imageSize().samples - 1) &&
                   image.line <= minimum + static_cast<double>(state.imageSize().lines - 1);
        }

    } // namespace

    EvaluationResult<LineScanProjectionDetails> projectLineScanNumericAtLine(const LineScanNumericState& state,
                                                                             const GroundCoordinate& ground,
                                                                             double line,
                                                                             const EvaluationOptions& options)
    {
        if (!validOptions(options) || !std::isfinite(line) ||
            !std::all_of(
                ground.position.begin(), ground.position.end(), [](double value) { return std::isfinite(value); }))
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(
                CameraErrorCode::InvalidArgument,
                "line-scan projection inputs and options must be finite and positive");
        }
        if (ground.frame != state.groundFrame())
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(
                CameraErrorCode::FrameMismatch, "ground coordinate frame does not match the line-scan state");
        }
        const auto nominal_time = state.timeForLine(line);
        if (!nominal_time)
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(nominal_time.error());
        }
        const double acquisition_seconds = nominal_time.value().seconds + state.trajectoryBias().timeOffsetSeconds;
        Vector3 center{};
        RotationMatrix camera_to_world{};
        if (!poseAt(state, acquisition_seconds, &center, &camera_to_world))
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(CameraErrorCode::OutsideModelDomain,
                                                                        "line is outside line-scan trajectory support");
        }

        const Vector3 offset{
            ground.position[0] - center[0], ground.position[1] - center[1], ground.position[2] - center[2]};
        const Vector3 sensor = transposeMultiply(camera_to_world, offset);
        if (!std::isfinite(sensor[2]) || sensor[2] <= 1.0e-9)
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(
                CameraErrorCode::OutsideModelDomain, "ground coordinate is not in front of the line-scan sensor");
        }

        const FocalPlaneCoordinate focal{state.optics().focalLengthMillimeters * sensor[0] / sensor[2],
                                         state.optics().focalLengthMillimeters * sensor[1] / sensor[2]};
        const auto detector =
            internal::lineScanUndistortedFocalToPixel(state.optics(), state.pixelConvention(), focal, options);
        if (!detector)
        {
            return EvaluationResult<LineScanProjectionDetails>::failure(detector.error());
        }
        const Projection projection{ImageCoordinate{detector.value().sample, line},
                                    sensor[2],
                                    TimeReference{nominal_time.value().scale, acquisition_seconds}};
        return EvaluationResult<LineScanProjectionDetails>::success(
            {projection, detector.value().lineResidualPixels, focal}, std::abs(detector.value().lineResidualPixels));
    }

    EvaluationResult<Projection> projectLineScanNumeric(const LineScanNumericState& state,
                                                        const GroundCoordinate& ground,
                                                        const EvaluationOptions& options)
    {
        if (!validOptions(options))
        {
            return EvaluationResult<Projection>::failure(CameraErrorCode::InvalidArgument,
                                                         "line-scan evaluation options must be finite and positive");
        }
        const double minimum_line = minimumCoordinate(state.pixelConvention());
        const double maximum_line = minimum_line + static_cast<double>(state.imageSize().lines - 1);
        double line0 = 0.5 * (minimum_line + maximum_line);
        double line1 = std::min(maximum_line, line0 + 0.1);
        auto first = projectLineScanNumericAtLine(state, ground, line0, options);
        if (!first)
        {
            return EvaluationResult<Projection>::failure(first.error());
        }
        if (line1 == line0)
        {
            if (std::abs(first.value().lineResidualPixels) <= options.desiredPrecisionPixels)
            {
                return EvaluationResult<Projection>::success(first.value().projection, first.achievedPrecisionPixels());
            }
            return EvaluationResult<Projection>::failure(CameraErrorCode::NonConvergence,
                                                         "single-line sensor residual exceeds the requested precision");
        }
        auto second = projectLineScanNumericAtLine(state, ground, line1, options);
        if (!second)
        {
            return EvaluationResult<Projection>::failure(second.error());
        }
        for (int iteration = 0; iteration < options.maximumIterations; ++iteration)
        {
            if (std::abs(second.value().lineResidualPixels) <= options.desiredPrecisionPixels)
            {
                if (options.requireInsideImage && !insideImage(state, second.value().projection.image))
                {
                    return EvaluationResult<Projection>::failure(CameraErrorCode::OutsideModelDomain,
                                                                 "line-scan projection lies outside the image");
                }
                return EvaluationResult<Projection>::success(second.value().projection,
                                                             second.achievedPrecisionPixels());
            }
            const double denominator = second.value().lineResidualPixels - first.value().lineResidualPixels;
            if (std::abs(denominator) < 1.0e-15)
            {
                break;
            }
            const double next_line = std::clamp(
                line1 - second.value().lineResidualPixels * (line1 - line0) / denominator, minimum_line, maximum_line);
            line0 = line1;
            first = second;
            line1 = next_line;
            second = projectLineScanNumericAtLine(state, ground, line1, options);
            if (!second)
            {
                return EvaluationResult<Projection>::failure(second.error());
            }
        }
        return EvaluationResult<Projection>::failure(CameraErrorCode::NonConvergence,
                                                     "line-scan projection did not reach the requested line precision");
    }

    EvaluationResult<ImagingLocus> lineScanNumericImagingLocus(const LineScanNumericState& state,
                                                               const ImageCoordinate& image,
                                                               const EvaluationOptions& options)
    {
        if (!validOptions(options) || !std::isfinite(image.sample) || !std::isfinite(image.line))
        {
            return EvaluationResult<ImagingLocus>::failure(
                CameraErrorCode::InvalidArgument, "line-scan image coordinate and options must be finite and positive");
        }
        if (options.requireInsideImage && !insideImage(state, image))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "image coordinate lies outside the line-scan image");
        }
        const auto nominal_time = state.timeForLine(image.line);
        const auto focal =
            internal::lineScanPixelToUndistortedFocal(state.optics(), state.pixelConvention(), image.sample, options);
        if (!nominal_time || !focal)
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "line-scan ray lies outside timing or optical support");
        }
        const double acquisition_seconds = nominal_time.value().seconds + state.trajectoryBias().timeOffsetSeconds;
        Vector3 center{};
        RotationMatrix camera_to_world{};
        if (!poseAt(state, acquisition_seconds, &center, &camera_to_world))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "line-scan ray lies outside trajectory support");
        }
        Vector3 sensor_direction{
            focal.value().xMillimeters, focal.value().yMillimeters, state.optics().focalLengthMillimeters};
        if (!normalize(&sensor_direction))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "line-scan sensor ray is invalid");
        }
        Vector3 direction = multiply(camera_to_world, sensor_direction);
        if (!normalize(&direction))
        {
            return EvaluationResult<ImagingLocus>::failure(CameraErrorCode::OutsideModelDomain,
                                                           "line-scan world ray is invalid");
        }
        return EvaluationResult<ImagingLocus>::success(
            {GroundCoordinate{state.groundFrame(), center},
             direction,
             TimeReference{nominal_time.value().scale, acquisition_seconds}});
    }

} // namespace placamera::internal
