#include "placamera/linescan_camera.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace placamera
{

    namespace
    {

        using Quaternion = std::array<double, 4>;

        Quaternion normalizeQuaternion(Quaternion value)
        {
            double norm = 0.0;
            for (const double component : value)
            {
                norm += component * component;
            }
            norm = std::sqrt(norm);
            if (!(norm > 0.0) || !std::isfinite(norm))
            {
                throw CameraValidationError(CameraErrorCode::InvalidPose,
                                            "line-scan trajectory quaternion must be finite and non-zero");
            }
            for (double& component : value)
            {
                component /= norm;
            }
            return value;
        }

        Quaternion matrixToQuaternion(const RotationMatrix& matrix)
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
            return normalizeQuaternion(result);
        }

        RotationMatrix quaternionToMatrix(const Quaternion& input)
        {
            const Quaternion q = normalizeQuaternion(input);
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

        Quaternion slerp(Quaternion first, Quaternion second, double fraction)
        {
            first = normalizeQuaternion(first);
            second = normalizeQuaternion(second);
            double dot = 0.0;
            for (int index = 0; index < 4; ++index)
            {
                dot += first[static_cast<std::size_t>(index)] * second[static_cast<std::size_t>(index)];
            }
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
                for (int index = 0; index < 4; ++index)
                {
                    result[static_cast<std::size_t>(index)] =
                        first[static_cast<std::size_t>(index)] +
                        fraction * (second[static_cast<std::size_t>(index)] - first[static_cast<std::size_t>(index)]);
                }
                return normalizeQuaternion(result);
            }

            const double angle = std::acos(dot);
            const double sine = std::sin(angle);
            const double first_weight = std::sin((1.0 - fraction) * angle) / sine;
            const double second_weight = std::sin(fraction * angle) / sine;
            Quaternion result{};
            for (int index = 0; index < 4; ++index)
            {
                result[static_cast<std::size_t>(index)] = first_weight * first[static_cast<std::size_t>(index)] +
                                                          second_weight * second[static_cast<std::size_t>(index)];
            }
            return result;
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

        RotationMatrix transpose(const RotationMatrix& matrix) noexcept
        {
            return {matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]};
        }

        Vector3 multiply(const RotationMatrix& matrix, const Vector3& vector) noexcept
        {
            return {matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
                    matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
                    matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]};
        }

        template <typename Sample> std::size_t lowerInterval(const std::vector<Sample>& samples, double seconds)
        {
            const auto upper = std::upper_bound(samples.begin(),
                                                samples.end(),
                                                seconds,
                                                [](double candidate, const Sample& sample)
                                                { return candidate < sample.time.seconds; });
            if (upper == samples.begin())
            {
                return 0;
            }
            return std::min<std::size_t>(samples.size() - 2,
                                         static_cast<std::size_t>(std::distance(samples.begin(), upper) - 1));
        }

        bool supports(const std::vector<QuaternionTrajectorySample>& samples, double seconds) noexcept
        {
            return samples.size() >= 2 && seconds >= samples.front().time.seconds - 1.0e-9 &&
                   seconds <= samples.back().time.seconds + 1.0e-9;
        }

        RotationMatrix rotationAt(const FrameRotationTrajectory& trajectory, double seconds)
        {
            const std::size_t index = lowerInterval(trajectory.samples, seconds);
            const QuaternionTrajectorySample& first = trajectory.samples[index];
            const QuaternionTrajectorySample& second = trajectory.samples[index + 1];
            const double fraction =
                std::clamp((seconds - first.time.seconds) / (second.time.seconds - first.time.seconds), 0.0, 1.0);
            return multiply(trajectory.constantRotation,
                            quaternionToMatrix(slerp(first.scalarFirst, second.scalarFirst, fraction)));
        }

        void validateQuaternionTrajectory(const FrameRotationTrajectory& trajectory, TimeScale scale, const char* label)
        {
            if (trajectory.samples.size() < 2)
            {
                throw CameraValidationError(CameraErrorCode::InvalidTime,
                                            std::string(label) + " requires at least two samples");
            }
            (void)Pose::create(FrameId("trajectory-validation"), {0.0, 0.0, 0.0}, trajectory.constantRotation);
            for (std::size_t index = 0; index < trajectory.samples.size(); ++index)
            {
                const QuaternionTrajectorySample& sample = trajectory.samples[index];
                if (sample.time.scale != scale || !std::isfinite(sample.time.seconds) ||
                    (index > 0 && sample.time.seconds <= trajectory.samples[index - 1].time.seconds))
                {
                    throw CameraValidationError(CameraErrorCode::InvalidTime,
                                                std::string(label) + " times are invalid");
                }
                (void)normalizeQuaternion(sample.scalarFirst);
            }
        }

        EvaluationResult<Pose> makePose(FrameId frame, Vector3 center, RotationMatrix rotation)
        {
            try
            {
                return EvaluationResult<Pose>::success(Pose::create(std::move(frame), center, rotation));
            }
            catch (const CameraValidationError& error)
            {
                return EvaluationResult<Pose>::failure(error.code(), error.what());
            }
        }

    } // namespace

    TrajectorySample::TrajectorySample(TimeReference sampleTime,
                                       Vector3 sampleCenter,
                                       RotationMatrix sampleCameraToWorldRotation,
                                       TrajectoryKnotConstraints sampleConstraints)
        : time(sampleTime), center(sampleCenter), cameraToWorldRotation(sampleCameraToWorldRotation),
          constraints(std::move(sampleConstraints))
    {
    }

    LineScanTrajectory LineScanTrajectory::create(std::vector<TrajectorySample> samples)
    {
        if (samples.size() < 2)
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime,
                                        "line-scan trajectory requires at least two samples");
        }
        const TimeScale scale = samples.front().time.scale;
        for (std::size_t index = 0; index < samples.size(); ++index)
        {
            const TrajectorySample& sample = samples[index];
            if (sample.time.scale != scale || !std::isfinite(sample.time.seconds) ||
                (index > 0 && sample.time.seconds <= samples[index - 1].time.seconds))
            {
                throw CameraValidationError(CameraErrorCode::InvalidTime,
                                            "line-scan trajectory times must be finite, same-scale, and increasing");
            }
            (void)Pose::create(FrameId("trajectory-validation"), sample.center, sample.cameraToWorldRotation);
            const auto valid_sigma = [](const std::optional<Vector3>& sigma)
            {
                return !sigma || std::all_of(sigma->begin(),
                                             sigma->end(),
                                             [](double value) { return std::isfinite(value) && value > 0.0; });
            };
            if (!valid_sigma(sample.constraints.positionSigmaMeters) ||
                !valid_sigma(sample.constraints.rotationSigmaRadians))
            {
                throw CameraValidationError(CameraErrorCode::InvalidModelState,
                                            "line-scan knot prior sigmas must be finite and positive");
            }
        }
        return LineScanTrajectory(std::move(samples));
    }

    LineScanTrajectory LineScanTrajectory::createFrameComposed(FrameComposedTrajectory trajectory)
    {
        if (trajectory.inertialStates.size() < 2)
        {
            throw CameraValidationError(CameraErrorCode::InvalidTime,
                                        "frame-composed trajectory requires at least two states");
        }
        const TimeScale scale = trajectory.inertialStates.front().time.scale;
        for (std::size_t index = 0; index < trajectory.inertialStates.size(); ++index)
        {
            const TranslationalStateSample& state = trajectory.inertialStates[index];
            const bool finite_state = std::all_of(state.positionMeters.begin(),
                                                  state.positionMeters.end(),
                                                  [](double value) { return std::isfinite(value); }) &&
                                      std::all_of(state.velocityMetersPerSecond.begin(),
                                                  state.velocityMetersPerSecond.end(),
                                                  [](double value) { return std::isfinite(value); });
            if (state.time.scale != scale || !std::isfinite(state.time.seconds) || !finite_state ||
                (index > 0 && state.time.seconds <= trajectory.inertialStates[index - 1].time.seconds))
            {
                throw CameraValidationError(CameraErrorCode::InvalidTime,
                                            "frame-composed translational states are invalid");
            }
        }
        validateQuaternionTrajectory(trajectory.inertialToWorld, scale, "inertial-to-world trajectory");
        validateQuaternionTrajectory(trajectory.inertialToSensor, scale, "inertial-to-sensor trajectory");
        return LineScanTrajectory(std::move(trajectory));
    }

    const std::vector<TrajectorySample>& LineScanTrajectory::samples() const noexcept
    {
        return _samples;
    }

    const FrameComposedTrajectory* LineScanTrajectory::frameComposed() const noexcept
    {
        return _frameComposed ? &*_frameComposed : nullptr;
    }

    TimeScale LineScanTrajectory::timeScale() const noexcept
    {
        return _timeScale;
    }

    EvaluationResult<Pose> LineScanTrajectory::poseAt(TimeReference time, const FrameId& frame) const
    {
        if (time.scale != _timeScale || !std::isfinite(time.seconds))
        {
            return EvaluationResult<Pose>::failure(CameraErrorCode::InvalidTime,
                                                   "line-scan pose time uses the wrong time scale");
        }
        if (_frameComposed)
        {
            const FrameComposedTrajectory& trajectory = *_frameComposed;
            if (time.seconds < trajectory.inertialStates.front().time.seconds ||
                time.seconds > trajectory.inertialStates.back().time.seconds ||
                !supports(trajectory.inertialToWorld.samples, time.seconds) ||
                !supports(trajectory.inertialToSensor.samples, time.seconds))
            {
                return EvaluationResult<Pose>::failure(CameraErrorCode::OutsideModelDomain,
                                                       "line-scan pose time is outside composed trajectory support");
            }

            const std::size_t state_index = lowerInterval(trajectory.inertialStates, time.seconds);
            const TranslationalStateSample& first = trajectory.inertialStates[state_index];
            const TranslationalStateSample& second = trajectory.inertialStates[state_index + 1];
            const double duration = second.time.seconds - first.time.seconds;
            const double fraction = std::clamp((time.seconds - first.time.seconds) / duration, 0.0, 1.0);
            const double h00 = 2.0 * fraction * fraction * fraction - 3.0 * fraction * fraction + 1.0;
            const double h10 = fraction * fraction * fraction - 2.0 * fraction * fraction + fraction;
            const double h01 = -2.0 * fraction * fraction * fraction + 3.0 * fraction * fraction;
            const double h11 = fraction * fraction * fraction - fraction * fraction;
            Vector3 inertial_center{};
            for (int axis = 0; axis < 3; ++axis)
            {
                inertial_center[static_cast<std::size_t>(axis)] =
                    h00 * first.positionMeters[static_cast<std::size_t>(axis)] +
                    h10 * duration * first.velocityMetersPerSecond[static_cast<std::size_t>(axis)] +
                    h01 * second.positionMeters[static_cast<std::size_t>(axis)] +
                    h11 * duration * second.velocityMetersPerSecond[static_cast<std::size_t>(axis)];
            }
            const RotationMatrix inertial_to_world = rotationAt(trajectory.inertialToWorld, time.seconds);
            const RotationMatrix inertial_to_sensor = rotationAt(trajectory.inertialToSensor, time.seconds);
            return makePose(frame,
                            multiply(inertial_to_world, inertial_center),
                            multiply(inertial_to_world, transpose(inertial_to_sensor)));
        }

        if (time.seconds < _samples.front().time.seconds || time.seconds > _samples.back().time.seconds)
        {
            return EvaluationResult<Pose>::failure(CameraErrorCode::OutsideModelDomain,
                                                   "line-scan pose time is outside trajectory support");
        }
        const auto upper = std::upper_bound(_samples.begin(),
                                            _samples.end(),
                                            time.seconds,
                                            [](double seconds, const TrajectorySample& sample)
                                            { return seconds < sample.time.seconds; });
        const std::size_t upper_index =
            upper == _samples.begin() ? 0 : static_cast<std::size_t>(std::distance(_samples.begin(), upper) - 1);
        if (upper_index >= _samples.size() - 1)
        {
            const TrajectorySample& last = _samples.back();
            return makePose(frame, last.center, last.cameraToWorldRotation);
        }

        const TrajectorySample& first = _samples[upper_index];
        const TrajectorySample& second = _samples[upper_index + 1];
        const double fraction = (time.seconds - first.time.seconds) / (second.time.seconds - first.time.seconds);
        const Quaternion orientation = slerp(matrixToQuaternion(first.cameraToWorldRotation),
                                             matrixToQuaternion(second.cameraToWorldRotation),
                                             std::clamp(fraction, 0.0, 1.0));
        const Vector3 center{first.center[0] + fraction * (second.center[0] - first.center[0]),
                             first.center[1] + fraction * (second.center[1] - first.center[1]),
                             first.center[2] + fraction * (second.center[2] - first.center[2])};
        return makePose(frame, center, quaternionToMatrix(orientation));
    }

    LineScanTrajectory::LineScanTrajectory(std::vector<TrajectorySample> samples)
        : _samples(std::move(samples)), _timeScale(_samples.front().time.scale)
    {
    }

    LineScanTrajectory::LineScanTrajectory(FrameComposedTrajectory trajectory)
        : _frameComposed(std::move(trajectory)), _timeScale(_frameComposed->inertialStates.front().time.scale)
    {
    }

} // namespace placamera
