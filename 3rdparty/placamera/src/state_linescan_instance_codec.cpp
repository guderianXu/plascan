#include "internal/state_json_common.h"

#include "placamera/linescan_camera.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace placamera::state_detail
{
    namespace
    {

        Json timeScale(TimeScale scale)
        {
            return timeReference(TimeReference::create(scale, 0.0)).at("scale");
        }

        bool readTimeScale(const Json& value, TimeScale* output)
        {
            if (!output || !value.is_string())
            {
                return false;
            }
            const Json encoded{{"scale", value}, {"seconds", 0.0}};
            TimeReference time = TimeReference::create(TimeScale::Relative, 0.0);
            if (!readTimeReference(encoded, &time))
            {
                return false;
            }
            *output = time.scale;
            return true;
        }

        Json directTrajectory(const std::vector<TrajectorySample>& samples)
        {
            Json encoded_samples = Json::array();
            for (const TrajectorySample& sample : samples)
            {
                const auto optional_vector = [](const std::optional<Vector3>& value)
                { return value ? finiteArray(*value) : Json(nullptr); };
                encoded_samples.push_back(
                    Json{{"time", timeReference(sample.time)},
                         {"center", finiteArray(sample.center)},
                         {"camera_to_world_rotation", finiteArray(sample.cameraToWorldRotation)},
                         {"constraints",
                          Json{{"position_fixed", sample.constraints.positionFixed},
                               {"rotation_fixed", sample.constraints.rotationFixed},
                               {"position_sigma_m", optional_vector(sample.constraints.positionSigmaMeters)},
                               {"rotation_sigma_rad", optional_vector(sample.constraints.rotationSigmaRadians)}}}});
            }
            return Json{{"kind", "samples"}, {"samples", std::move(encoded_samples)}};
        }

        Json rotationTrajectory(const FrameRotationTrajectory& trajectory)
        {
            Json samples = Json::array();
            for (const QuaternionTrajectorySample& sample : trajectory.samples)
            {
                samples.push_back(
                    Json{{"time", timeReference(sample.time)}, {"quaternion", finiteArray(sample.scalarFirst)}});
            }
            return Json{{"constant_rotation", finiteArray(trajectory.constantRotation)},
                        {"samples", std::move(samples)}};
        }

        Json composedTrajectory(const FrameComposedTrajectory& trajectory)
        {
            Json states = Json::array();
            for (const TranslationalStateSample& sample : trajectory.inertialStates)
            {
                states.push_back(Json{{"time", timeReference(sample.time)},
                                      {"position_m", finiteArray(sample.positionMeters)},
                                      {"velocity_mps", finiteArray(sample.velocityMetersPerSecond)}});
            }
            return Json{{"kind", "frame_composed"},
                        {"inertial_states", std::move(states)},
                        {"inertial_to_world", rotationTrajectory(trajectory.inertialToWorld)},
                        {"inertial_to_sensor", rotationTrajectory(trajectory.inertialToSensor)}};
        }

        Json trajectory(const LineScanTrajectory& value)
        {
            return value.frameComposed() ? composedTrajectory(*value.frameComposed())
                                         : directTrajectory(value.samples());
        }

        Json timing(const LineTiming& value)
        {
            Json segments = Json::array();
            for (const LineRateSegment& segment : value.segments)
            {
                segments.push_back(Json{{"start_line", segment.startLine},
                                        {"start_time_seconds", segment.startTimeSeconds},
                                        {"seconds_per_line", segment.secondsPerLine}});
            }
            return Json{{"line_zero", value.lineZero},
                        {"start_time_seconds", value.startTimeSeconds},
                        {"seconds_per_line", value.secondsPerLine},
                        {"time_scale", timeScale(value.timeScale)},
                        {"segments", std::move(segments)}};
        }

        bool readOptionalVector(const Json& value, std::optional<Vector3>* output)
        {
            if (!output)
            {
                return false;
            }
            if (value.is_null())
            {
                output->reset();
                return true;
            }
            Vector3 values{};
            if (!readFiniteArray(value, &values))
            {
                return false;
            }
            *output = values;
            return true;
        }

        bool readDirectTrajectory(const Json& root, int schemaVersion, std::optional<LineScanTrajectory>* output)
        {
            if (!output || !hasObjectKeys(root, {"kind", "samples"}) || !root.at("samples").is_array())
            {
                return false;
            }
            std::vector<TrajectorySample> samples;
            samples.reserve(root.at("samples").size());
            for (const Json& encoded : root.at("samples"))
            {
                const bool valid_keys =
                    schemaVersion == 1
                        ? hasObjectKeys(encoded, {"time", "center", "camera_to_world_rotation"})
                        : hasObjectKeys(encoded, {"time", "center", "camera_to_world_rotation", "constraints"});
                if (!valid_keys)
                {
                    return false;
                }
                TrajectorySample sample;
                if (!readTimeReference(encoded.at("time"), &sample.time) ||
                    !readFiniteArray(encoded.at("center"), &sample.center) ||
                    !readFiniteArray(encoded.at("camera_to_world_rotation"), &sample.cameraToWorldRotation))
                {
                    return false;
                }
                if (schemaVersion == 2)
                {
                    const Json& constraints = encoded.at("constraints");
                    if (!hasObjectKeys(
                            constraints,
                            {"position_fixed", "rotation_fixed", "position_sigma_m", "rotation_sigma_rad"}) ||
                        !constraints.at("position_fixed").is_boolean() ||
                        !constraints.at("rotation_fixed").is_boolean() ||
                        !readOptionalVector(constraints.at("position_sigma_m"),
                                            &sample.constraints.positionSigmaMeters) ||
                        !readOptionalVector(constraints.at("rotation_sigma_rad"),
                                            &sample.constraints.rotationSigmaRadians))
                    {
                        return false;
                    }
                    sample.constraints.positionFixed = constraints.at("position_fixed").get<bool>();
                    sample.constraints.rotationFixed = constraints.at("rotation_fixed").get<bool>();
                }
                samples.push_back(sample);
            }
            *output = LineScanTrajectory::create(std::move(samples));
            return true;
        }

        bool readRotationTrajectory(const Json& root, FrameRotationTrajectory* output)
        {
            if (!output || !hasObjectKeys(root, {"constant_rotation", "samples"}) || !root.at("samples").is_array() ||
                !readFiniteArray(root.at("constant_rotation"), &output->constantRotation))
            {
                return false;
            }
            output->samples.clear();
            output->samples.reserve(root.at("samples").size());
            for (const Json& encoded : root.at("samples"))
            {
                if (!hasObjectKeys(encoded, {"time", "quaternion"}))
                {
                    return false;
                }
                QuaternionTrajectorySample sample;
                if (!readTimeReference(encoded.at("time"), &sample.time) ||
                    !readFiniteArray(encoded.at("quaternion"), &sample.scalarFirst))
                {
                    return false;
                }
                output->samples.push_back(sample);
            }
            return true;
        }

        bool readComposedTrajectory(const Json& root, std::optional<LineScanTrajectory>* output)
        {
            if (!output ||
                !hasObjectKeys(root, {"kind", "inertial_states", "inertial_to_world", "inertial_to_sensor"}) ||
                !root.at("inertial_states").is_array())
            {
                return false;
            }
            FrameComposedTrajectory composed;
            composed.inertialStates.reserve(root.at("inertial_states").size());
            for (const Json& encoded : root.at("inertial_states"))
            {
                if (!hasObjectKeys(encoded, {"time", "position_m", "velocity_mps"}))
                {
                    return false;
                }
                TranslationalStateSample sample;
                if (!readTimeReference(encoded.at("time"), &sample.time) ||
                    !readFiniteArray(encoded.at("position_m"), &sample.positionMeters) ||
                    !readFiniteArray(encoded.at("velocity_mps"), &sample.velocityMetersPerSecond))
                {
                    return false;
                }
                composed.inertialStates.push_back(sample);
            }
            if (!readRotationTrajectory(root.at("inertial_to_world"), &composed.inertialToWorld) ||
                !readRotationTrajectory(root.at("inertial_to_sensor"), &composed.inertialToSensor))
            {
                return false;
            }
            *output = LineScanTrajectory::createFrameComposed(std::move(composed));
            return true;
        }

        bool readTrajectory(const Json& root, int schemaVersion, std::optional<LineScanTrajectory>* output)
        {
            std::string kind;
            if (!readString(root, "kind", &kind))
            {
                return false;
            }
            if (kind == "samples")
            {
                return readDirectTrajectory(root, schemaVersion, output);
            }
            if (kind == "frame_composed")
            {
                return readComposedTrajectory(root, output);
            }
            return false;
        }

        Json timeOffsetPrior(const std::optional<LineScanTimeOffsetPrior>& prior)
        {
            if (!prior)
            {
                return nullptr;
            }
            return Json{{"mean_seconds", prior->meanSeconds}, {"sigma_seconds", prior->sigmaSeconds}};
        }

        bool readTimeOffsetPrior(const Json& value, std::optional<LineScanTimeOffsetPrior>* output)
        {
            if (!output)
            {
                return false;
            }
            if (value.is_null())
            {
                output->reset();
                return true;
            }
            LineScanTimeOffsetPrior prior;
            if (!hasObjectKeys(value, {"mean_seconds", "sigma_seconds"}) ||
                !readFinite(value, "mean_seconds", &prior.meanSeconds) ||
                !readFinite(value, "sigma_seconds", &prior.sigmaSeconds) || !(prior.sigmaSeconds > 0.0))
            {
                return false;
            }
            *output = prior;
            return true;
        }

        bool readTiming(const Json& root, LineTiming* output)
        {
            if (!output ||
                !hasObjectKeys(root,
                               {"line_zero", "start_time_seconds", "seconds_per_line", "time_scale", "segments"}) ||
                !root.at("segments").is_array() || !readFinite(root, "line_zero", &output->lineZero) ||
                !readFinite(root, "start_time_seconds", &output->startTimeSeconds) ||
                !readFinite(root, "seconds_per_line", &output->secondsPerLine) ||
                !readTimeScale(root.at("time_scale"), &output->timeScale))
            {
                return false;
            }
            output->segments.clear();
            output->segments.reserve(root.at("segments").size());
            for (const Json& encoded : root.at("segments"))
            {
                if (!hasObjectKeys(encoded, {"start_line", "start_time_seconds", "seconds_per_line"}))
                {
                    return false;
                }
                LineRateSegment segment;
                if (!readFinite(encoded, "start_line", &segment.startLine) ||
                    !readFinite(encoded, "start_time_seconds", &segment.startTimeSeconds) ||
                    !readFinite(encoded, "seconds_per_line", &segment.secondsPerLine))
                {
                    return false;
                }
                output->segments.push_back(segment);
            }
            return true;
        }

    } // namespace

    Json lineScanInstanceState(const RasterModel& model)
    {
        const auto* line_scan = dynamic_cast<const LineScanModel*>(&model);
        if (!line_scan)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel,
                                        "line-scan codec received another instance type");
        }
        const LineScanTrajectoryBias& bias = line_scan->trajectoryBias();
        return Json{{"trajectory", trajectory(line_scan->trajectory())},
                    {"timing", timing(line_scan->lineTiming())},
                    {"bias",
                     Json{{"translation_m", finiteArray(bias.translationMeters)},
                          {"rotation_rad", finiteArray(bias.rotationVectorRadians)},
                          {"time_offset_seconds", bias.timeOffsetSeconds}}},
                    {"time_offset_prior", timeOffsetPrior(line_scan->timeOffsetPrior())}};
    }

    Result<ModelRegistry::ModelPointer> createLineScanInstance(const CameraInstanceState& state,
                                                               CameraModelFactory::DefinitionPointer definition)
    {
        if (state.state.schemaVersion != 1 && state.state.schemaVersion != LineScanModel::InstanceSchemaVersion)
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "unsupported line-scan instance schema");
        }
        const auto line_definition = std::dynamic_pointer_cast<const LineScanDefinition>(definition);
        if (!line_definition)
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "line-scan instance requires a line-scan definition");
        }
        const auto parsed = parseStateObject(state.state, InstanceStateJsonMediaType);
        if (!parsed)
        {
            return Result<ModelRegistry::ModelPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        const bool valid_root = state.state.schemaVersion == 1
                                    ? hasObjectKeys(root, {"trajectory", "timing", "bias"})
                                    : hasObjectKeys(root, {"trajectory", "timing", "bias", "time_offset_prior"});
        if (!valid_root || !root.at("bias").is_object() ||
            !hasObjectKeys(root.at("bias"), {"translation_m", "rotation_rad", "time_offset_seconds"}))
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "line-scan instance has missing or unknown fields");
        }

        std::optional<LineScanTrajectory> trajectory_value;
        LineTiming timing_value;
        LineScanTrajectoryBias bias;
        std::optional<LineScanTimeOffsetPrior> time_offset_prior;
        if (!readTrajectory(root.at("trajectory"), state.state.schemaVersion, &trajectory_value) ||
            !readTiming(root.at("timing"), &timing_value) ||
            !readFiniteArray(root.at("bias").at("translation_m"), &bias.translationMeters) ||
            !readFiniteArray(root.at("bias").at("rotation_rad"), &bias.rotationVectorRadians) ||
            !readFinite(root.at("bias"), "time_offset_seconds", &bias.timeOffsetSeconds) || !trajectory_value ||
            (state.state.schemaVersion == 2 && !readTimeOffsetPrior(root.at("time_offset_prior"), &time_offset_prior)))
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "line-scan trajectory, timing, or bias is invalid");
        }

        auto model = LineScanModel::create(state.instanceId,
                                           state.imageId,
                                           line_definition,
                                           state.imageSize,
                                           std::move(*trajectory_value),
                                           std::move(timing_value),
                                           bias,
                                           state.captureTime,
                                           time_offset_prior);
        return Result<ModelRegistry::ModelPointer>::success(std::make_shared<const LineScanModel>(std::move(model)));
    }

} // namespace placamera::state_detail
