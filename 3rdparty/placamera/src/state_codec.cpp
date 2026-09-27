#include "placamera/state_codec.h"

#include "internal/state_json_common.h"
#include "placamera/frame_camera.h"
#include "placamera/linescan_camera.h"
#include "placamera/rpc_camera.h"

#include <array>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace placamera
{
    namespace
    {

        using state_detail::Json;

        Result<Json> definitionParameters(const CameraDefinition& definition)
        {
            if (dynamic_cast<const FramePinholeDefinition*>(&definition))
            {
                return Result<Json>::success(state_detail::frameDefinitionParameters(definition));
            }
            if (definition.modelType() == "rpc00b")
            {
                return Result<Json>::success(state_detail::rpcDefinitionParameters(definition));
            }
            if (definition.modelType() == "planetary_linescan")
            {
                return Result<Json>::success(state_detail::lineScanDefinitionParameters(definition));
            }
            return Result<Json>::failure(CameraErrorCode::UnsupportedModel,
                                         "camera definition type has no built-in JSON codec");
        }

        Result<Json> instanceState(const RasterModel& model)
        {
            if (dynamic_cast<const FramePinholeModel*>(&model))
            {
                return Result<Json>::success(state_detail::frameInstanceState(model));
            }
            if (model.modelType() == "rpc00b")
            {
                return Result<Json>::success(state_detail::rpcInstanceState(model));
            }
            if (model.modelType() == "planetary_linescan")
            {
                return Result<Json>::success(state_detail::lineScanInstanceState(model));
            }
            return Result<Json>::failure(CameraErrorCode::UnsupportedModel,
                                         "camera instance type has no built-in JSON codec");
        }

        Result<std::string> encodeFailure(const std::exception& error)
        {
            return Result<std::string>::failure(CameraErrorCode::InvalidModelState,
                                                std::string("camera state encoding failed: ") + error.what());
        }

    } // namespace

    Result<std::string> encodeCameraDefinitionJson(const CameraDefinition& definition)
    {
        try
        {
            const auto parameters = definitionParameters(definition);
            if (!parameters)
            {
                return Result<std::string>::failure(parameters.error());
            }
            const Json envelope{{"kind", "placamera.definition"},
                                {"envelope_schema", CameraStateEnvelopeSchemaVersion},
                                {"model_type", definition.modelType()},
                                {"definition_id", definition.definitionId().value()},
                                {"ground_frame", definition.groundFrame().value()},
                                {"parameter_schema", definition.parameterSchemaVersion()},
                                {"parameters", parameters.value()}};
            return Result<std::string>::success(envelope.dump());
        }
        catch (const std::exception& error)
        {
            return encodeFailure(error);
        }
    }

    Result<std::string> encodeCameraInstanceJson(const RasterModel& model)
    {
        try
        {
            const auto state = instanceState(model);
            if (!state)
            {
                return Result<std::string>::failure(state.error());
            }
            const Json size{{"samples", model.imageSize().samples}, {"lines", model.imageSize().lines}};
            int instance_schema = 1;
            if (dynamic_cast<const FramePinholeModel*>(&model))
            {
                instance_schema = FramePinholeModel::InstanceSchemaVersion;
            }
            else if (dynamic_cast<const RpcModel*>(&model))
            {
                instance_schema = RpcModel::InstanceSchemaVersion;
            }
            else if (dynamic_cast<const LineScanModel*>(&model))
            {
                instance_schema = LineScanModel::InstanceSchemaVersion;
            }
            const Json envelope{{"kind", "placamera.instance"},
                                {"envelope_schema", CameraStateEnvelopeSchemaVersion},
                                {"model_type", model.modelType()},
                                {"instance_id", model.instanceId().value()},
                                {"image_id", model.imageId().value()},
                                {"definition_id", model.definitionId().value()},
                                {"instance_schema", instance_schema},
                                {"image_size", size},
                                {"capture_time", state_detail::optionalTimeReference(model.captureTime())},
                                {"state", state.value()}};
            return Result<std::string>::success(envelope.dump());
        }
        catch (const std::exception& error)
        {
            return encodeFailure(error);
        }
    }

    Result<CameraDefinitionState> decodeCameraDefinitionJson(std::string_view json)
    {
        try
        {
            const auto parsed = state_detail::parseObject(json);
            if (!parsed)
            {
                return Result<CameraDefinitionState>::failure(parsed.error());
            }
            const Json& envelope = parsed.value();
            if (!state_detail::hasObjectKeys(envelope,
                                             {"kind",
                                              "envelope_schema",
                                              "model_type",
                                              "definition_id",
                                              "ground_frame",
                                              "parameter_schema",
                                              "parameters"}))
            {
                return Result<CameraDefinitionState>::failure(
                    CameraErrorCode::InvalidModelState, "camera definition envelope has missing or unknown fields");
            }

            std::string kind;
            std::string model_type;
            std::string definition_id;
            std::string ground_frame;
            int envelope_schema = 0;
            int parameter_schema = 0;
            if (!state_detail::readString(envelope, "kind", &kind) || kind != "placamera.definition" ||
                !state_detail::readInt(envelope, "envelope_schema", &envelope_schema) ||
                envelope_schema != CameraStateEnvelopeSchemaVersion ||
                !state_detail::readString(envelope, "model_type", &model_type) ||
                !state_detail::readString(envelope, "definition_id", &definition_id) ||
                !state_detail::readString(envelope, "ground_frame", &ground_frame) ||
                !state_detail::readInt(envelope, "parameter_schema", &parameter_schema) || parameter_schema <= 0 ||
                !envelope.at("parameters").is_object())
            {
                return Result<CameraDefinitionState>::failure(CameraErrorCode::InvalidModelState,
                                                              "camera definition envelope values are invalid");
            }

            CameraDefinitionState state{CameraDefinitionId(definition_id),
                                        FrameId(ground_frame),
                                        state_detail::jsonState(model_type,
                                                                parameter_schema,
                                                                state_detail::DefinitionParametersJsonMediaType,
                                                                envelope.at("parameters"))};
            return Result<CameraDefinitionState>::success(std::move(state));
        }
        catch (const std::exception& error)
        {
            return Result<CameraDefinitionState>::failure(
                CameraErrorCode::InvalidModelState, std::string("camera definition decoding failed: ") + error.what());
        }
    }

    Result<CameraInstanceState> decodeCameraInstanceJson(std::string_view json)
    {
        try
        {
            const auto parsed = state_detail::parseObject(json);
            if (!parsed)
            {
                return Result<CameraInstanceState>::failure(parsed.error());
            }
            const Json& envelope = parsed.value();
            if (!state_detail::hasObjectKeys(envelope,
                                             {"kind",
                                              "envelope_schema",
                                              "model_type",
                                              "instance_id",
                                              "image_id",
                                              "definition_id",
                                              "instance_schema",
                                              "image_size",
                                              "capture_time",
                                              "state"}))
            {
                return Result<CameraInstanceState>::failure(CameraErrorCode::InvalidModelState,
                                                            "camera instance envelope has missing or unknown fields");
            }

            std::string kind;
            std::string model_type;
            std::string instance_id;
            std::string image_id;
            std::string definition_id;
            int envelope_schema = 0;
            int instance_schema = 0;
            if (!state_detail::readString(envelope, "kind", &kind) || kind != "placamera.instance" ||
                !state_detail::readInt(envelope, "envelope_schema", &envelope_schema) ||
                envelope_schema != CameraStateEnvelopeSchemaVersion ||
                !state_detail::readString(envelope, "model_type", &model_type) ||
                !state_detail::readString(envelope, "instance_id", &instance_id) ||
                !state_detail::readString(envelope, "image_id", &image_id) ||
                !state_detail::readString(envelope, "definition_id", &definition_id) ||
                !state_detail::readInt(envelope, "instance_schema", &instance_schema) || instance_schema <= 0 ||
                !envelope.at("image_size").is_object() || !envelope.at("state").is_object())
            {
                return Result<CameraInstanceState>::failure(CameraErrorCode::InvalidModelState,
                                                            "camera instance envelope values are invalid");
            }

            const Json& size = envelope.at("image_size");
            int samples = 0;
            int lines = 0;
            std::optional<TimeReference> capture_time;
            if (!state_detail::hasObjectKeys(size, {"samples", "lines"}) ||
                !state_detail::readInt(size, "samples", &samples) || !state_detail::readInt(size, "lines", &lines) ||
                samples <= 0 || lines <= 0 ||
                !state_detail::readOptionalTimeReference(envelope, "capture_time", &capture_time))
            {
                return Result<CameraInstanceState>::failure(CameraErrorCode::InvalidModelState,
                                                            "camera instance image size or capture time is invalid");
            }

            CameraInstanceState state{
                CameraInstanceId(instance_id),
                ImageId(image_id),
                CameraDefinitionId(definition_id),
                ImageSize{samples, lines},
                capture_time,
                state_detail::jsonState(
                    model_type, instance_schema, state_detail::InstanceStateJsonMediaType, envelope.at("state"))};
            return Result<CameraInstanceState>::success(std::move(state));
        }
        catch (const std::exception& error)
        {
            return Result<CameraInstanceState>::failure(
                CameraErrorCode::InvalidModelState, std::string("camera instance decoding failed: ") + error.what());
        }
    }

    Result<void> registerBuiltinJsonModelFactories(ModelRegistry& registry)
    {
        constexpr std::array<std::string_view, 6> frame_types{"frame_pinhole",
                                                              "frame_fisheye",
                                                              "frame_equidistant_fisheye",
                                                              "frame_equisolid_fisheye",
                                                              "frame_spherical",
                                                              "frame_cylindrical"};
        for (const std::string_view frame_type : frame_types)
        {
            const auto registered = registry.registerCameraFactory(
                std::string(frame_type), {state_detail::createFrameDefinition, state_detail::createFrameInstance});
            if (!registered)
            {
                return registered;
            }
        }
        const auto rpc = registry.registerCameraFactory(
            "rpc00b", {state_detail::createRpcDefinition, state_detail::createRpcInstance});
        if (!rpc)
        {
            return rpc;
        }
        const auto line_scan = registry.registerCameraFactory(
            "planetary_linescan", {state_detail::createLineScanDefinition, state_detail::createLineScanInstance});
        if (!line_scan)
        {
            return line_scan;
        }
        return Result<void>::success();
    }

} // namespace placamera
