#include "internal/state_json_common.h"

#include "placamera/rpc_camera.h"

#include <memory>
#include <optional>
#include <utility>

namespace placamera::state_detail
{
    namespace
    {

        Json optionalNumber(const std::optional<double>& value)
        {
            return value ? Json(*value) : Json(nullptr);
        }

        bool readOptionalFinite(const Json& object, std::string_view key, std::optional<double>* output)
        {
            if (!output || !object.contains(std::string(key)))
            {
                return false;
            }
            const Json& value = object.at(std::string(key));
            if (value.is_null())
            {
                *output = std::nullopt;
                return true;
            }
            if (!value.is_number())
            {
                return false;
            }
            const double number = value.get<double>();
            if (!std::isfinite(number))
            {
                return false;
            }
            *output = number;
            return true;
        }

    } // namespace

    Json rpcDefinitionParameters(const CameraDefinition& definition)
    {
        const auto* rpc = dynamic_cast<const RpcDefinition*>(&definition);
        if (!rpc)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel,
                                        "RPC codec received another definition type");
        }
        const RpcParameters& parameters = rpc->parameters();
        return Json{{"normalization",
                     Json{{"line_offset", parameters.lineOffset},
                          {"sample_offset", parameters.sampleOffset},
                          {"latitude_offset", parameters.latitudeOffset},
                          {"longitude_offset", parameters.longitudeOffset},
                          {"height_offset", parameters.heightOffset},
                          {"line_scale", parameters.lineScale},
                          {"sample_scale", parameters.sampleScale},
                          {"latitude_scale", parameters.latitudeScale},
                          {"longitude_scale", parameters.longitudeScale},
                          {"height_scale", parameters.heightScale}}},
                    {"coefficients",
                     Json{{"line_numerator", finiteArray(parameters.lineNumerator)},
                          {"line_denominator", finiteArray(parameters.lineDenominator)},
                          {"sample_numerator", finiteArray(parameters.sampleNumerator)},
                          {"sample_denominator", finiteArray(parameters.sampleDenominator)}}},
                    {"accuracy",
                     Json{{"error_bias_m", optionalNumber(parameters.errorBiasMeters)},
                          {"error_random_m", optionalNumber(parameters.errorRandomMeters)}}},
                    {"ellipsoid",
                     Json{{"semi_major_axis_m", rpc->ellipsoid().semiMajorAxisMeters},
                          {"inverse_flattening", rpc->ellipsoid().inverseFlattening}}}};
    }

    Json rpcInstanceState(const RasterModel& model)
    {
        const auto* rpc = dynamic_cast<const RpcModel*>(&model);
        if (!rpc)
        {
            throw CameraValidationError(CameraErrorCode::UnsupportedModel, "RPC codec received another instance type");
        }
        if (const RpcImageCorrection* correction = rpc->normalizedImageCorrection())
        {
            return Json{{"correction_domain", "normalized_image"},
                        {"normalized_image_correction",
                         Json{{"sample_offset", correction->sampleOffsetPixels},
                              {"sample_sample", correction->sampleSamplePixels},
                              {"sample_line", correction->sampleLinePixels},
                              {"line_offset", correction->lineOffsetPixels},
                              {"line_sample", correction->lineSamplePixels},
                              {"line_line", correction->lineLinePixels}}}};
        }

        const RpcGroundCorrection& correction = *rpc->groundCorrection();
        return Json{{"correction_domain", "ground_coordinates"},
                    {"ground_correction",
                     Json{{"sample_offset", correction.sampleOffsetPixels},
                          {"line_offset", correction.lineOffsetPixels},
                          {"sample_longitude", correction.sampleLongitudePixelsPerDegree},
                          {"sample_latitude", correction.sampleLatitudePixelsPerDegree},
                          {"sample_height", correction.sampleHeightPixelsPerMeter},
                          {"line_longitude", correction.lineLongitudePixelsPerDegree},
                          {"line_latitude", correction.lineLatitudePixelsPerDegree},
                          {"line_height", correction.lineHeightPixelsPerMeter}}}};
    }

    Result<CameraModelFactory::DefinitionPointer> createRpcDefinition(const CameraDefinitionState& state)
    {
        if (state.parameters.schemaVersion != RpcDefinition::ParameterSchemaVersion)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "unsupported RPC parameter schema");
        }
        const auto parsed = parseStateObject(state.parameters, DefinitionParametersJsonMediaType);
        if (!parsed)
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        if (!hasObjectKeys(root, {"normalization", "coefficients", "accuracy", "ellipsoid"}) ||
            !root.at("normalization").is_object() || !root.at("coefficients").is_object() ||
            !root.at("accuracy").is_object() || !root.at("ellipsoid").is_object())
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(
                CameraErrorCode::InvalidModelState, "RPC parameters have missing or unknown fields");
        }

        const Json& normalization = root.at("normalization");
        const Json& coefficients = root.at("coefficients");
        const Json& accuracy = root.at("accuracy");
        const Json& ellipsoid_json = root.at("ellipsoid");
        if (!hasObjectKeys(normalization,
                           {"line_offset",
                            "sample_offset",
                            "latitude_offset",
                            "longitude_offset",
                            "height_offset",
                            "line_scale",
                            "sample_scale",
                            "latitude_scale",
                            "longitude_scale",
                            "height_scale"}) ||
            !hasObjectKeys(coefficients,
                           {"line_numerator", "line_denominator", "sample_numerator", "sample_denominator"}) ||
            !hasObjectKeys(accuracy, {"error_bias_m", "error_random_m"}) ||
            !hasObjectKeys(ellipsoid_json, {"semi_major_axis_m", "inverse_flattening"}))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "RPC nested parameter objects are malformed");
        }

        RpcParameters parameters;
        ReferenceEllipsoid ellipsoid;
        if (!readFinite(normalization, "line_offset", &parameters.lineOffset) ||
            !readFinite(normalization, "sample_offset", &parameters.sampleOffset) ||
            !readFinite(normalization, "latitude_offset", &parameters.latitudeOffset) ||
            !readFinite(normalization, "longitude_offset", &parameters.longitudeOffset) ||
            !readFinite(normalization, "height_offset", &parameters.heightOffset) ||
            !readFinite(normalization, "line_scale", &parameters.lineScale) ||
            !readFinite(normalization, "sample_scale", &parameters.sampleScale) ||
            !readFinite(normalization, "latitude_scale", &parameters.latitudeScale) ||
            !readFinite(normalization, "longitude_scale", &parameters.longitudeScale) ||
            !readFinite(normalization, "height_scale", &parameters.heightScale) ||
            !readFiniteArray(coefficients.at("line_numerator"), &parameters.lineNumerator) ||
            !readFiniteArray(coefficients.at("line_denominator"), &parameters.lineDenominator) ||
            !readFiniteArray(coefficients.at("sample_numerator"), &parameters.sampleNumerator) ||
            !readFiniteArray(coefficients.at("sample_denominator"), &parameters.sampleDenominator) ||
            !readOptionalFinite(accuracy, "error_bias_m", &parameters.errorBiasMeters) ||
            !readOptionalFinite(accuracy, "error_random_m", &parameters.errorRandomMeters) ||
            !readFinite(ellipsoid_json, "semi_major_axis_m", &ellipsoid.semiMajorAxisMeters) ||
            !readFinite(ellipsoid_json, "inverse_flattening", &ellipsoid.inverseFlattening))
        {
            return Result<CameraModelFactory::DefinitionPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                          "RPC parameter values are invalid");
        }

        CameraModelFactory::DefinitionPointer definition =
            RpcDefinition::create(state.definitionId, state.groundFrame, parameters, ellipsoid);
        return Result<CameraModelFactory::DefinitionPointer>::success(std::move(definition));
    }

    Result<ModelRegistry::ModelPointer> createRpcInstance(const CameraInstanceState& state,
                                                          CameraModelFactory::DefinitionPointer definition)
    {
        if (state.state.schemaVersion != 1 && state.state.schemaVersion != RpcModel::InstanceSchemaVersion)
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "unsupported RPC instance schema");
        }
        const auto rpc_definition = std::dynamic_pointer_cast<const RpcDefinition>(definition);
        if (!rpc_definition)
        {
            return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                "RPC instance requires an RPC definition");
        }
        const auto parsed = parseStateObject(state.state, InstanceStateJsonMediaType);
        if (!parsed)
        {
            return Result<ModelRegistry::ModelPointer>::failure(parsed.error());
        }
        const Json& root = parsed.value();
        RpcCorrection correction = RpcCorrection::normalizedImage();
        if (state.state.schemaVersion == 1)
        {
            if (!hasObjectKeys(root, {"image_correction"}) ||
                !hasObjectKeys(
                    root.at("image_correction"),
                    {"sample_offset", "sample_sample", "sample_line", "line_offset", "line_sample", "line_line"}))
            {
                return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                    "RPC schema-1 image correction is malformed");
            }
            const Json& correction_json = root.at("image_correction");
            RpcImageCorrection image;
            if (!readFinite(correction_json, "sample_offset", &image.sampleOffsetPixels) ||
                !readFinite(correction_json, "sample_sample", &image.sampleSamplePixels) ||
                !readFinite(correction_json, "sample_line", &image.sampleLinePixels) ||
                !readFinite(correction_json, "line_offset", &image.lineOffsetPixels) ||
                !readFinite(correction_json, "line_sample", &image.lineSamplePixels) ||
                !readFinite(correction_json, "line_line", &image.lineLinePixels))
            {
                return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                    "RPC schema-1 image correction is invalid");
            }
            correction = RpcCorrection::normalizedImage(image);
        }
        else
        {
            std::string domain;
            if (!readString(root, "correction_domain", &domain))
            {
                return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                    "RPC correction domain is missing or invalid");
            }
            if (domain == "normalized_image")
            {
                if (!hasObjectKeys(root, {"correction_domain", "normalized_image_correction"}) ||
                    !hasObjectKeys(
                        root.at("normalized_image_correction"),
                        {"sample_offset", "sample_sample", "sample_line", "line_offset", "line_sample", "line_line"}))
                {
                    return Result<ModelRegistry::ModelPointer>::failure(
                        CameraErrorCode::InvalidModelState,
                        "normalized-image RPC correction fields do not match the declared domain");
                }
                const Json& value = root.at("normalized_image_correction");
                RpcImageCorrection image;
                if (!readFinite(value, "sample_offset", &image.sampleOffsetPixels) ||
                    !readFinite(value, "sample_sample", &image.sampleSamplePixels) ||
                    !readFinite(value, "sample_line", &image.sampleLinePixels) ||
                    !readFinite(value, "line_offset", &image.lineOffsetPixels) ||
                    !readFinite(value, "line_sample", &image.lineSamplePixels) ||
                    !readFinite(value, "line_line", &image.lineLinePixels))
                {
                    return Result<ModelRegistry::ModelPointer>::failure(
                        CameraErrorCode::InvalidModelState, "normalized-image RPC correction values are invalid");
                }
                correction = RpcCorrection::normalizedImage(image);
            }
            else if (domain == "ground_coordinates")
            {
                if (!hasObjectKeys(root, {"correction_domain", "ground_correction"}) ||
                    !hasObjectKeys(root.at("ground_correction"),
                                   {"sample_offset",
                                    "line_offset",
                                    "sample_longitude",
                                    "sample_latitude",
                                    "sample_height",
                                    "line_longitude",
                                    "line_latitude",
                                    "line_height"}))
                {
                    return Result<ModelRegistry::ModelPointer>::failure(
                        CameraErrorCode::InvalidModelState,
                        "ground-coordinate RPC correction fields do not match the declared domain");
                }
                const Json& value = root.at("ground_correction");
                RpcGroundCorrection ground;
                if (!readFinite(value, "sample_offset", &ground.sampleOffsetPixels) ||
                    !readFinite(value, "line_offset", &ground.lineOffsetPixels) ||
                    !readFinite(value, "sample_longitude", &ground.sampleLongitudePixelsPerDegree) ||
                    !readFinite(value, "sample_latitude", &ground.sampleLatitudePixelsPerDegree) ||
                    !readFinite(value, "sample_height", &ground.sampleHeightPixelsPerMeter) ||
                    !readFinite(value, "line_longitude", &ground.lineLongitudePixelsPerDegree) ||
                    !readFinite(value, "line_latitude", &ground.lineLatitudePixelsPerDegree) ||
                    !readFinite(value, "line_height", &ground.lineHeightPixelsPerMeter))
                {
                    return Result<ModelRegistry::ModelPointer>::failure(
                        CameraErrorCode::InvalidModelState, "ground-coordinate RPC correction values are invalid");
                }
                correction = RpcCorrection::groundCoordinates(ground);
            }
            else
            {
                return Result<ModelRegistry::ModelPointer>::failure(CameraErrorCode::InvalidModelState,
                                                                    "unsupported RPC correction domain");
            }
        }

        auto model = RpcModel::createWithCorrection(
            state.instanceId, state.imageId, rpc_definition, state.imageSize, std::move(correction), state.captureTime);
        return Result<ModelRegistry::ModelPointer>::success(std::make_shared<const RpcModel>(std::move(model)));
    }

} // namespace placamera::state_detail
