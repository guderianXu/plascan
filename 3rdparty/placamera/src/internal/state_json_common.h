#pragma once

#include "placamera/calibration.h"
#include "placamera/registry.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

namespace placamera::state_detail
{

    using Json = nlohmann::ordered_json;

    inline constexpr std::string_view DefinitionParametersJsonMediaType =
        "application/vnd.placamera.definition-parameters+json";
    inline constexpr std::string_view InstanceStateJsonMediaType = "application/vnd.placamera.instance-state+json";

    bool hasObjectKeys(const Json& value,
                       std::initializer_list<std::string_view> required,
                       std::initializer_list<std::string_view> optional = {});
    bool readString(const Json& object, std::string_view key, std::string* output);
    bool readInt(const Json& object, std::string_view key, int* output);
    bool readFinite(const Json& object, std::string_view key, double* output);
    bool readBool(const Json& object, std::string_view key, bool* output);

    template <std::size_t Size> bool readFiniteArray(const Json& value, std::array<double, Size>* output)
    {
        if (!output || !value.is_array() || value.size() != Size)
        {
            return false;
        }
        for (std::size_t index = 0; index < Size; ++index)
        {
            if (!value[index].is_number())
            {
                return false;
            }
            const double number = value[index].get<double>();
            if (!std::isfinite(number))
            {
                return false;
            }
            (*output)[index] = number;
        }
        return true;
    }

    template <std::size_t Size> Json finiteArray(const std::array<double, Size>& values)
    {
        Json result = Json::array();
        for (const double value : values)
        {
            result.push_back(value);
        }
        return result;
    }

    Json timeReference(const TimeReference& time);
    bool readTimeReference(const Json& value, TimeReference* output);
    Json optionalTimeReference(const std::optional<TimeReference>& time);
    bool readOptionalTimeReference(const Json& object, std::string_view key, std::optional<TimeReference>* output);

    Json principalPointDecomposition(const std::optional<PrincipalPointDecomposition>& principal);
    bool readPrincipalPointDecomposition(const Json& value,
                                         std::optional<PrincipalPointDecomposition>* output);
    Json metashapeCalibration(const std::optional<MetashapeCalibration>& calibration);
    bool readMetashapeCalibration(const Json& value, std::optional<MetashapeCalibration>* output);

    Result<Json> parseObject(std::string_view payload);
    Result<Json> parseStateObject(const ModelState& state, std::string_view expectedMediaType);
    ModelState jsonState(std::string modelType, int schemaVersion, std::string_view mediaType, const Json& payload);

    Result<CameraModelFactory::DefinitionPointer> createFrameDefinition(const CameraDefinitionState& state);
    Result<ModelRegistry::ModelPointer> createFrameInstance(const CameraInstanceState& state,
                                                            CameraModelFactory::DefinitionPointer definition);
    Json frameDefinitionParameters(const CameraDefinition& definition);
    Json frameInstanceState(const RasterModel& model);

    Result<CameraModelFactory::DefinitionPointer> createRpcDefinition(const CameraDefinitionState& state);
    Result<ModelRegistry::ModelPointer> createRpcInstance(const CameraInstanceState& state,
                                                          CameraModelFactory::DefinitionPointer definition);
    Json rpcDefinitionParameters(const CameraDefinition& definition);
    Json rpcInstanceState(const RasterModel& model);

    Result<CameraModelFactory::DefinitionPointer> createLineScanDefinition(const CameraDefinitionState& state);
    Result<ModelRegistry::ModelPointer> createLineScanInstance(const CameraInstanceState& state,
                                                               CameraModelFactory::DefinitionPointer definition);
    Json lineScanDefinitionParameters(const CameraDefinition& definition);
    Json lineScanInstanceState(const RasterModel& model);

} // namespace placamera::state_detail
