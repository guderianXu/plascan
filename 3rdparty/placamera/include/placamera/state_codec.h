#pragma once

#include "placamera/registry.h"

#include <string>
#include <string_view>

namespace placamera
{

    inline constexpr std::string_view CameraJsonMediaType = "application/vnd.placamera+json";
    inline constexpr int CameraStateEnvelopeSchemaVersion = 1;

    Result<std::string> encodeCameraDefinitionJson(const CameraDefinition& definition);
    Result<std::string> encodeCameraInstanceJson(const RasterModel& model);

    Result<CameraDefinitionState> decodeCameraDefinitionJson(std::string_view json);
    Result<CameraInstanceState> decodeCameraInstanceJson(std::string_view json);

    Result<void> registerBuiltinJsonModelFactories(ModelRegistry& registry);

} // namespace placamera
