#pragma once

#include "placoordinate/context/CoordinateContext.h"

#include <nlohmann/json_fwd.hpp>

#include <optional>
#include <string>

namespace placoordinate
{

    struct CoordinateContextJsonResult
    {
        std::optional<CoordinateContext> context;
        std::string error;

        bool ok() const noexcept;
    };

    nlohmann::json coordinateContextToJson(const CoordinateContext& context);
    CoordinateContextJsonResult coordinateContextFromJson(const nlohmann::json& document);

    void embedCoordinateContextInChunk(nlohmann::json* chunk, const CoordinateContext& context);
    CoordinateContextJsonResult coordinateContextFromChunk(const nlohmann::json& chunk);

} // namespace placoordinate
