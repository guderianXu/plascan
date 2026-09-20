#pragma once

#include "coordinate_system/context/CoordinateContext.h"

#include <nlohmann/json_fwd.hpp>

#include <optional>
#include <string>

namespace xjw::coordinate_system
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

} // namespace xjw::coordinate_system
