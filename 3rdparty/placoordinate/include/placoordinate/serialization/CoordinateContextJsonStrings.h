#pragma once

#include "placoordinate/context/CoordinateContext.h"

#include <optional>
#include <string_view>

namespace placoordinate::detail
{

    std::string_view enumName(SpatialReferenceKind value) noexcept;
    std::string_view enumName(CoordinateAxisOrder value) noexcept;
    std::string_view enumName(VerticalReference value) noexcept;
    std::string_view enumName(CoordinateFrameKind value) noexcept;
    std::string_view enumName(LinearUnit value) noexcept;
    std::string_view enumName(AngleUnit value) noexcept;
    std::string_view enumName(SolverScaleStatus value) noexcept;

    std::optional<SpatialReferenceKind> spatialReferenceKindFromName(std::string_view value) noexcept;
    std::optional<CoordinateAxisOrder> coordinateAxisOrderFromName(std::string_view value) noexcept;
    std::optional<VerticalReference> verticalReferenceFromName(std::string_view value) noexcept;
    std::optional<CoordinateFrameKind> coordinateFrameKindFromName(std::string_view value) noexcept;
    std::optional<LinearUnit> linearUnitFromName(std::string_view value) noexcept;
    std::optional<AngleUnit> angleUnitFromName(std::string_view value) noexcept;
    std::optional<SolverScaleStatus> solverScaleStatusFromName(std::string_view value) noexcept;

} // namespace placoordinate::detail
