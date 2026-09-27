#pragma once

#include "placoordinate/context/SpatialReferenceDefinition.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace placoordinate
{

    struct GdalSpatialReferenceResult
    {
        std::optional<SpatialReferenceDefinition> reference;
        std::string error;

        bool ok() const noexcept;
    };

    GdalSpatialReferenceResult normalizeGdalSpatialReference(SpatialReferenceId id,
                                                             CoordinateFrameId frameId,
                                                             std::string_view definition,
                                                             VerticalReference verticalReference);

    struct GdalCoordinateTransformResult
    {
        bool ok = false;
        std::array<double, 3> coordinate{{0.0, 0.0, 0.0}};
        std::string error;
    };

    GdalCoordinateTransformResult transformGdalCoordinate(const std::array<double, 3>& coordinate,
                                                          const SpatialReferenceDefinition& source,
                                                          CoordinateAxisOrder sourceAxisOrder,
                                                          const SpatialReferenceDefinition& target);

    struct GdalCoordinateUncertaintyResult
    {
        bool ok = false;
        std::array<double, 3> coordinate{{0.0, 0.0, 0.0}};
        std::array<double, 3> standardDeviation{{0.0, 0.0, 0.0}};
        std::string error;
    };

    GdalCoordinateUncertaintyResult
    transformGdalCoordinateWithDiagonalUncertainty(const std::array<double, 3>& coordinate,
                                                   const std::array<double, 3>& sourceStandardDeviation,
                                                   const SpatialReferenceDefinition& source,
                                                   CoordinateAxisOrder sourceAxisOrder,
                                                   const SpatialReferenceDefinition& target);

} // namespace placoordinate
