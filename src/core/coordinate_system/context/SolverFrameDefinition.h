#pragma once

#include "coordinate_system/types/CoordinateIds.h"

#include <string>

namespace xjw::coordinate_system
{

    enum class SolverScaleStatus
    {
        Metric,
        Unresolved,
    };

    struct SolverFrameDefinition
    {
        CoordinateFrameId frameId;
        SolverScaleStatus scaleStatus = SolverScaleStatus::Unresolved;
        std::string normalizationHash;

        static SolverFrameDefinition
        create(CoordinateFrameId frameId, SolverScaleStatus status, std::string normalizationHash);

        bool hasMetricScale() const noexcept;
    };

} // namespace xjw::coordinate_system
