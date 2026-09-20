#include "SolverFrameDefinition.h"

#include "coordinate_system/types/CoordinateErrors.h"

#include <utility>

namespace xjw::coordinate_system
{

    SolverFrameDefinition
    SolverFrameDefinition::create(CoordinateFrameId frameId, SolverScaleStatus status, std::string normalizationHash)
    {
        if (normalizationHash.empty())
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidSolverFrame,
                                            "solver frame normalization hash must not be empty");
        }
        return SolverFrameDefinition{std::move(frameId), status, std::move(normalizationHash)};
    }

    bool SolverFrameDefinition::hasMetricScale() const noexcept
    {
        return scaleStatus == SolverScaleStatus::Metric;
    }

} // namespace xjw::coordinate_system
