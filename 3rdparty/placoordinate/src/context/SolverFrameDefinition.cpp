#include "placoordinate/context/SolverFrameDefinition.h"

#include "placoordinate/types/CoordinateErrors.h"

#include <utility>

namespace placoordinate
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

} // namespace placoordinate
