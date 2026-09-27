#include "placoordinate/types/TimeReference.h"

#include "placoordinate/types/CoordinateErrors.h"

#include <cmath>

namespace placoordinate
{

    TimeReference TimeReference::create(TimeScale timeScale, double value)
    {
        if (!std::isfinite(value))
        {
            throw CoordinateValidationError(CoordinateErrorCode::InvalidTime,
                                            "time reference must contain finite seconds");
        }
        return TimeReference{timeScale, value};
    }

} // namespace placoordinate
