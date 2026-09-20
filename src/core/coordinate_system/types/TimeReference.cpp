#include "TimeReference.h"

#include "CoordinateErrors.h"

#include <cmath>

namespace xjw::coordinate_system
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

} // namespace xjw::coordinate_system
