#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace xjw::coordinate_system
{

    enum class CoordinateErrorCode
    {
        EmptyIdentifier,
        InvalidFrame,
        InvalidTime,
        InvalidSolverFrame,
        InvalidSpatialReference,
        InvalidCoordinateContext,
        InvalidCoordinatePersistence,
        CoordinateHashMismatch,
    };

    class CoordinateValidationError : public std::invalid_argument
    {
    public:
        CoordinateValidationError(CoordinateErrorCode code, std::string message)
            : std::invalid_argument(std::move(message)), _code(code)
        {
        }

        CoordinateErrorCode code() const noexcept
        {
            return _code;
        }

    private:
        CoordinateErrorCode _code;
    };

} // namespace xjw::coordinate_system
