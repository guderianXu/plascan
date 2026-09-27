#include "placoordinate/types/CoordinateIds.h"

#include "placoordinate/types/CoordinateErrors.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace placoordinate
{

    namespace
    {

        void requireIdentifier(const std::string& value, const char* label)
        {
            const bool has_non_whitespace = std::any_of(
                value.begin(), value.end(), [](unsigned char character) { return std::isspace(character) == 0; });
            if (value.empty() || !has_non_whitespace)
            {
                throw CoordinateValidationError(CoordinateErrorCode::EmptyIdentifier,
                                                std::string(label) + " must not be empty");
            }
        }

    } // namespace

    CoordinateFrameId::CoordinateFrameId(std::string value) : _value(std::move(value))
    {
        requireIdentifier(_value, "coordinate frame identifier");
    }

    const std::string& CoordinateFrameId::value() const noexcept
    {
        return _value;
    }

    SpatialReferenceId::SpatialReferenceId(std::string value) : _value(std::move(value))
    {
        requireIdentifier(_value, "spatial reference identifier");
    }

    const std::string& SpatialReferenceId::value() const noexcept
    {
        return _value;
    }

    CoordinateContextId::CoordinateContextId(std::string value) : _value(std::move(value))
    {
        requireIdentifier(_value, "coordinate context identifier");
    }

    const std::string& CoordinateContextId::value() const noexcept
    {
        return _value;
    }

} // namespace placoordinate
