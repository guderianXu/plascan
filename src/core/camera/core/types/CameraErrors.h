#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace xjw::camera_core
{

    enum class CameraErrorCode
    {
        EmptyIdentifier,
        InvalidFrame,
        InvalidRotation,
        InvalidCovariance,
        InvalidTime,
        InvalidIntrinsics,
        InvalidDistortion,
    };

    class CameraValidationError : public std::invalid_argument
    {
    public:
        CameraValidationError(CameraErrorCode code, std::string message)
            : std::invalid_argument(std::move(message)), _code(code)
        {
        }

        CameraErrorCode code() const noexcept
        {
            return _code;
        }

    private:
        CameraErrorCode _code;
    };

} // namespace xjw::camera_core
