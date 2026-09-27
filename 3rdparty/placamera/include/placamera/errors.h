#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace placamera
{

    enum class CameraErrorCode
    {
        None,
        EmptyIdentifier,
        InvalidFrame,
        FrameMismatch,
        InvalidPose,
        InvalidTime,
        InvalidImageSize,
        InvalidIntrinsics,
        InvalidDistortion,
        InvalidModelState,
        InvalidArgument,
        IoFailure,
        ParseFailure,
        OutsideModelDomain,
        NonConvergence,
        UnsupportedModel,
        UnsupportedFormat,
    };

    /** Structured diagnostic for recoverable PlaCamera operations. */
    struct CameraError
    {
        CameraErrorCode code = CameraErrorCode::None;
        std::string message;
        std::string source;
        std::optional<std::size_t> line;
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

} // namespace placamera
