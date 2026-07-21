#pragma once

#include <string>

namespace xjw
{

enum class SfmErrorCode
{
    None,
    InvalidInput,
    CameraUnavailable,
    InitializationFailed,
    RegistrationFailed,
    OptimizationFailed
};

struct SfmError
{
    SfmErrorCode code = SfmErrorCode::None;
    std::string diagnostic;

    explicit operator bool() const
    {
        return code != SfmErrorCode::None;
    }
};

} // namespace xjw
