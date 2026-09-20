#pragma once

#include "camera/core/model/CameraModelRegistry.h"

#include <optional>
#include <string_view>

namespace xjw::camera_models
{

    // Creates the built-in model registry. The registry owns only factories; each
    // factory returns an immutable model definition/instance value and validates
    // the canonical serialized model parameters at the registry boundary.
    camera_core::CameraModelRegistry makeBuiltinCameraModelRegistry();

    std::optional<int> builtinCameraParameterSchemaVersion(std::string_view modelType) noexcept;

} // namespace xjw::camera_models
