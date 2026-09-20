#pragma once

#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <string>

namespace xjw::camera_io
{

    /**
     * Load one ASP/Tsai text camera directly into the solver-owned numeric state.
     *
     * This is a file-format boundary. SfM/BA/MVS callers consume the same
     * numerical representation used by typed camera instances.
     */
    bool loadFramePinholeNumericStateFromTsaiFile(const std::string& path,
                                                  xjw::camera_models::frame_pinhole::FramePinholeNumericState* state,
                                                  std::string* error = nullptr);

} // namespace xjw::camera_io
