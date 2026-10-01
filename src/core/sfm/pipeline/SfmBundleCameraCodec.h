#pragma once

#include <placamera/frame_numeric_state.h>

#include <string>
#include <vector>

namespace xjw::sfm_bundle_camera
{
    /** Validate a PlaCamera state before passing it to PlaBundle. */
    bool encode(const placamera::FramePinholeNumericState& source,
                placamera::FramePinholeNumericState* target,
                std::string* error = nullptr);

    /** Accept a solved state only when its camera identity and image convention match. */
    bool decode(const placamera::FramePinholeNumericState& source,
                placamera::FramePinholeNumericState* target,
                std::string* error = nullptr);

    bool encodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
                   std::vector<placamera::FramePinholeNumericState>* targets,
                   std::string* error = nullptr);

    bool decodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
                   std::vector<placamera::FramePinholeNumericState>* targets,
                   std::string* error = nullptr);
} // namespace xjw::sfm_bundle_camera
