#pragma once

#include <placamera/frame_numeric_state.h>
#include <plabundle/camera.h>

#include <string>
#include <vector>

namespace xjw::sfm_bundle_camera
{

    /** Translate a canonical PlaCamera solver state into PlaBundle's numerical camera input. */
    bool encode(const placamera::FramePinholeNumericState& source,
                plabundle::FrameCamera* target,
                std::string* error = nullptr);

    /** Apply a PlaBundle result without changing camera identity, frame, image grid or optical convention. */
    bool decode(const plabundle::FrameCamera& source,
                placamera::FramePinholeNumericState* target,
                std::string* error = nullptr);

    bool encodeAll(const std::vector<placamera::FramePinholeNumericState>& sources,
                   std::vector<plabundle::FrameCamera>* targets,
                   std::string* error = nullptr);

    bool decodeAll(const std::vector<plabundle::FrameCamera>& sources,
                   std::vector<placamera::FramePinholeNumericState>* targets,
                   std::string* error = nullptr);

} // namespace xjw::sfm_bundle_camera
