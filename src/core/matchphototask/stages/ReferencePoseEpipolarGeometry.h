#pragma once

#include "camera/models/frame_pinhole/FramePinholeNumericState.h"

#include <array>

namespace xjw::matchphotos
{

    struct ReferencePoseEpipolarGeometry
    {
        std::array<double, 9> fundamental{};
        double baseline = 0.0;
        bool valid = false;
    };

    ReferencePoseEpipolarGeometry fundamentalFromReferenceCameras(
        const camera_models::frame_pinhole::FramePinholeNumericState& camera0,
        const camera_models::frame_pinhole::FramePinholeNumericState& camera1);

    double
    epipolarSampsonDistance(const std::array<double, 9>& fundamental, double x0, double y0, double x1, double y1);

} // namespace xjw::matchphotos
