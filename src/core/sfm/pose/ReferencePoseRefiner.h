#pragma once

#include "ReferenceP3p.h"

#include <placamera/frame_camera.h>

#include <array>
#include <cstddef>
#include <vector>

namespace xjw
{

    void refineReferencePose(const placamera::FramePinholeDefinition& camera,
                             const std::vector<std::array<double, 3>>& worldPoints,
                             const std::vector<std::array<double, 2>>& imagePoints,
                             const std::vector<std::size_t>& inlierIndices,
                             ReferenceWorldToCameraPose* pose,
                             std::size_t iterations = 10);

} // namespace xjw
