#pragma once

#include "FramePinholeCamera.h"
#include "ReferenceP3p.h"

#include <array>
#include <cstddef>
#include <vector>

namespace xjw
{

    void refineReferencePose(const FramePinholeCamera& camera,
                             const std::vector<std::array<double, 3>>& worldPoints,
                             const std::vector<std::array<double, 2>>& imagePoints,
                             const std::vector<std::size_t>& inlierIndices,
                             ReferenceWorldToCameraPose* pose,
                             std::size_t iterations = 10);

} // namespace xjw
