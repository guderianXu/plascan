#pragma once

#include <array>
#include <vector>

namespace xjw
{

    struct ReferenceWorldToCameraPose
    {
        std::array<double, 9> rotation{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0}};
        std::array<double, 3> translation{{0.0, 0.0, 0.0}};
    };

    std::vector<ReferenceWorldToCameraPose>
    solveReferenceP3p(const std::array<std::array<double, 3>, 3>& worldPoints,
                      const std::array<std::array<double, 3>, 3>& bearingVectors);

} // namespace xjw
