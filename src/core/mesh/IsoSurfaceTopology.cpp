#include "IsoSurfaceTopology.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xjw::mesh
{

IsoSurfaceFaceDecision decideIsoSurfaceFace(
    const std::array<float, 4> &values,
    float isoLevel,
    const GridFaceKey &key,
    double relativeEpsilon)
{
    IsoSurfaceFaceDecision decision;
    std::array<double, 4> shifted{};
    std::array<bool, 4> inside{};
    double maximum_absolute_value = 0.0;
    for (int index = 0; index < 4; ++index)
    {
        if (!std::isfinite(values[static_cast<std::size_t>(index)]))
        {
            throw std::invalid_argument("Iso-surface face values must be finite");
        }
        shifted[static_cast<std::size_t>(index)] =
            static_cast<double>(values[static_cast<std::size_t>(index)]) -
            static_cast<double>(isoLevel);
        inside[static_cast<std::size_t>(index)] =
            shifted[static_cast<std::size_t>(index)] < 0.0;
        maximum_absolute_value = std::max(
            maximum_absolute_value,
            std::abs(shifted[static_cast<std::size_t>(index)]));
    }

    decision.ambiguous =
        inside[0] == inside[2] &&
        inside[1] == inside[3] &&
        inside[0] != inside[1];
    if (!decision.ambiguous)
    {
        return decision;
    }

    decision.determinant =
        shifted[0] * shifted[2] - shifted[1] * shifted[3];
    const double tolerance =
        std::max(0.0, relativeEpsilon) *
        std::max(1.0, maximum_absolute_value * maximum_absolute_value);
    if (decision.determinant > tolerance)
    {
        decision.connectEdge01And23 = true;
        return decision;
    }
    if (decision.determinant < -tolerance)
    {
        decision.connectEdge01And23 = false;
        return decision;
    }

    decision.usedTieBreak = true;
    const std::uint64_t parity =
        static_cast<std::uint64_t>(key.axis) +
        static_cast<std::uint64_t>(std::max(0, key.x)) +
        static_cast<std::uint64_t>(std::max(0, key.y)) +
        static_cast<std::uint64_t>(std::max(0, key.z));
    decision.connectEdge01And23 = (parity & 1u) == 0u;
    return decision;
}

std::uint64_t encodeGridFaceKey(const GridFaceKey &key,
                                const std::array<int, 3> &cells)
{
    if (cells[0] <= 0 || cells[1] <= 0 || cells[2] <= 0 ||
        key.x < 0 || key.y < 0 || key.z < 0 ||
        key.x > cells[0] || key.y > cells[1] || key.z > cells[2])
    {
        throw std::invalid_argument("Invalid grid face key");
    }
    const std::uint64_t sx = static_cast<std::uint64_t>(cells[0]) + 1u;
    const std::uint64_t sy = static_cast<std::uint64_t>(cells[1]) + 1u;
    const std::uint64_t linear =
        (static_cast<std::uint64_t>(key.z) * sy +
         static_cast<std::uint64_t>(key.y)) * sx +
        static_cast<std::uint64_t>(key.x);
    return linear * 3u + static_cast<std::uint64_t>(key.axis);
}

} // namespace xjw::mesh
