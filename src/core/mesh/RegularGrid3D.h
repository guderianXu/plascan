#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace xjw::mesh::detail
{

    // Shared description for dense, axis-aligned scalar fields. The resolution
    // counts cells; sampled fields therefore contain resolution + 1 values per
    // axis. TSDF and visibility-occupancy code can reuse the same indexing rules.
    struct RegularGrid3D
    {
        std::array<float, 3> boundsMin{};
        std::array<float, 3> boundsMax{};
        std::array<int, 3> cellResolution{{0, 0, 0}};

        bool isValid() const noexcept
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                if (cellResolution[axis] <= 0 || !std::isfinite(boundsMin[axis]) || !std::isfinite(boundsMax[axis]) ||
                    !(boundsMax[axis] > boundsMin[axis]))
                {
                    return false;
                }
            }
            return sampleCount() != 0;
        }

        int sampleSize(int axis) const noexcept
        {
            return axis >= 0 && axis < 3 && cellResolution[axis] < std::numeric_limits<int>::max()
                       ? cellResolution[axis] + 1
                       : 0;
        }

        float spacing(int axis) const noexcept
        {
            if (axis < 0 || axis >= 3 || cellResolution[axis] <= 0)
            {
                return 0.0f;
            }
            return (boundsMax[axis] - boundsMin[axis]) / static_cast<float>(cellResolution[axis]);
        }

        std::size_t sampleCount() const noexcept
        {
            std::size_t count = 1;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (cellResolution[axis] <= 0 || cellResolution[axis] == std::numeric_limits<int>::max())
                {
                    return 0;
                }
                const std::size_t size = static_cast<std::size_t>(cellResolution[axis] + 1);
                if (count > std::numeric_limits<std::size_t>::max() / size)
                {
                    return 0;
                }
                count *= size;
            }
            return count;
        }

        std::size_t linearIndex(int x, int y, int z) const noexcept
        {
            const std::size_t size_x = static_cast<std::size_t>(sampleSize(0));
            const std::size_t size_y = static_cast<std::size_t>(sampleSize(1));
            return static_cast<std::size_t>(x) +
                   size_x * (static_cast<std::size_t>(y) + size_y * static_cast<std::size_t>(z));
        }

        std::array<float, 3> samplePosition(int x, int y, int z) const noexcept
        {
            return {boundsMin[0] + spacing(0) * static_cast<float>(x),
                    boundsMin[1] + spacing(1) * static_cast<float>(y),
                    boundsMin[2] + spacing(2) * static_cast<float>(z)};
        }

        bool isBoundarySample(int x, int y, int z) const noexcept
        {
            return x == 0 || y == 0 || z == 0 || x == cellResolution[0] || y == cellResolution[1] ||
                   z == cellResolution[2];
        }
    };

} // namespace xjw::mesh::detail
