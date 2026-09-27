#include "placamera/optimization.h"

#include <algorithm>

namespace placamera
{

    std::size_t OptimizationLayout::parameterCount() const noexcept
    {
        std::size_t count = 0;
        for (const OptimizationParameterBlock& block : blocks)
        {
            count = std::max(count, block.offset + block.size);
        }
        return count;
    }

    bool OptimizationLayout::isValid() const noexcept
    {
        std::size_t expected_offset = 0;
        for (const OptimizationParameterBlock& block : blocks)
        {
            if (block.name.empty() || block.unit.empty() || block.size == 0 || block.offset != expected_offset)
            {
                return false;
            }
            expected_offset += block.size;
        }
        return true;
    }

} // namespace placamera
