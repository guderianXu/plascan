#pragma once

#include "metmodel/types.hpp"

namespace metmodel
{
    // Tracked production selector; no legacy baseline fallback.
    std::vector<std::vector<std::size_t>> select_recovered_neighbors(const Scene& scene, std::size_t max_neighbors);
} // namespace metmodel
