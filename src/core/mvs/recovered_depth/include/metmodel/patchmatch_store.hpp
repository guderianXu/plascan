#pragma once

#include "metmodel/patchmatch.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metmodel {

struct RecoveredPatchMatchStoreBatch {
    std::vector<std::size_t> references;
    std::vector<std::size_t> closure;
};

// Stable, versioned per-camera persistence boundary for d4/d8/d16 PM products.
// Existing camera files are never overwritten. The reader validates identity,
// sizes, checksums and exact EOF before publishing output.
bool write_recovered_patchmatch_store_camera(
    const std::filesystem::path& root,
    const RecoveredPatchMatchD4PyramidOutput& camera,
    std::string& error);

bool read_recovered_patchmatch_store_camera(
    const std::filesystem::path& root,
    std::size_t expected_camera_index,
    RecoveredPatchMatchD4PyramidOutput& camera,
    std::string& error);

// Deterministic reference-order batches. Closure is the sorted union of each
// batch reference and its voting neighbours. No dataset-specific partition is
// consumed or inferred.
bool plan_recovered_patchmatch_store_batches(
    std::span<const std::size_t> references,
    std::span<const std::vector<std::size_t>> ranked_neighbors_by_camera,
    std::size_t maximum_references_per_batch,
    std::vector<RecoveredPatchMatchStoreBatch>& batches,
    std::string& error);

}  // namespace metmodel
