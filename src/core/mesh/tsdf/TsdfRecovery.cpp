#include "DepthTsdfInternals.h"
namespace xjw::mesh
{
    using namespace tsdf_detail;

    std::vector<std::uint8_t> buildVisualHullOccupancy(const DepthTsdfLayout& layout,
                                                       const QVector<DepthTsdfFrame>& frames,
                                                       int minimumVisibleViews,
                                                       int allowedSilhouetteViolations,
                                                       const std::function<bool()>& isCancelled)
    {
        std::vector<std::uint8_t> occupied(static_cast<std::size_t>(layout.sampleCount), 0);
        if (!layout.ok || frames.size() < 2)
        {
            return occupied;
        }

        const int minimum_visible_views = std::clamp(minimumVisibleViews, 2, static_cast<int>(frames.size()));
        const int allowed_violations = std::clamp(allowedSilhouetteViolations, 0, static_cast<int>(frames.size()) - 1);

        std::vector<VisualHullView> views;
        views.reserve(static_cast<std::size_t>(frames.size()));
        for (const DepthTsdfFrame& frame : frames)
        {
            VisualHullView view;
            view.camera = frame.camera;
            view.silhouetteMask = frame.supportMask;
            views.push_back(std::move(view));
        }

        VisualHullConfig config;
        config.minimumVisibleViews = minimum_visible_views;
        config.allowedSilhouetteViolations = allowed_violations;
        config.closeVolumeBoundary = false;
        config.computeBackend = VisualHullComputeBackend::Auto;
        config.isCancelled = isCancelled;
        const detail::RegularGrid3D grid{layout.boundsMin, layout.boundsMax, layout.cells};
        std::vector<float> field;
        if (!detail::evaluateVisualHullFieldGrid(views, config, grid, &field))
        {
            if (isCancelled && isCancelled())
            {
                occupied.clear();
            }
            return occupied;
        }
        const std::size_t count = std::min(occupied.size(), field.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            occupied[index] = field[index] < 0.0f ? 1 : 0;
        }
        return occupied;
    }

    std::uint64_t relaxVisualHullCompletionField(const DepthTsdfLayout& layout,
                                                 const std::vector<std::uint8_t>& occupied,
                                                 const std::vector<std::uint8_t>& completionMask,
                                                 const std::vector<std::uint8_t>& supported,
                                                 int iterations,
                                                 float lambda,
                                                 float maximumUpdate,
                                                 std::vector<float>* tsdf)
    {
        if (!tsdf || !layout.ok || occupied.size() != tsdf->size() || completionMask.size() != tsdf->size() ||
            supported.size() != tsdf->size())
        {
            return 0;
        }
        std::vector<std::size_t> completion_indices;
        completion_indices.reserve(
            static_cast<std::size_t>(std::count(completionMask.cbegin(), completionMask.cend(), std::uint8_t{1})));
        for (std::size_t index = 0; index < completionMask.size(); ++index)
        {
            if (completionMask[index] != 0)
            {
                completion_indices.push_back(index);
            }
        }
        if (completion_indices.empty())
        {
            return 0;
        }

        const int relaxation_iterations = std::clamp(iterations, 0, 24);
        const float relaxation_lambda = std::clamp(lambda, 0.0f, 0.49f);
        const float maximum_update = std::clamp(maximumUpdate, 0.01f, 1.0f);
        std::vector<float> relaxed(completion_indices.size(), 0.0f);
        const int row_size = layout.cells[0] + 1;
        const int layer_size = row_size * (layout.cells[1] + 1);
        for (int iteration = 0; iteration < relaxation_iterations; ++iteration)
        {
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (std::int64_t candidate = 0; candidate < static_cast<std::int64_t>(completion_indices.size());
                 ++candidate)
            {
                const std::size_t index = completion_indices[static_cast<std::size_t>(candidate)];
                const int z = static_cast<int>(index / layer_size);
                const int remainder = static_cast<int>(index % layer_size);
                const int y = remainder / row_size;
                const int x = remainder % row_size;
                float sum = 0.0f;
                int count = 0;
                constexpr std::array<std::array<int, 3>, 6> kOffsets = {
                    {{{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}}, {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
                for (const auto& offset : kOffsets)
                {
                    const int neighbor_x = x + offset[0];
                    const int neighbor_y = y + offset[1];
                    const int neighbor_z = z + offset[2];
                    if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 || neighbor_x > layout.cells[0] ||
                        neighbor_y > layout.cells[1] || neighbor_z > layout.cells[2])
                    {
                        continue;
                    }
                    const std::size_t neighbor = sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z);
                    if (supported[neighbor] == 0)
                    {
                        continue;
                    }
                    sum += (*tsdf)[neighbor];
                    ++count;
                }
                float value = (*tsdf)[index];
                if (count >= 2)
                {
                    const float target = sum / static_cast<float>(count);
                    const float update =
                        std::clamp((target - value) * relaxation_lambda, -maximum_update, maximum_update);
                    value += update;
                }
                const bool inside = occupied[index] != 0;
                if (inside)
                {
                    value = std::min(value, -1.0e-4f);
                }
                else
                {
                    value = std::max(value, 1.0e-4f);
                }
                relaxed[static_cast<std::size_t>(candidate)] = value;
            }
            for (std::size_t candidate = 0; candidate < completion_indices.size(); ++candidate)
            {
                (*tsdf)[completion_indices[candidate]] = relaxed[candidate];
            }
        }
        return static_cast<std::uint64_t>(completion_indices.size());
    }

    int DepthTsdfSurfaceBuilder::growGeometryVerifiedSingleViewSamples(const DepthTsdfLayout& layout,
                                                                       const std::vector<float>& tsdf,
                                                                       const std::vector<std::size_t>& candidateIndices,
                                                                       int minimumNeighborCount,
                                                                       int passes,
                                                                       float maximumTsdfDelta,
                                                                       std::vector<std::uint8_t>* supported)
    {
        if (!supported || supported->size() != tsdf.size() ||
            supported->size() != static_cast<std::size_t>(layout.sampleCount) || candidateIndices.empty())
        {
            return 0;
        }
        const int samples_x = layout.cells[0] + 1;
        const int samples_y = layout.cells[1] + 1;
        const int samples_z = layout.cells[2] + 1;
        if (samples_x <= 0 || samples_y <= 0 || samples_z <= 0)
        {
            return 0;
        }
        const std::size_t slice_size = static_cast<std::size_t>(samples_x) * samples_y;
        const int required_neighbors = std::clamp(minimumNeighborCount, 1, 26);
        const int maximum_passes = std::clamp(passes, 1, 6);
        const float maximum_delta = std::max(0.01f, maximumTsdfDelta);
        std::vector<std::size_t> pending = candidateIndices;
        std::vector<std::size_t> next_pending;
        std::vector<std::size_t> accepted;
        int accepted_count = 0;
        for (int pass = 0; pass < maximum_passes && !pending.empty(); ++pass)
        {
            accepted.clear();
            next_pending.clear();
            accepted.reserve(pending.size());
            next_pending.reserve(pending.size());
            for (const std::size_t index : pending)
            {
                if (index >= supported->size() || (*supported)[index])
                {
                    continue;
                }
                const int z = static_cast<int>(index / slice_size);
                const std::size_t within_slice = index % slice_size;
                const int y = static_cast<int>(within_slice / samples_x);
                const int x = static_cast<int>(within_slice % samples_x);
                int neighbor_count = 0;
                for (int dz = -1; dz <= 1 && neighbor_count < required_neighbors; ++dz)
                {
                    const int neighbor_z = z + dz;
                    if (neighbor_z < 0 || neighbor_z >= samples_z)
                    {
                        continue;
                    }
                    for (int dy = -1; dy <= 1 && neighbor_count < required_neighbors; ++dy)
                    {
                        const int neighbor_y = y + dy;
                        if (neighbor_y < 0 || neighbor_y >= samples_y)
                        {
                            continue;
                        }
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            const int neighbor_x = x + dx;
                            if ((dx == 0 && dy == 0 && dz == 0) || neighbor_x < 0 || neighbor_x >= samples_x)
                            {
                                continue;
                            }
                            const std::size_t neighbor_index = sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z);
                            if (!(*supported)[neighbor_index] ||
                                std::fabs(tsdf[neighbor_index] - tsdf[index]) > maximum_delta)
                            {
                                continue;
                            }
                            ++neighbor_count;
                            if (neighbor_count >= required_neighbors)
                            {
                                break;
                            }
                        }
                    }
                }
                if (neighbor_count >= required_neighbors)
                {
                    accepted.push_back(index);
                }
                else
                {
                    next_pending.push_back(index);
                }
            }
            if (accepted.empty())
            {
                break;
            }
            for (const std::size_t index : accepted)
            {
                (*supported)[index] = 1;
            }
            accepted_count += static_cast<int>(accepted.size());
            pending.swap(next_pending);
        }
        return accepted_count;
    }

    DepthTsdfZeroCrossingStatistics
    DepthTsdfSurfaceBuilder::analyzeZeroCrossings(const DepthTsdfLayout& layout,
                                                  const std::vector<float>& tsdf,
                                                  const std::vector<float>& weight,
                                                  const std::vector<std::uint8_t>& supported)
    {
        DepthTsdfZeroCrossingStatistics statistics;
        const std::size_t expected_size = static_cast<std::size_t>(layout.sampleCount);
        if (!layout.ok || tsdf.size() != expected_size || weight.size() != expected_size ||
            supported.size() != expected_size)
        {
            return statistics;
        }

        unsigned long long observed_cell_count = 0;
        unsigned long long raw_candidate_cell_count = 0;
        unsigned long long extractable_cell_count = 0;
        unsigned long long suppressed_by_support_cell_count = 0;
        unsigned long long positive_only_supported_cell_count = 0;
        unsigned long long negative_only_supported_cell_count = 0;
        unsigned long long partially_supported_cell_count = 0;
        unsigned long long fully_unsupported_observed_cell_count = 0;
#ifdef MESHING_OPENMP
#pragma omp parallel for schedule(static)                                                                              \
    reduction(+ : observed_cell_count, raw_candidate_cell_count, extractable_cell_count)                               \
    reduction(+ : suppressed_by_support_cell_count, positive_only_supported_cell_count)                                \
    reduction(+ : negative_only_supported_cell_count, partially_supported_cell_count)                                  \
    reduction(+ : fully_unsupported_observed_cell_count)
#endif
        for (int z = 0; z < layout.cells[2]; ++z)
        {
            for (int y = 0; y < layout.cells[1]; ++y)
            {
                for (int x = 0; x < layout.cells[0]; ++x)
                {
                    int observed_count = 0;
                    int supported_count = 0;
                    bool observed_positive = false;
                    bool observed_negative = false;
                    bool supported_positive = false;
                    bool supported_negative = false;
                    for (int dz = 0; dz <= 1; ++dz)
                    {
                        for (int dy = 0; dy <= 1; ++dy)
                        {
                            for (int dx = 0; dx <= 1; ++dx)
                            {
                                const std::size_t index = sampleIndex(layout, x + dx, y + dy, z + dz);
                                if (weight[index] > 0.0f)
                                {
                                    ++observed_count;
                                    observed_positive = observed_positive || tsdf[index] >= 0.0f;
                                    observed_negative = observed_negative || tsdf[index] < 0.0f;
                                }
                                if (supported[index] != 0)
                                {
                                    ++supported_count;
                                    supported_positive = supported_positive || tsdf[index] >= 0.0f;
                                    supported_negative = supported_negative || tsdf[index] < 0.0f;
                                }
                            }
                        }
                    }
                    if (observed_count == 0)
                    {
                        continue;
                    }
                    ++observed_cell_count;
                    const bool raw_candidate = observed_positive && observed_negative;
                    const bool extractable = supported_positive && supported_negative;
                    raw_candidate_cell_count += raw_candidate;
                    extractable_cell_count += extractable;
                    suppressed_by_support_cell_count += raw_candidate && !extractable;
                    positive_only_supported_cell_count += supported_positive && !supported_negative;
                    negative_only_supported_cell_count += supported_negative && !supported_positive;
                    partially_supported_cell_count += supported_count > 0 && supported_count < 8;
                    fully_unsupported_observed_cell_count += supported_count == 0;
                }
            }
        }
        statistics.observedCellCount = observed_cell_count;
        statistics.rawCandidateCellCount = raw_candidate_cell_count;
        statistics.extractableCellCount = extractable_cell_count;
        statistics.suppressedBySupportCellCount = suppressed_by_support_cell_count;
        statistics.positiveOnlySupportedCellCount = positive_only_supported_cell_count;
        statistics.negativeOnlySupportedCellCount = negative_only_supported_cell_count;
        statistics.partiallySupportedCellCount = partially_supported_cell_count;
        statistics.fullyUnsupportedObservedCellCount = fully_unsupported_observed_cell_count;
        return statistics;
    }

    DepthTsdfVisualHullCompletionStatistics
    DepthTsdfSurfaceBuilder::completeUnsupportedSamplesWithVisualHullSignedDistance(
        const DepthTsdfLayout& layout,
        const std::vector<std::uint8_t>& occupied,
        float bandVoxels,
        std::vector<float>* tsdf,
        std::vector<std::uint8_t>* supported,
        const std::vector<std::uint8_t>* immutableVeto)
    {
        DepthTsdfVisualHullCompletionStatistics statistics;
        const std::size_t expected_size = static_cast<std::size_t>(layout.sampleCount);
        if (!tsdf || !supported || !layout.ok || occupied.size() != expected_size || tsdf->size() != expected_size ||
            supported->size() != expected_size || (immutableVeto && immutableVeto->size() != expected_size) ||
            !std::isfinite(bandVoxels) || bandVoxels <= 0.0f)
        {
            return statistics;
        }

        statistics.occupiedSampleCount =
            static_cast<std::uint64_t>(std::count(occupied.cbegin(), occupied.cend(), std::uint8_t{1}));
        if (statistics.occupiedSampleCount == 0 || statistics.occupiedSampleCount == expected_size)
        {
            return statistics;
        }

        // Keep the original provenance fixed for the complete operation.  Values
        // synthesized below can never become new zero-crossing anchors.
        const std::vector<std::uint8_t> core_supported = *supported;
        std::vector<std::uint8_t> veto_neighborhood;
        if (immutableVeto)
        {
            veto_neighborhood.assign(expected_size, 0);
            for (int z = 0; z <= layout.cells[2]; ++z)
            {
                for (int y = 0; y <= layout.cells[1]; ++y)
                {
                    for (int x = 0; x <= layout.cells[0]; ++x)
                    {
                        const std::size_t index = sampleIndex(layout, x, y, z);
                        if ((*immutableVeto)[index] == 0)
                        {
                            continue;
                        }
                        for (int dz = -1; dz <= 1; ++dz)
                        {
                            for (int dy = -1; dy <= 1; ++dy)
                            {
                                for (int dx = -1; dx <= 1; ++dx)
                                {
                                    const int neighbor_x = x + dx;
                                    const int neighbor_y = y + dy;
                                    const int neighbor_z = z + dz;
                                    if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 ||
                                        neighbor_x > layout.cells[0] || neighbor_y > layout.cells[1] ||
                                        neighbor_z > layout.cells[2])
                                    {
                                        continue;
                                    }
                                    veto_neighborhood[sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z)] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        const auto is_vetoed = [&veto_neighborhood](std::size_t index)
        { return !veto_neighborhood.empty() && veto_neighborhood[index] != 0; };
        std::vector<std::uint8_t> requested(expected_size, 0);
        const std::size_t cell_count = static_cast<std::size_t>(layout.cells[0]) *
                                       static_cast<std::size_t>(layout.cells[1]) *
                                       static_cast<std::size_t>(layout.cells[2]);
        std::vector<std::uint8_t> frontier_cells(cell_count, 0);
        const auto cell_index = [&](int x, int y, int z)
        {
            return (static_cast<std::size_t>(z) * static_cast<std::size_t>(layout.cells[1]) +
                    static_cast<std::size_t>(y)) *
                       static_cast<std::size_t>(layout.cells[0]) +
                   static_cast<std::size_t>(x);
        };
        const auto inspect_core_signs = [&](int cell_x, int cell_y, int cell_z, bool* has_positive, bool* has_negative)
        {
            *has_positive = false;
            *has_negative = false;
            for (int dz = 0; dz <= 1; ++dz)
            {
                for (int dy = 0; dy <= 1; ++dy)
                {
                    for (int dx = 0; dx <= 1; ++dx)
                    {
                        const std::size_t corner = sampleIndex(layout, cell_x + dx, cell_y + dy, cell_z + dz);
                        if (core_supported[corner] == 0)
                        {
                            continue;
                        }
                        if ((*tsdf)[corner] < 0.0f)
                        {
                            *has_negative = true;
                        }
                        else
                        {
                            *has_positive = true;
                        }
                    }
                }
            }
        };
        for (int z = 0; z < layout.cells[2]; ++z)
        {
            for (int y = 0; y < layout.cells[1]; ++y)
            {
                for (int x = 0; x < layout.cells[0]; ++x)
                {
                    bool anchor_positive = false;
                    bool anchor_negative = false;
                    inspect_core_signs(x, y, z, &anchor_positive, &anchor_negative);
                    if (!anchor_positive || !anchor_negative)
                    {
                        continue;
                    }
                    ++statistics.anchorCellCount;

                    for (int neighbor_z = std::max(0, z - 1); neighbor_z <= std::min(layout.cells[2] - 1, z + 1);
                         ++neighbor_z)
                    {
                        for (int neighbor_y = std::max(0, y - 1); neighbor_y <= std::min(layout.cells[1] - 1, y + 1);
                             ++neighbor_y)
                        {
                            for (int neighbor_x = std::max(0, x - 1);
                                 neighbor_x <= std::min(layout.cells[0] - 1, x + 1);
                                 ++neighbor_x)
                            {
                                bool core_positive = false;
                                bool core_negative = false;
                                inspect_core_signs(neighbor_x, neighbor_y, neighbor_z, &core_positive, &core_negative);
                                if (core_positive == core_negative)
                                {
                                    continue;
                                }

                                bool requested_in_cell = false;
                                for (int dz = 0; dz <= 1; ++dz)
                                {
                                    for (int dy = 0; dy <= 1; ++dy)
                                    {
                                        for (int dx = 0; dx <= 1; ++dx)
                                        {
                                            const std::size_t corner =
                                                sampleIndex(layout, neighbor_x + dx, neighbor_y + dy, neighbor_z + dz);
                                            if (core_supported[corner] != 0)
                                            {
                                                continue;
                                            }
                                            // Preserve a one-sample clearance
                                            // around trusted free-space evidence;
                                            // otherwise neighboring hull samples
                                            // can still cap a real through-hole.
                                            if (is_vetoed(corner))
                                            {
                                                continue;
                                            }
                                            const bool hull_negative = occupied[corner] != 0;
                                            const bool supplies_missing_sign =
                                                (core_positive && hull_negative) || (core_negative && !hull_negative);
                                            if (!supplies_missing_sign)
                                            {
                                                continue;
                                            }
                                            requested[corner] = 1;
                                            requested_in_cell = true;
                                        }
                                    }
                                }
                                if (requested_in_cell)
                                {
                                    frontier_cells[cell_index(neighbor_x, neighbor_y, neighbor_z)] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        statistics.frontierCellCount =
            static_cast<std::uint64_t>(std::count(frontier_cells.cbegin(), frontier_cells.cend(), std::uint8_t{1}));
        if (statistics.anchorCellCount == 0 || statistics.frontierCellCount == 0)
        {
            return statistics;
        }

        constexpr std::uint16_t kInfinity = std::numeric_limits<std::uint16_t>::max();
        const int maximum_distance = std::clamp(static_cast<int>(std::ceil(bandVoxels * 3.0f)), 3, 192);
        std::vector<std::uint16_t> distance(expected_size, kInfinity);
        constexpr std::array<std::array<int, 3>, 6> kAxisOffsets = {
            {{{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}}, {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}}};
        for (int z = 0; z <= layout.cells[2]; ++z)
        {
            for (int y = 0; y <= layout.cells[1]; ++y)
            {
                for (int x = 0; x <= layout.cells[0]; ++x)
                {
                    const std::size_t index = sampleIndex(layout, x, y, z);
                    for (const auto& offset : kAxisOffsets)
                    {
                        const int neighbor_x = x + offset[0];
                        const int neighbor_y = y + offset[1];
                        const int neighbor_z = z + offset[2];
                        if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 || neighbor_x > layout.cells[0] ||
                            neighbor_y > layout.cells[1] || neighbor_z > layout.cells[2])
                        {
                            continue;
                        }
                        const std::size_t neighbor = sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z);
                        if (occupied[neighbor] != occupied[index])
                        {
                            distance[index] = 1;
                            ++statistics.boundarySampleCount;
                            break;
                        }
                    }
                }
            }
        }

        const auto relax = [&](int x, int y, int z, int dx, int dy, int dz, std::uint16_t cost)
        {
            const int neighbor_x = x + dx;
            const int neighbor_y = y + dy;
            const int neighbor_z = z + dz;
            if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0 || neighbor_x > layout.cells[0] ||
                neighbor_y > layout.cells[1] || neighbor_z > layout.cells[2])
            {
                return;
            }
            const std::size_t index = sampleIndex(layout, x, y, z);
            const std::uint16_t neighbor_distance = distance[sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z)];
            if (neighbor_distance == kInfinity)
            {
                return;
            }
            const int candidate = static_cast<int>(neighbor_distance) + cost;
            if (candidate < distance[index] && candidate <= maximum_distance)
            {
                distance[index] = static_cast<std::uint16_t>(candidate);
            }
        };

        for (int pass = 0; pass < 2; ++pass)
        {
            for (int z = 0; z <= layout.cells[2]; ++z)
            {
                for (int y = 0; y <= layout.cells[1]; ++y)
                {
                    for (int x = 0; x <= layout.cells[0]; ++x)
                    {
                        for (int dz = -1; dz <= 0; ++dz)
                        {
                            for (int dy = -1; dy <= 1; ++dy)
                            {
                                for (int dx = -1; dx <= 1; ++dx)
                                {
                                    if ((dz == 0 && dy > 0) || (dz == 0 && dy == 0 && dx >= 0))
                                    {
                                        continue;
                                    }
                                    const int axes = (dx != 0) + (dy != 0) + (dz != 0);
                                    relax(x, y, z, dx, dy, dz, axes == 1 ? 3 : (axes == 2 ? 4 : 5));
                                }
                            }
                        }
                    }
                }
            }
            for (int z = layout.cells[2]; z >= 0; --z)
            {
                for (int y = layout.cells[1]; y >= 0; --y)
                {
                    for (int x = layout.cells[0]; x >= 0; --x)
                    {
                        for (int dz = 0; dz <= 1; ++dz)
                        {
                            for (int dy = -1; dy <= 1; ++dy)
                            {
                                for (int dx = -1; dx <= 1; ++dx)
                                {
                                    if ((dz == 0 && dy < 0) || (dz == 0 && dy == 0 && dx <= 0))
                                    {
                                        continue;
                                    }
                                    const int axes = (dx != 0) + (dy != 0) + (dz != 0);
                                    relax(x, y, z, dx, dy, dz, axes == 1 ? 3 : (axes == 2 ? 4 : 5));
                                }
                            }
                        }
                    }
                }
            }
        }

        const float inverse_band = 1.0f / std::max(1.0f, bandVoxels * 3.0f);
        for (std::size_t index = 0; index < expected_size; ++index)
        {
            if (requested[index] == 0 || (*supported)[index] != 0 || is_vetoed(index) || distance[index] == kInfinity ||
                distance[index] > maximum_distance)
            {
                continue;
            }
            const float magnitude =
                std::clamp((static_cast<float>(distance[index]) - 0.5f) * inverse_band, 1.0e-4f, 1.0f);
            (*tsdf)[index] = occupied[index] != 0 ? -magnitude : magnitude;
            (*supported)[index] = 1;
            ++statistics.recoveredSampleCount;
        }
        return statistics;
    }

    DepthTsdfZeroCrossingRecoveryStatistics DepthTsdfSurfaceBuilder::recoverGeometryVerifiedZeroCrossingSamples(
        const DepthTsdfLayout& layout,
        const std::vector<float>& tsdf,
        const std::vector<float>& weight,
        const std::vector<DepthGeometrySourceMask>& geometrySourceMask,
        const std::vector<std::uint8_t>& eligible,
        int minimumSupportedCorners,
        int minimumCellVotes,
        std::vector<std::uint8_t>* supported)
    {
        DepthTsdfZeroCrossingRecoveryStatistics statistics;
        const std::size_t expected_size = static_cast<std::size_t>(layout.sampleCount);
        if (!supported || !layout.ok || tsdf.size() != expected_size || weight.size() != expected_size ||
            geometrySourceMask.size() != expected_size || eligible.size() != expected_size ||
            supported->size() != expected_size)
        {
            return statistics;
        }

        const int required_supported_corners = std::clamp(minimumSupportedCorners, 1, 7);
        const int required_votes = std::clamp(minimumCellVotes, 1, 8);
        const std::vector<std::uint8_t> core_supported = *supported;
        std::vector<std::uint8_t> votes(expected_size, 0);
        for (int z = 0; z < layout.cells[2]; ++z)
        {
            for (int y = 0; y < layout.cells[1]; ++y)
            {
                for (int x = 0; x < layout.cells[0]; ++x)
                {
                    std::array<std::size_t, 8> corners{};
                    int corner_count = 0;
                    int supported_count = 0;
                    bool observed_positive = false;
                    bool observed_negative = false;
                    bool supported_positive = false;
                    bool supported_negative = false;
                    for (int dz = 0; dz <= 1; ++dz)
                    {
                        for (int dy = 0; dy <= 1; ++dy)
                        {
                            for (int dx = 0; dx <= 1; ++dx)
                            {
                                const std::size_t index = sampleIndex(layout, x + dx, y + dy, z + dz);
                                corners[static_cast<std::size_t>(corner_count++)] = index;
                                if (weight[index] > 0.0f)
                                {
                                    observed_positive = observed_positive || tsdf[index] >= 0.0f;
                                    observed_negative = observed_negative || tsdf[index] < 0.0f;
                                }
                                if (core_supported[index] != 0)
                                {
                                    ++supported_count;
                                    supported_positive = supported_positive || tsdf[index] >= 0.0f;
                                    supported_negative = supported_negative || tsdf[index] < 0.0f;
                                }
                            }
                        }
                    }
                    if (!observed_positive || !observed_negative || supported_count < required_supported_corners ||
                        supported_positive == supported_negative)
                    {
                        continue;
                    }
                    const bool missing_negative = supported_positive;
                    for (const std::size_t candidate : corners)
                    {
                        if (core_supported[candidate] != 0 || eligible[candidate] == 0 || weight[candidate] <= 0.0f ||
                            ((tsdf[candidate] < 0.0f) != missing_negative))
                        {
                            continue;
                        }
                        bool shares_source = false;
                        for (const std::size_t neighbor : corners)
                        {
                            if (core_supported[neighbor] != 0 &&
                                (geometrySourceMask[candidate] & geometrySourceMask[neighbor]) != 0)
                            {
                                shares_source = true;
                                break;
                            }
                        }
                        bool connected_to_surface = true;
                        if (supported_count == 1)
                        {
                            const int samples_x = layout.cells[0] + 1;
                            const int samples_y = layout.cells[1] + 1;
                            const std::size_t samples_per_slice =
                                static_cast<std::size_t>(samples_x) * static_cast<std::size_t>(samples_y);
                            const int candidate_z = static_cast<int>(candidate / samples_per_slice);
                            const std::size_t slice_offset = candidate % samples_per_slice;
                            const int candidate_y =
                                static_cast<int>(slice_offset / static_cast<std::size_t>(samples_x));
                            const int candidate_x =
                                static_cast<int>(slice_offset % static_cast<std::size_t>(samples_x));
                            int source_connected_neighbor_count = 0;
                            bool has_candidate_sign_neighbor = false;
                            for (int dz = -1; dz <= 1; ++dz)
                            {
                                const int neighbor_z = candidate_z + dz;
                                if (neighbor_z < 0 || neighbor_z > layout.cells[2])
                                {
                                    continue;
                                }
                                for (int dy = -1; dy <= 1; ++dy)
                                {
                                    const int neighbor_y = candidate_y + dy;
                                    if (neighbor_y < 0 || neighbor_y > layout.cells[1])
                                    {
                                        continue;
                                    }
                                    for (int dx = -1; dx <= 1; ++dx)
                                    {
                                        const int neighbor_x = candidate_x + dx;
                                        if ((dx == 0 && dy == 0 && dz == 0) || neighbor_x < 0 ||
                                            neighbor_x > layout.cells[0])
                                        {
                                            continue;
                                        }
                                        const std::size_t neighbor =
                                            sampleIndex(layout, neighbor_x, neighbor_y, neighbor_z);
                                        if (core_supported[neighbor] == 0 ||
                                            (geometrySourceMask[candidate] & geometrySourceMask[neighbor]) == 0)
                                        {
                                            continue;
                                        }
                                        ++source_connected_neighbor_count;
                                        has_candidate_sign_neighbor = has_candidate_sign_neighbor ||
                                                                      ((tsdf[neighbor] < 0.0f) == missing_negative);
                                    }
                                }
                            }
                            connected_to_surface = source_connected_neighbor_count >= 2 && has_candidate_sign_neighbor;
                        }
                        if (shares_source && connected_to_surface)
                        {
                            votes[candidate] =
                                static_cast<std::uint8_t>(std::min(255, static_cast<int>(votes[candidate]) + 1));
                        }
                    }
                }
            }
        }
        for (std::size_t index = 0; index < expected_size; ++index)
        {
            if (votes[index] == 0)
            {
                continue;
            }
            ++statistics.candidateSampleCount;
            if (votes[index] >= required_votes)
            {
                (*supported)[index] = 1;
                ++statistics.recoveredSampleCount;
            }
        }
        return statistics;
    }
} // namespace xjw::mesh
