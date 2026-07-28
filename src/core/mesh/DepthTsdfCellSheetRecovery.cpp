#include "DepthTsdfCellSheetRecovery.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace xjw::mesh
{
namespace
{
struct CandidateCell
{
    int x = 0;
    int y = 0;
    int z = 0;
    bool missingNegative = false;
    bool anchor = false;
    std::uint16_t sourceMask = 0;
    std::array<std::size_t, 8> recoverableCorners{};
    int recoverableCornerCount = 0;
};
std::size_t sampleIndex(const DepthTsdfLayout &layout, int x, int y, int z)
{
    const std::size_t samples_x =
        static_cast<std::size_t>(layout.cells[0] + 1);
    const std::size_t samples_y =
        static_cast<std::size_t>(layout.cells[1] + 1);
    return (static_cast<std::size_t>(z) * samples_y +
            static_cast<std::size_t>(y)) *
            samples_x +
        static_cast<std::size_t>(x);
}
std::size_t cellIndex(const DepthTsdfLayout &layout, int x, int y, int z)
{
    return (static_cast<std::size_t>(z) *
                static_cast<std::size_t>(layout.cells[1]) +
            static_cast<std::size_t>(y)) *
            static_cast<std::size_t>(layout.cells[0]) +
        static_cast<std::size_t>(x);
}

bool isExtractableCell(const DepthTsdfLayout &layout,
                       const std::vector<float> &tsdf,
                       const std::vector<std::uint8_t> &supported,
                       int x,
                       int y,
                       int z)
{
    if (x < 0 || y < 0 || z < 0 ||
        x >= layout.cells[0] ||
        y >= layout.cells[1] ||
        z >= layout.cells[2])
    {
        return false;
    }

    bool positive = false;
    bool negative = false;
    for (int dz = 0; dz <= 1; ++dz)
    {
        for (int dy = 0; dy <= 1; ++dy)
        {
            for (int dx = 0; dx <= 1; ++dx)
            {
                const std::size_t index =
                    sampleIndex(layout, x + dx, y + dy, z + dz);
                if (supported[index] == 0)
                {
                    continue;
                }
                positive = positive || tsdf[index] >= 0.0f;
                negative = negative || tsdf[index] < 0.0f;
            }
        }
    }
    return positive && negative;
}

int findRoot(std::vector<int> *parents, int value)
{
    int root = value;
    while ((*parents)[static_cast<std::size_t>(root)] != root)
    {
        root = (*parents)[static_cast<std::size_t>(root)];
    }
    while ((*parents)[static_cast<std::size_t>(value)] != value)
    {
        const int next = (*parents)[static_cast<std::size_t>(value)];
        (*parents)[static_cast<std::size_t>(value)] = root;
        value = next;
    }
    return root;
}

void unite(std::vector<int> *parents,
           std::vector<std::uint8_t> *ranks,
           int lhs,
           int rhs)
{
    int lhs_root = findRoot(parents, lhs);
    int rhs_root = findRoot(parents, rhs);
    if (lhs_root == rhs_root)
    {
        return;
    }
    if ((*ranks)[static_cast<std::size_t>(lhs_root)] <
        (*ranks)[static_cast<std::size_t>(rhs_root)])
    {
        std::swap(lhs_root, rhs_root);
    }
    (*parents)[static_cast<std::size_t>(rhs_root)] = lhs_root;
    if ((*ranks)[static_cast<std::size_t>(lhs_root)] ==
        (*ranks)[static_cast<std::size_t>(rhs_root)])
    {
        ++(*ranks)[static_cast<std::size_t>(lhs_root)];
    }
}

bool compatible(const CandidateCell &lhs, const CandidateCell &rhs)
{
    return lhs.missingNegative == rhs.missingNegative &&
        (lhs.sourceMask & rhs.sourceMask) != 0;
}

} // namespace

DepthTsdfZeroCrossingRecoveryStatistics
recoverGeometryVerifiedZeroCrossingCellSheets(
    const DepthTsdfLayout &layout,
    const std::vector<float> &tsdf,
    const std::vector<float> &weight,
    const std::vector<std::uint16_t> &geometrySourceMask,
    const std::vector<std::uint8_t> &eligible,
    int minimumSupportedCorners,
    int minimumSheetCells,
    int minimumSheetAnchorCells,
    float maximumSingleVoteAbsoluteTsdf,
    std::vector<std::uint8_t> *supported)
{
    DepthTsdfZeroCrossingRecoveryStatistics statistics;
    const std::size_t sample_count =
        static_cast<std::size_t>(layout.sampleCount);
    if (!supported || !layout.ok ||
        tsdf.size() != sample_count ||
        weight.size() != sample_count ||
        geometrySourceMask.size() != sample_count ||
        eligible.size() != sample_count ||
        supported->size() != sample_count)
    {
        return statistics;
    }

    const int required_supported_corners =
        std::clamp(minimumSupportedCorners, 1, 7);
    const int required_sheet_cells = std::max(1, minimumSheetCells);
    const int required_anchor_cells = std::max(1, minimumSheetAnchorCells);
    const float maximum_single_vote_absolute_tsdf =
        std::clamp(maximumSingleVoteAbsoluteTsdf, 0.0f, 1.0f);
    const std::vector<std::uint8_t> core_supported = *supported;
    const std::size_t cell_count =
        static_cast<std::size_t>(layout.cells[0]) *
        static_cast<std::size_t>(layout.cells[1]) *
        static_cast<std::size_t>(layout.cells[2]);
    std::vector<int> candidate_by_cell(cell_count, -1);
    std::vector<CandidateCell> candidates;

    for (int z = 0; z < layout.cells[2]; ++z)
    {
        for (int y = 0; y < layout.cells[1]; ++y)
        {
            for (int x = 0; x < layout.cells[0]; ++x)
            {
                CandidateCell cell;
                cell.x = x;
                cell.y = y;
                cell.z = z;
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
                            const std::size_t index =
                                sampleIndex(layout, x + dx, y + dy, z + dz);
                            corners[static_cast<std::size_t>(corner_count++)] =
                                index;
                            if (weight[index] > 0.0f)
                            {
                                observed_positive =
                                    observed_positive || tsdf[index] >= 0.0f;
                                observed_negative =
                                    observed_negative || tsdf[index] < 0.0f;
                            }
                            if (core_supported[index] != 0)
                            {
                                ++supported_count;
                                supported_positive =
                                    supported_positive || tsdf[index] >= 0.0f;
                                supported_negative =
                                    supported_negative || tsdf[index] < 0.0f;
                            }
                        }
                    }
                }
                if (!observed_positive || !observed_negative ||
                    supported_count < required_supported_corners ||
                    supported_positive == supported_negative)
                {
                    continue;
                }

                cell.missingNegative = supported_positive;
                for (const std::size_t candidate : corners)
                {
                    if (core_supported[candidate] != 0 ||
                        eligible[candidate] == 0 ||
                        weight[candidate] <= 0.0f ||
                        ((tsdf[candidate] < 0.0f) != cell.missingNegative))
                    {
                        continue;
                    }
                    std::uint16_t shared_sources = 0;
                    for (const std::size_t neighbor : corners)
                    {
                        if (core_supported[neighbor] != 0)
                        {
                            shared_sources |= geometrySourceMask[candidate] &
                                geometrySourceMask[neighbor];
                        }
                    }
                    if (shared_sources == 0)
                    {
                        continue;
                    }
                    cell.sourceMask |= shared_sources;
                    cell.recoverableCorners[static_cast<std::size_t>(
                        cell.recoverableCornerCount++)] = candidate;
                }
                if (cell.recoverableCornerCount == 0)
                {
                    continue;
                }

                static constexpr std::array<std::array<int, 3>, 6>
                    neighbor_offsets{{
                        {-1, 0, 0},
                        {1, 0, 0},
                        {0, -1, 0},
                        {0, 1, 0},
                        {0, 0, -1},
                        {0, 0, 1}}};
                for (const auto &offset : neighbor_offsets)
                {
                    if (isExtractableCell(
                            layout,
                            tsdf,
                            core_supported,
                            x + offset[0],
                            y + offset[1],
                            z + offset[2]))
                    {
                        cell.anchor = true;
                        break;
                    }
                }

                const int candidate_index =
                    static_cast<int>(candidates.size());
                candidate_by_cell[cellIndex(layout, x, y, z)] =
                    candidate_index;
                candidates.push_back(cell);
            }
        }
    }

    statistics.candidateCellCount = candidates.size();
    if (candidates.empty())
    {
        return statistics;
    }

    std::vector<int> parents(candidates.size());
    std::iota(parents.begin(), parents.end(), 0);
    std::vector<std::uint8_t> ranks(candidates.size(), 0);
    for (int candidate_index = 0;
         candidate_index < static_cast<int>(candidates.size());
         ++candidate_index)
    {
        const CandidateCell &cell =
            candidates[static_cast<std::size_t>(candidate_index)];
        const std::array<std::array<int, 3>, 3> previous_offsets{{
            {-1, 0, 0},
            {0, -1, 0},
            {0, 0, -1}}};
        for (const auto &offset : previous_offsets)
        {
            const int neighbor_x = cell.x + offset[0];
            const int neighbor_y = cell.y + offset[1];
            const int neighbor_z = cell.z + offset[2];
            if (neighbor_x < 0 || neighbor_y < 0 || neighbor_z < 0)
            {
                continue;
            }
            const int neighbor_index = candidate_by_cell[cellIndex(
                layout, neighbor_x, neighbor_y, neighbor_z)];
            if (neighbor_index >= 0 &&
                compatible(cell, candidates[static_cast<std::size_t>(
                                     neighbor_index)]))
            {
                unite(&parents, &ranks, candidate_index, neighbor_index);
            }
        }
    }

    std::vector<int> component_sizes(candidates.size(), 0);
    std::vector<int> component_anchor_counts(candidates.size(), 0);
    for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
    {
        const int root = findRoot(&parents, index);
        ++component_sizes[static_cast<std::size_t>(root)];
        component_anchor_counts[static_cast<std::size_t>(root)] +=
            candidates[static_cast<std::size_t>(index)].anchor ? 1 : 0;
    }

    std::vector<std::uint8_t> accepted_components(candidates.size(), 0);
    for (int root = 0; root < static_cast<int>(candidates.size()); ++root)
    {
        if (component_sizes[static_cast<std::size_t>(root)] == 0)
        {
            continue;
        }
        ++statistics.componentCount;
        if (component_sizes[static_cast<std::size_t>(root)] <
            required_sheet_cells)
        {
            ++statistics.rejectedSmallComponentCount;
            continue;
        }
        if (component_anchor_counts[static_cast<std::size_t>(root)] <
            required_anchor_cells)
        {
            ++statistics.rejectedAnchorComponentCount;
            continue;
        }
        accepted_components[static_cast<std::size_t>(root)] = 1;
        ++statistics.acceptedComponentCount;
        statistics.acceptedCellCount += static_cast<std::uint64_t>(
            component_sizes[static_cast<std::size_t>(root)]);
    }

    std::vector<std::uint8_t> recover_votes(sample_count, 0);
    for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
    {
        const int root = findRoot(&parents, index);
        if (accepted_components[static_cast<std::size_t>(root)] == 0)
        {
            continue;
        }
        const CandidateCell &cell = candidates[static_cast<std::size_t>(index)];
        for (int corner = 0; corner < cell.recoverableCornerCount; ++corner)
        {
            const std::size_t candidate =
                cell.recoverableCorners[static_cast<std::size_t>(corner)];
            recover_votes[candidate] = static_cast<std::uint8_t>(
                std::min(255, static_cast<int>(recover_votes[candidate]) + 1));
        }
    }
    for (std::size_t index = 0; index < sample_count; ++index)
    {
        if (recover_votes[index] == 0)
        {
            continue;
        }
        ++statistics.candidateSampleCount;
        const bool strong_cell_vote = recover_votes[index] >= 2;
        const bool near_zero_single_vote =
            std::abs(tsdf[index]) <= maximum_single_vote_absolute_tsdf;
        if ((*supported)[index] == 0 &&
            (strong_cell_vote || near_zero_single_vote))
        {
            (*supported)[index] = 1;
            ++statistics.recoveredSampleCount;
        }
    }
    return statistics;
}

} // namespace xjw::mesh
