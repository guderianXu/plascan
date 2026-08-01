#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

struct VisibilityOccupancyHandleRepairOptions
{
    int maximumAcceptedCandidateCount = 32;
    std::size_t maximumCandidateSampleCount = 512;
    std::size_t maximumSubsetSampleCount = 96;
    int maximumSubsetSeedCount = 0;
    std::function<bool()> isCancelled;
};

struct VisibilityOccupancyHandleRepairStatistics
{
    std::uint64_t sampleCount = 0;
    std::uint64_t proposalAddedSampleCount = 0;
    std::uint64_t filledSampleCount = 0;
    std::uint64_t protectedExteriorSampleCountBefore = 0;
    std::uint64_t protectedExteriorSampleCountAfter = 0;
    int candidateComponentCount = 0;
    int acceptedCandidateCount = 0;
    int acceptedSubsetCandidateCount = 0;
    int acceptedPlateauSubsetCandidateCount = 0;
    int attemptedSubsetSeedCount = 0;
    int rejectedProtectedCandidateCount = 0;
    int rejectedOversizedCandidateCount = 0;
    int rejectedTopologyCandidateCount = 0;
    int rejectedProtectedReachabilityCandidateCount = 0;
    int bodyEulerBefore = 0;
    int bodyEulerAfter = 0;
};

struct VisibilityOccupancyHandleRepairResult
{
    bool ok = false;
    bool cancelled = false;
    std::string error;
    std::vector<std::uint8_t> occupied;
    VisibilityOccupancyHandleRepairStatistics statistics;
};

/**
 * @brief Selectively applies topology-improving additions from a closing proposal.
 *
 * The proposal must be a superset of occupied. Candidate additions are split
 * into deterministic face-connected components. Whole components are the fast
 * path; if their net Euler delta is not positive, topology-improving singleton
 * subsets are considered. A transaction is accepted only when it raises the
 * cubical-complex Euler characteristic without changing the occupied component
 * count or making any previously exterior-reachable empty sample interior.
 */
class VisibilityOccupancyHandleRepair
{
public:
    static VisibilityOccupancyHandleRepairResult repair(
        const std::array<int, 3> &dimensions,
        const std::vector<std::uint8_t> &occupied,
        const std::vector<std::uint8_t> &closingProposal,
        const std::vector<std::uint8_t> &protectedEmpty,
        const VisibilityOccupancyHandleRepairOptions &options = {});

    static int bodyEulerCharacteristic(
        const std::array<int, 3> &dimensions,
        const std::vector<std::uint8_t> &occupied);
};

} // namespace xjw::mesh
