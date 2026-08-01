#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh
{

using BinaryGridCapacity = std::int64_t;

enum class BinaryGridLabel : std::uint8_t
{
    Empty = 0,
    Full = 1
};

/**
 * @brief Integer-capacity binary energy defined on a regular 3D grid.
 *
 * A Full node belongs to the source side of the cut and an Empty node belongs
 * to the sink side. Therefore sourceCapacities[i] is paid when node i is Empty,
 * while sinkCapacities[i] is paid when node i is Full.
 *
 * Positive-axis pairwise arrays encode a Potts penalty paid when adjacent
 * labels differ. Each array is either empty (all-zero) or has nodeCount()
 * entries. Values stored on the corresponding positive grid boundary must be
 * zero because no neighbor exists there.
 */
struct BinaryGridMinCutProblem
{
    int sizeX = 0;
    int sizeY = 0;
    int sizeZ = 0;
    std::vector<BinaryGridCapacity> sourceCapacities;
    std::vector<BinaryGridCapacity> sinkCapacities;
    std::vector<BinaryGridCapacity> positiveXCapacities;
    std::vector<BinaryGridCapacity> positiveYCapacities;
    std::vector<BinaryGridCapacity> positiveZCapacities;

    std::size_t nodeCount() const;
    std::size_t index(int x, int y, int z) const;
};

struct BinaryGridMinCutStatistics
{
    std::uint64_t nodeCount = 0;
    std::uint64_t pairwiseEdgeCount = 0;
    std::uint64_t sourceSetNodeCount = 0;
    std::uint64_t pushCount = 0;
    std::uint64_t relabelCount = 0;
    std::uint64_t dischargeCount = 0;
    BinaryGridCapacity maximumFlow = 0;
    BinaryGridCapacity cutEnergy = 0;
};

struct BinaryGridMinCutResult
{
    bool solved = false;
    bool cancelled = false;
    std::string error;
    std::vector<BinaryGridLabel> labels;
    BinaryGridMinCutStatistics statistics;
};

/**
 * @brief Deterministic, zero-dependency regular-grid s-t minimum-cut solver.
 *
 * The implementation uses a deterministic highest-label push-relabel
 * maximum-flow algorithm with global relabel and gap heuristics. Cancellation
 * returns no partial labeling.
 */
class BinaryGridMinCutSolver
{
public:
    static BinaryGridMinCutResult solve(
        const BinaryGridMinCutProblem &problem,
        const std::function<bool()> &isCancelled = {});
};

} // namespace xjw::mesh
