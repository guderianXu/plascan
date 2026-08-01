#pragma once

#include "BinaryGridMinCutSolver.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace xjw::mesh::detail
{

struct ResidualEdge
{
    int destination = -1;
    int next = -1;
    BinaryGridCapacity capacity = 0;
};

class ResidualGraph
{
public:
    explicit ResidualGraph(int vertex_count);

    void reserve(std::size_t edge_count);
    void addDirected(int source, int destination, BinaryGridCapacity capacity);
    void addUndirected(int first, int second, BinaryGridCapacity capacity);

    std::vector<int> heads;
    std::vector<ResidualEdge> edges;

private:
    void addPair(
        int first,
        int second,
        BinaryGridCapacity forward_capacity,
        BinaryGridCapacity reverse_capacity);
};

struct MaximumFlowOutcome
{
    bool solved = false;
    bool cancelled = false;
    std::string error;
    BinaryGridCapacity maximumFlow = 0;
    std::uint64_t pushCount = 0;
    std::uint64_t relabelCount = 0;
    std::uint64_t dischargeCount = 0;
};

MaximumFlowOutcome solveMaximumFlow(
    ResidualGraph *graph,
    int source,
    int sink,
    const std::function<bool()> &is_cancelled);

} // namespace xjw::mesh::detail
