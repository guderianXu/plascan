#include "BinaryGridMinCutSolverInternal.h"

#include <algorithm>
#include <queue>

namespace xjw::mesh::detail
{

ResidualGraph::ResidualGraph(int vertex_count)
    : heads(static_cast<std::size_t>(vertex_count), -1)
{
}

void ResidualGraph::reserve(std::size_t edge_count)
{
    edges.reserve(edge_count);
}

void ResidualGraph::addDirected(
    int source,
    int destination,
    BinaryGridCapacity capacity)
{
    addPair(source, destination, capacity, 0);
}

void ResidualGraph::addUndirected(
    int first,
    int second,
    BinaryGridCapacity capacity)
{
    addPair(first, second, capacity, capacity);
}

void ResidualGraph::addPair(
    int first,
    int second,
    BinaryGridCapacity forward_capacity,
    BinaryGridCapacity reverse_capacity)
{
    const int forward_index = static_cast<int>(edges.size());
    edges.push_back({second, heads[static_cast<std::size_t>(first)], forward_capacity});
    heads[static_cast<std::size_t>(first)] = forward_index;
    edges.push_back({first, heads[static_cast<std::size_t>(second)], reverse_capacity});
    heads[static_cast<std::size_t>(second)] = forward_index + 1;
}

namespace
{

bool cancellationRequested(const std::function<bool()> &is_cancelled)
{
    return is_cancelled && is_cancelled();
}

class HighestLabelPushRelabel
{
public:
    HighestLabelPushRelabel(
        ResidualGraph *graph,
        int source,
        int sink,
        const std::function<bool()> &is_cancelled)
        : _graph(graph),
          _source(source),
          _sink(sink),
          _vertexCount(static_cast<int>(graph->heads.size())),
          _maximumHeight(_vertexCount * 2 + 1),
          _isCancelled(is_cancelled),
          _excess(static_cast<std::size_t>(_vertexCount), 0),
          _height(static_cast<std::size_t>(_vertexCount), 0),
          _current(graph->heads),
          _heightCount(static_cast<std::size_t>(_maximumHeight + 1), 0),
          _active(static_cast<std::size_t>(_vertexCount), 0),
          _buckets(static_cast<std::size_t>(_maximumHeight + 1))
    {
        _globalRelabelWorkLimit = std::max<std::uint64_t>(
            4096U,
            static_cast<std::uint64_t>(_graph->edges.size()));
    }

    MaximumFlowOutcome run()
    {
        if (cancellationRequested(_isCancelled))
        {
            _outcome.cancelled = true;
            return _outcome;
        }
        saturateSource();
        if (!globalRelabel())
        {
            return _outcome;
        }

        int node = -1;
        while (popHighest(&node))
        {
            ++_outcome.dischargeCount;
            if (!discharge(node))
            {
                return _outcome;
            }
            if (_workSinceGlobalRelabel >= _globalRelabelWorkLimit &&
                !globalRelabel())
            {
                return _outcome;
            }
        }

        _outcome.maximumFlow = _excess[static_cast<std::size_t>(_sink)];
        _outcome.solved = true;
        return _outcome;
    }

private:
    void saturateSource()
    {
        for (int edge_index = _graph->heads[static_cast<std::size_t>(_source)];
             edge_index >= 0;
             edge_index = _graph->edges[static_cast<std::size_t>(edge_index)].next)
        {
            auto &edge = _graph->edges[static_cast<std::size_t>(edge_index)];
            const BinaryGridCapacity amount = edge.capacity;
            if (amount <= 0)
            {
                continue;
            }
            edge.capacity = 0;
            _graph->edges[static_cast<std::size_t>(edge_index ^ 1)].capacity += amount;
            _excess[static_cast<std::size_t>(_source)] -= amount;
            _excess[static_cast<std::size_t>(edge.destination)] += amount;
        }
    }

    bool globalRelabel()
    {
        std::fill(_height.begin(), _height.end(), _vertexCount);
        std::fill(_heightCount.begin(), _heightCount.end(), 0);
        std::fill(_active.begin(), _active.end(), 0);
        _current = _graph->heads;
        for (auto &bucket : _buckets)
        {
            bucket.clear();
        }
        _highestActive = -1;

        std::queue<int> frontier;
        _height[static_cast<std::size_t>(_sink)] = 0;
        frontier.push(_sink);
        std::uint64_t scan_count = 0;
        while (!frontier.empty())
        {
            const int node = frontier.front();
            frontier.pop();
            for (int edge_index = _graph->heads[static_cast<std::size_t>(node)];
                 edge_index >= 0;
                 edge_index = _graph->edges[static_cast<std::size_t>(edge_index)].next)
            {
                if ((++scan_count & 0x0FFFU) == 0U &&
                    cancellationRequested(_isCancelled))
                {
                    _outcome.cancelled = true;
                    return false;
                }
                const auto &edge = _graph->edges[static_cast<std::size_t>(edge_index)];
                const auto &reverse =
                    _graph->edges[static_cast<std::size_t>(edge_index ^ 1)];
                if (reverse.capacity <= 0 || edge.destination == _source ||
                    _height[static_cast<std::size_t>(edge.destination)] != _vertexCount)
                {
                    continue;
                }
                _height[static_cast<std::size_t>(edge.destination)] =
                    _height[static_cast<std::size_t>(node)] + 1;
                frontier.push(edge.destination);
            }
        }
        _height[static_cast<std::size_t>(_source)] = _vertexCount;

        for (int node = 0; node < _vertexCount; ++node)
        {
            if (node == _source || node == _sink)
            {
                continue;
            }
            ++_heightCount[static_cast<std::size_t>(_height[static_cast<std::size_t>(node)])];
            activate(node);
        }
        _workSinceGlobalRelabel = 0;
        return true;
    }

    void activate(int node)
    {
        const std::size_t index = static_cast<std::size_t>(node);
        if (node == _source || node == _sink || _excess[index] <= 0 || _active[index] != 0)
        {
            return;
        }
        const int node_height = _height[index];
        if (node_height < 0 || node_height > _maximumHeight)
        {
            return;
        }
        _active[index] = 1;
        _buckets[static_cast<std::size_t>(node_height)].push_back(node);
        _highestActive = std::max(_highestActive, node_height);
    }

    bool popHighest(int *node)
    {
        while (_highestActive >= 0)
        {
            auto &bucket = _buckets[static_cast<std::size_t>(_highestActive)];
            while (!bucket.empty())
            {
                const int candidate = bucket.back();
                bucket.pop_back();
                const std::size_t index = static_cast<std::size_t>(candidate);
                if (_active[index] != 0 && _excess[index] > 0 &&
                    _height[index] == _highestActive)
                {
                    _active[index] = 0;
                    *node = candidate;
                    return true;
                }
            }
            --_highestActive;
        }
        return false;
    }

    bool discharge(int node)
    {
        const std::size_t node_index = static_cast<std::size_t>(node);
        while (_excess[node_index] > 0)
        {
            if ((++_cancellationWork & 0x0FFFU) == 0U &&
                cancellationRequested(_isCancelled))
            {
                _outcome.cancelled = true;
                return false;
            }
            int &edge_index = _current[node_index];
            if (edge_index < 0)
            {
                if (!relabel(node))
                {
                    return false;
                }
                continue;
            }

            auto &edge = _graph->edges[static_cast<std::size_t>(edge_index)];
            ++_workSinceGlobalRelabel;
            if (edge.capacity > 0 &&
                _height[node_index] ==
                    _height[static_cast<std::size_t>(edge.destination)] + 1)
            {
                const BinaryGridCapacity amount = std::min(_excess[node_index], edge.capacity);
                edge.capacity -= amount;
                _graph->edges[static_cast<std::size_t>(edge_index ^ 1)].capacity += amount;
                _excess[node_index] -= amount;
                _excess[static_cast<std::size_t>(edge.destination)] += amount;
                activate(edge.destination);
                ++_outcome.pushCount;
            }
            else
            {
                edge_index = edge.next;
            }
        }
        return true;
    }

    bool relabel(int node)
    {
        const std::size_t node_index = static_cast<std::size_t>(node);
        const int old_height = _height[node_index];
        int minimum_height = _maximumHeight;
        for (int edge_index = _graph->heads[node_index];
             edge_index >= 0;
             edge_index = _graph->edges[static_cast<std::size_t>(edge_index)].next)
        {
            ++_workSinceGlobalRelabel;
            const auto &edge = _graph->edges[static_cast<std::size_t>(edge_index)];
            if (edge.capacity > 0)
            {
                minimum_height = std::min(
                    minimum_height,
                    _height[static_cast<std::size_t>(edge.destination)]);
            }
        }
        if (minimum_height >= _maximumHeight)
        {
            _outcome.error = "active node has no usable residual edge";
            return false;
        }

        const int new_height = minimum_height + 1;
        --_heightCount[static_cast<std::size_t>(old_height)];
        _height[node_index] = new_height;
        ++_heightCount[static_cast<std::size_t>(new_height)];
        _current[node_index] = _graph->heads[node_index];
        ++_outcome.relabelCount;
        if (old_height < _vertexCount &&
            _heightCount[static_cast<std::size_t>(old_height)] == 0)
        {
            applyGap(old_height, node);
        }
        return true;
    }

    void applyGap(int gap_height, int discharging_node)
    {
        for (int node = 0; node < _vertexCount; ++node)
        {
            if (node == _source || node == _sink)
            {
                continue;
            }
            const std::size_t index = static_cast<std::size_t>(node);
            const int node_height = _height[index];
            if (node_height <= gap_height || node_height >= _vertexCount)
            {
                continue;
            }
            --_heightCount[static_cast<std::size_t>(node_height)];
            _height[index] = _vertexCount;
            ++_heightCount[static_cast<std::size_t>(_vertexCount)];
            _current[index] = _graph->heads[index];
            _active[index] = 0;
            if (node != discharging_node)
            {
                activate(node);
            }
        }
    }

    ResidualGraph *_graph = nullptr;
    int _source = -1;
    int _sink = -1;
    int _vertexCount = 0;
    int _maximumHeight = 0;
    const std::function<bool()> &_isCancelled;
    std::vector<BinaryGridCapacity> _excess;
    std::vector<int> _height;
    std::vector<int> _current;
    std::vector<int> _heightCount;
    std::vector<std::uint8_t> _active;
    std::vector<std::vector<int>> _buckets;
    int _highestActive = -1;
    std::uint64_t _workSinceGlobalRelabel = 0;
    std::uint64_t _globalRelabelWorkLimit = 0;
    std::uint64_t _cancellationWork = 0;
    MaximumFlowOutcome _outcome;
};

} // namespace

MaximumFlowOutcome solveMaximumFlow(
    ResidualGraph *graph,
    int source,
    int sink,
    const std::function<bool()> &is_cancelled)
{
    HighestLabelPushRelabel solver(graph, source, sink, is_cancelled);
    return solver.run();
}

} // namespace xjw::mesh::detail
