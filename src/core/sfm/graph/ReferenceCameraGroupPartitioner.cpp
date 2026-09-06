#include "ReferenceCameraGroupPartitioner.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace xjw
{
    namespace
    {

        using Group = std::vector<std::size_t>;
        using PairWeights = std::map<std::pair<std::size_t, std::size_t>, int>;

        struct GraphEdge
        {
            int weight = 0;
            std::size_t first = 0;
            std::size_t second = 0;
        };

        PairWeights collapsePairWeights(const std::vector<Group>& groups, const PairWeights& originalWeights)
        {
            std::vector<std::size_t> groupOf;
            for (const Group& group : groups)
            {
                for (std::size_t camera : group)
                {
                    groupOf.resize(std::max(groupOf.size(), camera + 1));
                }
            }
            for (std::size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
            {
                for (std::size_t camera : groups[groupIndex])
                {
                    groupOf[camera] = groupIndex;
                }
            }

            std::map<std::pair<std::size_t, std::size_t>, std::vector<int>> candidates;
            for (const auto& [pair, weight] : originalWeights)
            {
                std::size_t first = groupOf[pair.first];
                std::size_t second = groupOf[pair.second];
                if (first == second || weight == 0)
                {
                    continue;
                }
                if (first > second)
                {
                    std::swap(first, second);
                }
                std::vector<int>& values = candidates[{first, second}];
                values.insert(std::lower_bound(values.begin(), values.end(), weight), weight);
                if (values.size() > 11)
                {
                    values.erase(values.begin());
                }
            }

            PairWeights collapsed;
            for (const auto& [pair, values] : candidates)
            {
                collapsed[pair] = std::accumulate(values.begin(), values.end(), 0);
            }
            return collapsed;
        }

        std::vector<Group> partitionOnce(const std::vector<Group>& groups,
                                         const PairWeights& weights,
                                         const ReferenceCameraGroupPartitionOptions& options)
        {
            const std::size_t vertexCount = groups.size();
            if (vertexCount == 0)
            {
                return {};
            }

            std::vector<std::vector<std::pair<std::size_t, int>>> adjacency(vertexCount);
            for (const auto& [pair, weight] : weights)
            {
                adjacency[pair.first].push_back({pair.second, weight});
                adjacency[pair.second].push_back({pair.first, weight});
            }

            constexpr int noEdge = std::numeric_limits<int>::min();
            std::vector<int> best(vertexCount, noEdge);
            std::vector<std::size_t> parent(vertexCount, 0);
            std::vector<bool> visited(vertexCount, false);
            std::vector<GraphEdge> forest;
            forest.reserve(vertexCount - 1);
            const auto updateNeighbors = [&](std::size_t vertex)
            {
                for (const auto& [neighbor, weight] : adjacency[vertex])
                {
                    if (!visited[neighbor] && best[neighbor] < weight)
                    {
                        best[neighbor] = weight;
                        parent[neighbor] = vertex;
                    }
                }
            };
            visited[0] = true;
            updateNeighbors(0);
            for (std::size_t step = 1; step < vertexCount; ++step)
            {
                int strongest = noEdge;
                std::size_t selected = 0;
                for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
                {
                    if (!visited[vertex] && best[vertex] > strongest)
                    {
                        strongest = best[vertex];
                        selected = vertex;
                    }
                }
                if (strongest == noEdge)
                {
                    while (selected < vertexCount && visited[selected])
                    {
                        ++selected;
                    }
                }
                else
                {
                    forest.push_back({strongest, parent[selected], selected});
                }
                visited[selected] = true;
                updateNeighbors(selected);
            }

            std::sort(forest.begin(),
                      forest.end(),
                      [](const GraphEdge& left, const GraphEdge& right) { return left.weight < right.weight; });
            std::vector<std::size_t> degree(vertexCount, 0);
            for (const GraphEdge& edge : forest)
            {
                ++degree[edge.first];
                ++degree[edge.second];
            }
            const std::vector<std::size_t> originalDegree = degree;
            std::vector<bool> retained(forest.size(), true);
            for (std::size_t edgeIndex = 0; edgeIndex < forest.size(); ++edgeIndex)
            {
                const GraphEdge& edge = forest[edgeIndex];
                if (degree[edge.first] > 1 && degree[edge.second] > 1)
                {
                    --degree[edge.first];
                    --degree[edge.second];
                    retained[edgeIndex] = false;
                }
            }

            constexpr std::size_t unassigned = std::numeric_limits<std::size_t>::max();
            std::vector<std::size_t> assignment(vertexCount, unassigned);
            std::vector<std::vector<std::size_t>> groupedVertices;
            std::vector<std::size_t> groupedCameraCounts;
            for (std::size_t reverse = forest.size(); reverse > 0; --reverse)
            {
                const std::size_t edgeIndex = reverse - 1;
                if (!retained[edgeIndex])
                {
                    continue;
                }
                const GraphEdge& edge = forest[edgeIndex];
                const std::size_t firstSize = assignment[edge.first] == unassigned
                                                  ? groups[edge.first].size()
                                                  : groupedCameraCounts[assignment[edge.first]];
                const std::size_t secondSize = assignment[edge.second] == unassigned
                                                   ? groups[edge.second].size()
                                                   : groupedCameraCounts[assignment[edge.second]];
                const std::size_t combinedSize = firstSize + secondSize;
                if (combinedSize > options.maximumGroupSize &&
                    std::min(firstSize, secondSize) > options.minimumLargeSide)
                {
                    continue;
                }

                std::size_t outputGroup = unassigned;
                if (assignment[edge.first] != unassigned)
                {
                    outputGroup = assignment[edge.first];
                }
                if (assignment[edge.second] != unassigned)
                {
                    outputGroup = outputGroup == unassigned ? assignment[edge.second]
                                                            : std::min(outputGroup, assignment[edge.second]);
                }
                if (outputGroup == unassigned)
                {
                    outputGroup = groupedVertices.size();
                    groupedVertices.emplace_back();
                    groupedCameraCounts.push_back(0);
                }
                groupedCameraCounts[outputGroup] = combinedSize;
                for (std::size_t vertex : {edge.first, edge.second})
                {
                    if (assignment[vertex] == unassigned)
                    {
                        assignment[vertex] = outputGroup;
                        if (originalDegree[vertex] > 1)
                        {
                            groupedVertices[outputGroup].insert(groupedVertices[outputGroup].begin(), vertex);
                        }
                        else
                        {
                            groupedVertices[outputGroup].push_back(vertex);
                        }
                    }
                    else if (assignment[vertex] != outputGroup)
                    {
                        throw std::runtime_error("reference camera-group forest invariant violated");
                    }
                }
            }

            for (std::size_t vertex = 0; vertex < vertexCount; ++vertex)
            {
                if (assignment[vertex] == unassigned)
                {
                    assignment[vertex] = groupedVertices.size();
                    groupedVertices.push_back({vertex});
                }
            }

            std::vector<Group> expanded;
            expanded.reserve(groupedVertices.size());
            for (const std::vector<std::size_t>& vertices : groupedVertices)
            {
                Group group;
                for (std::size_t vertex : vertices)
                {
                    group.insert(group.end(), groups[vertex].begin(), groups[vertex].end());
                }
                expanded.push_back(std::move(group));
            }
            return expanded;
        }

    } // namespace

    std::vector<ReferenceCameraGroup>
    partitionReferenceCameraGroups(const std::vector<ImageId>& imageIds,
                                   const std::vector<Track>& tracks,
                                   const ReferenceCameraGroupPartitionOptions& options)
    {
        if (options.maximumGroupSize == 0 || options.minimumLargeSide == 0)
        {
            throw std::invalid_argument("reference camera-group size limits must be positive");
        }
        std::unordered_map<ImageId, std::size_t> cameraIndex;
        cameraIndex.reserve(imageIds.size());
        for (std::size_t index = 0; index < imageIds.size(); ++index)
        {
            cameraIndex.emplace(imageIds[index], index);
        }

        PairWeights originalWeights;
        for (const Track& track : tracks)
        {
            std::vector<std::size_t> cameras;
            for (const TrackElement& observation : track.elements)
            {
                const auto it = cameraIndex.find(observation.imageId);
                if (it != cameraIndex.end())
                {
                    cameras.push_back(it->second);
                }
            }
            std::sort(cameras.begin(), cameras.end());
            cameras.erase(std::unique(cameras.begin(), cameras.end()), cameras.end());
            for (std::size_t first = 0; first < cameras.size(); ++first)
            {
                for (std::size_t second = first + 1; second < cameras.size(); ++second)
                {
                    ++originalWeights[{cameras[first], cameras[second]}];
                }
            }
        }

        std::vector<Group> groups(imageIds.size());
        for (std::size_t camera = 0; camera < imageIds.size(); ++camera)
        {
            groups[camera].push_back(camera);
        }
        while (groups.size() > 1)
        {
            const PairWeights collapsed = collapsePairWeights(groups, originalWeights);
            std::vector<Group> next = partitionOnce(groups, collapsed, options);
            const bool converged = next.size() == groups.size();
            groups = std::move(next);
            if (groups.size() == 1 || converged)
            {
                break;
            }
        }

        std::vector<ReferenceCameraGroup> result;
        result.reserve(groups.size());
        for (const Group& group : groups)
        {
            ReferenceCameraGroup imageGroup;
            imageGroup.reserve(group.size());
            for (std::size_t index : group)
            {
                imageGroup.push_back(imageIds[index]);
            }
            result.push_back(std::move(imageGroup));
        }
        return result;
    }

} // namespace xjw
