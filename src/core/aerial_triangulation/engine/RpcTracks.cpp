#include "engine/RpcEngineInternals.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xjw::aerial_triangulation::engine
{
    camera_models::rpc::ImagePoint rpcImageCoordinate(const TiePointGraph& graph, const RpcObservation& observation)
    {
        const auto& keypoint = graph.keypointsByImage.at(observation.imageId).at(observation.featureIdx);
        return {keypoint.x, keypoint.y};
    }
    namespace
    {
        class DisjointSet
        {
        public:
            std::size_t add()
            {
                const std::size_t index = _parent.size();
                _parent.push_back(index);
                _rank.push_back(0);
                return index;
            }

            std::size_t find(std::size_t index)
            {
                if (_parent[index] != index)
                {
                    _parent[index] = find(_parent[index]);
                }
                return _parent[index];
            }

            void unite(std::size_t first, std::size_t second)
            {
                first = find(first);
                second = find(second);
                if (first == second)
                {
                    return;
                }
                if (_rank[first] < _rank[second])
                {
                    std::swap(first, second);
                }
                _parent[second] = first;
                if (_rank[first] == _rank[second])
                {
                    ++_rank[first];
                }
            }

        private:
            std::vector<std::size_t> _parent;
            std::vector<unsigned char> _rank;
        };

    } // namespace
    namespace detail
    {
        std::vector<std::vector<RpcObservation>> buildRpcTracks(const TiePointGraph& graph)
        {
            std::map<RpcObservation, std::size_t> indexByRpcObservation;
            std::vector<RpcObservation> observations;
            DisjointSet components;

            const auto observationIndex = [&](const RpcObservation& observation)
            {
                const auto existing = indexByRpcObservation.find(observation);
                if (existing != indexByRpcObservation.end())
                {
                    return existing->second;
                }
                const std::size_t index = components.add();
                indexByRpcObservation.emplace(observation, index);
                observations.push_back(observation);
                return index;
            };

            for (const TiePointPair& pair : graph.matchPairs)
            {
                const auto keypointsA = graph.keypointsByImage.find(pair.imageA);
                const auto keypointsB = graph.keypointsByImage.find(pair.imageB);
                if (keypointsA == graph.keypointsByImage.cend() || keypointsB == graph.keypointsByImage.cend())
                {
                    continue;
                }
                for (const FeatureMatch& match : pair.matches)
                {
                    if (match.idx1 >= keypointsA->second.size() || match.idx2 >= keypointsB->second.size())
                    {
                        continue;
                    }
                    components.unite(observationIndex({pair.imageA, match.idx1}),
                                     observationIndex({pair.imageB, match.idx2}));
                }
            }

            std::map<std::size_t, std::vector<RpcObservation>> grouped;
            for (std::size_t index = 0; index < observations.size(); ++index)
            {
                grouped[components.find(index)].push_back(observations[index]);
            }

            std::vector<std::vector<RpcObservation>> tracks;
            tracks.reserve(grouped.size());
            for (auto& [root, track] : grouped)
            {
                (void)root;
                std::sort(track.begin(), track.end());
                bool duplicateImage = false;
                for (std::size_t index = 1; index < track.size(); ++index)
                {
                    duplicateImage |= track[index - 1].imageId == track[index].imageId;
                }
                if (!duplicateImage && track.size() >= 2)
                {
                    tracks.push_back(std::move(track));
                }
            }
            return tracks;
        }

    } // namespace detail
} // namespace xjw::aerial_triangulation::engine
