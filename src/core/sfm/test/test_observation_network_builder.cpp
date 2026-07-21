#include "graph/ObservationNetworkBuilder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace
{

std::vector<std::pair<int, int>> edgeKeys(const xjw::ObservationNetwork &network)
{
    std::vector<std::pair<int, int>> keys;
    keys.reserve(network.edges.size());
    for (const xjw::NetworkEdge &edge : network.edges)
    {
        keys.emplace_back(std::min(edge.idx0, edge.idx1), std::max(edge.idx0, edge.idx1));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

std::vector<std::pair<int, int>> matchEdgeKeys(const std::vector<xjw::MatchEdge> &edges)
{
    std::vector<std::pair<int, int>> keys;
    keys.reserve(edges.size());
    for (const xjw::MatchEdge &edge : edges)
    {
        keys.emplace_back(std::min(edge.idx0, edge.idx1), std::max(edge.idx0, edge.idx1));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

} // namespace

TEST(ObservationNetworkBuilderTest, KdTreeKeepsNearestMatchedNeighbors)
{
    const std::vector<std::string> names{"0", "1", "2", "3"};
    const std::vector<xjw::GpsCoord> gps{
        {0.0, 0.0, true},
        {2.0, 0.0, true},
        {5.0, 0.0, true},
        {9.0, 0.0, true},
    };
    const std::vector<xjw::MatchEdge> edges{
        {0, 1, 80, 0.5, 0.9},
        {0, 2, 70, 0.5, 0.9},
        {0, 3, 60, 0.5, 0.9},
        {1, 2, 50, 0.5, 0.9},
        {1, 3, 40, 0.5, 0.9},
        {2, 3, 30, 0.5, 0.9},
    };

    xjw::ObservationNetworkConfig config;
    config.algorithm = xjw::ObservationNetworkConfig::KDTree;
    config.k = 1;
    config.minMatches = 1;
    config.minOverlap = 0.0;
    config.pruneWeak = false;

    const xjw::ObservationNetwork result =
        xjw::ObservationNetworkBuilder::build(names, edges, gps, config);

    const std::vector<std::pair<int, int>> expected{{0, 1}, {1, 2}, {2, 3}};
    EXPECT_EQ(edgeKeys(result), expected);
    EXPECT_EQ(result.degrees, (std::vector<int>{1, 2, 2, 1}));
}

TEST(ObservationNetworkBuilderTest, KdTreeFallsBackToSequenceCoordinates)
{
    const std::vector<std::string> names{"0", "1", "2", "3"};
    const std::vector<xjw::MatchEdge> edges{
        {0, 1, 40, 0.0, 1.0},
        {1, 2, 40, 0.0, 1.0},
        {2, 3, 40, 0.0, 1.0},
    };

    xjw::ObservationNetworkConfig config;
    config.algorithm = xjw::ObservationNetworkConfig::KDTree;
    config.k = 1;
    config.minMatches = 1;
    config.minOverlap = 0.0;
    config.pruneWeak = false;

    const xjw::ObservationNetwork result =
        xjw::ObservationNetworkBuilder::build(names, edges, {}, config);

    EXPECT_EQ(edgeKeys(result), (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}}));
}

TEST(ObservationNetworkBuilderTest, StrongConnectedCoreDropsRedundantWeakEdges)
{
    const std::vector<xjw::MatchEdge> edges{
        {0, 1, 120, 0.0, 1.0},
        {1, 2, 100, 0.0, 1.0},
        {2, 3, 80, 0.0, 1.0},
        {0, 2, 35, 0.0, 1.0},
        {0, 3, 25, 0.0, 1.0},
    };

    const std::vector<xjw::MatchEdge> result =
        xjw::ObservationNetworkBuilder::selectStrongConnectedCore(4, edges, 60);

    EXPECT_EQ(matchEdgeKeys(result),
              (std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 3}}));
}

TEST(ObservationNetworkBuilderTest, StrongCoreKeepsWeakEdgesWhenRequiredForConnectivity)
{
    const std::vector<xjw::MatchEdge> edges{
        {0, 1, 120, 0.0, 1.0},
        {2, 3, 100, 0.0, 1.0},
        {1, 2, 35, 0.0, 1.0},
    };

    const std::vector<xjw::MatchEdge> result =
        xjw::ObservationNetworkBuilder::selectStrongConnectedCore(4, edges, 60);

    EXPECT_EQ(matchEdgeKeys(result), matchEdgeKeys(edges));
}
