#include <gtest/gtest.h>

#include "OverlapPairGraphPlanner.h"

#include <algorithm>
#include <vector>

namespace
{

xjw::OverlapPairGraphInputEdge edge(int indexA, int indexB, double score)
{
    xjw::OverlapPairGraphInputEdge result;
    result.indexA = indexA;
    result.indexB = indexB;
    result.bowScore = score;
    result.sharedWordCount = static_cast<int>(score * 100.0);
    return result;
}

const xjw::OverlapPairGraphEdge *findEdge(const std::vector<xjw::OverlapPairGraphEdge> &edges,
                                          int indexA,
                                          int indexB)
{
    const int a = std::min(indexA, indexB);
    const int b = std::max(indexA, indexB);
    for (const xjw::OverlapPairGraphEdge &candidate : edges)
    {
        if (candidate.indexA == a && candidate.indexB == b)
        {
            return &candidate;
        }
    }
    return nullptr;
}

bool hasSource(const xjw::OverlapPairGraphEdge &edge, xjw::OverlapPairGraphSource source)
{
    return std::find(edge.sources.begin(), edge.sources.end(), source) != edge.sources.end();
}

} // namespace

TEST(OverlapPairGraphPlannerTest, KeepsAtLeastMinPairsPerImageFromOneWayTopK)
{
    const std::vector<xjw::OverlapPairGraphInputEdge> input{
        edge(0, 1, 0.90),
        edge(0, 2, 0.80),
        edge(1, 2, 0.20),
    };

    xjw::OverlapPairGraphPlannerOptions options;
    options.imageCount = 3;
    options.topK = 1;
    options.minPairsPerImage = 1;
    options.mutualTopK = true;
    options.keepOneWayTopK = true;
    options.connectComponents = false;
    options.closeSequenceLoop = false;

    const xjw::OverlapPairGraphPlan plan = xjw::OverlapPairGraphPlanner::plan(input, options);

    ASSERT_NE(findEdge(plan.edges, 0, 1), nullptr);
    const xjw::OverlapPairGraphEdge *oneWay = findEdge(plan.edges, 0, 2);
    ASSERT_NE(oneWay, nullptr);
    EXPECT_TRUE(hasSource(*oneWay, xjw::OverlapPairGraphSource::BowOneWayTopK));
}

TEST(OverlapPairGraphPlannerTest, BridgesDisconnectedBowComponentsWithBestCrossComponentEdge)
{
    const std::vector<xjw::OverlapPairGraphInputEdge> input{
        edge(0, 1, 0.90),
        edge(2, 3, 0.90),
        edge(1, 2, 0.40),
        edge(0, 3, 0.20),
    };

    xjw::OverlapPairGraphPlannerOptions options;
    options.imageCount = 4;
    options.topK = 1;
    options.minPairsPerImage = 0;
    options.mutualTopK = true;
    options.keepOneWayTopK = false;
    options.connectComponents = true;
    options.useSequenceFallback = false;
    options.closeSequenceLoop = false;

    const xjw::OverlapPairGraphPlan plan = xjw::OverlapPairGraphPlanner::plan(input, options);

    EXPECT_EQ(plan.componentCountBeforeRepair, 2);
    EXPECT_EQ(plan.componentCountAfterRepair, 1);
    EXPECT_EQ(plan.bowBridgeCount, 1);

    const xjw::OverlapPairGraphEdge *bridge = findEdge(plan.edges, 1, 2);
    ASSERT_NE(bridge, nullptr);
    EXPECT_TRUE(hasSource(*bridge, xjw::OverlapPairGraphSource::BowComponentBridge));
}

TEST(OverlapPairGraphPlannerTest, FallsBackToSequenceBridgeWhenBowHasNoCrossComponentScore)
{
    const std::vector<xjw::OverlapPairGraphInputEdge> input{
        edge(0, 1, 0.90),
        edge(3, 4, 0.90),
    };

    xjw::OverlapPairGraphPlannerOptions options;
    options.imageCount = 5;
    options.topK = 1;
    options.minPairsPerImage = 0;
    options.mutualTopK = true;
    options.keepOneWayTopK = false;
    options.connectComponents = true;
    options.useSequenceFallback = true;
    options.sequenceWindow = 1;
    options.closeSequenceLoop = false;

    const xjw::OverlapPairGraphPlan plan = xjw::OverlapPairGraphPlanner::plan(input, options);

    EXPECT_EQ(plan.componentCountBeforeRepair, 3);
    EXPECT_EQ(plan.componentCountAfterRepair, 1);
    EXPECT_EQ(plan.sequenceBridgeCount, 2);

    const xjw::OverlapPairGraphEdge *leftBridge = findEdge(plan.edges, 1, 2);
    const xjw::OverlapPairGraphEdge *rightBridge = findEdge(plan.edges, 2, 3);
    ASSERT_NE(leftBridge, nullptr);
    ASSERT_NE(rightBridge, nullptr);
    EXPECT_TRUE(hasSource(*leftBridge, xjw::OverlapPairGraphSource::SequenceBridge));
    EXPECT_TRUE(hasSource(*rightBridge, xjw::OverlapPairGraphSource::SequenceBridge));
}

TEST(OverlapPairGraphPlannerTest, AddsSequenceWindowEvenWhenBowGraphIsAlreadyConnected)
{
    const std::vector<xjw::OverlapPairGraphInputEdge> input{
        edge(0, 2, 0.90),
        edge(2, 4, 0.88),
        edge(4, 1, 0.86),
        edge(1, 3, 0.84),
    };

    xjw::OverlapPairGraphPlannerOptions options;
    options.imageCount = 5;
    options.topK = 4;
    options.minPairsPerImage = 0;
    options.mutualTopK = false;
    options.keepOneWayTopK = true;
    options.connectComponents = true;
    options.useSequenceFallback = true;
    options.sequenceWindow = 1;
    options.closeSequenceLoop = false;

    const xjw::OverlapPairGraphPlan plan = xjw::OverlapPairGraphPlanner::plan(input, options);

    EXPECT_EQ(plan.componentCountBeforeRepair, 1);
    EXPECT_EQ(plan.componentCountAfterRepair, 1);

    const xjw::OverlapPairGraphEdge *sequenceNeighbor = findEdge(plan.edges, 0, 1);
    ASSERT_NE(sequenceNeighbor, nullptr);
    EXPECT_TRUE(hasSource(*sequenceNeighbor, xjw::OverlapPairGraphSource::SequenceBridge));
}

TEST(OverlapPairGraphPlannerTest, AddsRingClosureForSequenceLoop)
{
    xjw::OverlapPairGraphPlannerOptions options;
    options.imageCount = 5;
    options.connectComponents = false;
    options.useSequenceFallback = false;
    options.sequenceWindow = 1;
    options.closeSequenceLoop = true;

    const xjw::OverlapPairGraphPlan plan = xjw::OverlapPairGraphPlanner::plan({}, options);

    const xjw::OverlapPairGraphEdge *loop = findEdge(plan.edges, 0, 4);
    ASSERT_NE(loop, nullptr);
    EXPECT_TRUE(hasSource(*loop, xjw::OverlapPairGraphSource::SequenceLoop));
}
