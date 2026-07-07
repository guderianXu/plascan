#pragma once

#include <string>
#include <vector>

namespace xjw
{

enum class OverlapPairGraphSource
{
    BowMutualTopK,
    BowOneWayTopK,
    BowComponentBridge,
    SequenceBridge,
    SequenceLoop
};

struct OverlapPairGraphInputEdge
{
    int indexA = -1;
    int indexB = -1;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
};

struct OverlapPairGraphPlannerOptions
{
    int imageCount = 0;
    int topK = 8;
    int minPairsPerImage = 4;
    double minSimilarity = 0.05;
    bool mutualTopK = true;
    bool keepOneWayTopK = true;
    bool connectComponents = true;
    bool useSequenceFallback = true;
    int sequenceWindow = 1;
    bool closeSequenceLoop = true;
    int componentBridgeMaxPairs = 0;
};

struct OverlapPairGraphEdge
{
    int indexA = -1;
    int indexB = -1;
    double bowScore = 0.0;
    int sharedWordCount = 0;
    int geometricInliers = 0;
    std::vector<OverlapPairGraphSource> sources;
};

struct OverlapPairGraphPlan
{
    std::vector<OverlapPairGraphEdge> edges;
    int componentCountBeforeRepair = 0;
    int componentCountAfterRepair = 0;
    int bowBridgeCount = 0;
    int sequenceBridgeCount = 0;
    std::string detail;
};

const char *overlapPairGraphSourceId(OverlapPairGraphSource source);

class OverlapPairGraphPlanner
{
public:
    static OverlapPairGraphPlan plan(const std::vector<OverlapPairGraphInputEdge> &edges,
                                     const OverlapPairGraphPlannerOptions &options);
};

} // namespace xjw
