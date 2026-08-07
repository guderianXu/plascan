#include <gtest/gtest.h>

#include "graph/CovisibilityPartitioner.h"
#include "pipeline/HierarchicalBundleAdjuster.h"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{

void addWeightedEdge(xjw::CorrespondenceGraph *graph,
                     xjw::ImageId first,
                     xjw::ImageId second,
                     std::size_t weight)
{
    std::vector<xjw::FeatureMatch> matches(weight);
    for (xjw::FeatureMatch &match : matches)
    {
        match.idx1 = 0;
        match.idx2 = 0;
        match.score = 1.0F;
    }
    graph->addMatches(first, second, matches);
}

struct SyntheticGraph
{
    std::vector<xjw::ImageId> imageIds;
    xjw::CorrespondenceGraph graph;
};

SyntheticGraph makeLocalGraph(int image_count, xjw::ImageId first_id, xjw::ImageId stride)
{
    SyntheticGraph result;
    result.imageIds.reserve(static_cast<std::size_t>(image_count));
    for (int index = 0; index < image_count; ++index)
    {
        const xjw::ImageId image_id = first_id + static_cast<xjw::ImageId>(index) * stride;
        result.imageIds.push_back(image_id);
        result.graph.addImage(image_id, 1);
    }
    for (int index = 0; index < image_count; ++index)
    {
        for (int offset = 1; offset <= 3 && index + offset < image_count; ++offset)
        {
            addWeightedEdge(&result.graph,
                            result.imageIds[static_cast<std::size_t>(index)],
                            result.imageIds[static_cast<std::size_t>(index + offset)],
                            static_cast<std::size_t>(120 - offset * 25));
        }
    }
    return result;
}

void expectValidPartition(const SyntheticGraph &input,
                          const std::vector<xjw::CovisibilityBlock> &blocks,
                          std::size_t maximum_core_size,
                          std::size_t maximum_overlap_size)
{
    std::unordered_map<xjw::ImageId, int> core_count;
    for (const xjw::CovisibilityBlock &block : blocks)
    {
        EXPECT_FALSE(block.coreImageIds.empty());
        EXPECT_LE(block.coreImageIds.size(), maximum_core_size);
        EXPECT_LE(block.overlapImageIds.size(), maximum_overlap_size);
        const std::unordered_set<xjw::ImageId> core(
            block.coreImageIds.begin(), block.coreImageIds.end());
        for (xjw::ImageId image_id : block.coreImageIds)
        {
            ++core_count[image_id];
        }
        for (xjw::ImageId overlap_id : block.overlapImageIds)
        {
            EXPECT_EQ(core.count(overlap_id), 0U);
            const bool connected_to_core = std::any_of(
                block.coreImageIds.begin(), block.coreImageIds.end(),
                [&](xjw::ImageId core_id)
                {
                    return input.graph.numMatchesBetween(core_id, overlap_id) > 0;
                });
            EXPECT_TRUE(connected_to_core);
        }
    }
    EXPECT_EQ(core_count.size(), input.imageIds.size());
    for (xjw::ImageId image_id : input.imageIds)
    {
        EXPECT_EQ(core_count[image_id], 1);
    }
}

} // namespace

TEST(CovisibilityPartitionerTest, Partitions444CameraNetworkIntoBalancedOverlappingBlocks)
{
    const SyntheticGraph input = makeLocalGraph(444, 5, 2);
    xjw::CovisibilityPartitionOptions options;
    options.targetCoreSize = 64;
    options.overlapSize = 12;

    const auto blocks = xjw::CovisibilityPartitioner::partition(
        input.imageIds, input.graph, options);

    ASSERT_EQ(blocks.size(), 7U);
    expectValidPartition(input, blocks, 64, 12);
}

TEST(CovisibilityPartitionerTest, IsDeterministicFor1000ArbitraryCameraIds)
{
    SyntheticGraph input = makeLocalGraph(1000, 7, 3);
    std::reverse(input.imageIds.begin(), input.imageIds.end());
    xjw::CovisibilityPartitionOptions options;
    options.targetCoreSize = 64;
    options.overlapSize = 10;

    const auto first = xjw::CovisibilityPartitioner::partition(
        input.imageIds, input.graph, options);
    const auto second = xjw::CovisibilityPartitioner::partition(
        input.imageIds, input.graph, options);

    ASSERT_EQ(first.size(), 16U);
    ASSERT_EQ(first.size(), second.size());
    expectValidPartition(input, first, 63, 10);
    for (std::size_t index = 0; index < first.size(); ++index)
    {
        EXPECT_EQ(first[index].coreImageIds, second[index].coreImageIds);
        EXPECT_EQ(first[index].overlapImageIds, second[index].overlapImageIds);
    }
}

TEST(CovisibilityPartitionerTest, NeverMixesDisconnectedComponentsInsideOneCore)
{
    SyntheticGraph first = makeLocalGraph(80, 1, 1);
    SyntheticGraph second = makeLocalGraph(40, 1001, 1);
    for (xjw::ImageId image_id : second.imageIds)
    {
        first.imageIds.push_back(image_id);
        first.graph.addImage(image_id, 1);
    }
    for (const xjw::ImagePair &pair : second.graph.imagePairs())
    {
        addWeightedEdge(&first.graph,
                        pair.first,
                        pair.second,
                        second.graph.numMatchesBetween(pair.first, pair.second));
    }

    xjw::CovisibilityPartitionOptions options;
    options.targetCoreSize = 32;
    options.overlapSize = 6;
    const auto blocks = xjw::CovisibilityPartitioner::partition(
        first.imageIds, first.graph, options);

    for (const xjw::CovisibilityBlock &block : blocks)
    {
        const bool low_component = block.coreImageIds.front() < 1000;
        EXPECT_TRUE(std::all_of(block.coreImageIds.begin(), block.coreImageIds.end(),
                                [low_component](xjw::ImageId image_id)
                                {
                                    return (image_id < 1000) == low_component;
                                }));
    }
}

TEST(HierarchicalBundleAdjusterPolicyTest, UsesProblemScaleWithoutSceneTypeAssumption)
{
    EXPECT_FALSE(xjw::HierarchicalBundleAdjuster::shouldRun(true, 127, 128, true));
    EXPECT_TRUE(xjw::HierarchicalBundleAdjuster::shouldRun(true, 128, 128, true));
    EXPECT_FALSE(xjw::HierarchicalBundleAdjuster::shouldRun(false, 1000, 128, true));
    EXPECT_FALSE(xjw::HierarchicalBundleAdjuster::shouldRun(true, 1000, 128, false));
}

TEST(HierarchicalBundleAdjusterPolicyTest, AllocatesWorkersFromActualThreadBudget)
{
    EXPECT_EQ(xjw::HierarchicalBundleAdjuster::resolveWorkerCount(7, 32, 0, true), 7);
    EXPECT_EQ(xjw::HierarchicalBundleAdjuster::resolveWorkerCount(20, 8, 0, true), 8);
    EXPECT_EQ(xjw::HierarchicalBundleAdjuster::resolveWorkerCount(20, 32, 4, true), 4);
    EXPECT_EQ(xjw::HierarchicalBundleAdjuster::resolveWorkerCount(20, 32, 0, false), 1);
    EXPECT_EQ(xjw::HierarchicalBundleAdjuster::resolveWorkerCount(0, 32, 0, true), 0);
}
