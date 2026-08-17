#include "DepthAuxiliaryBridgeSelector.h"

#include <gtest/gtest.h>

#include <vector>

namespace
{

using xjw::mesh::DepthAuxiliaryBridgeNode;
using xjw::mesh::DepthAuxiliaryBridgeSelector;

DepthAuxiliaryBridgeNode primary(
    int frame_index, int ref_index, std::vector<int> sources = {})
{
    DepthAuxiliaryBridgeNode node;
    node.frameIndex = frame_index;
    node.refIndex = ref_index;
    node.primary = true;
    node.geometrySourceIndices = std::move(sources);
    return node;
}

DepthAuxiliaryBridgeNode auxiliary(
    int frame_index,
    int ref_index,
    std::vector<int> sources,
    double sparse_error = 0.005)
{
    DepthAuxiliaryBridgeNode node;
    node.frameIndex = frame_index;
    node.refIndex = ref_index;
    node.geometrySourceIndices = std::move(sources);
    node.sparseAbsoluteDepthMedianLogError = sparse_error;
    node.validWithinMaskRatio = 0.98;
    node.consistencyRetentionRatio = 0.99;
    node.largestComponentRatio = 0.99;
    node.meanConfidence = 0.75;
    node.sourceViewCount = 4;
    node.qualityReasonCount = 1;
    node.trustedPixelCount = 100000;
    return node;
}

TEST(DepthAuxiliaryBridgeSelectorTest, ReturnsNoBridgeForConnectedPrimaryGraph)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10, {11}),
        primary(1, 11, {10})});

    EXPECT_EQ(result.primaryComponentCount, 1);
    EXPECT_TRUE(result.connected);
    EXPECT_FALSE(result.failClosed);
    EXPECT_TRUE(result.selectedAuxiliaryFrameIndices.empty());
}

TEST(DepthAuxiliaryBridgeSelectorTest, ConnectsMultipleComponentsThroughMetricMst)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10),
        primary(1, 20),
        primary(2, 30),
        auxiliary(3, 101, {10, 20}),
        auxiliary(4, 102, {20, 30}),
        auxiliary(5, 103, {10}),
        auxiliary(6, 104, {103, 30})});

    EXPECT_EQ(result.primaryComponentCount, 3);
    EXPECT_TRUE(result.connected);
    EXPECT_FALSE(result.failClosed);
    EXPECT_EQ(result.selectedAuxiliaryFrameIndices,
              (std::vector<int>{3, 4}));
    EXPECT_EQ(result.selectedAuxiliaryRefIndices,
              (std::vector<int>{101, 102}));
}

TEST(DepthAuxiliaryBridgeSelectorTest, PrefersLowerQualityCostForEqualHopBridge)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10),
        primary(1, 20),
        auxiliary(7, 101, {10, 20}, 0.015),
        auxiliary(3, 102, {10, 20}, 0.002)});

    ASSERT_TRUE(result.connected);
    EXPECT_EQ(result.selectedAuxiliaryFrameIndices,
              (std::vector<int>{3}));
}

TEST(DepthAuxiliaryBridgeSelectorTest, MinimizesBridgeFramesBeforeQualityCost)
{
    auto first_short = auxiliary(2, 101, {10, 102}, 0.019);
    auto second_short = auxiliary(3, 102, {101, 20}, 0.019);
    first_short.meanConfidence = 0.60;
    first_short.validWithinMaskRatio = 0.90;
    first_short.largestComponentRatio = 0.95;
    first_short.qualityReasonCount = 10;
    second_short.meanConfidence = 0.60;
    second_short.validWithinMaskRatio = 0.90;
    second_short.largestComponentRatio = 0.95;
    second_short.qualityReasonCount = 10;

    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10),
        primary(1, 20),
        first_short,
        second_short,
        auxiliary(4, 201, {10, 202}, 0.0),
        auxiliary(5, 202, {201, 203}, 0.0),
        auxiliary(6, 203, {202, 20}, 0.0)});

    ASSERT_TRUE(result.connected);
    EXPECT_EQ(result.selectedAuxiliaryFrameIndices,
              (std::vector<int>{2, 3}));
}

TEST(DepthAuxiliaryBridgeSelectorTest, ResolvesExactQualityTieByFrameIndex)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10),
        primary(1, 20),
        auxiliary(8, 101, {10, 20}),
        auxiliary(3, 102, {10, 20})});

    ASSERT_TRUE(result.connected);
    EXPECT_EQ(result.selectedAuxiliaryFrameIndices,
              (std::vector<int>{3}));
}

TEST(DepthAuxiliaryBridgeSelectorTest, AcceptsReverseOnlyPrimaryObservationEdge)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10),
        primary(1, 20, {101}),
        auxiliary(2, 101, {10})});

    ASSERT_TRUE(result.connected);
    EXPECT_EQ(result.selectedAuxiliaryFrameIndices,
              (std::vector<int>{2}));
}

TEST(DepthAuxiliaryBridgeSelectorTest, FailsClosedWhenOnlyBridgeFailsQuality)
{
    auto weak = auxiliary(2, 101, {10, 20});
    weak.trustedPixelCount = 100;
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10), primary(1, 20), weak});

    EXPECT_EQ(result.primaryComponentCount, 2);
    EXPECT_FALSE(result.connected);
    EXPECT_TRUE(result.failClosed);
    EXPECT_TRUE(result.selectedAuxiliaryFrameIndices.empty());
    EXPECT_TRUE(result.selectedAuxiliaryRefIndices.empty());
}

TEST(DepthAuxiliaryBridgeSelectorTest, FailsClosedOnDuplicateReferenceIdentity)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10), primary(1, 10)});

    EXPECT_FALSE(result.connected);
    EXPECT_TRUE(result.failClosed);
}

TEST(DepthAuxiliaryBridgeSelectorTest, FailsClosedOnDuplicateFrameIdentity)
{
    const auto result = DepthAuxiliaryBridgeSelector::select({
        primary(0, 10), primary(0, 20)});

    EXPECT_FALSE(result.connected);
    EXPECT_TRUE(result.failClosed);
}

} // namespace
