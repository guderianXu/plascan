#include "DepthGeometrySourceEncoding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{

using xjw::mesh::DepthGeometrySourceEncoding;
using xjw::mesh::DepthGeometrySourceMask;

DepthGeometrySourceMask bitFor(
    const DepthGeometrySourceEncoding &encoding,
    int source_index)
{
    const int slot = encoding.slotForSourceIndex(source_index);
    return slot < 0
        ? DepthGeometrySourceMask{0}
        : DepthGeometrySourceMask::singleBit(
              static_cast<std::size_t>(slot));
}

} // namespace

TEST(DepthGeometrySourceEncodingTest, EncodesSourceIdsAboveFifteen)
{
    const std::vector<std::vector<int>> frames{
        {74, 75, 108, 109},
        {108, 109},
        {74, 108}
    };
    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build(frames);

    EXPECT_EQ(
        encoding.slotToSourceIndex(),
        std::vector<int>({108, 74, 109, 75}));
    EXPECT_EQ(encoding.statistics().validReferenceCount, 8U);
    EXPECT_EQ(encoding.statistics().distinctSourceCount, 4U);

    const DepthGeometrySourceMask mask = encoding.encodeLocalMask(
        frames.front(), std::uint16_t{0x000f});
    EXPECT_EQ(mask, bitFor(encoding, 74) |
                    bitFor(encoding, 75) |
                    bitFor(encoding, 108) |
                    bitFor(encoding, 109));
    EXPECT_EQ(mask, DepthGeometrySourceMask{0x0f});
}

TEST(DepthGeometrySourceEncodingTest, DetectsOnlyExactSourceOverlap)
{
    const std::vector<std::vector<int>> frames{
        {74, 108},
        {108, 109},
        {200, 201}
    };
    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build(frames);

    const DepthGeometrySourceMask first = encoding.encodeLocalMask(
        frames[0], std::uint16_t{0x0003});
    const DepthGeometrySourceMask overlapping = encoding.encodeLocalMask(
        frames[1], std::uint16_t{0x0003});
    const DepthGeometrySourceMask disjoint = encoding.encodeLocalMask(
        frames[2], std::uint16_t{0x0003});

    EXPECT_EQ(first & overlapping, bitFor(encoding, 108));
    EXPECT_EQ(first & disjoint, DepthGeometrySourceMask{0});
    EXPECT_EQ(overlapping & disjoint, DepthGeometrySourceMask{0});
}

TEST(DepthGeometrySourceEncodingTest, MaskSupportsAllFourWords)
{
    const DepthGeometrySourceMask bit_63 =
        DepthGeometrySourceMask::singleBit(63);
    const DepthGeometrySourceMask bit_64 =
        DepthGeometrySourceMask::singleBit(64);
    const DepthGeometrySourceMask bit_255 =
        DepthGeometrySourceMask{1} << 255U;
    const DepthGeometrySourceMask combined = bit_63 | bit_64 | bit_255;

    EXPECT_TRUE(bit_63.any());
    EXPECT_FALSE(bit_63.none());
    EXPECT_EQ(combined.count(), 3U);
    EXPECT_EQ(combined & bit_63, bit_63);
    EXPECT_EQ(combined & bit_64, bit_64);
    EXPECT_EQ(combined & bit_255, bit_255);
    EXPECT_TRUE((bit_63 & bit_64).none());
    EXPECT_NE(bit_63, bit_64);
    EXPECT_EQ(DepthGeometrySourceMask{}, DepthGeometrySourceMask{0});
    EXPECT_TRUE((DepthGeometrySourceMask{1} << 256U).none());
}

TEST(DepthGeometrySourceEncodingTest, CoversAllOneHundredEightyEightSources)
{
    std::vector<std::vector<int>> frames;
    for (int source_index = 16; source_index < 204; ++source_index)
    {
        frames.push_back({source_index});
    }

    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build(frames);

    ASSERT_EQ(encoding.slotToSourceIndex().size(), 188U);
    EXPECT_EQ(encoding.statistics().mappedSourceCount, 188U);
    EXPECT_EQ(encoding.statistics().unmappedSourceCount, 0U);
    EXPECT_EQ(encoding.slotForSourceIndex(16), 0);
    EXPECT_EQ(encoding.slotForSourceIndex(203), 187);
    for (int source_index = 16; source_index < 204; ++source_index)
    {
        EXPECT_TRUE(bitFor(encoding, source_index).any());
        EXPECT_EQ(bitFor(encoding, source_index).count(), 1U);
    }
}

TEST(DepthGeometrySourceEncodingTest, SelectsTopTwoHundredFiftySixByCountThenId)
{
    std::vector<std::vector<int>> frames;
    for (int source_index = 1000; source_index < 1262; ++source_index)
    {
        frames.push_back({source_index});
    }
    frames.push_back({1261});
    frames.push_back({1260});

    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build(frames);

    ASSERT_EQ(encoding.slotToSourceIndex().size(), 256U);
    EXPECT_EQ(encoding.slotToSourceIndex()[0], 1260);
    EXPECT_EQ(encoding.slotToSourceIndex()[1], 1261);
    for (int offset = 0; offset < 254; ++offset)
    {
        EXPECT_EQ(
            encoding.slotToSourceIndex()[static_cast<std::size_t>(offset + 2)],
            1000 + offset);
    }
    EXPECT_EQ(encoding.slotForSourceIndex(1254), -1);
    EXPECT_EQ(encoding.slotForSourceIndex(1259), -1);
    EXPECT_EQ(encoding.statistics().mappedSourceCount, 256U);
    EXPECT_EQ(encoding.statistics().unmappedSourceCount, 6U);
    EXPECT_EQ(encoding.statistics().mappedReferenceCount, 258U);
    EXPECT_EQ(encoding.statistics().unmappedReferenceCount, 6U);
    EXPECT_EQ(
        encoding.encodeLocalMask({1253}, std::uint16_t{0x0001}),
        DepthGeometrySourceMask::singleBit(255));

    const auto &counts = encoding.sourceReferenceCounts();
    ASSERT_EQ(counts.size(), 262U);
    EXPECT_EQ(counts[0].sourceIndex, 1260);
    EXPECT_EQ(counts[0].referenceCount, 2U);
    EXPECT_EQ(counts[0].mappedSlot, 0);
    EXPECT_EQ(counts[256].mappedSlot, -1);
}

TEST(DepthGeometrySourceEncodingTest, IsDeterministicAcrossInputOrdering)
{
    const std::vector<std::vector<int>> first_frames{
        {300, 100, 200},
        {400, 100},
        {300, 500},
        {600, 200}
    };
    std::vector<std::vector<int>> reordered_frames = first_frames;
    std::reverse(reordered_frames.begin(), reordered_frames.end());
    for (std::vector<int> &sources : reordered_frames)
    {
        std::reverse(sources.begin(), sources.end());
    }

    const DepthGeometrySourceEncoding first =
        DepthGeometrySourceEncoding::build(first_frames);
    const DepthGeometrySourceEncoding second =
        DepthGeometrySourceEncoding::build(reordered_frames);

    EXPECT_EQ(first.slotToSourceIndex(), second.slotToSourceIndex());
    ASSERT_EQ(
        first.sourceReferenceCounts().size(),
        second.sourceReferenceCounts().size());
    for (std::size_t index = 0;
         index < first.sourceReferenceCounts().size();
         ++index)
    {
        EXPECT_EQ(
            first.sourceReferenceCounts()[index].sourceIndex,
            second.sourceReferenceCounts()[index].sourceIndex);
        EXPECT_EQ(
            first.sourceReferenceCounts()[index].referenceCount,
            second.sourceReferenceCounts()[index].referenceCount);
        EXPECT_EQ(
            first.sourceReferenceCounts()[index].mappedSlot,
            second.sourceReferenceCounts()[index].mappedSlot);
    }
}

TEST(DepthGeometrySourceEncodingTest, UnmappedSourcesNeverCreateFalseSharing)
{
    std::vector<std::vector<int>> frames;
    for (int source_index = 0; source_index < 258; ++source_index)
    {
        frames.push_back({source_index});
    }
    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build(frames);

    ASSERT_EQ(encoding.slotForSourceIndex(256), -1);
    ASSERT_EQ(encoding.slotForSourceIndex(257), -1);
    const DepthGeometrySourceMask first_unmapped =
        encoding.encodeLocalMask({256}, std::uint16_t{0x0001});
    const DepthGeometrySourceMask second_unmapped =
        encoding.encodeLocalMask({257}, std::uint16_t{0x0001});
    const DepthGeometrySourceMask mapped_and_unmapped =
        encoding.encodeLocalMask({255, 256}, std::uint16_t{0x0003});

    EXPECT_TRUE(first_unmapped.none());
    EXPECT_TRUE(second_unmapped.none());
    EXPECT_TRUE((first_unmapped & second_unmapped).none());
    EXPECT_EQ(mapped_and_unmapped, bitFor(encoding, 255));

    const auto local_encoding = encoding.makeLocalEncoding({255, 256, 257});
    EXPECT_EQ(local_encoding.mappedLocalMask(0x0001), 0x0001);
    EXPECT_EQ(local_encoding.mappedLocalMask(0x0007), 0x0001);
    EXPECT_EQ(local_encoding.mappedLocalMask(0x0006), 0x0000);
    EXPECT_EQ(local_encoding.encode(0x0007), bitFor(encoding, 255));
}

TEST(DepthGeometrySourceEncodingTest, IgnoresNegativeSourcesConservatively)
{
    const DepthGeometrySourceEncoding encoding =
        DepthGeometrySourceEncoding::build({{-1, 74}, {-5, 75}});

    EXPECT_EQ(encoding.slotToSourceIndex(), std::vector<int>({74, 75}));
    EXPECT_EQ(encoding.statistics().validReferenceCount, 2U);
    EXPECT_EQ(encoding.statistics().ignoredNegativeReferenceCount, 2U);
    EXPECT_EQ(
        encoding.encodeLocalMask({-1, 74}, std::uint16_t{0x0003}),
        bitFor(encoding, 74));
}
