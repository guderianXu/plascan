#include "DepthMemoryPolicy.h"

#include <gtest/gtest.h>

#include <vector>

namespace
{

constexpr uint64_t gibibytes(uint64_t value)
{
    return value * 1024ull * 1024ull * 1024ull;
}

std::vector<xjw::mvs::DepthMemoryFrameSize> frames(
    int count, int width, int height)
{
    return std::vector<xjw::mvs::DepthMemoryFrameSize>(
        static_cast<size_t>(count), {width, height});
}

} // namespace

TEST(DepthMemoryPolicyTest, LargeOrbitalBatchStreamsOnNinetySixGiBMachine)
{
    const auto frameSizes = frames(113, 4608, 3456);
    const auto decision = xjw::mvs::decideDepthMemoryPolicy(
        frameSizes, 8, true, false, gibibytes(96), gibibytes(90), 0.60f, gibibytes(2));

    EXPECT_FALSE(decision.retainAllFrames);
    EXPECT_GT(decision.estimate.peakBytes, decision.budgetBytes);
    EXPECT_GE(decision.reserveBytes, gibibytes(19));
}

TEST(DepthMemoryPolicyTest, SmallBatchCanRemainInMemory)
{
    const auto frameSizes = frames(12, 4000, 3000);
    const auto decision = xjw::mvs::decideDepthMemoryPolicy(
        frameSizes, 8, true, false, gibibytes(96), gibibytes(90), 0.60f, gibibytes(2));

    EXPECT_TRUE(decision.retainAllFrames);
    EXPECT_LT(decision.estimate.peakBytes, decision.budgetBytes);
}

TEST(DepthMemoryPolicyTest, IntermediatePyramidIncreasesPeakEstimate)
{
    const auto frameSizes = frames(20, 4096, 3072);
    const auto withoutPyramid = xjw::mvs::estimateDepthConsistencyMemory(
        frameSizes, 8, true, false);
    const auto withPyramid = xjw::mvs::estimateDepthConsistencyMemory(
        frameSizes, 8, true, true);

    EXPECT_GT(withPyramid.intermediatePyramidBytes, 0u);
    EXPECT_GT(withPyramid.peakBytes, withoutPyramid.peakBytes);
}

TEST(DepthMemoryPolicyTest, LowAvailableMemoryForcesStreaming)
{
    const auto frameSizes = frames(12, 4000, 3000);
    const auto decision = xjw::mvs::decideDepthMemoryPolicy(
        frameSizes, 8, true, false, gibibytes(128), gibibytes(18), 0.60f, gibibytes(2));

    EXPECT_FALSE(decision.retainAllFrames);
    EXPECT_EQ(decision.budgetBytes, 0u);
}

TEST(DepthMemoryPolicyTest, InvalidFrameDimensionsUseConservativeStreaming)
{
    const auto frameSizes = frames(8, 0, 0);
    const auto decision = xjw::mvs::decideDepthMemoryPolicy(
        frameSizes, 8, true, false, gibibytes(96), gibibytes(90), 0.60f, gibibytes(2));

    EXPECT_FALSE(decision.retainAllFrames);
    EXPECT_EQ(decision.estimate.totalPixels, 0u);
}
