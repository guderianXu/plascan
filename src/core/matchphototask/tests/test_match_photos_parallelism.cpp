#include "MatchPhotosParallelism.h"

#include <gtest/gtest.h>

namespace
{

constexpr std::uint64_t gib(std::uint64_t value)
{
    return value * 1024ULL * 1024ULL * 1024ULL;
}

} // namespace

TEST(MatchPhotosParallelismTest, CpuMatchingAlwaysUsesOneWorker)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = gib(24);
    memory.totalBytes = gib(24);

    const auto decision = xjw::matchphotos::resolveLightGlueParallelism(
        4, 120, false, 4096, memory);

    EXPECT_EQ(decision.requestedWorkers, 4);
    EXPECT_EQ(decision.effectiveWorkers, 1);
    EXPECT_EQ(decision.maxWorkersByMemory, 1);
    EXPECT_TRUE(decision.reason.contains(QStringLiteral("CPU")));
}

TEST(MatchPhotosParallelismTest, AutoModeUsesMemoryAwareBoundedConcurrency)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = gib(20);
    memory.totalBytes = gib(24);

    const auto decision = xjw::matchphotos::resolveLightGlueParallelism(
        0, 120, true, 4096, memory);

    EXPECT_TRUE(decision.autoSelected);
    EXPECT_EQ(decision.requestedWorkers, 0);
    EXPECT_EQ(decision.effectiveWorkers, 4);
    EXPECT_EQ(decision.maxWorkersByMemory, 4);
    EXPECT_GT(decision.estimatedBytesPerWorker, 0u);
}

TEST(MatchPhotosParallelismTest, ExplicitConcurrencyIsCappedByMemory)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = gib(4);
    memory.totalBytes = gib(8);

    const auto decision = xjw::matchphotos::resolveLightGlueParallelism(
        4, 120, true, 8192, memory);

    EXPECT_EQ(decision.effectiveWorkers, 1);
    EXPECT_EQ(decision.maxWorkersByMemory, 1);
    EXPECT_TRUE(decision.memoryLimited);
}

TEST(MatchPhotosParallelismTest, PairCountCapsWorkerCount)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;
    memory.freeBytes = gib(20);
    memory.totalBytes = gib(24);

    const auto decision = xjw::matchphotos::resolveLightGlueParallelism(
        4, 2, true, 4096, memory);

    EXPECT_EQ(decision.effectiveWorkers, 2);
}

TEST(MatchPhotosParallelismTest, MissingMemoryTelemetryFallsBackToSerial)
{
    const auto decision = xjw::matchphotos::resolveLightGlueParallelism(
        0, 120, true, 4096, {});

    EXPECT_EQ(decision.effectiveWorkers, 1);
    EXPECT_TRUE(decision.memoryLimited);
}

TEST(MatchPhotosParallelismTest, DetectsCudaOutOfMemoryDiagnostics)
{
    EXPECT_TRUE(xjw::matchphotos::isCudaOutOfMemoryError(
        QStringLiteral("CUDA out of memory. Tried to allocate 2.00 GiB")));
    EXPECT_TRUE(xjw::matchphotos::isCudaOutOfMemoryError(
        QStringLiteral("CUDNN_STATUS_ALLOC_FAILED")));
    EXPECT_FALSE(xjw::matchphotos::isCudaOutOfMemoryError(
        QStringLiteral("LightGlue 模型输出维度错误")));
}

TEST(MatchPhotosParallelismTest, SelectsLoMaRBucketFromStableTotalGpuMemory)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;

    memory.totalBytes = gib(6);
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(40000, 0, memory), 1024);
    memory.totalBytes = gib(10);
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(40000, 0, memory), 2048);
    memory.totalBytes = gib(16);
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(40000, 0, memory), 3840);
}

TEST(MatchPhotosParallelismTest, ManualLoMaRBucketOverridesAutomaticTier)
{
    xjw::matchphotos::MatchPhotosGpuMemoryInfo memory;
    memory.available = true;
    memory.totalBytes = gib(6);

    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(40000, 3840, memory), 3840);
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(1500, 3840, memory), 1500);
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(0, 2048, memory), 2048);
}

TEST(MatchPhotosParallelismTest, MissingTelemetryUsesConservativeLoMaRBucket)
{
    EXPECT_EQ(xjw::matchphotos::resolveLoMaRKeypointBudget(40000, 0, {}), 1024);
}

TEST(MatchPhotosParallelismTest, GeometryVerificationUsesBoundedCpuPairParallelism)
{
    EXPECT_EQ(
        xjw::matchphotos::resolveGeometryVerificationWorkers(120, 32),
        8);
    EXPECT_EQ(
        xjw::matchphotos::resolveGeometryVerificationWorkers(8, 32),
        4);
}

TEST(MatchPhotosParallelismTest, GeometryVerificationKeepsSmallOrUnknownHostsSerial)
{
    EXPECT_EQ(
        xjw::matchphotos::resolveGeometryVerificationWorkers(2, 32),
        1);
    EXPECT_EQ(
        xjw::matchphotos::resolveGeometryVerificationWorkers(120, 0),
        1);
    EXPECT_EQ(
        xjw::matchphotos::resolveGeometryVerificationWorkers(0, 32),
        1);
}
