#pragma once

#include <QString>

#include <cstdint>

namespace xjw
{
namespace matchphotos
{

struct MatchPhotosGpuMemoryInfo
{
    bool available = false;
    std::uint64_t freeBytes = 0;
    std::uint64_t totalBytes = 0;
    int deviceIndex = -1;
};

struct LightGlueParallelismDecision
{
    int requestedWorkers = 0;
    int effectiveWorkers = 1;
    int maxWorkersByMemory = 1;
    std::uint64_t estimatedBytesPerWorker = 0;
    bool autoSelected = false;
    bool memoryLimited = false;
    QString reason;
};

MatchPhotosGpuMemoryInfo queryMatchPhotosGpuMemory(int deviceIndex);

LightGlueParallelismDecision resolveLightGlueParallelism(
    int requestedWorkers,
    int pairCount,
    bool useCuda,
    int keypointBudget,
    const MatchPhotosGpuMemoryInfo &memory);

// USAC 自身保持单线程和固定随机种子，在像对层做有界 CPU 并行。
// 每个 worker 至少分配两个像对，避免小任务被线程启动开销吞噬。
int resolveGeometryVerificationWorkers(int pairCount,
                                       unsigned int hardwareThreads);

bool isCudaOutOfMemoryError(const QString &message);

} // namespace matchphotos
} // namespace xjw
