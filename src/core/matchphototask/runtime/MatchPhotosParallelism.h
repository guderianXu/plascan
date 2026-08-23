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

/**
 * @brief 解析 LoMa-R 静态 TensorRT 特征桶。
 *
 * 手动档位优先；自动模式只依据稳定的 GPU 总显存分档，避免特征阶段加载 engine
 * 后可用显存下降，导致匹配阶段误选另一个不兼容 bucket。
 */
int resolveLoMaRKeypointBudget(int requestedKeypoints,
                               int configuredBudget,
                               const MatchPhotosGpuMemoryInfo &memory);

// USAC 自身保持单线程和固定随机种子，在像对层使用一半逻辑 CPU 并行。
// 每个 worker 至少分配两个像对，避免小任务被线程启动开销吞噬。
int resolveGeometryVerificationWorkers(int pairCount,
                                       unsigned int hardwareThreads);

// SIFT 引导匹配的描述子搜索在单个像对内保持串行，在像对层使用一半逻辑 CPU 并行。
int resolveGuidedMatchingWorkers(int pairCount,
                                 unsigned int hardwareThreads);

bool isCudaOutOfMemoryError(const QString &message);

} // namespace matchphotos
} // namespace xjw
