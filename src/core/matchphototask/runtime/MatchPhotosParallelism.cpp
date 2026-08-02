#include "MatchPhotosParallelism.h"

#include <algorithm>
#include <limits>

#if defined(PLASCAN_HAS_CUDA_SIFT)
#  include <cuda_runtime_api.h>
#endif

namespace xjw
{
namespace matchphotos
{
namespace
{

constexpr int kMaximumLightGlueWorkers = 4;
constexpr int kMaximumGeometryWorkers = 8;
constexpr int kMinimumGeometryPairsPerWorker = 2;
constexpr int kLoMaRNormalKeypoints = 1024;
constexpr int kLoMaRHighKeypoints = 2048;
constexpr int kLoMaRHighestKeypoints = 3840;
constexpr std::uint64_t kMebibyte = 1024ULL * 1024ULL;
constexpr std::uint64_t kGibibyte = 1024ULL * kMebibyte;

std::uint64_t estimateLightGlueWorkerBytes(int keypointBudget)
{
    const std::uint64_t count = static_cast<std::uint64_t>(std::max(1024, keypointBudget));
    const std::uint64_t quadraticBytes = count * count * 20ULL;
    return 512ULL * kMebibyte + quadraticBytes;
}

} // namespace

MatchPhotosGpuMemoryInfo queryMatchPhotosGpuMemory(int deviceIndex)
{
    MatchPhotosGpuMemoryInfo memory;
    memory.deviceIndex = std::max(0, deviceIndex);

#if defined(PLASCAN_HAS_CUDA_SIFT)
    int previousDevice = 0;
    const bool restoreDevice = cudaGetDevice(&previousDevice) == cudaSuccess;
    if (cudaSetDevice(memory.deviceIndex) == cudaSuccess)
    {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess)
        {
            memory.freeBytes = static_cast<std::uint64_t>(freeBytes);
            memory.totalBytes = static_cast<std::uint64_t>(totalBytes);
            memory.available = freeBytes > 0 && totalBytes > 0;
        }
        if (restoreDevice && previousDevice != memory.deviceIndex)
        {
            cudaSetDevice(previousDevice);
        }
    }
#endif

    return memory;
}

LightGlueParallelismDecision resolveLightGlueParallelism(
    int requestedWorkers,
    int pairCount,
    bool useCuda,
    int keypointBudget,
    const MatchPhotosGpuMemoryInfo &memory)
{
    LightGlueParallelismDecision decision;
    decision.requestedWorkers = std::max(0, requestedWorkers);
    decision.autoSelected = decision.requestedWorkers == 0;
    decision.estimatedBytesPerWorker = estimateLightGlueWorkerBytes(keypointBudget);

    const int boundedPairCount = std::max(1, pairCount);
    if (!useCuda)
    {
        decision.reason = QStringLiteral("CPU LightGlue 固定使用单 worker");
        return decision;
    }

    if (!memory.available || memory.freeBytes == 0 || memory.totalBytes == 0)
    {
        decision.memoryLimited = true;
        decision.reason = QStringLiteral("无法读取 CUDA 显存，保守使用单 worker");
        return decision;
    }

    const std::uint64_t reserveBytes =
        std::max(kGibibyte, memory.totalBytes / 10ULL);
    const std::uint64_t usableBytes =
        memory.freeBytes > reserveBytes ? memory.freeBytes - reserveBytes : 0;
    const std::uint64_t workersByMemory =
        decision.estimatedBytesPerWorker > 0
            ? usableBytes / decision.estimatedBytesPerWorker
            : 1;
    decision.maxWorkersByMemory = std::clamp(
        static_cast<int>(std::min<std::uint64_t>(
            workersByMemory,
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()))),
        1,
        kMaximumLightGlueWorkers);

    const int desiredWorkers = decision.autoSelected
        ? kMaximumLightGlueWorkers
        : std::max(1, decision.requestedWorkers);
    decision.effectiveWorkers = std::max(
        1,
        std::min({desiredWorkers, decision.maxWorkersByMemory, boundedPairCount}));
    decision.memoryLimited = decision.effectiveWorkers < std::min(
        desiredWorkers,
        boundedPairCount);
    decision.reason = decision.memoryLimited
        ? QStringLiteral("LightGlue 并发受显存预算限制")
        : QStringLiteral("LightGlue 并发已按显存预算启用");
    return decision;
}

int resolveLoMaRKeypointBudget(int requestedKeypoints,
                               int configuredBudget,
                               const MatchPhotosGpuMemoryInfo &memory)
{
    int budget = configuredBudget;
    if (budget <= 0)
    {
        // LoMa-R 的 TensorRT 图是静态 K。这里使用总显存而不是当前空闲显存，
        // 防止特征 engine 加载后匹配阶段因为 freeBytes 变化而选择另一个档位。
        if (!memory.available || memory.totalBytes == 0 ||
            memory.totalBytes < 8ULL * kGibibyte)
        {
            budget = kLoMaRNormalKeypoints;
        }
        else if (memory.totalBytes < 12ULL * kGibibyte)
        {
            budget = kLoMaRHighKeypoints;
        }
        else
        {
            budget = kLoMaRHighestKeypoints;
        }
    }

    if (requestedKeypoints > 0)
    {
        budget = std::min(budget, requestedKeypoints);
    }
    return std::max(1, budget);
}

int resolveGeometryVerificationWorkers(int pairCount,
                                       unsigned int hardwareThreads)
{
    if (pairCount <= 0 || hardwareThreads == 0)
    {
        return 1;
    }

    const int workersByItems = pairCount / kMinimumGeometryPairsPerWorker;
    return std::max(
        1,
        std::min({
            workersByItems,
            static_cast<int>(hardwareThreads),
            kMaximumGeometryWorkers,
        }));
}

bool isCudaOutOfMemoryError(const QString &message)
{
    const QString normalized = message.trimmed().toLower();
    return normalized.contains(QStringLiteral("cuda out of memory")) ||
        normalized.contains(QStringLiteral("cuda_error_out_of_memory")) ||
        normalized.contains(QStringLiteral("cudnn_status_alloc_failed")) ||
        normalized.contains(QStringLiteral("cuda allocation failed")) ||
        normalized.contains(QStringLiteral("cuda alloc failed"));
}

} // namespace matchphotos
} // namespace xjw
