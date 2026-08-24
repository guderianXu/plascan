#include "MatchPhotosParallelism.h"

#include <algorithm>
#include <limits>

#if defined(PLASCAN_HAS_CUDA_SIFT)
#include <cuda_runtime_api.h>
#endif

namespace xjw
{
    namespace matchphotos
    {
        namespace
        {

            constexpr int kMaximumCudaSiftWorkers = 8;
            constexpr int kMaximumLightGlueWorkers = 4;
            constexpr int kMaximumLoMaRWorkers = 3;
            constexpr int kMinimumGeometryPairsPerWorker = 2;
            constexpr int kLoMaRNormalKeypoints = 1024;
            constexpr int kLoMaRHighKeypoints = 2048;
            constexpr int kLoMaRHighestKeypoints = 3840;
            constexpr std::uint64_t kMebibyte = 1024ULL * 1024ULL;
            constexpr std::uint64_t kGibibyte = 1024ULL * kMebibyte;
            constexpr std::uint64_t kMemoryReserveDivisor = 10;

            std::uint64_t estimateLightGlueWorkerBytes(int keypointBudget)
            {
                const std::uint64_t count = static_cast<std::uint64_t>(std::max(1024, keypointBudget));
                const std::uint64_t quadraticBytes = count * count * 20ULL;
                return 512ULL * kMebibyte + quadraticBytes;
            }

            std::uint64_t estimateCudaSiftWorkerBytes(int keypointBudget)
            {
                // 当前 CUDA SIFT 匹配器使用 O(K) 的无相关矩阵实现。SiftPoint 在第三方
                // ABI 中约 576 字节；每个 worker 复用 query/train 两块设备缓冲区。
                constexpr std::uint64_t siftPointBytes = 576ULL;
                const std::uint64_t count = static_cast<std::uint64_t>(std::max(1024, keypointBudget));
                return 64ULL * kMebibyte + 2ULL * count * siftPointBytes;
            }

            std::uint64_t estimateLoMaRWorkerBytes(int keypointBudget, int descriptorDimension)
            {
                // LoMa-R matcher 显式产生 KxK score 张量，此外还持有两套关键点、描述子和
                // TensorRT activation/workspace。它与 LightGlue 的 engine 图不同，必须使用
                // 独立的保守模型，不能复用 LightGlue 的经验系数。
                const std::uint64_t count = static_cast<std::uint64_t>(std::max(1024, keypointBudget));
                const std::uint64_t dimension = static_cast<std::uint64_t>(std::max(1, descriptorDimension));
                const std::uint64_t scoreAndActivationBytes = count * count * 12ULL;
                const std::uint64_t descriptorBytes = 2ULL * count * dimension * sizeof(float);
                return 768ULL * kMebibyte + scoreAndActivationBytes + descriptorBytes;
            }

            std::uint64_t
            estimateWorkerBytes(GpuMatchingAlgorithm algorithm, int keypointBudget, int descriptorDimension)
            {
                switch (algorithm)
                {
                case GpuMatchingAlgorithm::CudaSift:
                    return estimateCudaSiftWorkerBytes(keypointBudget);
                case GpuMatchingAlgorithm::LightGlue:
                    return estimateLightGlueWorkerBytes(keypointBudget);
                case GpuMatchingAlgorithm::LoMaR:
                    return estimateLoMaRWorkerBytes(keypointBudget, descriptorDimension);
                }
                return estimateCudaSiftWorkerBytes(keypointBudget);
            }

            int maximumWorkers(GpuMatchingAlgorithm algorithm)
            {
                switch (algorithm)
                {
                case GpuMatchingAlgorithm::CudaSift:
                    return kMaximumCudaSiftWorkers;
                case GpuMatchingAlgorithm::LightGlue:
                    return kMaximumLightGlueWorkers;
                case GpuMatchingAlgorithm::LoMaR:
                    return kMaximumLoMaRWorkers;
                }
                return 1;
            }

            QString algorithmDisplayName(GpuMatchingAlgorithm algorithm)
            {
                switch (algorithm)
                {
                case GpuMatchingAlgorithm::CudaSift:
                    return QStringLiteral("CUDA SIFT");
                case GpuMatchingAlgorithm::LightGlue:
                    return QStringLiteral("LightGlue");
                case GpuMatchingAlgorithm::LoMaR:
                    return QStringLiteral("LoMa-R");
                }
                return QStringLiteral("GPU");
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

        GpuMatchingParallelismDecision resolveGpuMatchingParallelism(GpuMatchingAlgorithm algorithm,
                                                                     int requestedWorkers,
                                                                     int pairCount,
                                                                     bool useCuda,
                                                                     int keypointBudget,
                                                                     int descriptorDimension,
                                                                     const MatchPhotosGpuMemoryInfo& memory)
        {
            GpuMatchingParallelismDecision decision;
            decision.algorithm = algorithm;
            decision.requestedWorkers = std::max(0, requestedWorkers);
            decision.autoSelected = decision.requestedWorkers == 0;
            decision.estimatedBytesPerWorker = estimateWorkerBytes(algorithm, keypointBudget, descriptorDimension);

            const int boundedPairCount = std::max(1, pairCount);
            if (!useCuda)
            {
                decision.reason = QStringLiteral("CPU 匹配固定使用单 worker");
                return decision;
            }

            if (!memory.available || memory.freeBytes == 0 || memory.totalBytes == 0)
            {
                decision.memoryLimited = true;
                decision.reason = QStringLiteral("无法读取 CUDA 显存，保守使用单 worker");
                return decision;
            }

            const std::uint64_t reserveBytes = std::max(kGibibyte, memory.totalBytes / kMemoryReserveDivisor);
            const std::uint64_t usableBytes = memory.freeBytes > reserveBytes ? memory.freeBytes - reserveBytes : 0;
            const std::uint64_t workersByMemory =
                decision.estimatedBytesPerWorker > 0 ? usableBytes / decision.estimatedBytesPerWorker : 1;
            const int workerLimit = maximumWorkers(algorithm);
            decision.maxWorkersByMemory =
                std::clamp(static_cast<int>(std::min<std::uint64_t>(
                               workersByMemory, static_cast<std::uint64_t>(std::numeric_limits<int>::max()))),
                           1,
                           workerLimit);

            const int desiredWorkers = decision.autoSelected ? workerLimit : std::max(1, decision.requestedWorkers);
            decision.effectiveWorkers =
                std::max(1, std::min({desiredWorkers, decision.maxWorkersByMemory, boundedPairCount}));
            decision.memoryLimited = decision.effectiveWorkers < std::min(desiredWorkers, boundedPairCount);
            const QString displayName = algorithmDisplayName(algorithm);
            decision.reason = decision.memoryLimited ? QStringLiteral("%1 并发受独立显存模型限制").arg(displayName)
                                                     : QStringLiteral("%1 并发已按独立显存模型启用").arg(displayName);
            return decision;
        }

        int
        resolveLoMaRKeypointBudget(int requestedKeypoints, int configuredBudget, const MatchPhotosGpuMemoryInfo& memory)
        {
            int budget = configuredBudget;
            if (budget <= 0)
            {
                // LoMa-R 的 TensorRT 图是静态 K。这里使用总显存而不是当前空闲显存，
                // 防止特征 engine 加载后匹配阶段因为 freeBytes 变化而选择另一个档位。
                if (!memory.available || memory.totalBytes == 0 || memory.totalBytes < 8ULL * kGibibyte)
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

        int resolveGeometryVerificationWorkers(int pairCount, unsigned int hardwareThreads)
        {
            if (pairCount <= 0 || hardwareThreads == 0)
            {
                return 1;
            }

            const int workersByItems = pairCount / kMinimumGeometryPairsPerWorker;
            const int workersByCpu = std::max(1, static_cast<int>(hardwareThreads / 2));
            return std::max(1, std::min(workersByItems, workersByCpu));
        }

        int resolveGuidedMatchingWorkers(int pairCount, unsigned int hardwareThreads)
        {
            return resolveGeometryVerificationWorkers(pairCount, hardwareThreads);
        }

        bool isCudaOutOfMemoryError(const QString& message)
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
