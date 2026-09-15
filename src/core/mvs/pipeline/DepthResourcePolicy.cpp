#include "MvsPipelineInternals.h"

namespace xjw::mvs::pipeline_detail
{
    using namespace pipeline_detail;
    using common::string_utils::asciiLowerCopy;

    double elapsedMs(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    double bytesToGiB(uint64_t bytes)
    {
        return static_cast<double>(bytes) / static_cast<double>(kBytesPerGiB);
    }

    SystemMemorySnapshot querySystemMemorySnapshot()
    {
        SystemMemorySnapshot snapshot;
#ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status))
        {
            snapshot.totalPhysicalBytes = static_cast<uint64_t>(status.ullTotalPhys);
            snapshot.availablePhysicalBytes = static_cast<uint64_t>(status.ullAvailPhys);
            snapshot.valid = snapshot.totalPhysicalBytes > 0 && snapshot.availablePhysicalBytes > 0;
        }
#elif defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string key;
        uint64_t valueKb = 0;
        std::string unit;
        uint64_t totalKb = 0;
        uint64_t availableKb = 0;
        while (meminfo >> key >> valueKb >> unit)
        {
            if (key == "MemTotal:")
            {
                totalKb = valueKb;
            }
            else if (key == "MemAvailable:")
            {
                availableKb = valueKb;
            }
        }
        if (totalKb > 0 && availableKb > 0)
        {
            snapshot.totalPhysicalBytes = totalKb * 1024ull;
            snapshot.availablePhysicalBytes = availableKb * 1024ull;
            snapshot.valid = true;
        }
#endif
        return snapshot;
    }

    uint64_t depthFramePixelStorageBytes(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            return 0;
        }

        const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
        return pixels * sizeof(float) * 2ull; // depth + confidence
    }

    std::vector<DepthMemoryFrameSize> depthMemoryFrameSizes(const std::vector<CameraView>& views)
    {
        std::vector<DepthMemoryFrameSize> frameSizes;
        frameSizes.reserve(views.size());
        for (const CameraView& view : views)
        {
            frameSizes.push_back({view.imageWidth, view.imageHeight});
        }
        return frameSizes;
    }

    std::vector<MvsImageMemoryFrame> imageMemoryFrames(const std::vector<CameraView>& views)
    {
        std::vector<MvsImageMemoryFrame> frames;
        frames.reserve(views.size());
        for (const CameraView& view : views)
        {
            frames.push_back(
                {view.imageWidth, view.imageHeight, !mvsImagePreparationRequiresDistinctPixels(view.camera)});
        }
        return frames;
    }

    DepthMemoryPolicyDecision evaluateDepthMemoryPolicy(const std::vector<CameraView>& views,
                                                        const DepthGenConfig& config,
                                                        MvsSceneProfile sceneProfile,
                                                        const SystemMemorySnapshot& snapshot)
    {
        const std::vector<DepthMemoryFrameSize> frameSizes = depthMemoryFrameSizes(views);
        return decideDepthMemoryPolicy(frameSizes,
                                       maximumConsistencySourceViews(config),
                                       usesAdaptiveGeometryEvidence(config, sceneProfile),
                                       config.saveIntermediatePyramidLevels,
                                       snapshot.valid ? snapshot.totalPhysicalBytes : 0,
                                       snapshot.valid ? snapshot.availablePhysicalBytes : 0,
                                       config.maxDepthCacheRamFraction,
                                       config.minFreeRamBytes);
    }

    uint64_t largestDepthFrameBytes(const std::vector<CameraView>& views)
    {
        uint64_t largest = 0;
        for (const CameraView& view : views)
        {
            largest = std::max(largest, depthFramePixelStorageBytes(view.imageWidth, view.imageHeight));
        }
        return largest;
    }

    uint64_t saturatingMultiplyBytes(uint64_t value, uint64_t factor)
    {
        if (value != 0 && factor > std::numeric_limits<uint64_t>::max() / value)
        {
            return std::numeric_limits<uint64_t>::max();
        }
        return value * factor;
    }

    uint64_t retainedDepthMemoryBudgetBytes(const SystemMemorySnapshot& snapshot,
                                            const DepthGenConfig& config,
                                            uint64_t largestFrameBytes,
                                            uint64_t transientFrameBytes,
                                            size_t concurrentFrameWorkers)
    {
        if (!snapshot.valid)
        {
            return 0;
        }

        const uint64_t producerWorkingSetReserve = saturatingMultiplyBytes(largestFrameBytes, 8);
        return calculateDepthSaveQueueBudgetBytes(snapshot.totalPhysicalBytes,
                                                  snapshot.availablePhysicalBytes,
                                                  config.maxDepthCacheRamFraction,
                                                  config.minFreeRamBytes,
                                                  transientFrameBytes,
                                                  concurrentFrameWorkers,
                                                  producerWorkingSetReserve);
    }

    QString depthMemoryPolicyReason(const DepthMemoryPolicyDecision& decision, const SystemMemorySnapshot& snapshot)
    {
        if (decision.estimate.totalPixels == 0)
        {
            return QStringLiteral("无有效影像尺寸，采用保守流式模式");
        }
        if (!snapshot.valid)
        {
            return QStringLiteral("无法读取系统内存，采用保守流式模式");
        }
        return QStringLiteral("预计峰值 %1 GiB %2 内存预算 %3 GiB（常驻=%4，快照=%5，证据=%6，"
                              "金字塔=%7，单帧临时=%8，动态保留=%9 GiB）")
            .arg(bytesToGiB(decision.estimate.peakBytes), 0, 'f', 2)
            .arg(decision.retainAllFrames ? QStringLiteral("<=") : QStringLiteral(">"))
            .arg(bytesToGiB(decision.budgetBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.estimate.residentFrameBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.estimate.consistencySnapshotBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.estimate.retainedEvidenceBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.estimate.intermediatePyramidBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.estimate.transientFrameBytes), 0, 'f', 2)
            .arg(bytesToGiB(decision.reserveBytes), 0, 'f', 2);
    }

    bool memoryPressureRequiresStreaming(const DepthGenConfig& config,
                                         const SystemMemorySnapshot& snapshot,
                                         const DepthMemoryPolicyDecision& decision)
    {
        if (!config.adaptiveDepthCacheMemory || !snapshot.valid)
        {
            return false;
        }

        const uint64_t runtimeReserve =
            std::max(decision.reserveBytes, saturatingMultiplyBytes(decision.estimate.transientFrameBytes, 2));
        if (snapshot.availablePhysicalBytes <= runtimeReserve)
        {
            return true;
        }

        const uint64_t availableBudget = snapshot.availablePhysicalBytes - runtimeReserve;
        const uint64_t totalBudget =
            static_cast<uint64_t>(static_cast<double>(snapshot.totalPhysicalBytes) *
                                  std::clamp(static_cast<double>(config.maxDepthCacheRamFraction), 0.10, 0.90));
        return decision.estimate.peakBytes > std::min(totalBudget, availableBudget);
    }

    size_t adaptiveSaveQueueCapacity(const SystemMemorySnapshot& snapshot,
                                     const DepthGenConfig& config,
                                     uint64_t largestFrameBytes,
                                     uint64_t transientFrameBytes,
                                     size_t concurrentFrameWorkers)
    {
        if (!config.adaptiveDepthCacheMemory || !snapshot.valid || largestFrameBytes == 0)
        {
            return 2;
        }

        const uint64_t budgetBytes = retainedDepthMemoryBudgetBytes(
            snapshot, config, largestFrameBytes, transientFrameBytes, concurrentFrameWorkers);
        if (budgetBytes >= saturatingMultiplyBytes(largestFrameBytes, 8))
        {
            return 4;
        }
        return 2;
    }

    uint64_t estimatedSaveQueueProducerBytes(uint64_t largestFrameBytes)
    {
        constexpr uint64_t kFallbackResidentBytes = 512ull * 1024ull * 1024ull;
        constexpr uint64_t kEstimatedResultToDepthConfidenceRatio = 8;

        if (largestFrameBytes == 0)
        {
            return kFallbackResidentBytes;
        }
        return saturatingMultiplyBytes(largestFrameBytes, kEstimatedResultToDepthConfidenceRatio);
    }

    uint64_t adaptiveSaveQueueResidentByteCapacity(const SystemMemorySnapshot& snapshot,
                                                   const DepthGenConfig& config,
                                                   uint64_t largestFrameBytes,
                                                   size_t maxResidentTasks,
                                                   uint64_t transientFrameBytes,
                                                   size_t concurrentFrameWorkers)
    {
        constexpr uint64_t kFallbackMaximumResidentBytes = 4ull * 1024ull * 1024ull * 1024ull;

        const uint64_t estimatedTaskBytes = estimatedSaveQueueProducerBytes(largestFrameBytes);

        const uint64_t residentTaskCount = static_cast<uint64_t>(std::max<size_t>(1, maxResidentTasks));
        uint64_t byteCapacity = saturatingMultiplyBytes(estimatedTaskBytes, residentTaskCount);

        const uint64_t memoryBudget = retainedDepthMemoryBudgetBytes(
            snapshot, config, largestFrameBytes, transientFrameBytes, concurrentFrameWorkers);
        if (snapshot.valid)
        {
            byteCapacity = std::min(byteCapacity, memoryBudget);
        }
        else
        {
            // Keep the historical conservative cap only when no trustworthy
            // system-memory snapshot is available. On known-memory systems the
            // dynamic budget already reserves both OS headroom and transient MVS
            // working sets; a second fixed 4 GiB cap can otherwise collapse a
            // multi-GPU preparation pipeline to one frame per device.
            byteCapacity = std::min(byteCapacity, kFallbackMaximumResidentBytes);
        }
        return byteCapacity;
    }

    int preloadImagesWorkerCount(int viewCount, int requestedThreads)
    {
        if (viewCount <= 1)
        {
            return std::max(0, viewCount);
        }

        const int hwThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int requested = std::max(1, requestedThreads);
        return std::clamp(std::min(requested, hwThreads), 1, std::min(viewCount, 8));
    }

    int resolvedTotalCpuThreadBudget(const DepthGenConfig& config)
    {
        const int hardware_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const int active_frame_workers = std::max(1, config.gpuFrameWorkerCount + config.cpuFrameWorkerCount);
        const int derived_budget = std::max(1, config.cpuWorkerCount * active_frame_workers);
        const int requested_budget = config.totalCpuThreadBudget > 0 ? config.totalCpuThreadBudget : derived_budget;
        return std::clamp(requested_budget, 1, hardware_threads);
    }

    bool isCudaMemoryFailure(const std::string& message)
    {
        const std::string lower = asciiLowerCopy(message);

        return lower.find("out of memory") != std::string::npos ||
               lower.find("cuda_error_memory") != std::string::npos ||
               (lower.find("cuda") != std::string::npos && lower.find("memory") != std::string::npos);
    }
} // namespace xjw::mvs::pipeline_detail
