#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xjw
{
namespace mvs
{

enum class DepthComputeBackend
{
    Cpu,
    Cuda,
    OpenCl
};

const char *depthComputeBackendName(DepthComputeBackend backend);

/// Highest-quality orbital reconstruction is sensitive to backend-dependent
/// photometric numerics. When CUDA and OpenCL are both available in automatic
/// mode, keep every frame on CUDA so a faster integrated GPU cannot claim an
/// arbitrary difficult view and lower the batch quality.
bool preferCudaOnlyForHighestQualityOrbital(bool automaticBackend,
                                            bool cudaAvailable,
                                            bool openClAvailable,
                                            bool orbitalScene,
                                            const std::string &qualityProfile);

struct DepthComputeWorker
{
    DepthComputeBackend backend = DepthComputeBackend::Cpu;
    int deviceIndex = -1;

    std::string id() const;
};

/// Builds frame-level host workers while keeping one worker for every physical
/// accelerator before adding at most one preparation lane for each device.
std::vector<DepthComputeWorker> buildDepthComputeWorkerPool(
    const std::vector<DepthComputeWorker> &physicalWorkers,
    int cudaHostSlotCount,
    int openClHostSlotCount,
    std::size_t maximumWorkerCount);

struct DepthFrameTask
{
    int viewIndex = -1;
    float priority = 0.0f;
};

struct DepthComputeWorkerStats
{
    int completedTasks = 0;
    int failedTasks = 0;
    double elapsedMilliseconds = 0.0;
};

class DepthComputeScheduler
{
public:
    explicit DepthComputeScheduler(std::vector<DepthFrameTask> tasks);

    std::optional<int> takeNext(const DepthComputeWorker &worker);
    void complete(const DepthComputeWorker &worker,
                  std::chrono::duration<double, std::milli> elapsed,
                  bool success);

    std::size_t pendingTaskCount() const;
    std::unordered_map<std::string, DepthComputeWorkerStats> workerStats() const;

private:
    mutable std::mutex _mutex;
    std::deque<int> _pendingViewIndices;
    std::unordered_map<std::string, DepthComputeWorkerStats> _workerStats;
};

} // namespace mvs
} // namespace xjw
