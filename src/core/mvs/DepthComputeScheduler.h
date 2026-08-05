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
    OpenCl,
    Vulkan
};

const char *depthComputeBackendName(DepthComputeBackend backend);

struct DepthComputeWorker
{
    DepthComputeBackend backend = DepthComputeBackend::Cpu;
    int deviceIndex = -1;

    std::string id() const;
};

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
