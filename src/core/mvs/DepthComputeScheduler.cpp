#include "DepthComputeScheduler.h"

#include <algorithm>

namespace xjw
{
namespace mvs
{

const char *depthComputeBackendName(DepthComputeBackend backend)
{
    switch (backend)
    {
    case DepthComputeBackend::Cpu:
        return "CPU";
    case DepthComputeBackend::Cuda:
        return "CUDA";
    case DepthComputeBackend::OpenCl:
        return "OpenCL";
    }
    return "Unknown";
}

std::string DepthComputeWorker::id() const
{
    std::string result = depthComputeBackendName(backend);
    if (deviceIndex >= 0)
    {
        result += ":" + std::to_string(deviceIndex);
    }
    return result;
}

DepthComputeScheduler::DepthComputeScheduler(std::vector<DepthFrameTask> tasks)
{
    std::stable_sort(tasks.begin(), tasks.end(), [](const DepthFrameTask &left,
                                                     const DepthFrameTask &right)
    {
        return left.priority > right.priority;
    });
    for (const DepthFrameTask &task : tasks)
    {
        if (task.viewIndex >= 0)
        {
            _pendingViewIndices.push_back(task.viewIndex);
        }
    }
}

std::optional<int> DepthComputeScheduler::takeNext(const DepthComputeWorker &worker)
{
    std::lock_guard<std::mutex> lock(_mutex);
    _workerStats.try_emplace(worker.id());
    if (_pendingViewIndices.empty())
    {
        return std::nullopt;
    }
    const int view_index = _pendingViewIndices.front();
    _pendingViewIndices.pop_front();
    return view_index;
}

void DepthComputeScheduler::complete(
    const DepthComputeWorker &worker,
    std::chrono::duration<double, std::milli> elapsed,
    bool success)
{
    std::lock_guard<std::mutex> lock(_mutex);
    DepthComputeWorkerStats &stats = _workerStats[worker.id()];
    ++stats.completedTasks;
    stats.failedTasks += success ? 0 : 1;
    stats.elapsedMilliseconds += elapsed.count();
}

std::size_t DepthComputeScheduler::pendingTaskCount() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _pendingViewIndices.size();
}

std::unordered_map<std::string, DepthComputeWorkerStats>
DepthComputeScheduler::workerStats() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _workerStats;
}

} // namespace mvs
} // namespace xjw
