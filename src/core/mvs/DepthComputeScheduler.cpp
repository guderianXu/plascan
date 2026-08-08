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

bool preferCudaOnlyForHighestQualityOrbital(bool automaticBackend,
                                            bool cudaAvailable,
                                            bool openClAvailable,
                                            bool orbitalScene,
                                            const std::string &qualityProfile)
{
    return automaticBackend && cudaAvailable && openClAvailable &&
        orbitalScene && qualityProfile == "highest";
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

std::vector<DepthComputeWorker> buildDepthComputeWorkerPool(
    const std::vector<DepthComputeWorker> &physicalWorkers,
    int cudaHostSlotCount,
    int openClHostSlotCount,
    std::size_t maximumWorkerCount)
{
    if (maximumWorkerCount == 0)
    {
        return {};
    }

    std::vector<DepthComputeWorker> workers;
    workers.reserve(std::min(maximumWorkerCount, physicalWorkers.size()));
    for (const DepthComputeWorker &worker : physicalWorkers)
    {
        if (workers.size() >= maximumWorkerCount)
        {
            return workers;
        }
        workers.push_back(worker);
    }

    const auto add_host_slots = [&](DepthComputeBackend backend, int requestedCount)
    {
        std::vector<DepthComputeWorker> candidates;
        for (const DepthComputeWorker &worker : physicalWorkers)
        {
            if (worker.backend == backend)
            {
                candidates.push_back(worker);
            }
        }
        if (candidates.empty())
        {
            return;
        }

        std::size_t current_count = static_cast<std::size_t>(std::count_if(
            workers.begin(), workers.end(), [backend](const DepthComputeWorker &worker)
            {
                return worker.backend == backend;
            }));
        const std::size_t maximum_pipeline_count = candidates.size() * 2;
        const std::size_t target_count = std::min(
            maximum_pipeline_count,
            std::max(
                current_count,
                static_cast<std::size_t>(std::max(0, requestedCount))));
        std::size_t candidate_index = 0;
        while (current_count < target_count && workers.size() < maximumWorkerCount)
        {
            workers.push_back(candidates[candidate_index % candidates.size()]);
            ++candidate_index;
            ++current_count;
        }
    };

    add_host_slots(DepthComputeBackend::Cuda, cudaHostSlotCount);
    add_host_slots(DepthComputeBackend::OpenCl, openClHostSlotCount);
    return workers;
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
