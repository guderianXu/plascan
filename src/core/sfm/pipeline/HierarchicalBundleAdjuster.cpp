#include "HierarchicalBundleAdjuster.h"

#include "HierarchicalBaBlockSolver.h"
#include "IncrementalSfm.h"
#include "graph/CovisibilityPartitioner.h"

#include "log/Logger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xjw
{

using hierarchical_ba_detail::BlockOutcome;
using hierarchical_ba_detail::solveBlock;

namespace
{

struct GlobalReprojectionState
{
    double rms = std::numeric_limits<double>::infinity();
    std::size_t observations = 0;
};

GlobalReprojectionState evaluateGlobalReprojection(
    const SfmReconstruction &reconstruction)
{
    GlobalReprojectionState state;
    long double squared_error_sum = 0.0L;
    for (Point3DId point_id : reconstruction.allPoint3DIds())
    {
        if (!reconstruction.hasPoint3D(point_id))
        {
            continue;
        }
        const ScenePoint3D &point = reconstruction.point3D(point_id);
        const double world[3] = {point.xyz[0], point.xyz[1], point.xyz[2]};
        for (const TrackElement &element : point.track.elements)
        {
            if (!reconstruction.isRegistered(element.imageId) ||
                !reconstruction.hasCamera(element.imageId) ||
                !reconstruction.hasImage(element.imageId))
            {
                continue;
            }
            const ImageData &image = reconstruction.image(element.imageId);
            if (element.featureIdx >= image.keypoints.size())
            {
                continue;
            }
            double projected[2] = {0.0, 0.0};
            if (!reconstruction.camera(element.imageId).projectWorldPoint(
                    world, projected))
            {
                continue;
            }
            const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
            const double dx = projected[0] - keypoint.x;
            const double dy = projected[1] - keypoint.y;
            const double squared_error = dx * dx + dy * dy;
            if (!std::isfinite(squared_error))
            {
                continue;
            }
            squared_error_sum += static_cast<long double>(squared_error);
            ++state.observations;
        }
    }
    if (state.observations > 0)
    {
        state.rms = std::sqrt(
            static_cast<double>(squared_error_sum /
                                static_cast<long double>(state.observations)));
    }
    return state;
}

struct PointSnapshot
{
    Point3DId id = kInvalidPoint3DId;
    std::array<double, 3> xyz{{0.0, 0.0, 0.0}};
    double error = 0.0;
};

} // namespace

HierarchicalBundleAdjuster::HierarchicalBundleAdjuster(IncrementalSfm &owner)
    : _owner(owner)
{
}

bool HierarchicalBundleAdjuster::shouldRun(bool enabled,
                                           int registeredImageCount,
                                           int minimumImageCount,
                                           bool refineCameraPose)
{
    return enabled && refineCameraPose && minimumImageCount >= 2 &&
        registeredImageCount >= minimumImageCount;
}

int HierarchicalBundleAdjuster::resolveWorkerCount(int blockCount,
                                                   int totalThreadCount,
                                                   int configuredMaximum,
                                                   bool concurrentBackendAvailable)
{
    if (blockCount <= 0)
    {
        return 0;
    }
    if (!concurrentBackendAvailable)
    {
        return 1;
    }
    const int total_threads = std::max(1, totalThreadCount);
    const int configured_limit = configuredMaximum > 0
        ? configuredMaximum
        : total_threads;
    return std::max(1, std::min({blockCount, total_threads, configured_limit}));
}

int HierarchicalBundleAdjuster::resolveWorkerThreadCount(int totalThreadCount,
                                                         int activeWorkerCount,
                                                         int workerIndex)
{
    const int total_threads = std::max(1, totalThreadCount);
    const int active_workers = std::clamp(
        activeWorkerCount, 1, total_threads);
    const int normalized_index = std::clamp(
        workerIndex, 0, active_workers - 1);
    const int base_threads = total_threads / active_workers;
    const int extra_threads = total_threads % active_workers;
    return base_threads + (normalized_index < extra_threads ? 1 : 0);
}

bool HierarchicalBundleAdjuster::shouldWriteBackPoint(
    std::size_t blockObservationCount,
    std::size_t totalRegisteredObservationCount)
{
    return totalRegisteredObservationCount >= 2 &&
           blockObservationCount == totalRegisteredObservationCount;
}

bool HierarchicalBundleAdjuster::isGlobalWriteBackConsistent(
    double rmsBefore,
    std::size_t observationsBefore,
    double rmsAfter,
    std::size_t observationsAfter)
{
    if (!std::isfinite(rmsBefore) || !std::isfinite(rmsAfter) ||
        observationsBefore == 0 || observationsAfter == 0)
    {
        return false;
    }
    const double coverage_ratio = static_cast<double>(observationsAfter) /
        static_cast<double>(observationsBefore);
    if (!std::isfinite(coverage_ratio) || coverage_ratio < 0.95)
    {
        return false;
    }
    const double allowed_growth = std::max(0.25, std::abs(rmsBefore) * 0.25);
    return rmsAfter <= rmsBefore + allowed_growth;
}

HierarchicalBaRunSummary HierarchicalBundleAdjuster::run()
{
    HierarchicalBaRunSummary summary;
    const int registered_count = static_cast<int>(_owner._reconstruction->numRegisteredImages());
    if (!shouldRun(_owner._sfmOptions.enableHierarchicalBA,
                   registered_count,
                   _owner._sfmOptions.hierarchicalBAMinImages,
                   _owner._sfmOptions.baOptions.refineCameraPose))
    {
        return summary;
    }
    if (_owner._sfmOptions.useKnownCameraPoses || _owner._controlNetworkApplied ||
        !_owner._pendingPriorTracks.empty() || !_owner._pendingPriorScaleBars.empty())
    {
        Logger::instance()->info(
            "[BA] hierarchical skipped reason=absolute_or_manual_constraints_present");
        return summary;
    }

    const int repeat_threshold = std::max(
        2, _owner._sfmOptions.hierarchicalBATargetBlockSize / 2);
    const int previous_attempt_count = std::max(
        _owner._lastHierarchicalBAImageCount,
        _owner._lastHierarchicalBAAttemptImageCount);
    if (previous_attempt_count > 0 &&
        registered_count - previous_attempt_count < repeat_threshold)
    {
        Logger::instance()->infof(
            "[BA] hierarchical skipped reason=model_growth_too_small registered=%d previous=%d threshold=%d",
            registered_count,
            previous_attempt_count,
            repeat_threshold);
        return summary;
    }

    CovisibilityPartitionOptions partition_options;
    partition_options.targetCoreSize = static_cast<std::size_t>(
        std::max(2, _owner._sfmOptions.hierarchicalBATargetBlockSize));
    partition_options.overlapSize = static_cast<std::size_t>(
        std::max(2, _owner._sfmOptions.hierarchicalBAOverlapImages));
    const std::vector<CovisibilityBlock> blocks = CovisibilityPartitioner::partition(
        _owner._reconstruction->registeredImageIds(),
        _owner._correspondenceGraph,
        partition_options);
    if (blocks.size() < 2)
    {
        return summary;
    }

    summary.attempted = true;
    _owner._lastHierarchicalBAAttemptImageCount = registered_count;
    summary.plannedBlocks = static_cast<int>(blocks.size());
    const auto started = std::chrono::steady_clock::now();
    const unsigned int hardware_threads = std::max(1u, std::thread::hardware_concurrency());
    const int total_threads = _owner._sfmOptions.baOptions.numThreads > 0
        ? _owner._sfmOptions.baOptions.numThreads
        : static_cast<int>(hardware_threads);
    const bool ceres_available = BundleAdjust::isBackendAvailable(BABackend::CeresCpu);
    const int worker_count = resolveWorkerCount(
        static_cast<int>(blocks.size()),
        total_threads,
        _owner._sfmOptions.hierarchicalBAMaxConcurrentBlocks,
        ceres_available);
    const int minimum_threads_per_block = resolveWorkerThreadCount(
        total_threads, worker_count, worker_count - 1);
    const int maximum_threads_per_block = resolveWorkerThreadCount(
        total_threads, worker_count, 0);

    BAOptions block_options = _owner._sfmOptions.baOptions;
    block_options.maxIterations = std::clamp(
        _owner._sfmOptions.hierarchicalBAMaxIterations,
        1,
        std::max(1, _owner._sfmOptions.baOptions.maxIterations));
    Logger::instance()->infof(
        "[BA] hierarchical start cameras=%d blocks=%zu targetCore=%zu overlap=%zu "
        "workers=%d threadsPerBlock=%d-%d backend=%s iterations=%d",
        registered_count,
        blocks.size(),
        partition_options.targetCoreSize,
        partition_options.overlapSize,
        worker_count,
        minimum_threads_per_block,
        maximum_threads_per_block,
        ceres_available ? "ceres_cpu" : BundleAdjust::backendName(block_options.backend),
        block_options.maxIterations);

    std::unordered_map<ImageId, std::size_t> core_owner;
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index)
    {
        for (ImageId image_id : blocks[block_index].coreImageIds)
        {
            core_owner[image_id] = block_index;
        }
    }
    std::unordered_map<Point3DId, std::size_t> point_owner;
    for (Point3DId point_id : _owner._reconstruction->allPoint3DIds())
    {
        const ScenePoint3D &point = _owner._reconstruction->point3D(point_id);
        std::unordered_map<std::size_t, int> observations_per_block;
        for (const TrackElement &element : point.track.elements)
        {
            const auto owner = core_owner.find(element.imageId);
            if (owner != core_owner.end())
            {
                ++observations_per_block[owner->second];
            }
        }
        std::size_t best_block = 0;
        int best_count = 0;
        for (const auto &[block_index, count] : observations_per_block)
        {
            if (count > best_count || (count == best_count && block_index < best_block))
            {
                best_block = block_index;
                best_count = count;
            }
        }
        if (best_count > 0)
        {
            point_owner[point_id] = best_block;
        }
    }

    std::vector<BlockOutcome> outcomes;
    outcomes.reserve(blocks.size());
    bool continue_processing = true;
    for (std::size_t first = 0; first < blocks.size(); first += worker_count)
    {
        const std::size_t end = std::min(blocks.size(), first + static_cast<std::size_t>(worker_count));
        const int active_worker_count = static_cast<int>(end - first);
        std::vector<std::future<BlockOutcome>> futures;
        futures.reserve(end - first);
        for (std::size_t block_index = first; block_index < end; ++block_index)
        {
            const int worker_index = static_cast<int>(block_index - first);
            const int block_thread_count = resolveWorkerThreadCount(
                total_threads, active_worker_count, worker_index);
            futures.push_back(std::async(
                std::launch::async,
                [&, block_index, block_thread_count]()
                {
                    return solveBlock(block_index,
                                      blocks[block_index],
                                      *_owner._reconstruction,
                                      block_options,
                                      block_thread_count,
                                      ceres_available);
                }));
        }
        for (std::future<BlockOutcome> &future : futures)
        {
            try
            {
                outcomes.push_back(future.get());
            }
            catch (const std::exception &error)
            {
                Logger::instance()->warnf(
                    "[BA] hierarchical block failed exception=%s", error.what());
            }
        }
        Logger::instance()->infof(
            "[BA] hierarchical progress completed=%zu/%zu",
            end,
            blocks.size());
        if (block_options.progressCallback)
        {
            double rms_sum = 0.0;
            int rms_count = 0;
            int valid_points = 0;
            for (const BlockOutcome &outcome : outcomes)
            {
                if (outcome.accepted)
                {
                    rms_sum += outcome.result.meanRmsAfter;
                    ++rms_count;
                    valid_points += outcome.result.optimizedTracks;
                }
            }
            continue_processing = block_options.progressCallback(
                static_cast<int>(end),
                static_cast<int>(blocks.size()),
                rms_count > 0 ? rms_sum / static_cast<double>(rms_count) : 0.0,
                valid_points);
            if (!continue_processing)
            {
                if (block_options.cancelFlag)
                {
                    block_options.cancelFlag->store(true);
                }
                break;
            }
        }
    }

    std::sort(outcomes.begin(), outcomes.end(), [](const BlockOutcome &left, const BlockOutcome &right)
    {
        return left.blockIndex < right.blockIndex;
    });
    const GlobalReprojectionState global_before =
        evaluateGlobalReprojection(*_owner._reconstruction);
    std::unordered_map<ImageId, Camera> camera_snapshots;
    std::vector<PointSnapshot> point_snapshots;
    int candidate_applied_blocks = 0;
    int candidate_updated_cameras = 0;
    int candidate_updated_points = 0;
    for (const BlockOutcome &outcome : outcomes)
    {
        if (!outcome.accepted)
        {
            Logger::instance()->warnf(
                "[BA] hierarchical block rejected block=%zu status=%s rms=%.6f->%.6f "
                "fixedBoundaryTracks=%d",
                outcome.blockIndex,
                BundleAdjust::solveStatusName(outcome.result.solveStatus),
                outcome.result.meanRmsBefore,
                outcome.result.meanRmsAfter,
                outcome.fixedTrackCount);
            continue;
        }
        ++candidate_applied_blocks;
        std::unordered_map<ImageId, std::size_t> camera_index;
        for (std::size_t index = 0; index < outcome.cameraIds.size(); ++index)
        {
            camera_index[outcome.cameraIds[index]] = index;
        }
        for (ImageId image_id : blocks[outcome.blockIndex].coreImageIds)
        {
            const auto index = camera_index.find(image_id);
            if (index != camera_index.end() &&
                index->second < outcome.result.refinedCameras.size())
            {
                camera_snapshots.try_emplace(
                    image_id, _owner._reconstruction->camera(image_id));
                _owner._reconstruction->camera(image_id) =
                    outcome.result.refinedCameras[index->second];
                ++candidate_updated_cameras;
            }
        }
        const std::unordered_set<ImageId> active_image_ids(
            outcome.cameraIds.begin(), outcome.cameraIds.end());
        for (std::size_t index = 0;
             index < outcome.pointIds.size() && index < outcome.result.points.size();
             ++index)
        {
            const Point3DId point_id = outcome.pointIds[index];
            const auto owner = point_owner.find(point_id);
            const BARefinedPoint &refined = outcome.result.points[index];
            if (owner == point_owner.end() || owner->second != outcome.blockIndex ||
                !refined.valid || !_owner._reconstruction->hasPoint3D(point_id))
            {
                continue;
            }
            ScenePoint3D &point = _owner._reconstruction->point3D(point_id);
            std::size_t registered_observations = 0;
            std::size_t block_observations = 0;
            for (const TrackElement &element : point.track.elements)
            {
                if (!_owner._reconstruction->isRegistered(element.imageId))
                {
                    continue;
                }
                ++registered_observations;
                if (active_image_ids.count(element.imageId) > 0)
                {
                    ++block_observations;
                }
            }
            if (!shouldWriteBackPoint(block_observations, registered_observations))
            {
                continue;
            }
            point_snapshots.push_back({point_id, point.xyz, point.error});
            point.xyz = refined.point;
            point.error = refined.rmsAfter;
            ++candidate_updated_points;
        }
    }

    const GlobalReprojectionState global_after =
        evaluateGlobalReprojection(*_owner._reconstruction);
    const bool globally_consistent = candidate_applied_blocks > 0 &&
        isGlobalWriteBackConsistent(global_before.rms,
                                    global_before.observations,
                                    global_after.rms,
                                    global_after.observations);
    Logger::instance()->infof(
        "[BA] hierarchical global_consistency accepted=%s rms=%.6f->%.6f "
        "observations=%zu->%zu candidateBlocks=%d candidateCameras=%d "
        "candidatePoints=%d",
        globally_consistent ? "true" : "false",
        global_before.rms,
        global_after.rms,
        global_before.observations,
        global_after.observations,
        candidate_applied_blocks,
        candidate_updated_cameras,
        candidate_updated_points);
    if (globally_consistent)
    {
        summary.appliedBlocks = candidate_applied_blocks;
        summary.updatedCameras = candidate_updated_cameras;
        summary.updatedPoints = candidate_updated_points;
    }
    else
    {
        for (const auto &[image_id, camera] : camera_snapshots)
        {
            _owner._reconstruction->camera(image_id) = camera;
        }
        for (const PointSnapshot &snapshot : point_snapshots)
        {
            if (_owner._reconstruction->hasPoint3D(snapshot.id))
            {
                ScenePoint3D &point = _owner._reconstruction->point3D(snapshot.id);
                point.xyz = snapshot.xyz;
                point.error = snapshot.error;
            }
        }
        if (candidate_applied_blocks > 0)
        {
            Logger::instance()->warn(
                "[BA] hierarchical merge rejected reason=global_reprojection_inconsistent; "
                "restored pre-block cameras and points");
        }
    }

    summary.totalSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (summary.applied())
    {
        _owner._lastHierarchicalBAImageCount = registered_count;
    }
    _owner._lastHierarchicalBAPlannedBlocks = summary.plannedBlocks;
    _owner._lastHierarchicalBAAppliedBlocks = summary.appliedBlocks;
    _owner._lastHierarchicalBAUpdatedCameras = summary.updatedCameras;
    _owner._lastHierarchicalBATotalSeconds = summary.totalSeconds;
    Logger::instance()->infof(
        "[BA] hierarchical result appliedBlocks=%d/%d cameras=%d points=%d totalSeconds=%.3f",
        summary.appliedBlocks,
        summary.plannedBlocks,
        summary.updatedCameras,
        summary.updatedPoints,
        summary.totalSeconds);
    return summary;
}

} // namespace xjw
