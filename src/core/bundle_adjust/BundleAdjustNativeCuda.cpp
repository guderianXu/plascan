/**
 * @file BundleAdjustNativeCuda.cpp
 * @brief Native CUDA 固定相机、独立三维点 BA 后端的公共适配实现。
 *
 * 该后端不构造全局 Schur 系统，也不优化相机变量。主机先把有效轨迹压平成
 * Workset，设备上一线程优化一个三维点，下载后再按公共 BAResult 契约执行
 * 正深度、RMS 和有效轨迹比例门控。需要联合相机/焦距优化时应选择 Ceres 后端。
 */

#include "BundleAdjustNativeCuda.h"

#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
#  include "BundleAdjustNativeCudaKernels.cuh"
#endif

#include "BundleAdjustNativeCudaMath.h"
#include "BundleAdjustNativeCudaWorkset.h"
#include "BundleAdjustQuality.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace xjw::detail
{

namespace
{

/// 在公共结果装配前检查下载点的三个坐标均为有限值。
bool finitePoint(const std::array<double, 3> &point)
{
    return std::isfinite(point[0]) &&
           std::isfinite(point[1]) &&
           std::isfinite(point[2]);
}

/**
 * @brief 用 CPU 参考投影计算一个点的加权像素 RMS。
 *
 * 该复核独立于 CUDA 核内部目标，用于生成可与 Legacy/Ceres 比较的 rmsBefore/
 * rmsAfter。无有效正深度观测时返回 infinity。
 */
double computePointRms(const native_cuda::Workset &workset,
                       const native_cuda::HostPoint &point)
{
    double sum = 0.0;
    int count = 0;
    for (int local = 0; local < point.observationCount; ++local)
    {
        const int observationIndex = point.observationBegin + local;
        if (observationIndex < 0 ||
            observationIndex >= static_cast<int>(workset.observations.size()))
        {
            continue;
        }

        const native_cuda::HostObservation &observation =
            workset.observations[static_cast<size_t>(observationIndex)];
        if (observation.cameraIndex < 0 ||
            observation.cameraIndex >= static_cast<int>(workset.cameras.size()))
        {
            continue;
        }

        const native_cuda::ProjectionResult projection =
            native_cuda::projectHost(workset.cameras[static_cast<size_t>(observation.cameraIndex)],
                                     point.xyz);
        if (!projection.ok)
        {
            continue;
        }

        const double du = projection.pixel[0] - observation.u;
        const double dv = projection.pixel[1] - observation.v;
        const double weight = std::isfinite(observation.weight) ? std::max(0.0, observation.weight) : 0.0;
        sum += weight * (du * du + dv * dv);
        count += 2;
    }
    return count > 0 ? std::sqrt(sum / static_cast<double>(count)) : std::numeric_limits<double>::infinity();
}

} // namespace

bool isNativeCudaBackendCompiled()
{
#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
    return true;
#else
    return false;
#endif
}

bool isNativeCudaRuntimeAvailable(int deviceId, std::string *message)
{
    if (!isNativeCudaBackendCompiled())
    {
        if (message)
        {
            *message = "native_cuda 后端未编译";
        }
        return false;
    }

    if (deviceId < 0)
    {
        if (message)
        {
            *message = "native_cuda GPU 设备 ID 无效";
        }
        return false;
    }

    return true;
}

BAResult optimizePointsWithNativeCuda(const std::vector<Camera> &cameras,
                                      const std::vector<BATrack> &tracks,
                                      const BAOptions &options)
{
    const auto totalStart = std::chrono::steady_clock::now();
    BAResult result;
    result.requestedBackend = options.backend;
    result.usedBackend = BABackend::NativeCuda;
    result.usedGpu = false;
    result.totalTracks = static_cast<int>(tracks.size());
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());
    for (size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
    {
        result.points[trackIndex].point = tracks[trackIndex].initialPoint;
    }

    if (options.cancelFlag && options.cancelFlag->load())
    {
        result.solveStatus = BASolveStatus::Cancelled;
        result.backendMessage = "native_cuda BA 在启动前被取消";
        return result;
    }

    // 阶段 1：验证输入并仅保留具有足量有效观测的轨迹，建立连续观测区间。
    auto build = native_cuda::buildWorkset(cameras, tracks, options);
    const auto setupEnd = std::chrono::steady_clock::now();
    result.setupSeconds = std::chrono::duration<double>(setupEnd - totalStart).count();
    if (!build.ok)
    {
        result.solveStatus = BASolveStatus::InvalidInput;
        result.backendMessage = build.message;
        result.totalSeconds = result.setupSeconds;
        return result;
    }

    result.nativeCudaActiveCameras = static_cast<int>(build.workset.cameras.size());
    result.nativeCudaActiveTracks = static_cast<int>(build.workset.points.size());
    result.nativeCudaActiveObservations = static_cast<int>(build.workset.observations.size());
    result.observationCount = result.nativeCudaActiveObservations;

#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
    // 阶段 2：设备函数原位更新 workset.points；初始点副本用于统一质量对比。
    const auto solveStart = std::chrono::steady_clock::now();
    const std::vector<native_cuda::HostPoint> initialPoints = build.workset.points;
    const native_cuda::KernelRunSummary summary =
        native_cuda::runNativeCudaBundleAdjust(&build.workset,
                                               options.nativeCudaDevice,
                                               options.maxIterations,
                                               options.huberDelta,
                                               options.damping,
                                               options.nativeCudaMaxPointStepNorm);
    const auto solveEnd = std::chrono::steady_clock::now();
    result.solveSeconds = std::chrono::duration<double>(solveEnd - solveStart).count();
    result.totalSeconds = std::chrono::duration<double>(solveEnd - totalStart).count();
    if (!summary.ok)
    {
        result.solveStatus = BASolveStatus::NumericalFailure;
        result.backendMessage = summary.message;
        return result;
    }

    result.solveStatus = BASolveStatus::Success;
    result.solutionUsable = true;
    result.usedGpu = true;
    result.nativeCudaInitialCost = summary.initialCost;
    result.nativeCudaFinalCost = summary.finalCost;
    result.nativeCudaAcceptedSteps = summary.acceptedSteps;
    result.nativeCudaRejectedSteps = summary.rejectedSteps;
    result.nativeCudaUploadSeconds = summary.uploadSeconds;
    result.nativeCudaKernelSeconds = summary.kernelSeconds;
    result.nativeCudaDownloadSeconds = summary.downloadSeconds;
    result.nativeCudaHostCostSeconds = summary.hostCostSeconds;
    result.nativeCudaDeviceSelectSeconds = summary.deviceSelectSeconds;
    result.nativeCudaStagingSeconds = summary.stagingSeconds;
    result.nativeCudaReleaseSeconds = summary.releaseSeconds;
    result.backendMessage = summary.message;
    result.refinedCameraCount = 0;

    // 阶段 3：将压缩工作集映射回原始 track 顺序，并在主机复算每点 RMS。
    double sumBefore = 0.0;
    double sumAfter = 0.0;
    int countBefore = 0;
    int countAfter = 0;
    for (size_t pointIndex = 0; pointIndex < build.workset.points.size(); ++pointIndex)
    {
        const native_cuda::HostPoint &initialPoint = initialPoints[pointIndex];
        const native_cuda::HostPoint &optimizedPoint = build.workset.points[pointIndex];
        if (optimizedPoint.originalTrackIndex < 0 ||
            optimizedPoint.originalTrackIndex >= static_cast<int>(result.points.size()))
        {
            continue;
        }

        BARefinedPoint refined;
        refined.point = optimizedPoint.xyz;
        refined.valid = finitePoint(refined.point);
        refined.converged = summary.ok;
        refined.iterations = options.maxIterations;
        refined.rmsBefore = computePointRms(build.workset, initialPoint);
        refined.rmsAfter = computePointRms(build.workset, optimizedPoint);
        if (std::isfinite(refined.rmsBefore))
        {
            sumBefore += refined.rmsBefore;
            ++countBefore;
        }
        if (refined.valid)
        {
            refined.valid = std::isfinite(refined.rmsAfter);
        }
        if (refined.valid)
        {
            sumAfter += refined.rmsAfter;
            ++countAfter;
            ++result.optimizedTracks;
        }
        result.points[static_cast<size_t>(optimizedPoint.originalTrackIndex)] = refined;
    }

    result.meanRmsBefore = countBefore > 0 ? sumBefore / static_cast<double>(countBefore) : 0.0;
    result.meanRmsAfter = countAfter > 0 ? sumAfter / static_cast<double>(countAfter) : 0.0;
    // Native CUDA 当前不改变相机。保留显式映射是为了维持后端统一结果契约。
    result.refinedCameras = cameras;
    for (const native_cuda::HostCamera &hostCamera : build.workset.cameras)
    {
        if (hostCamera.originalIndex < 0 ||
            hostCamera.originalIndex >= static_cast<int>(result.refinedCameras.size()))
        {
            continue;
        }
        Camera camera = result.refinedCameras[static_cast<size_t>(hostCamera.originalIndex)];
        camera.setPose(hostCamera.cameraToWorldRotation, hostCamera.cameraCenter);
        result.refinedCameras[static_cast<size_t>(hostCamera.originalIndex)] = camera;
    }
    // 阶段 4：应用所有后端共享的质量门，数值完成不等于结果可安全写回。
    finalizeBundleAdjustResult(cameras, tracks, options, &result);
    return result;
#endif

    result.backendMessage = "native_cuda 后端尚未完成求解路径";
    result.solveStatus = BASolveStatus::BackendUnavailable;
    result.totalSeconds = result.setupSeconds;
    return result;
}

} // namespace xjw::detail
