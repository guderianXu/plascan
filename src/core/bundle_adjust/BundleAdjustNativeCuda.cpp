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

#include "BundleAdjustNativeCudaWorkset.h"
#include "BundleAdjustQuality.h"

#include <chrono>
#include <cmath>

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

#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
    return native_cuda::queryNativeCudaRuntime(deviceId, message);
#else
    (void)message;
    return false;
#endif
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
    // 阶段 2：设备函数原位更新 workset.points。
    const auto solveStart = std::chrono::steady_clock::now();
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

    // 阶段 3：将压缩工作集映射回原始 track 顺序。
    for (size_t pointIndex = 0; pointIndex < build.workset.points.size(); ++pointIndex)
    {
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
        result.points[static_cast<size_t>(optimizedPoint.originalTrackIndex)] = refined;
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
