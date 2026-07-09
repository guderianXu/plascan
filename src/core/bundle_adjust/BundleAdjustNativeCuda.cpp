#include "BundleAdjustNativeCuda.h"

#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
#  include "BundleAdjustNativeCudaKernels.cuh"
#endif

#include "BundleAdjustNativeCudaWorkset.h"

namespace xjw::detail
{

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
    BAResult result;
    result.requestedBackend = options.backend;
    result.usedBackend = BABackend::NativeCuda;
    result.usedGpu = false;
    result.totalTracks = static_cast<int>(tracks.size());
    result.refinedCameras = cameras;
    result.points.resize(tracks.size());

    const auto build = native_cuda::buildWorkset(cameras, tracks, options);
    if (!build.ok)
    {
        result.backendMessage = build.message;
        return result;
    }

    result.nativeCudaActiveCameras = static_cast<int>(build.workset.cameras.size());
    result.nativeCudaActiveTracks = static_cast<int>(build.workset.points.size());
    result.nativeCudaActiveObservations = static_cast<int>(build.workset.observations.size());

#ifdef PLASCAN_BA_HAS_NATIVE_CUDA
    auto workset = build.workset;
    const native_cuda::KernelRunSummary summary =
        native_cuda::runNativeCudaBundleAdjust(&workset,
                                               options.nativeCudaDevice,
                                               options.maxIterations,
                                               options.nativeCudaMaxPcgIterations,
                                               options.nativeCudaPcgTolerance,
                                               options.huberDelta,
                                               options.damping);
    if (!summary.ok)
    {
        result.backendMessage = summary.message;
        return result;
    }

    result.usedGpu = true;
    result.nativeCudaPcgIterations = summary.pcgIterations;
    result.nativeCudaLinearResidual = summary.linearResidual;
    result.nativeCudaAcceptedSteps = summary.acceptedSteps;
    result.nativeCudaRejectedSteps = summary.rejectedSteps;
    result.backendMessage = "native_cuda kernel smoke path completed";
    return result;
#endif

    result.backendMessage = "native_cuda 后端尚未完成求解路径";
    return result;
}

} // namespace xjw::detail
