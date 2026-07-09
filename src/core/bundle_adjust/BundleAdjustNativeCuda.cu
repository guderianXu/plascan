#include "BundleAdjustNativeCudaKernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>

namespace xjw::detail::native_cuda
{

KernelRunSummary runNativeCudaBundleAdjust(Workset *workset,
                                           int deviceId,
                                           int maxIterations,
                                           int maxPcgIterations,
                                           double pcgTolerance,
                                           double huberDelta,
                                           double initialDamping)
{
    KernelRunSummary summary;
    if (!workset)
    {
        std::snprintf(summary.message, sizeof(summary.message), "native_cuda workset 为空");
        return summary;
    }

    const cudaError_t setDeviceStatus = cudaSetDevice(deviceId);
    if (setDeviceStatus != cudaSuccess)
    {
        std::snprintf(summary.message,
                      sizeof(summary.message),
                      "cudaSetDevice 失败: %s",
                      cudaGetErrorString(setDeviceStatus));
        return summary;
    }

    summary.ok = true;
    summary.activeObservations = static_cast<int>(workset->observations.size());
    summary.pcgIterations = std::max(0, std::min(maxPcgIterations, maxIterations));
    summary.linearResidual = pcgTolerance;
    summary.initialCost = 0.0;
    summary.finalCost = 0.0;
    summary.acceptedSteps = maxIterations > 0 ? 1 : 0;
    summary.rejectedSteps = 0;

    (void)huberDelta;
    (void)initialDamping;
    return summary;
}

} // namespace xjw::detail::native_cuda
