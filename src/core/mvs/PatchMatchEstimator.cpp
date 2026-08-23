// =============================================================================
// 文件: PatchMatchEstimator.cpp
// 模块: MVS - PatchMatch 后端选择与公共输入校验
// =============================================================================

#include "PatchMatchCUDA.h"

#include "Logger.h"

#include <algorithm>
#include <atomic>

namespace xjw
{
namespace mvs
{

PatchMatchBackend resolvePatchMatchEstimatorBackend(PatchMatchBackend requestedBackend,
                                                     bool cudaAvailable,
                                                     bool openClAvailable) noexcept
{
    if (requestedBackend != PatchMatchBackend::Auto)
    {
        return requestedBackend;
    }
    if (cudaAvailable)
    {
        return PatchMatchBackend::Cuda;
    }
    if (openClAvailable)
    {
        return PatchMatchBackend::OpenCl;
    }
    return PatchMatchBackend::Cpu;
}

bool isUsableOpenClPatchMatchDevice(bool availabilityQuerySucceeded,
                                    bool available,
                                    bool compilerQuerySucceeded,
                                    bool compilerAvailable) noexcept
{
    return availabilityQuerySucceeded && available &&
        compilerQuerySucceeded && compilerAvailable;
}

bool PatchMatchDepthEstimator::estimate(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const FramePinholeCamera                   &refCam,
    const std::vector<FramePinholeCamera>      &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth,
    const cv::Mat                *hintRadius,
    const cv::Mat                *refValidMask,
    const std::vector<cv::Mat>   *srcValidMasks,
    const PatchMatchAuxiliaryInput *auxiliaryInput,
    PatchMatchAuxiliaryOutput *auxiliaryOutput)
{
    if (config.numIterations <= 0)
    {
        if (errorMsg) *errorMsg = "PatchMatch iteration count must be positive";
        return false;
    }
    if (srcGrays.empty() || srcCams.empty() || srcGrays.size() != srcCams.size())
    {
        if (errorMsg) *errorMsg = "source frame count mismatch or empty";
        return false;
    }
    if (refValidMask && !refValidMask->empty() && refValidMask->channels() != 1)
    {
        if (errorMsg) *errorMsg = "reference valid mask must be single-channel";
        return false;
    }
    if (srcValidMasks && srcValidMasks->size() != srcGrays.size())
    {
        if (errorMsg) *errorMsg = "source valid mask count does not match source frame count";
        return false;
    }
    if (auxiliaryInput && auxiliaryInput->sourceDepthMaps &&
        auxiliaryInput->sourceDepthMaps->size() != srcGrays.size())
    {
        if (errorMsg) *errorMsg = "source depth map count does not match source frame count";
        return false;
    }
    if (auxiliaryInput && auxiliaryInput->sourceDepthMaps)
    {
        for (const cv::Mat &source_depth : *auxiliaryInput->sourceDepthMaps)
        {
            if (!source_depth.empty() && source_depth.channels() != 1)
            {
                if (errorMsg) *errorMsg = "source depth maps must be single-channel";
                return false;
            }
        }
    }
    if (srcValidMasks)
    {
        for (const cv::Mat &source_mask : *srcValidMasks)
        {
            if (!source_mask.empty() && source_mask.channels() != 1)
            {
                if (errorMsg) *errorMsg = "source valid masks must be single-channel";
                return false;
            }
        }
    }
    if (!refCam.isValid())
    {
        if (errorMsg) *errorMsg = "reference camera parameters are invalid";
        return false;
    }
    zNear = std::max(zNear, 0.01f);
    zFar  = std::max(zFar, zNear + 0.1f);

    bool cuda_available = false;
    bool opencl_available = false;
    if (config.backend == PatchMatchBackend::Auto)
    {
        cuda_available = isCudaAvailable();
        opencl_available = !cuda_available && isOpenClAvailable();
    }
    const PatchMatchBackend backend = resolvePatchMatchEstimatorBackend(
        config.backend, cuda_available, opencl_available);
    const bool has_geometric_guidance_input = auxiliaryInput &&
        auxiliaryInput->sourceDepthMaps &&
        !auxiliaryInput->sourceDepthMaps->empty();
    if (backend == PatchMatchBackend::OpenCl && has_geometric_guidance_input)
    {
        if (errorMsg)
        {
            *errorMsg = "OpenCL PatchMatch does not support frozen source-depth guidance; "
                        "select CUDA or CPU explicitly";
        }
        return false;
    }

    bool accelerator_ok = false;
    bool fallback_to_cpu = false;
    const char *backend_name = nullptr;
    if (backend == PatchMatchBackend::Cuda)
    {
        backend_name = "CUDA";
        fallback_to_cpu = config.cudaFallbackToCpu;
        if (isCudaAvailable())
        {
            accelerator_ok = estimateGPU(refGray, srcGrays, refCam, srcCams,
                                         zNear, zFar, config, depthOut, confOut, errorMsg,
                                         hintDepth, hintRadius, refValidMask, srcValidMasks,
                                         auxiliaryInput, auxiliaryOutput);
        }
        else if (errorMsg)
        {
            *errorMsg = "CUDA backend requested but no CUDA device is available";
        }
    }
    else if (backend == PatchMatchBackend::OpenCl)
    {
        backend_name = "OpenCL";
        fallback_to_cpu = config.openClFallbackToCpu;
        if (isOpenClAvailable())
        {
            accelerator_ok = estimateOpenCL(refGray, srcGrays, refCam, srcCams,
                                            zNear, zFar, config, depthOut, confOut, errorMsg,
                                            hintDepth, hintRadius, refValidMask, srcValidMasks,
                                            auxiliaryInput, auxiliaryOutput);
        }
        else if (errorMsg)
        {
            *errorMsg = "OpenCL backend requested but no OpenCL GPU device is available";
        }
    }
    if (backend_name)
    {
        if (accelerator_ok)
        {
            return true;
        }
        if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMsg && errorMsg->empty()) *errorMsg = "PatchMatch cancelled";
            return false;
        }
        if (!fallback_to_cpu)
        {
            return false;
        }
        LOG_WARN("[MVS][PatchMatch] %s failed; falling back to CPU", backend_name);
    }
    return estimateCPU(refGray, srcGrays, refCam, srcCams,
                       zNear, zFar, config, depthOut, confOut, errorMsg,
                       hintDepth, hintRadius, refValidMask, srcValidMasks,
                       auxiliaryInput, auxiliaryOutput);
}

} // namespace mvs
} // namespace xjw
