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

bool PatchMatchDepthEstimator::estimate(
    const cv::Mat                &refGray,
    const std::vector<cv::Mat>   &srcGrays,
    const Camera                   &refCam,
    const std::vector<Camera>      &srcCams,
    float zNear, float zFar,
    const PatchMatchConfig       &config,
    cv::Mat                      &depthOut,
    cv::Mat                      *confOut,
    std::string                  *errorMsg,
    const cv::Mat                *hintDepth,
    const cv::Mat                *hintRadius,
    const cv::Mat                *refValidMask,
    const std::vector<cv::Mat>   *srcValidMasks)
{
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

    if (config.useCuda && isCudaAvailable()) 
    {
        if (estimateGPU(refGray, srcGrays, refCam, srcCams,
                        zNear, zFar, config, depthOut, confOut, errorMsg,
                        hintDepth, hintRadius, refValidMask, srcValidMasks))
        {
            return true;
        }
        if (config.cancelFlag && config.cancelFlag->load(std::memory_order_relaxed))
        {
            if (errorMsg && errorMsg->empty()) *errorMsg = "PatchMatch cancelled";
            return false;
        }
        if (!config.cudaFallbackToCpu)
        {
            return false;
        }
        LOG_WARN("[MVS][PatchMatch] GPU failed; falling back to CPU");
    }
    return estimateCPU(refGray, srcGrays, refCam, srcCams,
                       zNear, zFar, config, depthOut, confOut, errorMsg,
                       hintDepth, hintRadius, refValidMask, srcValidMasks);
}

} // namespace mvs
} // namespace xjw

