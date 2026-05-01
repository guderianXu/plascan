// =============================================================================
// 文件: PatchMatchNoCUDA.cpp
// 说明: 当 CUDA 编译器不可用时，提供 GPU 接口的 CPU 回退存根。
//       PatchMatchDepthEstimator::estimate() 在调用 estimateGPU() 前会先检查
//       isCudaAvailable()，因此 estimateGPU() 实际上不会被执行。
// =============================================================================

#include "PatchMatchCUDA.h"
#include "DensePointCloudCUDA.h"

namespace xjw {
namespace mvs {

// ── PatchMatchDepthEstimator ──────────────────────────────────────────────────

bool PatchMatchDepthEstimator::isCudaAvailable()
{
    return false;
}

void PatchMatchDepthEstimator::cleanupGpuImageCache()
{
    // 无 CUDA 构建中无缓存，空操作
}

bool PatchMatchDepthEstimator::estimateGPU(
    const cv::Mat &,
    const std::vector<cv::Mat> &,
    const PositiveDepthCameraModel &,
    const std::vector<PositiveDepthCameraModel> &,
    float, float,
    const PatchMatchConfig &,
    cv::Mat &, cv::Mat *,
    std::string *errorMsg,
    const cv::Mat *)
{
    if (errorMsg) *errorMsg = "CUDA 不可用（编译时未启用）";
    return false;
}

// ── DensePointCloudCUDA ───────────────────────────────────────────────────────

std::vector<DensePoint> DensePointCloudCUDA::unprojectGPU(
    const cv::Mat &,
    const cv::Mat &,
    const PositiveDepthCameraModel &,
    const cv::Mat &,
    float, float,
    std::string *errorMsg,
    const DenseCloudOptions *)
{
    if (errorMsg) *errorMsg = "CUDA 不可用（编译时未启用）";
    return {};
}

} // namespace mvs
} // namespace xjw
