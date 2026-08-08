#include "PatchMatchCUDA.h"

namespace xjw
{
namespace mvs
{

bool PatchMatchDepthEstimator::isOpenClAvailable()
{
    return false;
}

std::vector<OpenClDeviceInfo> PatchMatchDepthEstimator::openClDevices()
{
    return {};
}

void PatchMatchDepthEstimator::resetOpenClExecutionStats()
{
}

std::vector<OpenClExecutionStats> PatchMatchDepthEstimator::openClExecutionStats()
{
    return {};
}

void PatchMatchDepthEstimator::cleanupOpenClResources()
{
}

bool PatchMatchDepthEstimator::estimateOpenCL(
    const cv::Mat &,
    const std::vector<cv::Mat> &,
    const Camera &,
    const std::vector<Camera> &,
    float,
    float,
    const PatchMatchConfig &,
    cv::Mat &,
    cv::Mat *,
    std::string *errorMsg,
    const cv::Mat *,
    const cv::Mat *,
    const cv::Mat *,
    const std::vector<cv::Mat> *)
{
    if (errorMsg)
    {
        *errorMsg = "OpenCL is unavailable in this build";
    }
    return false;
}

} // namespace mvs
} // namespace xjw
