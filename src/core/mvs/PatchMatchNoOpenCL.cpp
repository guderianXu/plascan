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

bool PatchMatchDepthEstimator::prepareOpenClDevice(int,
                                                   std::string *errorMsg)
{
    if (errorMsg)
    {
        *errorMsg = "OpenCL is unavailable in this build";
    }
    return false;
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
    const FramePinholeCamera &,
    const std::vector<FramePinholeCamera> &,
    float,
    float,
    const PatchMatchConfig &,
    cv::Mat &,
    cv::Mat *,
    std::string *errorMsg,
    const cv::Mat *,
    const cv::Mat *,
    const cv::Mat *,
    const std::vector<cv::Mat> *,
    const PatchMatchAuxiliaryInput *,
    PatchMatchAuxiliaryOutput *)
{
    if (errorMsg)
    {
        *errorMsg = "OpenCL is unavailable in this build";
    }
    return false;
}

} // namespace mvs
} // namespace xjw
