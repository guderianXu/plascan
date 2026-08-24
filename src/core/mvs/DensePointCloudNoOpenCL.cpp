#include "DensePointCloudOpenCL.h"

namespace xjw::mvs
{

    bool DensePointCloudOpenCL::isAvailable(int, std::string* errorMsg)
    {
        if (errorMsg)
        {
            *errorMsg = "OpenCL 不可用（编译时未启用）";
        }
        return false;
    }

    std::string DensePointCloudOpenCL::deviceName(int)
    {
        return {};
    }

    std::vector<DensePoint> DensePointCloudOpenCL::unproject(const cv::Mat&,
                                                             const cv::Mat&,
                                                             const FramePinholeCamera&,
                                                             const cv::Mat&,
                                                             float,
                                                             float,
                                                             std::string* errorMsg,
                                                             const DenseCloudOptions*)
    {
        if (errorMsg)
        {
            *errorMsg = "OpenCL 不可用（编译时未启用）";
        }
        return {};
    }

} // namespace xjw::mvs
