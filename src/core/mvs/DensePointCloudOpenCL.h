#pragma once

#include "DenseCloudBuilder.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace xjw::mvs
{

    class DensePointCloudOpenCL
    {
    public:
        static bool isAvailable(int deviceIndex = 0, std::string* errorMsg = nullptr);
        static std::string deviceName(int deviceIndex = 0);

        static std::vector<DensePoint> unproject(const cv::Mat& depth,
                                                 const cv::Mat& mask,
                                                 const FramePinholeCamera& cameraModel,
                                                 const cv::Mat& colorImg,
                                                 float minDepth,
                                                 float maxDepth,
                                                 std::string* errorMsg = nullptr,
                                                 const DenseCloudOptions* options = nullptr);
    };

} // namespace xjw::mvs
