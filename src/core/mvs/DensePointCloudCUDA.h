#pragma once
// =============================================================================
// 文件: DensePointCloudCUDA.h
// 模块: MVS - GPU 稠密点云生成
// 说明:
//   利用 CUDA kernel 并行将深度图反投影为世界坐标系点云。
// =============================================================================

#include "MvsTypes.h"
#include "DenseCloudBuilder.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace xjw
{
    namespace mvs
    {

        class DensePointCloudCUDA
        {
        public:
            static bool isAvailable(int deviceIndex = 0, std::string* errorMsg = nullptr);
            static std::string deviceName(int deviceIndex = 0);

            /// GPU 反投影接口
            /// @param depth     深度图 (CV_32F)
            /// @param mask      有效掩码 (CV_8U)，可为空（全部有效）
            /// @param cam       相机参数
            /// @param colorImg  颜色图 (CV_8UC3 BGR) 或灰度，可为空
            /// @param minDepth  下深度截断
            /// @param maxDepth  上深度截断
            /// @param errorMsg  出错时填充
            /// @return 点云；CUDA 不可用或执行失败时返回空并填写 errorMsg，不静默回退
            static std::vector<DensePoint> unprojectGPU(const cv::Mat& depth,
                                                        const cv::Mat& mask,
                                                        const FramePinholeCamera& cameraModel,
                                                        const cv::Mat& colorImg,
                                                        float minDepth = 0.01f,
                                                        float maxDepth = 1e6f,
                                                        std::string* errorMsg = nullptr,
                                                        const DenseCloudOptions* options = nullptr);
        };

    } // namespace mvs
} // namespace xjw
