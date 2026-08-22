#pragma once

/**
 * @file SiftComputeBackend.h
 * @brief SIFT 提取与描述子匹配使用的统一计算后端接口。
 */

#include "SiftMatchFilter.h"
#include "SiftBackendType.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <QString>

#include <vector>

namespace xjw::image_matching
{

    struct SiftRawFeatures
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    struct SiftExtractionRequest
    {
        cv::Mat image;
        int maximumFeatures = 0;
        float contrastThreshold = 0.02f;
        int deviceIndex = 0;
    };

    const char* siftBackendName(SiftComputeBackend backend);
    QString siftBackendDisplayName(SiftComputeBackend backend);
    /// 返回运行设备的硬件名称；CPU 后端没有单独设备名，返回空字符串。
    QString siftBackendDeviceName(SiftComputeBackend backend, int deviceIndex = 0);
    /// 面向 UI/报告的实际运行设备，例如“CUDA · NVIDIA RTX ...”或“OpenCV CPU”。
    QString siftBackendRuntimeDisplayName(SiftComputeBackend backend, int deviceIndex = 0);

    bool isSiftBackendAvailable(SiftComputeBackend backend, int deviceIndex = 0);
    SiftComputeBackend resolveSiftBackend(SiftComputeBackend requested, int deviceIndex = 0);

    SiftRawFeatures extractSiftOnGpu(SiftComputeBackend backend, const SiftExtractionRequest& request);

    std::vector<SiftNearestMatch> matchSiftOnGpu(SiftComputeBackend backend,
                                                 const cv::Mat& queryDescriptors,
                                                 const cv::Mat& trainDescriptors,
                                                 int deviceIndex = 0);

} // namespace xjw::image_matching
