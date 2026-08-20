#pragma once

/**
 * @file SiftFeatureExtractor.h
 * @brief PlaScan 影像匹配模块唯一保留的特征提取实现。
 */

#include "FeatureSet.h"
#include "ImageMatchingAlgorithm.h"
#include "SiftComputeBackend.h"

namespace xjw::image_matching
{

    class SiftFeatureExtractor
    {
    public:
        static bool isBackendAvailable(SiftComputeBackend backend, int deviceIndex = 0);
        static SiftComputeBackend resolveBackend(SiftComputeBackend requested, int deviceIndex = 0);

        /**
         * @brief 提取 SIFT 并恢复到原始影像坐标系。
         *
         * auto_sift 按 CUDA、Metal、OpenCL、CPU 的顺序选择可用后端；
         * 自适应尺度、阈值重试、网格均匀化和 RootSIFT 由 runtime 显式启用。
         */
        static FeatureSet extract(const ImageFeatureInput& input, const ImageMatchingRuntimeConfig& runtime);
    };

} // namespace xjw::image_matching
