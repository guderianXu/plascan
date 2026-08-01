#pragma once

/**
 * @file SiftFeatureExtractor.h
 * @brief PlaScan 影像匹配模块唯一保留的特征提取实现。
 */

#include "FeatureSet.h"
#include "ImageMatchingAlgorithm.h"

namespace xjw::image_matching
{

class SiftFeatureExtractor
{
public:
    static bool isCudaAvailable();

    /**
     * @brief 提取 SIFT 并恢复到原始影像坐标系。
     *
     * 默认必须使用 CUDA SIFT。只有 runtime.allowCpuSiftFallback=true 时，CUDA
     * 不可用或执行失败才会使用 OpenCV SIFT；降级行为由调用方显式记录到报告。
     */
    static FeatureSet extract(const ImageFeatureInput &input,
                              const ImageMatchingRuntimeConfig &runtime,
                              bool *usedCuda = nullptr);
};

} // namespace xjw::image_matching
