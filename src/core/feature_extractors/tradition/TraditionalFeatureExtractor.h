#pragma once

#include <opencv2/opencv.hpp>

#include <string>

#include "SuperPoint.h"

namespace xjw::feature_extractors
{

struct TraditionalFeatureConfig
{
    int maxKeypoints = 20000;
    int maxImageSize = 0;
    int removeBorders = 4;
    int descriptorDim = 128;
    float grayscaleMin = 0.0f;
    float grayscaleMax = 1.0f;
    bool allowDeviceFallback = true;
    float detectionThreshold = 0.005f;
};

TraditionalFeatureConfig traditionalFeatureConfigFromSuperPoint(const SuperPointConfig &config);

/**
 * @brief 传统特征提取器（ORB/SIFT）统一入口。
 *
 * 该类用于在不依赖 GUI 的情况下提供传统特征提取算法，
 * 并将结果转换为与 SuperPoint 相同的输出结构 `FeatureOutput`，
 * 以复用现有 `.sp` 文件写入与下游处理流程。
 */
class TraditionalFeatureExtractor
{
public:
    /**
     * @brief 规范化算法名。
     * @param algorithmName 输入算法名（大小写不敏感）。
     * @return 规范名：`superpoint` / `orb` / `sift`。
     */
    static std::string normalizeAlgorithmName(const std::string &algorithmName);

    /**
     * @brief 判断是否为传统算法。
     * @param normalizedName 已规范化算法名。
     * @return true 表示 `orb` 或 `sift`。
     */
    static bool isTraditionalAlgorithm(const std::string &normalizedName);

    /**
     * @brief 运行传统特征提取，并输出统一结构。
     * @param grayImage 单通道灰度图（CV_8U）。
     * @param config 传统特征参数。
     * @param normalizedName 已规范化算法名（`orb`/`sift`）。
     * @return 与 SuperPoint 兼容的输出结构。
     */
    static FeatureOutput detect(const cv::Mat &grayImage,
                                const TraditionalFeatureConfig &config,
                                const std::string &normalizedName);

    static FeatureOutput detect(const cv::Mat &grayImage,
                                const TraditionalFeatureConfig &config,
                                const std::string &normalizedName,
                                bool useCuda,
                                int cudaDevice);

    // 旧调用方兼容入口。新代码应使用 TraditionalFeatureConfig，避免把
    // SuperPoint 语义泄漏到 SIFT/ORB/AKAZE 等传统特征阶段。
    static FeatureOutput detect(const cv::Mat &grayImage,
                                   const SuperPointConfig &config,
                                   const std::string &normalizedName);

    static FeatureOutput detect(const cv::Mat &grayImage,
                                const SuperPointConfig &config,
                                const std::string &normalizedName,
                                bool useCuda,
                                int cudaDevice);
};

} // namespace xjw::feature_extractors
