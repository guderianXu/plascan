#pragma once

#include <opencv2/opencv.hpp>

#include <string>

#include "SuperPoint.h"

namespace xjw::feature_extractors
{

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
     * @param config 特征参数（沿用 SuperPointConfig 的公共字段）。
     * @param normalizedName 已规范化算法名（`orb`/`sift`）。
     * @return 与 SuperPoint 兼容的输出结构。
     */
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
