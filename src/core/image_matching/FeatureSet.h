#pragma once

/**
 * @file FeatureSet.h
 * @brief 一次匹配任务内部使用的、不会持久化的特征集合。
 */

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace xjw::image_matching
{

    class IFeaturePayload
    {
    public:
        virtual ~IFeaturePayload() = default;

        virtual std::string schemaId() const = 0;
        virtual std::size_t approximateBytes() const = 0;
        virtual bool isConsistent(std::size_t featureCount) const = 0;
    };

    struct FeatureSet
    {
        std::vector<cv::KeyPoint> keypoints;  ///< 与描述子行严格一一对应。
        std::vector<float> scores;            ///< 检测响应，用于显存预算下的稳定筛选。
        cv::Mat descriptors;                  ///< 浮点描述子为 CV_32F，二进制描述子为 CV_8U。
        bool descriptorsL2Normalized = false; ///< 可直接用于余弦/点积匹配，无需逐像对再次归一化。
        std::string sourceAlgorithm = "sift"; ///< 运行时校验字段，不参与文件持久化。
        std::string computeBackend = "cpu";   ///< 本次提取实际使用的计算后端，不参与持久化。
        int imageWidth = 0;                   ///< 坐标所属原始影像宽度。
        int imageHeight = 0;                  ///< 坐标所属原始影像高度。
        std::shared_ptr<const IFeaturePayload> payload; ///< 算法私有、任务期只读扩展数据。

        bool empty() const;
        int size() const;
        int descriptorDimension() const;
        std::size_t approximateBytes() const;
        bool isConsistent() const;
    };

} // namespace xjw::image_matching
