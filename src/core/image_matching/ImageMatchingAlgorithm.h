#pragma once

/**
 * @file ImageMatchingAlgorithm.h
 * @brief 影像匹配算法的能力描述和统一扩展接口。
 */

#include "FeatureSet.h"
#include "MatchResult.h"
#include "sift/SiftBackendType.h"

#include <QByteArray>
#include <QString>

#include <opencv2/core.hpp>

#include <cstdint>

namespace xjw::image_matching
{

enum class AlgorithmInputModel
{
    ReusableFeatures, ///< 先按影像提取一次特征，再在多个像对间复用。
    EndToEndImagePair ///< 算法直接消费两幅影像，不承诺稳定的跨像对特征索引。
};

/// 算法静态能力；version 变化会使旧 `.pimatch` 变体失效。
struct ImageMatchingAlgorithmDescriptor
{
    QString id;
    QString displayName;
    std::uint32_t version = 0;
    AlgorithmInputModel inputModel = AlgorithmInputModel::ReusableFeatures;
    bool requiresCuda = false;
    bool suppliesStableFeatureIds = true;
    bool requiresColorInput = false; ///< 特征前端需要 RGB，而不是仅灰度输入。
};

/**
 * @brief 一次算法实例的运行配置。
 *
 * configFingerprint/modelFingerprint 由任务层计算并随匹配结果持久化，用来区分
 * 相同算法版本下的门限、蒙版、图像尺度和模型 engine 变化。
 */
struct ImageMatchingRuntimeConfig
{
    QString tensorRtEnginePath;
    QString tensorRtFeatureEnginePath;
    QString tensorRtMatcherEnginePath;
    int cudaDevice = 0;
    int maxKeypoints = 40000;
    int maxMatcherKeypoints = 0;
    int featureKeypointCount = 0;
    int removeBorders = 16;
    int maxImageDimension = 2048;
    float siftDetectionThreshold = 0.0005f;
    float siftContrastThreshold = 0.02f;
    float grayscaleMin = 0.0f;
    float grayscaleMax = 1.0f;
    float matchThreshold = 0.15f;
    int modelInputWidth = 0;
    int modelInputHeight = 0;
    int descriptorDimension = 0;
    SiftComputeBackend siftBackend = SiftComputeBackend::Automatic;
    bool adaptiveSift = false;
    bool rootSift = false;
    QByteArray configFingerprint;
    QByteArray modelFingerprint;
};

/// 单幅影像进入特征提取阶段的内存输入，不参与任何文件持久化。
struct ImageFeatureInput
{
    QString imagePath;
    cv::Mat grayImage;
    cv::Mat colorImage; ///< BGR/BGRA/RGB 由具体算法在边界处规范化。
    cv::Mat validMask; ///< 空矩阵表示不应用蒙版；非零像素表示允许提取。
    int originalWidth = 0;
    int originalHeight = 0;
    double coordinateScale = 1.0; ///< 输入坐标乘以该值恢复到原始影像。
};

/**
 * @brief 匹配算法统一接口。
 *
 * ReusableFeatures 算法实现 extract()/matchFeatures()；端到端算法实现
 * matchImages()。默认实现显式抛出不支持错误，调度层必须先检查 descriptor()
 * 中的 inputModel，不能通过捕获异常猜测算法能力。
 */
class IImageMatchingAlgorithm
{
public:
    virtual ~IImageMatchingAlgorithm() = default;

    virtual ImageMatchingAlgorithmDescriptor descriptor() const = 0;
    virtual FeatureSet extract(const ImageFeatureInput &input) const;
    virtual MatchResult matchFeatures(const FeatureSet &features0,
                                      const FeatureSet &features1);
    virtual MatchResult matchImages(const cv::Mat &image0, const cv::Mat &image1);
};

} // namespace xjw::image_matching
