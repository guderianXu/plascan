#pragma once

/**
 * @file MatchPhotosMaskSupport.h
 * @brief 创建连接点流程统一使用的蒙版语义和路径解析工具。
 *
 * 本文件只处理影像、坐标和蒙版本身，不依赖任何具体特征或匹配器类型。算法层
 * 需要的“非零=有效”蒙版在 FeatureStage 边界转换；项目持久化语义始终保持
 * “0=有效、非 0=排除”。
 */

#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"

#include <QString>

#include <opencv2/core.hpp>

namespace xjw::matchphotos
{

enum class MatchPhotosMaskApplyMode
{
    None,
    Keypoints,
    Tiepoints
};

MatchPhotosMaskApplyMode maskApplyModeFromToken(const QString &token);
bool shouldApplyMasksToKeypoints(const MatchPhotosOptions &options);
bool shouldApplyMasksToTiepoints(const MatchPhotosOptions &options);

/// 将任意通道/尺寸蒙版规范成 CV_8U；0=允许，非 0=排除。
cv::Mat normalizedMaskForImage(const cv::Mat &mask, const cv::Size &imageSize);
/// 将排除概率蒙版向内部收缩，保留分割边界和低置信度区域。
cv::Mat softenedExclusionMask(const cv::Mat &mask,
                              const MatchPhotosOptions &options);
cv::Mat makeExtractorValidMask(const cv::Mat &mask,
                               const cv::Size &extractorSize,
                               const MatchPhotosOptions &options);
float maskPointWeight(const cv::Mat &mask,
                      const cv::Point2f &point,
                      const MatchPhotosOptions &options);
bool isPointAllowedByMask(const cv::Mat &mask,
                          const cv::Point2f &point,
                          const MatchPhotosOptions &options);

QString maskPathForImage(const MatchPhotosContext &context, const QString &imagePath);
cv::Mat loadMaskForImage(const MatchPhotosContext &context,
                         const QString &imagePath,
                         const cv::Size &imageSize);

} // namespace xjw::matchphotos
