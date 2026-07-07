#pragma once

#include "FeatureData.h"
#include "FeatureOutput.h"
#include "match.h"
#include "MatchPhotosContext.h"
#include "MatchPhotosOptions.h"

#include <QString>

#include <opencv2/core.hpp>

namespace xjw
{
namespace matchphotos
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

// 蒙版约定：0 表示有效成像区域，非 0 表示需要排除的区域。
cv::Mat normalizedMaskForImage(const cv::Mat &mask, const cv::Size &imageSize);
bool isPointAllowedByMask(const cv::Mat &mask, const cv::Point2f &point);

FeatureOutput filterFeatureOutputByMask(const FeatureOutput &output, const cv::Mat &mask);
xjw::feature_match::MatchResult filterMatchResultByMasks(
    const xjw::feature_match::MatchResult &matchResult,
    const xjw::feature_extractors::FeatureData &feature0,
    const xjw::feature_extractors::FeatureData &feature1,
    const cv::Mat &mask0,
    const cv::Mat &mask1);

QString maskPathForImage(const MatchPhotosContext &context, const QString &imagePath);
cv::Mat loadMaskForImage(const MatchPhotosContext &context,
                         const QString &imagePath,
                         const cv::Size &imageSize);

} // namespace matchphotos
} // namespace xjw
