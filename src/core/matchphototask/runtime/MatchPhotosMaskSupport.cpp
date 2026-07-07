#include "MatchPhotosMaskSupport.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xjw
{
namespace matchphotos
{
namespace
{

QString normalizedToken(const QString &token)
{
    QString value = QDir::cleanPath(QDir::fromNativeSeparators(token.trimmed()));
#if defined(Q_OS_WIN)
    value = value.toLower();
#endif
    return value;
}

bool imageTokensReferToSameImage(const QString &lhs, const QString &rhs)
{
    const QString left = lhs.trimmed();
    const QString right = rhs.trimmed();
    if (left.isEmpty() || right.isEmpty())
    {
        return false;
    }

    if (normalizedToken(left) == normalizedToken(right))
    {
        return true;
    }

    const QFileInfo leftInfo(left);
    const QFileInfo rightInfo(right);
    if (!leftInfo.fileName().isEmpty() &&
        leftInfo.fileName().compare(rightInfo.fileName(), Qt::CaseInsensitive) == 0)
    {
        return true;
    }

    return !leftInfo.completeBaseName().isEmpty() &&
           leftInfo.completeBaseName().compare(rightInfo.completeBaseName(), Qt::CaseInsensitive) == 0;
}

int descriptorRowCount(const torch::Tensor &descriptors)
{
    if (!descriptors.defined() || descriptors.dim() != 2)
    {
        return 0;
    }
    return static_cast<int>(descriptors.size(0));
}

} // namespace

MatchPhotosMaskApplyMode maskApplyModeFromToken(const QString &token)
{
    const QString value = token.trimmed().toLower();
    if (value == QStringLiteral("keypoints") || value == QStringLiteral("keypoint"))
    {
        return MatchPhotosMaskApplyMode::Keypoints;
    }
    if (value == QStringLiteral("tiepoints") ||
        value == QStringLiteral("tie_points") ||
        value == QStringLiteral("matches"))
    {
        return MatchPhotosMaskApplyMode::Tiepoints;
    }
    return MatchPhotosMaskApplyMode::None;
}

bool shouldApplyMasksToKeypoints(const MatchPhotosOptions &options)
{
    return maskApplyModeFromToken(options.maskApplyMode) == MatchPhotosMaskApplyMode::Keypoints;
}

bool shouldApplyMasksToTiepoints(const MatchPhotosOptions &options)
{
    return maskApplyModeFromToken(options.maskApplyMode) == MatchPhotosMaskApplyMode::Tiepoints;
}

cv::Mat normalizedMaskForImage(const cv::Mat &mask, const cv::Size &imageSize)
{
    if (mask.empty())
    {
        return cv::Mat();
    }

    cv::Mat grayMask;
    if (mask.channels() == 1)
    {
        grayMask = mask;
    }
    else
    {
        cv::cvtColor(mask,
                     grayMask,
                     mask.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
    }

    if (grayMask.depth() != CV_8U)
    {
        cv::Mat converted;
        grayMask.convertTo(converted, CV_8U);
        grayMask = converted;
    }

    if (imageSize.width > 0 &&
        imageSize.height > 0 &&
        grayMask.size() != imageSize)
    {
        cv::Mat resized;
        cv::resize(grayMask, resized, imageSize, 0.0, 0.0, cv::INTER_NEAREST);
        return resized;
    }

    return grayMask.clone();
}

bool isPointAllowedByMask(const cv::Mat &mask, const cv::Point2f &point)
{
    if (mask.empty())
    {
        return true;
    }

    const int x = static_cast<int>(std::lround(point.x));
    const int y = static_cast<int>(std::lround(point.y));
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows)
    {
        return false;
    }

    return mask.at<uchar>(y, x) == 0;
}

FeatureOutput filterFeatureOutputByMask(const FeatureOutput &output, const cv::Mat &mask)
{
    if (mask.empty() || output.keypoints.empty())
    {
        return output;
    }

    FeatureOutput filtered;
    filtered.imageWidth = output.imageWidth;
    filtered.imageHeight = output.imageHeight;
    filtered.keypoints.reserve(output.keypoints.size());
    filtered.scores.reserve(output.scores.size());

    std::vector<int64_t> keptIndices;
    keptIndices.reserve(output.keypoints.size());
    for (std::size_t index = 0; index < output.keypoints.size(); ++index)
    {
        const cv::KeyPoint &keypoint = output.keypoints[index];
        if (!isPointAllowedByMask(mask, keypoint.pt))
        {
            continue;
        }

        filtered.keypoints.push_back(keypoint);
        if (index < output.scores.size())
        {
            filtered.scores.push_back(output.scores[index]);
        }
        keptIndices.push_back(static_cast<int64_t>(index));
    }

    if (output.scores.size() < output.keypoints.size())
    {
        filtered.scores.resize(filtered.keypoints.size(), 0.0f);
    }

    const int descriptorRows = descriptorRowCount(output.descriptors);
    if (descriptorRows >= static_cast<int>(output.keypoints.size()) && !keptIndices.empty())
    {
        const torch::Tensor indices =
            torch::from_blob(keptIndices.data(),
                             {static_cast<int64_t>(keptIndices.size())},
                             torch::kInt64)
                .clone();
        filtered.descriptors = output.descriptors.to(torch::kCPU).index_select(0, indices).contiguous();
    }
    else if (descriptorRows > 0 && keptIndices.empty())
    {
        filtered.descriptors =
            torch::empty({0, output.descriptors.size(1)}, output.descriptors.options().device(torch::kCPU));
    }

    return filtered;
}

xjw::feature_match::MatchResult filterMatchResultByMasks(
    const xjw::feature_match::MatchResult &matchResult,
    const xjw::feature_extractors::FeatureData &feature0,
    const xjw::feature_extractors::FeatureData &feature1,
    const cv::Mat &mask0,
    const cv::Mat &mask1)
{
    if (mask0.empty() && mask1.empty())
    {
        return matchResult;
    }

    xjw::feature_match::MatchResult normalized = matchResult;
    if (normalized.cvMatches.empty() && !normalized.matches0.empty())
    {
        normalized.buildCvMatchesFromIndices();
    }

    std::vector<cv::DMatch> keptMatches;
    keptMatches.reserve(normalized.cvMatches.size());
    for (const cv::DMatch &match : normalized.cvMatches)
    {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= feature0.size() ||
            match.trainIdx >= feature1.size())
        {
            continue;
        }

        const cv::KeyPoint &keypoint0 = feature0.keypoints[static_cast<std::size_t>(match.queryIdx)];
        const cv::KeyPoint &keypoint1 = feature1.keypoints[static_cast<std::size_t>(match.trainIdx)];
        if (!isPointAllowedByMask(mask0, keypoint0.pt) ||
            !isPointAllowedByMask(mask1, keypoint1.pt))
        {
            continue;
        }
        keptMatches.push_back(match);
    }

    xjw::feature_match::MatchResult filtered;
    filtered.cvMatches = std::move(keptMatches);
    filtered.sourceAlgorithm = matchResult.sourceAlgorithm;
    filtered.buildIndicesFromCvMatches(feature0.size(), feature1.size());
    return filtered;
}

QString maskPathForImage(const MatchPhotosContext &context, const QString &imagePath)
{
    if (imagePath.trimmed().isEmpty())
    {
        return QString();
    }

    const QString direct = context.maskPaths.value(imagePath).trimmed();
    if (!direct.isEmpty())
    {
        return direct;
    }

    const QString cleanImage = normalizedToken(imagePath);
    const QString clean = context.maskPaths.value(cleanImage).trimmed();
    if (!clean.isEmpty())
    {
        return clean;
    }

    for (auto it = context.maskPaths.constBegin(); it != context.maskPaths.constEnd(); ++it)
    {
        if (it.value().trimmed().isEmpty())
        {
            continue;
        }
        if (imageTokensReferToSameImage(it.key(), imagePath))
        {
            return it.value();
        }
    }

    return QString();
}

cv::Mat loadMaskForImage(const MatchPhotosContext &context,
                         const QString &imagePath,
                         const cv::Size &imageSize)
{
    const QString maskPath = maskPathForImage(context, imagePath);
    if (maskPath.isEmpty())
    {
        return cv::Mat();
    }

    const cv::Mat mask = xjw::common::io::readImage(maskPath, cv::IMREAD_GRAYSCALE);
    return normalizedMaskForImage(mask, imageSize);
}

} // namespace matchphotos
} // namespace xjw
