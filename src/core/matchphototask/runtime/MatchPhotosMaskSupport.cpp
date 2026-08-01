#include "MatchPhotosMaskSupport.h"

#include "io/PathIO.h"

#include <QDir>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

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
