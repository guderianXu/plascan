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
        cv::resize(grayMask, resized, imageSize, 0.0, 0.0, cv::INTER_LINEAR);
        return resized;
    }

    return grayMask.clone();
}

cv::Mat softenedExclusionMask(const cv::Mat &mask,
                              const MatchPhotosOptions &options)
{
    if (mask.empty())
    {
        return {};
    }
    cv::Mat softened = mask;
    const int radius = std::clamp(options.maskRelaxationRadius, 0, 32);
    if (radius > 0)
    {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
        cv::erode(mask, softened, kernel);
    }
    return softened;
}

cv::Mat makeExtractorValidMask(const cv::Mat &mask,
                               const cv::Size &extractorSize,
                               const MatchPhotosOptions &options)
{
    if (mask.empty())
    {
        return {};
    }
    cv::Mat resized;
    cv::resize(mask, resized, extractorSize, 0.0, 0.0, cv::INTER_LINEAR);
    const cv::Mat softened = softenedExclusionMask(resized, options);
    cv::Mat valid;
    const double threshold = 255.0 * std::clamp(options.maskHardExclusionThreshold, 0.0f, 1.0f);
    cv::compare(softened, cv::Scalar(threshold), valid, cv::CMP_LT);
    return valid;
}

float maskPointWeight(const cv::Mat &mask,
                      const cv::Point2f &point,
                      const MatchPhotosOptions &options)
{
    if (mask.empty())
    {
        return 1.0f;
    }

    const int x = static_cast<int>(std::lround(point.x));
    const int y = static_cast<int>(std::lround(point.y));
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows)
    {
        return 0.0f;
    }
    const float exclusion = static_cast<float>(mask.at<uchar>(y, x)) / 255.0f;
    if (exclusion >= std::clamp(options.maskHardExclusionThreshold, 0.0f, 1.0f))
    {
        return 0.0f;
    }
    return std::clamp(1.0f - exclusion,
                      std::clamp(options.maskMinimumTiepointWeight, 0.0f, 1.0f),
                      1.0f);
}

bool isPointAllowedByMask(const cv::Mat &mask,
                          const cv::Point2f &point,
                          const MatchPhotosOptions &options)
{
    return maskPointWeight(mask, point, options) > 0.0f;
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
