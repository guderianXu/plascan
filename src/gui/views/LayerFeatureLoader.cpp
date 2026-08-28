#include "LayerFeatureLoader.h"

#include "FeatureResidualLoader.h"
#include "ImageFeaturePointFile.h"
#include "ImageMatchFile.h"
#include "project/ProjectIO.h"

#include <QDebug>
#include <QFileInfo>
#include <QSet>

#include <algorithm>
#include <limits>

namespace xjw::gui::views
{

namespace
{

cv::KeyPoint keypointFromObservation(
    const xjw::image_matching::KeypointObservation &observation)
{
    const float size = std::max(1.0f, observation.scale);
    const std::uint32_t maximumClassId =
        static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    const int classId = observation.featureId <= maximumClassId
        ? static_cast<int>(observation.featureId)
        : -1;
    return cv::KeyPoint(observation.x,
                        observation.y,
                        size,
                        observation.orientation,
                        observation.response,
                        0,
                        classId);
}

std::vector<cv::KeyPoint> keypointsFromCatalog(
    const xjw::image_matching::ImageFeaturePointCatalog &catalog)
{
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(catalog.observations.size());
    for (const xjw::image_matching::KeypointObservation &observation : catalog.observations)
    {
        keypoints.push_back(keypointFromObservation(observation));
    }
    return keypoints;
}

} // namespace

QString featurePointSourceToken(FeaturePointSource source)
{
    switch (source)
    {
    case FeaturePointSource::ExtractedFeatures:
        return QStringLiteral("extracted_features");
    case FeaturePointSource::RawMatches:
        return QStringLiteral("raw_matches");
    case FeaturePointSource::ValidTiePoints:
        return QStringLiteral("valid_tie_points");
    }
    return QStringLiteral("valid_tie_points");
}

FeaturePointSource featurePointSourceFromToken(const QString &token)
{
    const QString normalized = token.trimmed().toLower();
    if (normalized == QLatin1String("extracted_features"))
    {
        return FeaturePointSource::ExtractedFeatures;
    }
    if (normalized == QLatin1String("raw_matches"))
    {
        return FeaturePointSource::RawMatches;
    }
    return FeaturePointSource::ValidTiePoints;
}

QString featurePointSourceDisplayName(FeaturePointSource source)
{
    switch (source)
    {
    case FeaturePointSource::ExtractedFeatures:
        return QStringLiteral("提取特征点");
    case FeaturePointSource::RawMatches:
        return QStringLiteral("原始匹配点");
    case FeaturePointSource::ValidTiePoints:
        return QStringLiteral("有效连接点");
    }
    return QStringLiteral("有效连接点");
}

std::vector<cv::KeyPoint> loadMatchedKeypointsFromFile(const QString &matchFilePath)
{
    if (matchFilePath.trimmed().isEmpty())
    {
        return {};
    }
    if (!QFileInfo::exists(matchFilePath))
    {
        return {};
    }

    xjw::image_matching::ImageMatchShard shard;
    QString error_message;
    if (!xjw::image_matching::ImageMatchFile::read(matchFilePath, &shard, &error_message))
    {
        qWarning() << "读取影像匹配分片失败:" << matchFilePath << error_message;
        return {};
    }

    std::size_t observationCount = 0;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        observationCount += block.matches.size();
    }

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(observationCount);
    QSet<QString> seenObservations;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        // 同一算法变体的一个特征可能出现在多个邻接块中。显示层按“算法版本、
        // 配置指纹、特征编号”去重，但不合并不同算法变体中碰巧相同的编号。
        const QString variantPrefix = block.algorithmId.toLower() + QLatin1Char('\n') +
            QString::number(block.algorithmVersion) + QLatin1Char('\n') +
            QString::fromLatin1(block.configFingerprint.toHex()) + QLatin1Char('\n');
        for (const xjw::image_matching::MatchRecord &match : block.matches)
        {
            const QString observationKey =
                variantPrefix + QString::number(match.ownerFeatureId);
            if (seenObservations.contains(observationKey))
            {
                continue;
            }
            const xjw::image_matching::KeypointObservation *observation =
                block.findOwnerObservation(match.ownerFeatureId);
            if (!observation)
            {
                continue;
            }
            seenObservations.insert(observationKey);
            keypoints.push_back(keypointFromObservation(*observation));
        }
    }

    return keypoints;
}

std::vector<cv::KeyPoint> loadMatchedKeypointsForImage(const QString &plascanPath,
                                                       const QString &imagePath)
{
    const QString match_directory =
        xjw::common::project::ProjectIO::imageMatchOutputDir(plascanPath);
    const QString match_file_path =
        xjw::image_matching::ImageMatchFile::filePathForImage(match_directory, imagePath);
    if (match_file_path.isEmpty())
    {
        qDebug() << "未能为影像计算匹配分片路径:" << imagePath;
        return {};
    }

    return loadMatchedKeypointsFromFile(match_file_path);
}

FeaturePointLoadResult loadFeaturePointsForImage(const QString &plascanPath,
                                                 const QString &imagePath,
                                                 FeaturePointSource source)
{
    FeaturePointLoadResult result;
    if (plascanPath.trimmed().isEmpty() || imagePath.trimmed().isEmpty())
    {
        result.message = QStringLiteral("当前项目或照片路径为空");
        return result;
    }

    if (source == FeaturePointSource::ValidTiePoints)
    {
        ValidTiePointDiagnostics diagnostics =
            loadValidTiePointDiagnosticsForImage(plascanPath, imagePath);
        result.keypoints = std::move(diagnostics.keypoints);
        result.sourcePath = diagnostics.sidecarPath;
        result.message = diagnostics.message;
        result.available = diagnostics.available;
        return result;
    }

    const QString matchDirectory =
        xjw::common::project::ProjectIO::imageMatchOutputDir(plascanPath);
    if (source == FeaturePointSource::ExtractedFeatures)
    {
        result.sourcePath = xjw::image_matching::ImageFeaturePointFile::filePathForImage(
            matchDirectory, imagePath);
        if (!QFileInfo::exists(result.sourcePath))
        {
            result.message =
                QStringLiteral("当前照片没有提取特征点数据；请重新运行影像匹配后查看");
            return result;
        }
        xjw::image_matching::ImageFeaturePointCatalog catalog;
        QString errorMessage;
        if (!xjw::image_matching::ImageFeaturePointFile::read(
                result.sourcePath, &catalog, &errorMessage))
        {
            result.message = errorMessage;
            return result;
        }
        result.keypoints = keypointsFromCatalog(catalog);
    }
    else
    {
        result.sourcePath = xjw::image_matching::ImageMatchFile::filePathForImage(
            matchDirectory, imagePath);
        if (!QFileInfo::exists(result.sourcePath))
        {
            result.message = QStringLiteral("当前照片没有原始匹配分片");
            return result;
        }
        result.keypoints = loadMatchedKeypointsFromFile(result.sourcePath);
    }

    result.available = !result.keypoints.empty();
    result.message = result.available
        ? QStringLiteral("%1 %2 个")
              .arg(featurePointSourceDisplayName(source))
              .arg(result.keypoints.size())
        : QStringLiteral("当前照片没有可显示的%1").arg(featurePointSourceDisplayName(source));
    return result;
}

} // namespace xjw::gui::views
