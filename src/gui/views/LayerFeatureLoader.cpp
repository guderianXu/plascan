#include "LayerFeatureLoader.h"

#include "ImageMatchFile.h"
#include "project/ProjectIO.h"

#include <QDebug>
#include <QFileInfo>
#include <QSet>

#include <algorithm>
#include <limits>

namespace xjw::gui::views
{

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

    std::size_t observation_count = 0;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        observation_count += block.ownerObservations.size();
    }

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(observation_count);
    QSet<QString> seen_observations;
    for (const xjw::image_matching::NeighborMatchBlock &block : shard.neighbors)
    {
        // 同一算法变体的一个特征可能出现在多个邻接块中。显示层按“算法版本、
        // 配置指纹、特征编号”去重，但不合并不同算法变体中碰巧相同的编号。
        const QString variant_prefix = block.algorithmId.toLower() + QLatin1Char('\n') +
            QString::number(block.algorithmVersion) + QLatin1Char('\n') +
            QString::fromLatin1(block.configFingerprint.toHex()) + QLatin1Char('\n');
        for (const xjw::image_matching::KeypointObservation &observation :
             block.ownerObservations)
        {
            const QString observation_key =
                variant_prefix + QString::number(observation.featureId);
            if (seen_observations.contains(observation_key))
            {
                continue;
            }
            seen_observations.insert(observation_key);

            // OpenCV 要求关键点尺度为正数；损坏数据中的非正值按 1 像素显示。
            const float size = std::max(1.0f, observation.scale);
            const std::uint32_t max_class_id =
                static_cast<std::uint32_t>(std::numeric_limits<int>::max());
            const int class_id = observation.featureId <= max_class_id
                ? static_cast<int>(observation.featureId)
                : -1;
            keypoints.emplace_back(observation.x,
                                   observation.y,
                                   size,
                                   observation.orientation,
                                   observation.response,
                                   0,
                                   class_id);
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

} // namespace xjw::gui::views
