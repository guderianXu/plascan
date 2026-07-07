#include "TiePointTrackManager.h"

// Avoid Qt keyword macros rewriting LibTorch's slots() member name.
#ifdef slots
#undef slots
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef signals
#undef signals
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef emit
#undef emit
#define PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureData.h"

#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#define slots Q_SLOTS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SLOTS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#define signals Q_SIGNALS
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_SIGNALS
#endif
#ifdef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#define emit Q_EMIT
#undef PLASCAN_MATCHPHOTOS_RESTORE_QT_EMIT
#endif

#include "FeatureFileIO.h"
#include "MatchFileIO.h"
#include "MatchPhotosRuntime.h"
#include "tracks/MultiViewTrackBuilder.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>

#include <algorithm>
#include <cmath>
#include <map>

namespace xjw
{
namespace matchphotos
{
namespace
{

QString canonicalPath(const QString &path)
{
    const QString trimmed = path.trimmed();
    if (trimmed.isEmpty())
    {
        return QString();
    }
    return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
}

std::vector<FeatureKeypoint> toSfmKeypoints(const std::vector<cv::KeyPoint> &keypoints)
{
    std::vector<FeatureKeypoint> converted;
    converted.reserve(keypoints.size());
    for (const cv::KeyPoint &keypoint : keypoints)
    {
        converted.push_back(FeatureKeypoint{keypoint.pt.x, keypoint.pt.y});
    }
    return converted;
}

float matchScoreAt(const xjw::feature_match::MatchResult &matchResult, int index0, float distance)
{
    if (index0 >= 0 && index0 < static_cast<int>(matchResult.matchingScores0.size()))
    {
        return matchResult.matchingScores0[static_cast<std::size_t>(index0)];
    }
    if (std::isfinite(distance))
    {
        return 1.0f / (1.0f + std::max(0.0f, distance));
    }
    return 1.0f;
}

QJsonObject makeTrackSummary(const MultiViewTrackBuildResult &buildResult)
{
    QJsonObject summary;
    summary[QStringLiteral("tracks")] = static_cast<int>(buildResult.tracks.size());
    summary[QStringLiteral("total_components")] = buildResult.totalComponents;
    summary[QStringLiteral("accepted_components")] = buildResult.acceptedComponents;
    summary[QStringLiteral("rejected_conflict_components")] = buildResult.rejectedConflictComponents;
    summary[QStringLiteral("rejected_conflict_edges")] = buildResult.rejectedConflictEdges;
    summary[QStringLiteral("pruned_by_quality_thinning")] = buildResult.prunedByQualityThinning;
    summary[QStringLiteral("pruned_stationary_tracks")] = buildResult.prunedStationaryTracks;
    summary[QStringLiteral("mean_track_confidence")] = buildResult.meanTrackConfidence;

    QJsonObject histogram;
    for (const auto &entry : buildResult.trackLengthHistogram)
    {
        histogram[QString::number(entry.first)] = entry.second;
    }
    summary[QStringLiteral("track_length_histogram")] = histogram;
    return summary;
}

QJsonObject makePersistedSettings(const MatchPhotosOptions &options)
{
    QJsonObject settings;
    settings[QStringLiteral("keypoint_limit")] = options.maxKeypoints;
    settings[QStringLiteral("keypoint_limit_per_mpx")] = options.keypointLimitPerMegapixel;
    settings[QStringLiteral("tiepoint_limit")] = options.maxTiePointsPerImage;
    settings[QStringLiteral("tiepoint_grid_columns")] = options.tiePointGridColumns;
    settings[QStringLiteral("tiepoint_grid_rows")] = options.tiePointGridRows;
    settings[QStringLiteral("exclude_stationary_tie_points")] = options.excludeStationaryTiePoints;
    settings[QStringLiteral("stationary_tie_point_max_pixel_motion")] =
        static_cast<double>(options.stationaryTiePointMaxPixelMotion);
    settings[QStringLiteral("guided_image_matching")] = options.enableGuidedMatching;
    settings[QStringLiteral("generic_preselection")] = options.useGenericPreselection;
    settings[QStringLiteral("reference_preselection")] = options.useReferencePreselection;
    return settings;
}

QString tiePointOutputPath(const MatchPhotosContext &context)
{
    const QString assetsDir = context.workingDirectory.trimmed();
    if (assetsDir.isEmpty())
    {
        return QString();
    }

    return QDir(QDir(assetsDir).filePath(QStringLiteral("tie_points")))
        .filePath(QStringLiteral("latest_tie_points.json"));
}

QByteArray jsonString(const QString &value)
{
    QJsonArray wrapper;
    wrapper.append(value);
    const QByteArray encoded = QJsonDocument(wrapper).toJson(QJsonDocument::Compact);
    return encoded.size() >= 2 ? encoded.mid(1, encoded.size() - 2) : QByteArray("\"\"");
}

QByteArray jsonNumber(double value)
{
    return QByteArray::number(value, 'g', 12);
}

QByteArray compactJson(const QJsonObject &object)
{
    return QJsonDocument(object).toJson(QJsonDocument::Compact);
}

bool writeRaw(QSaveFile *file, const QByteArray &data, QString *errorMessage)
{
    if (!file)
    {
        return false;
    }

    if (file->write(data) != static_cast<qint64>(data.size()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("写入连接点文件失败");
        }
        return false;
    }
    return true;
}

bool writeField(QSaveFile *file,
                const QByteArray &name,
                const QByteArray &value,
                bool trailingComma,
                QString *errorMessage)
{
    QByteArray line = QByteArrayLiteral("  \"") + name + QByteArrayLiteral("\": ") + value;
    line += trailingComma ? QByteArrayLiteral(",\n") : QByteArrayLiteral("\n");
    return writeRaw(file, line, errorMessage);
}

bool writeImages(QSaveFile *file, const MatchPhotosContext &context, QString *errorMessage)
{
    if (!writeRaw(file, QByteArrayLiteral("  \"images\": [\n"), errorMessage))
    {
        return false;
    }

    for (int index = 0; index < context.pairInput.images.size(); ++index)
    {
        QByteArray line = QByteArrayLiteral("    {\"image_id\":") + QByteArray::number(index) +
            QByteArrayLiteral(",\"path\":") + jsonString(canonicalPath(context.pairInput.images.at(index))) +
            QByteArrayLiteral("}");
        line += index + 1 < context.pairInput.images.size()
            ? QByteArrayLiteral(",\n")
            : QByteArrayLiteral("\n");
        if (!writeRaw(file, line, errorMessage))
        {
            return false;
        }
    }

    return writeRaw(file, QByteArrayLiteral("  ],\n"), errorMessage);
}

QByteArray observationJson(
    const TrackElement &element,
    const MatchPhotosContext &context,
    const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage)
{
    QByteArray object = QByteArrayLiteral("{\"image_id\":") +
        QByteArray::number(static_cast<int>(element.imageId)) +
        QByteArrayLiteral(",\"feature_idx\":") +
        QByteArray::number(static_cast<int>(element.featureIdx));

    if (element.imageId < static_cast<ImageId>(context.pairInput.images.size()))
    {
        object += QByteArrayLiteral(",\"image_path\":") +
            jsonString(canonicalPath(context.pairInput.images.at(static_cast<int>(element.imageId))));
    }

    const auto keypointsIt = keypointsByImage.find(element.imageId);
    if (keypointsIt != keypointsByImage.end() &&
        element.featureIdx < static_cast<FeatureIdx>(keypointsIt->second.size()))
    {
        const FeatureKeypoint &keypoint = keypointsIt->second[static_cast<std::size_t>(element.featureIdx)];
        object += QByteArrayLiteral(",\"xy\":[") + jsonNumber(keypoint.x) +
            QByteArrayLiteral(",") + jsonNumber(keypoint.y) + QByteArrayLiteral("]");
    }

    object += QByteArrayLiteral("}");
    return object;
}

bool writeTrack(QSaveFile *file,
                const Track &track,
                int trackId,
                const MatchPhotosContext &context,
                const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage,
                bool trailingComma,
                QString *errorMessage)
{
    QByteArray header = QByteArrayLiteral("    {\"track_id\":") + QByteArray::number(trackId) +
        QByteArrayLiteral(",\"track_len\":") + QByteArray::number(static_cast<int>(track.length())) +
        QByteArrayLiteral(",\"confidence\":") + jsonNumber(track.confidence) +
        QByteArrayLiteral(",\"observations\":[");
    if (!writeRaw(file, header, errorMessage))
    {
        return false;
    }

    for (std::size_t index = 0; index < track.elements.size(); ++index)
    {
        QByteArray observation = observationJson(track.elements[index], context, keypointsByImage);
        if (index + 1 < track.elements.size())
        {
            observation += QByteArrayLiteral(",");
        }
        if (!writeRaw(file, observation, errorMessage))
        {
            return false;
        }
    }

    QByteArray footer = QByteArrayLiteral("]}");
    footer += trailingComma ? QByteArrayLiteral(",\n") : QByteArrayLiteral("\n");
    return writeRaw(file, footer, errorMessage);
}

bool writeTracks(QSaveFile *file,
                 const MatchPhotosContext &context,
                 const TiePointTrackBuildResult &result,
                 const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage,
                 QString *errorMessage)
{
    if (!writeRaw(file, QByteArrayLiteral("  \"tracks\": [\n"), errorMessage))
    {
        return false;
    }

    for (std::size_t index = 0; index < result.tracks.size(); ++index)
    {
        if (!writeTrack(file,
                        result.tracks[index],
                        static_cast<int>(index),
                        context,
                        keypointsByImage,
                        index + 1 < result.tracks.size(),
                        errorMessage))
        {
            return false;
        }
    }

    return writeRaw(file, QByteArrayLiteral("  ]\n"), errorMessage);
}

bool writeTiePointFile(
    const QString &path,
    const MatchPhotosContext &context,
    const MatchPhotosOptions &options,
    const TiePointTrackBuildResult &result,
    const std::map<ImageId, std::vector<FeatureKeypoint>> &keypointsByImage,
    QString *errorMessage)
{
    if (path.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("连接点输出路径为空");
        }
        return false;
    }

    const QFileInfo info(path);
    if (!QDir().mkpath(info.absolutePath()))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法创建连接点输出目录: %1").arg(info.absolutePath());
        }
        return false;
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("无法写入连接点文件: %1").arg(path);
        }
        return false;
    }

    if (!writeRaw(&file, QByteArrayLiteral("{\n"), errorMessage) ||
        !writeField(&file, QByteArrayLiteral("format"), jsonString(QStringLiteral("plascan_tie_points")), true,
                    errorMessage) ||
        !writeField(&file, QByteArrayLiteral("format_version"), QByteArrayLiteral("1"), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("created_at"),
                    jsonString(QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("track_count"),
                    QByteArray::number(static_cast<int>(result.tracks.size())), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("consumed_pair_count"),
                    QByteArray::number(result.consumedPairCount), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("skipped_pair_count"),
                    QByteArray::number(result.skippedPairCount), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("summary"), compactJson(result.trackSummary), true, errorMessage) ||
        !writeField(&file, QByteArrayLiteral("settings"), compactJson(makePersistedSettings(options)), true,
                    errorMessage) ||
        !writeImages(&file, context, errorMessage) ||
        !writeTracks(&file, context, result, keypointsByImage, errorMessage) ||
        !writeRaw(&file, QByteArrayLiteral("}\n"), errorMessage))
    {
        return false;
    }

    if (!file.commit())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("提交连接点文件失败: %1").arg(path);
        }
        return false;
    }
    return true;
}

MultiViewTrackBuilder::BuildOptions makeBuildOptions(const MatchPhotosOptions &options,
                                                     float imageWidth,
                                                     float imageHeight)
{
    MultiViewTrackBuilder::BuildOptions buildOptions;
    buildOptions.enableQualityThinning = options.maxTiePointsPerImage > 0;
    buildOptions.maxTracksPerImage = options.maxTiePointsPerImage;
    buildOptions.maxTracksPerGridCell = options.maxTiePointsPerImage > 0
        ? options.maxTiePointsPerGridCell
        : 0;
    buildOptions.gridColumns = options.tiePointGridColumns;
    buildOptions.gridRows = options.tiePointGridRows;
    buildOptions.imageWidth = imageWidth;
    buildOptions.imageHeight = imageHeight;
    buildOptions.excludeStationaryTracks = options.excludeStationaryTiePoints;
    buildOptions.stationaryTrackMaxPixelMotion = options.stationaryTiePointMaxPixelMotion;
    return buildOptions;
}

void copyBuildResult(const MultiViewTrackBuildResult &buildResult, TiePointTrackBuildResult *result)
{
    if (!result)
    {
        return;
    }

    result->tracks = buildResult.tracks;
    result->totalComponents = buildResult.totalComponents;
    result->acceptedComponents = buildResult.acceptedComponents;
    result->rejectedConflictComponents = buildResult.rejectedConflictComponents;
    result->rejectedConflictEdges = buildResult.rejectedConflictEdges;
    result->prunedByQualityThinning = buildResult.prunedByQualityThinning;
    result->prunedStationaryTracks = buildResult.prunedStationaryTracks;
    result->meanTrackConfidence = buildResult.meanTrackConfidence;
    result->trackSummary = makeTrackSummary(buildResult);
}

bool appendInlierIndexPairs(const MatchPhotosMatchRecord &record,
                            int keypointCount0,
                            int keypointCount1,
                            std::vector<MultiViewTrackBuilder::MatchIndexPair> *indexedMatches)
{
    if (!indexedMatches || record.inlierIndexPairs.empty())
    {
        return false;
    }

    indexedMatches->reserve(record.inlierIndexPairs.size());
    for (const std::array<int, 2> &pair : record.inlierIndexPairs)
    {
        if (pair[0] < 0 || pair[1] < 0 || pair[0] >= keypointCount0 || pair[1] >= keypointCount1)
        {
            continue;
        }
        indexedMatches->emplace_back(static_cast<FeatureIdx>(pair[0]),
                                     static_cast<FeatureIdx>(pair[1]),
                                     1.0f);
    }
    return true;
}

void appendRawIndexedMatches(const QString &matchPath,
                             int keypointCount0,
                             int keypointCount1,
                             std::vector<MultiViewTrackBuilder::MatchIndexPair> *indexedMatches)
{
    if (!indexedMatches)
    {
        return;
    }

    QString image0Name;
    QString image1Name;
    xjw::feature_match::MatchResult matchResult;
    if (!xjw::feature_match::readIndexedMatchFile(matchPath, image0Name, image1Name, matchResult))
    {
        return;
    }

    indexedMatches->reserve(matchResult.cvMatches.size());
    for (const cv::DMatch &match : matchResult.cvMatches)
    {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= keypointCount0 || match.trainIdx >= keypointCount1)
        {
            continue;
        }
        indexedMatches->emplace_back(static_cast<FeatureIdx>(match.queryIdx),
                                     static_cast<FeatureIdx>(match.trainIdx),
                                     matchScoreAt(matchResult, match.queryIdx, match.distance));
    }
}

} // namespace

TiePointTrackBuildResult TiePointTrackManager::build(const MatchPhotosContext &context,
                                                     const MatchPhotosOptions &options,
                                                     const std::vector<MatchPhotosMatchRecord> &matchRecords) const
{
    TiePointTrackBuildResult result;
    result.success = true;

    std::map<QString, ImageId> imageIdByPath;
    for (int index = 0; index < context.pairInput.images.size(); ++index)
    {
        imageIdByPath[canonicalPath(context.pairInput.images.at(index))] = static_cast<ImageId>(index);
    }

    MultiViewTrackBuilder builder;
    std::map<ImageId, bool> keypointsLoaded;
    std::map<ImageId, std::vector<FeatureKeypoint>> keypointsByImage;
    float imageWidth = 0.0f;
    float imageHeight = 0.0f;

    for (const MatchPhotosMatchRecord &record : matchRecords)
    {
        if (shouldCancelMatchPhotos(context))
        {
            result.success = false;
            result.errorMessage = QStringLiteral("用户取消连接点轨迹构建");
            return result;
        }

        if (options.enableGeometryVerification && !record.passedGeometry)
        {
            ++result.skippedPairCount;
            continue;
        }

        const QString image0Path = canonicalPath(record.image0Path);
        const QString image1Path = canonicalPath(record.image1Path);
        const auto image0It = imageIdByPath.find(image0Path);
        const auto image1It = imageIdByPath.find(image1Path);
        if (image0It == imageIdByPath.end() || image1It == imageIdByPath.end())
        {
            ++result.skippedPairCount;
            continue;
        }

        const ImageId image0Id = image0It->second;
        const ImageId image1Id = image1It->second;
        const QString feature0Path = record.settings.value(QStringLiteral("feature0_path")).toString();
        const QString feature1Path = record.settings.value(QStringLiteral("feature1_path")).toString();
        QString featureImageName0;
        QString featureImageName1;
        xjw::feature_extractors::FeatureData feature0;
        xjw::feature_extractors::FeatureData feature1;
        if (feature0Path.isEmpty() ||
            feature1Path.isEmpty() ||
            !FeatureFileIO::readData(feature0Path, featureImageName0, feature0) ||
            !FeatureFileIO::readData(feature1Path, featureImageName1, feature1))
        {
            ++result.skippedPairCount;
            continue;
        }

        if (!keypointsLoaded[image0Id])
        {
            keypointsByImage[image0Id] = toSfmKeypoints(feature0.keypoints);
            builder.setImageKeypoints(image0Id, keypointsByImage[image0Id]);
            keypointsLoaded[image0Id] = true;
        }
        if (!keypointsLoaded[image1Id])
        {
            keypointsByImage[image1Id] = toSfmKeypoints(feature1.keypoints);
            builder.setImageKeypoints(image1Id, keypointsByImage[image1Id]);
            keypointsLoaded[image1Id] = true;
        }
        imageWidth = std::max(imageWidth, static_cast<float>(std::max(feature0.imageWidth, feature1.imageWidth)));
        imageHeight = std::max(imageHeight, static_cast<float>(std::max(feature0.imageHeight, feature1.imageHeight)));

        std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
        if (!appendInlierIndexPairs(record,
                                    static_cast<int>(feature0.keypoints.size()),
                                    static_cast<int>(feature1.keypoints.size()),
                                    &indexedMatches) &&
            !options.enableGeometryVerification)
        {
            appendRawIndexedMatches(record.matchPath,
                                    static_cast<int>(feature0.keypoints.size()),
                                    static_cast<int>(feature1.keypoints.size()),
                                    &indexedMatches);
        }

        if (indexedMatches.empty())
        {
            ++result.skippedPairCount;
            continue;
        }

        builder.addMatchPair(image0Id, image1Id, indexedMatches);
        ++result.consumedPairCount;
    }

    const MultiViewTrackBuildResult buildResult =
        builder.build(makeBuildOptions(options, imageWidth, imageHeight));
    copyBuildResult(buildResult, &result);
    if (result.consumedPairCount <= 0)
    {
        result.success = false;
        result.errorMessage = QStringLiteral("未消费任何有效匹配对，无法生成连接点轨迹");
        return result;
    }
    if (result.tracks.empty())
    {
        result.success = false;
        result.errorMessage = QStringLiteral("未生成可用连接点轨迹");
        return result;
    }

    const QString outputPath = tiePointOutputPath(context);
    if (!outputPath.isEmpty())
    {
        QString writeError;
        if (!writeTiePointFile(outputPath, context, options, result, keypointsByImage, &writeError))
        {
            result.success = false;
            result.errorMessage = writeError;
            return result;
        }
        result.tiePointPath = outputPath;
    }

    return result;
}

} // namespace matchphotos
} // namespace xjw
