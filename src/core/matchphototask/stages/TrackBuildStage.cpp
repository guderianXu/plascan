#include "TrackBuildStage.h"

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

#include <QDir>
#include <QJsonObject>

#include <algorithm>
#include <cmath>
#include <map>

namespace xjw
{
namespace matchphotos
{
namespace
{

MatchPhotosStageReport makeTrackReport(MatchPhotosStageStatus status,
                                       const QString &message,
                                       int itemCount = 0)
{
    MatchPhotosStageReport report;
    report.stageId = QStringLiteral("track_build");
    report.displayName = QStringLiteral("连接点轨迹");
    report.status = status;
    report.message = message;
    report.itemCount = itemCount;
    return report;
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
    summary[QStringLiteral("mean_track_confidence")] = buildResult.meanTrackConfidence;

    QJsonObject histogram;
    for (const auto &entry : buildResult.trackLengthHistogram)
    {
        histogram[QString::number(entry.first)] = entry.second;
    }
    summary[QStringLiteral("track_length_histogram")] = histogram;
    return summary;
}

} // namespace

MatchPhotosStageReport TrackBuildStage::run(const MatchPhotosContext &context,
                                            const MatchPhotosOptions &options,
                                            const std::vector<MatchPhotosMatchRecord> &matchRecords,
                                            MatchPhotosResult *result) const
{
    if (options.planOnly)
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("plan-only 模式，跳过连接点轨迹构建"));
    }
    if (!options.enableTrackBuild)
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("连接点轨迹构建已禁用"));
    }
    if (matchRecords.empty())
    {
        return makeTrackReport(MatchPhotosStageStatus::Skipped,
                               QStringLiteral("没有可用于构建 track 的匹配结果"));
    }

    std::map<QString, ImageId> imageIdByPath;
    for (int index = 0; index < context.pairInput.images.size(); ++index)
    {
        imageIdByPath[QDir::cleanPath(context.pairInput.images.at(index))] = static_cast<ImageId>(index);
    }

    MultiViewTrackBuilder builder;
    std::map<ImageId, bool> keypointsLoaded;
    float imageWidth = 0.0f;
    float imageHeight = 0.0f;
    int consumedPairs = 0;
    int skippedPairs = 0;

    for (const MatchPhotosMatchRecord &record : matchRecords)
    {
        if (shouldCancelMatchPhotos(context))
        {
            return makeTrackReport(MatchPhotosStageStatus::Failed,
                                   QStringLiteral("用户取消连接点轨迹构建"),
                                   consumedPairs);
        }

        if (options.enableGeometryVerification && !record.passedGeometry)
        {
            ++skippedPairs;
            continue;
        }

        const QString image0Path = QDir::cleanPath(record.image0Path);
        const QString image1Path = QDir::cleanPath(record.image1Path);
        const auto image0It = imageIdByPath.find(image0Path);
        const auto image1It = imageIdByPath.find(image1Path);
        if (image0It == imageIdByPath.end() || image1It == imageIdByPath.end())
        {
            ++skippedPairs;
            continue;
        }

        const ImageId image0Id = image0It->second;
        const ImageId image1Id = image1It->second;
        const QString feature0Path = record.settings.value(QStringLiteral("feature0_path")).toString();
        const QString feature1Path = record.settings.value(QStringLiteral("feature1_path")).toString();
        QString featureImageName;
        xjw::feature_extractors::FeatureData feature0;
        xjw::feature_extractors::FeatureData feature1;
        if (!FeatureFileIO::readData(feature0Path, featureImageName, feature0) ||
            !FeatureFileIO::readData(feature1Path, featureImageName, feature1))
        {
            ++skippedPairs;
            continue;
        }

        if (!keypointsLoaded[image0Id])
        {
            builder.setImageKeypoints(image0Id, toSfmKeypoints(feature0.keypoints));
            keypointsLoaded[image0Id] = true;
        }
        if (!keypointsLoaded[image1Id])
        {
            builder.setImageKeypoints(image1Id, toSfmKeypoints(feature1.keypoints));
            keypointsLoaded[image1Id] = true;
        }
        imageWidth = std::max(imageWidth, static_cast<float>(std::max(feature0.imageWidth, feature1.imageWidth)));
        imageHeight = std::max(imageHeight, static_cast<float>(std::max(feature0.imageHeight, feature1.imageHeight)));

        std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
        if (!record.inlierIndexPairs.empty())
        {
            indexedMatches.reserve(record.inlierIndexPairs.size());
            for (const std::array<int, 2> &pair : record.inlierIndexPairs)
            {
                if (pair[0] < 0 || pair[1] < 0)
                {
                    continue;
                }
                indexedMatches.emplace_back(static_cast<FeatureIdx>(pair[0]),
                                            static_cast<FeatureIdx>(pair[1]),
                                            1.0f);
            }
        }
        else if (!options.enableGeometryVerification)
        {
            QString image0Name;
            QString image1Name;
            xjw::feature_match::MatchResult matchResult;
            if (xjw::feature_match::readIndexedMatchFile(record.matchPath, image0Name, image1Name, matchResult))
            {
                indexedMatches.reserve(matchResult.cvMatches.size());
                for (const cv::DMatch &match : matchResult.cvMatches)
                {
                    indexedMatches.emplace_back(static_cast<FeatureIdx>(match.queryIdx),
                                                static_cast<FeatureIdx>(match.trainIdx),
                                                matchScoreAt(matchResult, match.queryIdx, match.distance));
                }
            }
        }

        if (indexedMatches.empty())
        {
            ++skippedPairs;
            continue;
        }

        builder.addMatchPair(image0Id, image1Id, indexedMatches);
        ++consumedPairs;
    }

    MultiViewTrackBuilder::BuildOptions buildOptions;
    buildOptions.enableQualityThinning = true;
    buildOptions.maxTracksPerImage = options.maxTiePointsPerImage;
    buildOptions.maxTracksPerGridCell = options.maxTiePointsPerGridCell;
    buildOptions.gridColumns = options.tiePointGridColumns;
    buildOptions.gridRows = options.tiePointGridRows;
    buildOptions.imageWidth = imageWidth;
    buildOptions.imageHeight = imageHeight;
    const MultiViewTrackBuildResult buildResult = builder.build(buildOptions);

    if (result)
    {
        result->trackCount = static_cast<int>(buildResult.tracks.size());
        result->acceptedTrackComponents = buildResult.acceptedComponents;
        result->rejectedTrackConflictComponents = buildResult.rejectedConflictComponents;
        result->trackSummary = makeTrackSummary(buildResult);
    }

    return makeTrackReport(MatchPhotosStageStatus::Completed,
                           QStringLiteral("连接点轨迹完成：track %1，消费匹配对 %2，跳过 %3")
                               .arg(static_cast<int>(buildResult.tracks.size()))
                               .arg(consumedPairs)
                               .arg(skippedPairs),
                           static_cast<int>(buildResult.tracks.size()));
}

} // namespace matchphotos
} // namespace xjw
