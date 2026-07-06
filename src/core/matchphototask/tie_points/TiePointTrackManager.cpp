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
        imageIdByPath[QDir::cleanPath(context.pairInput.images.at(index))] = static_cast<ImageId>(index);
    }

    MultiViewTrackBuilder builder;
    std::map<ImageId, bool> keypointsLoaded;
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

        const QString image0Path = QDir::cleanPath(record.image0Path);
        const QString image1Path = QDir::cleanPath(record.image1Path);
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
    return result;
}

} // namespace matchphotos
} // namespace xjw
