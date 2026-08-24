#include "TiePointTrackManager.h"

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
#include <cstdint>
#include <map>
#include <set>

namespace xjw
{
    namespace matchphotos
    {
        namespace
        {

            QString canonicalPath(const QString& path)
            {
                const QString trimmed = path.trimmed();
                if (trimmed.isEmpty())
                {
                    return QString();
                }
                return QDir::cleanPath(QFileInfo(trimmed).absoluteFilePath());
            }

            QJsonObject makeTrackSummary(const MultiViewTrackBuildResult& buildResult)
            {
                QJsonObject summary;
                summary[QStringLiteral("tracks")] = static_cast<int>(buildResult.tracks.size());
                summary[QStringLiteral("total_components")] = buildResult.totalComponents;
                summary[QStringLiteral("accepted_components")] = buildResult.acceptedComponents;
                summary[QStringLiteral("rejected_conflict_components")] = buildResult.rejectedConflictComponents;
                summary[QStringLiteral("rejected_conflict_edges")] = buildResult.rejectedConflictEdges;
                summary[QStringLiteral("rejected_inconsistent_bridge_edges")] =
                    buildResult.rejectedInconsistentBridgeEdges;
                summary[QStringLiteral("accepted_supported_bridge_edges")] = buildResult.acceptedSupportedBridgeEdges;
                summary[QStringLiteral("pruned_by_quality_thinning")] = buildResult.prunedByQualityThinning;
                summary[QStringLiteral("pruned_stationary_tracks")] = buildResult.prunedStationaryTracks;
                summary[QStringLiteral("mean_track_confidence")] = buildResult.meanTrackConfidence;

                QJsonObject histogram;
                for (const auto& entry : buildResult.trackLengthHistogram)
                {
                    histogram[QString::number(entry.first)] = entry.second;
                }
                summary[QStringLiteral("track_length_histogram")] = histogram;
                return summary;
            }

            QJsonObject makePersistedSettings(const MatchPhotosOptions& options)
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
                settings[QStringLiteral("guided_image_matching_mode")] =
                    guidedMatchingModeName(options.guidedMatchingMode);
                settings[QStringLiteral("guided_image_matching")] = guidedMatchingEnabled(options.guidedMatchingMode);
                settings[QStringLiteral("generic_preselection")] = options.useGenericPreselection;
                settings[QStringLiteral("reference_preselection")] = options.useReferencePreselection;
                return settings;
            }

            QString tiePointOutputPath(const MatchPhotosContext& context)
            {
                const QString assetsDir = context.workingDirectory.trimmed();
                if (assetsDir.isEmpty())
                {
                    return QString();
                }

                return QDir(QDir(assetsDir).filePath(QStringLiteral("tie_points")))
                    .filePath(QStringLiteral("latest_tie_points.json"));
            }

            QByteArray jsonString(const QString& value)
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

            QByteArray compactJson(const QJsonObject& object)
            {
                return QJsonDocument(object).toJson(QJsonDocument::Compact);
            }

            bool writeRaw(QSaveFile* file, const QByteArray& data, QString* errorMessage)
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

            bool writeField(QSaveFile* file,
                            const QByteArray& name,
                            const QByteArray& value,
                            bool trailingComma,
                            QString* errorMessage)
            {
                QByteArray line = QByteArrayLiteral("  \"") + name + QByteArrayLiteral("\": ") + value;
                line += trailingComma ? QByteArrayLiteral(",\n") : QByteArrayLiteral("\n");
                return writeRaw(file, line, errorMessage);
            }

            bool writeImages(QSaveFile* file, const MatchPhotosContext& context, QString* errorMessage)
            {
                if (!writeRaw(file, QByteArrayLiteral("  \"images\": [\n"), errorMessage))
                {
                    return false;
                }

                for (int index = 0; index < context.pairInput.images.size(); ++index)
                {
                    QByteArray line = QByteArrayLiteral("    {\"image_id\":") + QByteArray::number(index) +
                                      QByteArrayLiteral(",\"path\":") +
                                      jsonString(canonicalPath(context.pairInput.images.at(index))) +
                                      QByteArrayLiteral("}");
                    line += index + 1 < context.pairInput.images.size() ? QByteArrayLiteral(",\n")
                                                                        : QByteArrayLiteral("\n");
                    if (!writeRaw(file, line, errorMessage))
                    {
                        return false;
                    }
                }

                return writeRaw(file, QByteArrayLiteral("  ],\n"), errorMessage);
            }

            QByteArray observationJson(const TrackElement& element,
                                       const MatchPhotosContext& context,
                                       const std::map<ImageId, std::vector<FeatureKeypoint>>& keypointsByImage)
            {
                QByteArray object =
                    QByteArrayLiteral("{\"image_id\":") + QByteArray::number(static_cast<int>(element.imageId)) +
                    QByteArrayLiteral(",\"feature_idx\":") + QByteArray::number(static_cast<int>(element.featureIdx));

                if (element.imageId < static_cast<ImageId>(context.pairInput.images.size()))
                {
                    object += QByteArrayLiteral(",\"image_path\":") +
                              jsonString(canonicalPath(context.pairInput.images.at(static_cast<int>(element.imageId))));
                }

                const auto keypointsIt = keypointsByImage.find(element.imageId);
                if (keypointsIt != keypointsByImage.end() &&
                    element.featureIdx < static_cast<FeatureIdx>(keypointsIt->second.size()))
                {
                    const FeatureKeypoint& keypoint = keypointsIt->second[static_cast<std::size_t>(element.featureIdx)];
                    object += QByteArrayLiteral(",\"xy\":[") + jsonNumber(keypoint.x) + QByteArrayLiteral(",") +
                              jsonNumber(keypoint.y) + QByteArrayLiteral("]");
                }

                object += QByteArrayLiteral("}");
                return object;
            }

            bool writeTrack(QSaveFile* file,
                            const Track& track,
                            const std::vector<std::pair<std::size_t, std::size_t>>& directEdges,
                            int trackId,
                            const MatchPhotosContext& context,
                            const std::map<ImageId, std::vector<FeatureKeypoint>>& keypointsByImage,
                            bool trailingComma,
                            QString* errorMessage)
            {
                QByteArray header = QByteArrayLiteral("    {\"track_id\":") + QByteArray::number(trackId) +
                                    QByteArrayLiteral(",\"track_len\":") +
                                    QByteArray::number(static_cast<int>(track.length())) +
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

                if (!writeRaw(file, QByteArrayLiteral("],\"direct_edges\":["), errorMessage))
                {
                    return false;
                }
                for (std::size_t index = 0; index < directEdges.size(); ++index)
                {
                    const auto& [first, second] = directEdges[index];
                    QByteArray edge = QByteArrayLiteral("[") + QByteArray::number(static_cast<qulonglong>(first)) +
                                      QByteArrayLiteral(",") + QByteArray::number(static_cast<qulonglong>(second)) +
                                      QByteArrayLiteral("]");
                    if (index + 1 < directEdges.size())
                    {
                        edge += QByteArrayLiteral(",");
                    }
                    if (!writeRaw(file, edge, errorMessage))
                    {
                        return false;
                    }
                }

                QByteArray footer = QByteArrayLiteral("]}");
                footer += trailingComma ? QByteArrayLiteral(",\n") : QByteArrayLiteral("\n");
                return writeRaw(file, footer, errorMessage);
            }

            bool writeTracks(QSaveFile* file,
                             const MatchPhotosContext& context,
                             const TiePointTrackBuildResult& result,
                             const std::map<ImageId, std::vector<FeatureKeypoint>>& keypointsByImage,
                             QString* errorMessage)
            {
                if (!writeRaw(file, QByteArrayLiteral("  \"tracks\": [\n"), errorMessage))
                {
                    return false;
                }

                for (std::size_t index = 0; index < result.tracks.size(); ++index)
                {
                    static const std::vector<std::pair<std::size_t, std::size_t>> emptyEdges;
                    const auto& directEdges =
                        index < result.directEdgesByTrack.size() ? result.directEdgesByTrack[index] : emptyEdges;
                    if (!writeTrack(file,
                                    result.tracks[index],
                                    directEdges,
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

            bool writeTiePointFile(const QString& path,
                                   const MatchPhotosContext& context,
                                   const MatchPhotosOptions& options,
                                   const TiePointTrackBuildResult& result,
                                   const std::map<ImageId, std::vector<FeatureKeypoint>>& keypointsByImage,
                                   QString* errorMessage)
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
                    !writeField(&file,
                                QByteArrayLiteral("format"),
                                jsonString(QStringLiteral("plascan_tie_points")),
                                true,
                                errorMessage) ||
                    !writeField(
                        &file, QByteArrayLiteral("format_version"), QByteArrayLiteral("2"), true, errorMessage) ||
                    !writeField(&file,
                                QByteArrayLiteral("created_at"),
                                jsonString(QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)),
                                true,
                                errorMessage) ||
                    !writeField(&file,
                                QByteArrayLiteral("track_count"),
                                QByteArray::number(static_cast<int>(result.tracks.size())),
                                true,
                                errorMessage) ||
                    !writeField(&file,
                                QByteArrayLiteral("consumed_pair_count"),
                                QByteArray::number(result.consumedPairCount),
                                true,
                                errorMessage) ||
                    !writeField(&file,
                                QByteArrayLiteral("skipped_pair_count"),
                                QByteArray::number(result.skippedPairCount),
                                true,
                                errorMessage) ||
                    !writeField(
                        &file, QByteArrayLiteral("summary"), compactJson(result.trackSummary), true, errorMessage) ||
                    !writeField(&file,
                                QByteArrayLiteral("settings"),
                                compactJson(makePersistedSettings(options)),
                                true,
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

            MultiViewTrackBuilder::BuildOptions
            makeBuildOptions(const MatchPhotosOptions& options, float imageWidth, float imageHeight)
            {
                MultiViewTrackBuilder::BuildOptions buildOptions;
                buildOptions.enableQualityThinning = options.maxTiePointsPerImage > 0;
                buildOptions.maxTracksPerImage = options.maxTiePointsPerImage;
                buildOptions.maxTracksPerGridCell =
                    options.maxTiePointsPerImage > 0 ? options.maxTiePointsPerGridCell : 0;
                buildOptions.gridColumns = options.tiePointGridColumns;
                buildOptions.gridRows = options.tiePointGridRows;
                buildOptions.imageWidth = imageWidth;
                buildOptions.imageHeight = imageHeight;
                buildOptions.excludeStationaryTracks = options.excludeStationaryTiePoints;
                buildOptions.stationaryTrackMaxPixelMotion = options.stationaryTiePointMaxPixelMotion;
                return buildOptions;
            }

            void copyBuildResult(const MultiViewTrackBuildResult& buildResult, TiePointTrackBuildResult* result)
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
                result->rejectedInconsistentBridgeEdges = buildResult.rejectedInconsistentBridgeEdges;
                result->acceptedSupportedBridgeEdges = buildResult.acceptedSupportedBridgeEdges;
                result->prunedByQualityThinning = buildResult.prunedByQualityThinning;
                result->prunedStationaryTracks = buildResult.prunedStationaryTracks;
                result->meanTrackConfidence = buildResult.meanTrackConfidence;
                result->trackSummary = makeTrackSummary(buildResult);
            }

            using ObservationKey = std::pair<ImageId, FeatureIdx>;

            struct TrackObservationMembership
            {
                std::size_t trackIndex = 0;
                std::size_t observationIndex = 0;
            };

            std::vector<std::vector<std::pair<std::size_t, std::size_t>>>
            collectDirectTrackEdges(const std::vector<Track>& tracks,
                                    const std::map<QString, ImageId>& imageIdByPath,
                                    const MatchPhotosOptions& options,
                                    const std::vector<MatchPhotosMatchRecord>& records)
            {
                std::map<ObservationKey, TrackObservationMembership> membership;
                for (std::size_t trackIndex = 0; trackIndex < tracks.size(); ++trackIndex)
                {
                    const Track& track = tracks[trackIndex];
                    for (std::size_t observationIndex = 0; observationIndex < track.elements.size(); ++observationIndex)
                    {
                        const TrackElement& element = track.elements[observationIndex];
                        membership[{element.imageId, element.featureIdx}] = {trackIndex, observationIndex};
                    }
                }

                std::vector<std::vector<std::pair<std::size_t, std::size_t>>> directEdges(tracks.size());
                for (const MatchPhotosMatchRecord& record : records)
                {
                    if (!record.pairData || (options.enableGeometryVerification && !record.passedGeometry))
                    {
                        continue;
                    }
                    const auto image0 = imageIdByPath.find(canonicalPath(record.image0Path));
                    const auto image1 = imageIdByPath.find(canonicalPath(record.image1Path));
                    if (image0 == imageIdByPath.end() || image1 == imageIdByPath.end())
                    {
                        continue;
                    }

                    for (const image_matching::PairCorrespondence& correspondence : record.pairData->correspondences)
                    {
                        if (options.enableGeometryVerification &&
                            !image_matching::hasFlag(correspondence.flags,
                                                     image_matching::MatchRecordFlag::GeometryInlier))
                        {
                            continue;
                        }
                        const auto first = membership.find(
                            {image0->second, static_cast<FeatureIdx>(correspondence.observation0.featureId)});
                        const auto second = membership.find(
                            {image1->second, static_cast<FeatureIdx>(correspondence.observation1.featureId)});
                        if (first == membership.end() || second == membership.end() ||
                            first->second.trackIndex != second->second.trackIndex)
                        {
                            continue;
                        }

                        std::size_t left = first->second.observationIndex;
                        std::size_t right = second->second.observationIndex;
                        if (left > right)
                        {
                            std::swap(left, right);
                        }
                        directEdges[first->second.trackIndex].emplace_back(left, right);
                    }
                }

                for (auto& edges : directEdges)
                {
                    std::sort(edges.begin(), edges.end());
                    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
                }
                return directEdges;
            }

            image_matching::MatchRecordFlag withTrackFlag(image_matching::MatchRecordFlag value, bool enabled)
            {
                const auto raw = static_cast<std::uint32_t>(value);
                const auto bit = static_cast<std::uint32_t>(image_matching::MatchRecordFlag::InTiePointTrack);
                return static_cast<image_matching::MatchRecordFlag>(enabled ? raw | bit : raw & ~bit);
            }

            void rememberKeypoint(ImageId imageId,
                                  const image_matching::KeypointObservation& observation,
                                  std::map<ImageId, std::map<FeatureIdx, FeatureKeypoint>>* sparseKeypoints)
            {
                if (!sparseKeypoints)
                {
                    return;
                }
                (*sparseKeypoints)[imageId][static_cast<FeatureIdx>(observation.featureId)] =
                    FeatureKeypoint{observation.x, observation.y};
            }

            std::vector<FeatureKeypoint> denseKeypointVector(const std::map<FeatureIdx, FeatureKeypoint>& sparse)
            {
                if (sparse.empty())
                {
                    return {};
                }
                const FeatureIdx maximumIndex = sparse.rbegin()->first;
                std::vector<FeatureKeypoint> dense(static_cast<std::size_t>(maximumIndex) + 1U);
                for (const auto& [featureIndex, keypoint] : sparse)
                {
                    dense[static_cast<std::size_t>(featureIndex)] = keypoint;
                }
                return dense;
            }

            void markTrackMembership(const std::vector<Track>& tracks,
                                     const std::map<QString, ImageId>& imageIdByPath,
                                     std::vector<MatchPhotosMatchRecord>* records)
            {
                std::map<ObservationKey, int> trackByObservation;
                for (int trackId = 0; trackId < static_cast<int>(tracks.size()); ++trackId)
                {
                    for (const TrackElement& element : tracks[static_cast<std::size_t>(trackId)].elements)
                    {
                        trackByObservation[{element.imageId, element.featureIdx}] = trackId;
                    }
                }

                for (MatchPhotosMatchRecord& record : *records)
                {
                    if (!record.pairData)
                    {
                        continue;
                    }
                    const auto image0It = imageIdByPath.find(canonicalPath(record.image0Path));
                    const auto image1It = imageIdByPath.find(canonicalPath(record.image1Path));
                    if (image0It == imageIdByPath.end() || image1It == imageIdByPath.end())
                    {
                        continue;
                    }

                    int trackMatches = 0;
                    for (image_matching::PairCorrespondence& correspondence : record.pairData->correspondences)
                    {
                        const ObservationKey key0{image0It->second,
                                                  static_cast<FeatureIdx>(correspondence.observation0.featureId)};
                        const ObservationKey key1{image1It->second,
                                                  static_cast<FeatureIdx>(correspondence.observation1.featureId)};
                        const auto track0 = trackByObservation.find(key0);
                        const auto track1 = trackByObservation.find(key1);
                        const bool inSameTrack = track0 != trackByObservation.end() &&
                                                 track1 != trackByObservation.end() && track0->second == track1->second;
                        correspondence.flags = withTrackFlag(correspondence.flags, inSameTrack);
                        if (inSameTrack)
                        {
                            ++trackMatches;
                        }
                    }
                    record.pairData->tiePointMatchCount = static_cast<std::uint32_t>(trackMatches);
                    record.settings[QStringLiteral("tie_point_matches")] = trackMatches;
                }
            }

        } // namespace

        TiePointTrackBuildResult TiePointTrackManager::build(const MatchPhotosContext& context,
                                                             const MatchPhotosOptions& options,
                                                             std::vector<MatchPhotosMatchRecord>* matchRecords) const
        {
            TiePointTrackBuildResult result;
            result.success = true;
            if (!matchRecords)
            {
                result.success = false;
                result.errorMessage = QStringLiteral("内部错误：连接点匹配记录为空");
                return result;
            }

            std::map<QString, ImageId> imageIdByPath;
            for (int index = 0; index < context.pairInput.images.size(); ++index)
            {
                imageIdByPath[canonicalPath(context.pairInput.images.at(index))] = static_cast<ImageId>(index);
            }

            MultiViewTrackBuilder builder;
            std::map<ImageId, std::map<FeatureIdx, FeatureKeypoint>> sparseKeypoints;
            std::map<ImageId, std::vector<FeatureKeypoint>> keypointsByImage;
            float imageWidth = 0.0f;
            float imageHeight = 0.0f;

            // 第一遍只整理匹配真正引用的观测。featureId 保留原始 SIFT 索引，因此数组
            // 可能稀疏；先用 map 收集，再一次性展开，避免对每条边反复扩容。
            for (const MatchPhotosMatchRecord& record : *matchRecords)
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

                if (!record.pairData)
                {
                    ++result.skippedPairCount;
                    continue;
                }
                const auto image0It = imageIdByPath.find(canonicalPath(record.image0Path));
                const auto image1It = imageIdByPath.find(canonicalPath(record.image1Path));
                if (image0It == imageIdByPath.end() || image1It == imageIdByPath.end())
                {
                    ++result.skippedPairCount;
                    continue;
                }

                const ImageId image0Id = image0It->second;
                const ImageId image1Id = image1It->second;
                for (const image_matching::PairCorrespondence& correspondence : record.pairData->correspondences)
                {
                    if (options.enableGeometryVerification &&
                        !image_matching::hasFlag(correspondence.flags, image_matching::MatchRecordFlag::GeometryInlier))
                    {
                        continue;
                    }
                    rememberKeypoint(image0Id, correspondence.observation0, &sparseKeypoints);
                    rememberKeypoint(image1Id, correspondence.observation1, &sparseKeypoints);
                }
                imageWidth = std::max(
                    imageWidth,
                    static_cast<float>(std::max(record.pairData->image0.width, record.pairData->image1.width)));
                imageHeight = std::max(
                    imageHeight,
                    static_cast<float>(std::max(record.pairData->image0.height, record.pairData->image1.height)));
            }

            for (const auto& [imageId, sparse] : sparseKeypoints)
            {
                keypointsByImage[imageId] = denseKeypointVector(sparse);
                builder.setImageKeypoints(imageId, keypointsByImage[imageId]);
            }

            // 第二遍添加经过几何验证的边。置信度来自 LightGlue，供冲突消解和质量抽稀
            // 使用；不再从单独的 `.match` 文件恢复索引。
            for (const MatchPhotosMatchRecord& record : *matchRecords)
            {
                if (!record.pairData || (options.enableGeometryVerification && !record.passedGeometry))
                {
                    continue;
                }
                const auto image0It = imageIdByPath.find(canonicalPath(record.image0Path));
                const auto image1It = imageIdByPath.find(canonicalPath(record.image1Path));
                if (image0It == imageIdByPath.end() || image1It == imageIdByPath.end())
                {
                    continue;
                }

                std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
                indexedMatches.reserve(record.pairData->correspondences.size());
                for (const image_matching::PairCorrespondence& correspondence : record.pairData->correspondences)
                {
                    if (options.enableGeometryVerification &&
                        !image_matching::hasFlag(correspondence.flags, image_matching::MatchRecordFlag::GeometryInlier))
                    {
                        continue;
                    }
                    indexedMatches.emplace_back(static_cast<FeatureIdx>(correspondence.observation0.featureId),
                                                static_cast<FeatureIdx>(correspondence.observation1.featureId),
                                                correspondence.confidence);
                }
                if (indexedMatches.empty())
                {
                    continue;
                }

                builder.addMatchPair(image0It->second, image1It->second, indexedMatches);
                ++result.consumedPairCount;
            }

            const MultiViewTrackBuildResult buildResult =
                builder.build(makeBuildOptions(options, imageWidth, imageHeight));
            copyBuildResult(buildResult, &result);
            result.directEdgesByTrack = collectDirectTrackEdges(result.tracks, imageIdByPath, options, *matchRecords);
            markTrackMembership(result.tracks, imageIdByPath, matchRecords);
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
