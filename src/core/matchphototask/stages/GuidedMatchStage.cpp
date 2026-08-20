#include "GuidedMatchStage.h"

#include "GeometryVerifyStage.h"
#include "MatchGeometryVerifier.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftGuidedMatcher.h"

#include <algorithm>
#include <cstdint>
#include <cmath>

#include <QDir>
#include <QFileInfo>
#include <QHash>
#include <QSet>

namespace xjw::matchphotos
{
    namespace
    {

        image_matching::MatchRecordFlag withGeometryFlag(image_matching::MatchRecordFlag value, bool enabled)
        {
            const auto raw = static_cast<std::uint32_t>(value);
            const auto bit = static_cast<std::uint32_t>(image_matching::MatchRecordFlag::GeometryInlier);
            return static_cast<image_matching::MatchRecordFlag>(enabled ? raw | bit : raw & ~bit);
        }

        image_matching::KeypointObservation observationFor(const cv::KeyPoint& keypoint, int featureId)
        {
            image_matching::KeypointObservation observation;
            observation.featureId = static_cast<std::uint32_t>(featureId);
            observation.x = keypoint.pt.x;
            observation.y = keypoint.pt.y;
            observation.scale = keypoint.size;
            observation.orientation = keypoint.angle;
            observation.response = keypoint.response;
            return observation;
        }

        QString observationKey(const QString& imagePath, std::uint32_t featureId)
        {
            return QDir::cleanPath(QFileInfo(imagePath).absoluteFilePath()) +
                QChar(0x1f) + QString::number(featureId);
        }

        using ObservationLinks = QHash<QString, QSet<QString>>;

        ObservationLinks buildReliableObservationLinks(
            const std::vector<MatchPhotosMatchRecord>& records)
        {
            ObservationLinks links;
            for (const MatchPhotosMatchRecord& record : records)
            {
                if (!record.passedGeometry || !record.pairData)
                {
                    continue;
                }
                for (const auto& correspondence : record.pairData->correspondences)
                {
                    if (!image_matching::hasFlag(
                            correspondence.flags,
                            image_matching::MatchRecordFlag::GeometryInlier))
                    {
                        continue;
                    }
                    const QString key0 = observationKey(
                        record.image0Path, correspondence.observation0.featureId);
                    const QString key1 = observationKey(
                        record.image1Path, correspondence.observation1.featureId);
                    links[key0].insert(key1);
                    links[key1].insert(key0);
                }
            }
            return links;
        }

        bool passesMultiViewConsistency(const ObservationLinks& links,
                                        const QString& image0Path,
                                        int index0,
                                        const QString& image1Path,
                                        int index1)
        {
            const QSet<QString> peers0 = links.value(
                observationKey(image0Path, static_cast<std::uint32_t>(index0)));
            const QSet<QString> peers1 = links.value(
                observationKey(image1Path, static_cast<std::uint32_t>(index1)));
            if (peers0.isEmpty() && peers1.isEmpty())
            {
                return true;
            }
            if (peers0.isEmpty() || peers1.isEmpty())
            {
                return false;
            }
            for (const QString& peer : peers0)
            {
                if (peers1.contains(peer))
                {
                    return true;
                }
            }
            return false;
        }

        bool hasReliableFundamental(const MatchPhotosContext& context,
                                    const MatchPhotosOptions& options,
                                    const MatchPhotosMatchRecord& record)
        {
            if (!record.passedGeometry || !record.pairData ||
                record.pairData->geometryModel != image_matching::GeometryModel::Fundamental)
            {
                return false;
            }
            const auto& values = record.pairData->geometryMatrix;
            double squaredNorm = 0.0;
            for (const double value : values)
            {
                if (!std::isfinite(value))
                {
                    return false;
                }
                squaredNorm += value * value;
            }
            const double norm = std::sqrt(squaredNorm);
            if (norm <= 1e-12)
            {
                return false;
            }
            const cv::Matx33d fundamental(values[0], values[1], values[2],
                                          values[3], values[4], values[5],
                                          values[6], values[7], values[8]);
            if (std::abs(cv::determinant(fundamental)) / (norm * norm * norm) > 1e-3)
            {
                return false;
            }
            const bool adjacent = areSequenceAdjacent(
                context, options, record.image0Path, record.image1Path);
            return evaluateGeometryQuality(
                measureGeometryQuality(*record.pairData, options, adjacent), options).passed;
        }

        bool applyGuidedGeometry(const image_matching::FeatureSet& features0,
                                 const image_matching::FeatureSet& features1,
                                 const MatchPhotosContext& context,
                                 const MatchPhotosOptions& options,
                                 MatchPhotosMatchRecord* record)
        {
            if (!record || !record->pairData)
            {
                return false;
            }
            image_matching::PairMatchData& pair = *record->pairData;
            image_matching::MatchResult matches;
            matches.sourceAlgorithm = image_matching::kAutoSiftAlgorithmId;
            matches.cvMatches.reserve(pair.correspondences.size());
            for (const auto& correspondence : pair.correspondences)
            {
                matches.cvMatches.emplace_back(static_cast<int>(correspondence.observation0.featureId),
                                               static_cast<int>(correspondence.observation1.featureId),
                                               1.0f - correspondence.confidence);
            }
            matches.numMatches = static_cast<int>(matches.cvMatches.size());

            image_matching::MatchGeometryOptions geometryOptions;
            geometryOptions.model = image_matching::GeometryModel::Fundamental;
            geometryOptions.reprojectionThresholdPixels = options.geometryReprojThreshold;
            geometryOptions.minimumInliers = options.geometryMinInliers;
            geometryOptions.maximumIterations = std::max(100, options.geometryMaxIterations);
            geometryOptions.confidence = 0.9999;
            geometryOptions.randomSeed = 0;
            const image_matching::MatchGeometryResult geometry =
                image_matching::MatchGeometryVerifier::verify(matches, features0, features1, geometryOptions);

            const int rawCount = static_cast<int>(pair.correspondences.size());
            pair.rawMatchCount = static_cast<std::uint32_t>(rawCount);
            pair.geometryInlierCount = static_cast<std::uint32_t>(geometry.inlierCount);
            pair.geometryModel = geometry.modelEstimated ? geometry.model : image_matching::GeometryModel::None;
            pair.geometryMatrix = geometry.matrix;
            for (int index = 0; index < rawCount; ++index)
            {
                auto& correspondence = pair.correspondences[static_cast<std::size_t>(index)];
                const bool inlier = index < static_cast<int>(geometry.inlierMask.size()) &&
                                    geometry.inlierMask[static_cast<std::size_t>(index)];
                correspondence.flags = withGeometryFlag(correspondence.flags, inlier);
                correspondence.residualPixels = index < static_cast<int>(geometry.residualPixels.size())
                                                    ? geometry.residualPixels[static_cast<std::size_t>(index)]
                                                    : -1.0f;
            }

            const bool adjacent = areSequenceAdjacent(
                context, options, record->image0Path, record->image1Path);
            const GeometryQualityMetrics quality = measureGeometryQuality(pair, options, adjacent);
            const GeometryQualityDecision decision = evaluateGeometryQuality(quality, options);
            pair.geometryPassed = geometry.modelEstimated && decision.passed;

            record->matchCount = rawCount;
            record->geometricInlierCount = geometry.inlierCount;
            record->passedGeometry = pair.geometryPassed;
            record->settings[QStringLiteral("geometry_verified")] = true;
            record->settings[QStringLiteral("geometry_cache_reusable")] = true;
            record->settings[QStringLiteral("geometry_raw_matches")] = rawCount;
            record->settings[QStringLiteral("geometric_inliers")] = geometry.inlierCount;
            record->settings[QStringLiteral("geometry_inlier_ratio")] =
                rawCount > 0 ? static_cast<double>(geometry.inlierCount) / static_cast<double>(rawCount) : 0.0;
            record->settings[QStringLiteral("geometry_passed")] = pair.geometryPassed;
            record->settings[QStringLiteral("geometry_grid_coverage_image0")] =
                quality.image0GridCoverage;
            record->settings[QStringLiteral("geometry_grid_coverage_image1")] =
                quality.image1GridCoverage;
            record->settings[QStringLiteral("geometry_adjacent_images")] = quality.adjacentImages;
            record->settings[QStringLiteral("geometry_quality_score")] = decision.score;
            return geometry.modelEstimated;
        }

    } // namespace

    MatchPhotosStageReport GuidedMatchStage::run(const MatchPhotosContext& context,
                                                 const MatchPhotosOptions& options,
                                                 const MatchPhotosAlgorithmPlan& algorithmPlan,
                                                 std::vector<MatchPhotosMatchRecord>* matchRecords) const
    {
        MatchPhotosStageReport report;
        report.stageId = QStringLiteral("guided_match");
        report.displayName = QStringLiteral("引导匹配");
        if (!options.enableGuidedMatching)
        {
            report.status = MatchPhotosStageStatus::Skipped;
            report.message = QStringLiteral("引导匹配未启用");
            return report;
        }
        if (algorithmPlan.algorithmId != QLatin1String(image_matching::kAutoSiftAlgorithmId))
        {
            report.status = MatchPhotosStageStatus::Skipped;
            report.message = QStringLiteral("当前算法不使用 SIFT 几何引导重匹配");
            return report;
        }
        if (!options.enableGeometryVerification || !context.featureCache || !matchRecords || matchRecords->empty())
        {
            report.status = MatchPhotosStageStatus::Skipped;
            report.message = QStringLiteral("没有可用于几何引导重匹配的特征或模型");
            return report;
        }

        int processedPairs = 0;
        int addedMatches = 0;
        int improvedPairs = 0;
        int unreliableModels = 0;
        int consistencyRejected = 0;
        const ObservationLinks observationLinks = buildReliableObservationLinks(*matchRecords);
        for (MatchPhotosMatchRecord& record : *matchRecords)
        {
            if (!hasReliableFundamental(context, options, record))
            {
                ++unreliableModels;
                continue;
            }
            const auto features0 = context.featureCache->find(record.image0Path);
            const auto features1 = context.featureCache->find(record.image1Path);
            if (!features0 || !features1)
            {
                continue;
            }

            std::vector<int> existing0;
            std::vector<int> existing1;
            existing0.reserve(record.pairData->correspondences.size());
            existing1.reserve(record.pairData->correspondences.size());
            for (const auto& correspondence : record.pairData->correspondences)
            {
                existing0.push_back(static_cast<int>(correspondence.observation0.featureId));
                existing1.push_back(static_cast<int>(correspondence.observation1.featureId));
            }

            image_matching::SiftGuidedMatchOptions guidedOptions;
            guidedOptions.epipolarThresholdPixels = std::max(2.0, options.geometryReprojThreshold * 2.0);
            guidedOptions.maximumDescriptorRatio = std::min(0.95f, options.siftMaximumRatio);
            guidedOptions.maximumAdditionalMatches =
                std::max(256, options.maxTiePointsPerImage > 0 ? options.maxTiePointsPerImage : 5000);
            std::vector<image_matching::SiftGuidedMatch> additions = image_matching::findGuidedSiftMatches(
                *features0, *features1, record.pairData->geometryMatrix, existing0, existing1, guidedOptions);
            ++processedPairs;
            if (additions.empty())
            {
                continue;
            }

            const bool applyTiepointMask = shouldApplyMasksToTiepoints(options);
            const cv::Mat mask0 = applyTiepointMask
                                      ? softenedExclusionMask(loadMaskForImage(
                                            context,
                                            record.image0Path,
                                            cv::Size(features0->imageWidth, features0->imageHeight)),
                                            options)
                                      : cv::Mat();
            const cv::Mat mask1 = applyTiepointMask
                                      ? softenedExclusionMask(loadMaskForImage(
                                            context,
                                            record.image1Path,
                                            cv::Size(features1->imageWidth, features1->imageHeight)),
                                            options)
                                      : cv::Mat();
            int accepted = 0;
            for (const image_matching::SiftGuidedMatch& match : additions)
            {
                const cv::KeyPoint& keypoint0 = features0->keypoints[static_cast<std::size_t>(match.index0)];
                const cv::KeyPoint& keypoint1 = features1->keypoints[static_cast<std::size_t>(match.index1)];
                if (options.guidedRequireMultiViewConsistency &&
                    !passesMultiViewConsistency(observationLinks,
                                                record.image0Path,
                                                match.index0,
                                                record.image1Path,
                                                match.index1))
                {
                    ++consistencyRejected;
                    continue;
                }
                const float maskWeight = applyTiepointMask
                    ? std::min(maskPointWeight(mask0, keypoint0.pt, options),
                               maskPointWeight(mask1, keypoint1.pt, options))
                    : 1.0f;
                if (maskWeight <= 0.0f)
                {
                    continue;
                }
                image_matching::PairCorrespondence correspondence;
                correspondence.observation0 = observationFor(keypoint0, match.index0);
                correspondence.observation1 = observationFor(keypoint1, match.index1);
                correspondence.confidence = match.confidence * maskWeight;
                correspondence.flags =
                    image_matching::MatchRecordFlag::MaskAccepted | image_matching::MatchRecordFlag::Guided;
                record.pairData->correspondences.push_back(correspondence);
                ++accepted;
            }
            if (accepted == 0)
            {
                continue;
            }

            const int previousInliers = record.geometricInlierCount;
            applyGuidedGeometry(*features0, *features1, context, options, &record);

            // 引导候选必须再次成为可靠 F 的内点。把未闭合的 guided 边移除，
            // 防止其进入后续并查集并污染多视图轨迹。
            const auto beforeCleanup = record.pairData->correspondences.size();
            std::erase_if(record.pairData->correspondences, [](const auto& correspondence)
            {
                return image_matching::hasFlag(correspondence.flags,
                                                image_matching::MatchRecordFlag::Guided) &&
                    !image_matching::hasFlag(correspondence.flags,
                                             image_matching::MatchRecordFlag::GeometryInlier);
            });
            const int removed = static_cast<int>(
                beforeCleanup - record.pairData->correspondences.size());
            if (removed > 0)
            {
                applyGuidedGeometry(*features0, *features1, context, options, &record);
            }
            const int retained = accepted - removed;
            record.settings[QStringLiteral("guided_matches_added")] = retained;
            record.settings[QStringLiteral("guided_matches_rejected_by_geometry")] = removed;
            record.settings[QStringLiteral("guided_matching_completed")] = true;
            addedMatches += retained;
            improvedPairs += record.geometricInlierCount > previousInliers ? 1 : 0;
        }

        report.status = MatchPhotosStageStatus::Completed;
        report.itemCount = addedMatches;
        report.message = QStringLiteral("SIFT 几何引导重匹配完成：检查 %1 对，保留 %2 个新增内点，"
                                        "%3 对内点增加；跳过不可靠 F %4 对，多视图一致性拒绝 %5 个")
                             .arg(processedPairs)
                             .arg(addedMatches)
                             .arg(improvedPairs)
                             .arg(unreliableModels)
                             .arg(consistencyRejected);
        return report;
    }

} // namespace xjw::matchphotos
