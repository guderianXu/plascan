#include "GuidedMatchStage.h"

#include "GeometryVerifyStage.h"
#include "MatchGeometryVerifier.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftGuidedMatcher.h"

#include <algorithm>
#include <cstdint>

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

        bool applyGuidedGeometry(const image_matching::FeatureSet& features0,
                                 const image_matching::FeatureSet& features1,
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
            pair.geometryPassed = geometry.modelEstimated &&
                                  passesGeometryQualityGate(rawCount, geometry.inlierCount, options.geometryMinInliers);
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
        for (MatchPhotosMatchRecord& record : *matchRecords)
        {
            if (!record.pairData || record.pairData->geometryModel != image_matching::GeometryModel::Fundamental)
            {
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
                                      ? loadMaskForImage(context,
                                                         record.image0Path,
                                                         cv::Size(features0->imageWidth, features0->imageHeight))
                                      : cv::Mat();
            const cv::Mat mask1 = applyTiepointMask
                                      ? loadMaskForImage(context,
                                                         record.image1Path,
                                                         cv::Size(features1->imageWidth, features1->imageHeight))
                                      : cv::Mat();
            int accepted = 0;
            for (const image_matching::SiftGuidedMatch& match : additions)
            {
                const cv::KeyPoint& keypoint0 = features0->keypoints[static_cast<std::size_t>(match.index0)];
                const cv::KeyPoint& keypoint1 = features1->keypoints[static_cast<std::size_t>(match.index1)];
                if (applyTiepointMask &&
                    (!isPointAllowedByMask(mask0, keypoint0.pt) || !isPointAllowedByMask(mask1, keypoint1.pt)))
                {
                    continue;
                }
                image_matching::PairCorrespondence correspondence;
                correspondence.observation0 = observationFor(keypoint0, match.index0);
                correspondence.observation1 = observationFor(keypoint1, match.index1);
                correspondence.confidence = match.confidence;
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
            applyGuidedGeometry(*features0, *features1, options, &record);
            record.settings[QStringLiteral("guided_matches_added")] = accepted;
            record.settings[QStringLiteral("guided_matching_completed")] = true;
            addedMatches += accepted;
            improvedPairs += record.geometricInlierCount > previousInliers ? 1 : 0;
        }

        report.status = MatchPhotosStageStatus::Completed;
        report.itemCount = addedMatches;
        report.message = QStringLiteral("SIFT 几何引导重匹配完成：检查 %1 对，新增 %2 个候选，%3 对内点增加")
                             .arg(processedPairs)
                             .arg(addedMatches)
                             .arg(improvedPairs);
        return report;
    }

} // namespace xjw::matchphotos
