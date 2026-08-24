#include "GuidedMatchStage.h"

#include "GeometryVerifyStage.h"
#include "GuidedMatchPolicy.h"
#include "MatchGeometryVerifier.h"
#include "MatchPhotosFeatureCache.h"
#include "MatchPhotosMaskSupport.h"
#include "MatchPhotosParallelism.h"
#include "MatchPhotosRuntime.h"
#include "concurrency/SafeWorkerGroup.h"
#include "log/Logger.h"
#include "sift/AutoSiftAlgorithm.h"
#include "sift/SiftGuidedMatcher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QHash>

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

        using ObservationId = std::uint64_t;

        ObservationId observationId(std::uint32_t image_index, std::uint32_t feature_id)
        {
            return (static_cast<ObservationId>(image_index) << 32U) | static_cast<ObservationId>(feature_id);
        }

        struct ObservationLinks
        {
            static constexpr std::uint32_t invalidImageIndex = std::numeric_limits<std::uint32_t>::max();

            std::uint32_t ensureImage(const QString& image_path)
            {
                const auto raw = imageIndexByPath.constFind(image_path);
                if (raw != imageIndexByPath.cend())
                {
                    return raw.value();
                }
                const QString normalized = QDir::cleanPath(QFileInfo(image_path).absoluteFilePath());
                auto canonical = imageIndexByNormalizedPath.constFind(normalized);
                std::uint32_t image_index = invalidImageIndex;
                if (canonical == imageIndexByNormalizedPath.cend())
                {
                    image_index = static_cast<std::uint32_t>(imageIndexByNormalizedPath.size());
                    imageIndexByNormalizedPath.insert(normalized, image_index);
                }
                else
                {
                    image_index = canonical.value();
                }
                imageIndexByPath.insert(image_path, image_index);
                return image_index;
            }

            std::uint32_t imageIndex(const QString& image_path) const
            {
                const auto found = imageIndexByPath.constFind(image_path);
                return found == imageIndexByPath.cend() ? invalidImageIndex : found.value();
            }

            QHash<QString, std::uint32_t> imageIndexByPath;
            QHash<QString, std::uint32_t> imageIndexByNormalizedPath;
            std::unordered_map<ObservationId, std::vector<ObservationId>> peers;
        };

        ObservationLinks buildReliableObservationLinks(const std::vector<MatchPhotosMatchRecord>& records)
        {
            ObservationLinks links;
            for (const MatchPhotosMatchRecord& record : records)
            {
                const std::uint32_t image_index0 = links.ensureImage(record.image0Path);
                const std::uint32_t image_index1 = links.ensureImage(record.image1Path);
                if (!record.passedGeometry || !record.pairData)
                {
                    continue;
                }
                for (const auto& correspondence : record.pairData->correspondences)
                {
                    if (!image_matching::hasFlag(correspondence.flags, image_matching::MatchRecordFlag::GeometryInlier))
                    {
                        continue;
                    }
                    const ObservationId key0 = observationId(image_index0, correspondence.observation0.featureId);
                    const ObservationId key1 = observationId(image_index1, correspondence.observation1.featureId);
                    links.peers[key0].push_back(key1);
                    links.peers[key1].push_back(key0);
                }
            }
            for (auto& [observation, peers] : links.peers)
            {
                static_cast<void>(observation);
                std::sort(peers.begin(), peers.end());
                peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
            }
            return links;
        }

        bool passesMultiViewConsistency(const ObservationLinks& links,
                                        std::uint32_t image_index0,
                                        int feature_index0,
                                        std::uint32_t image_index1,
                                        int feature_index1)
        {
            const auto found0 =
                links.peers.find(observationId(image_index0, static_cast<std::uint32_t>(feature_index0)));
            const auto found1 =
                links.peers.find(observationId(image_index1, static_cast<std::uint32_t>(feature_index1)));
            if (found0 == links.peers.end() && found1 == links.peers.end())
            {
                return true;
            }
            if (found0 == links.peers.end() || found1 == links.peers.end())
            {
                return false;
            }
            const auto& peers0 = found0->second;
            const auto& peers1 = found1->second;
            std::size_t index0 = 0;
            std::size_t index1 = 0;
            while (index0 < peers0.size() && index1 < peers1.size())
            {
                if (peers0[index0] == peers1[index1])
                {
                    return true;
                }
                if (peers0[index0] < peers1[index1])
                {
                    ++index0;
                }
                else
                {
                    ++index1;
                }
            }
            return false;
        }

        class GuidedMaskCache
        {
        public:
            GuidedMaskCache(const MatchPhotosContext& context, const MatchPhotosOptions& options)
                : _context(context), _options(options)
            {
            }

            cv::Mat find(const QString& image_path, const cv::Size& image_size)
            {
                std::shared_ptr<Entry> entry;
                {
                    std::lock_guard lock(_mutex);
                    auto found = _entries.find(image_path);
                    if (found == _entries.end())
                    {
                        entry = std::make_shared<Entry>();
                        _entries.insert(image_path, entry);
                    }
                    else
                    {
                        entry = found.value();
                    }
                }
                std::call_once(entry->once,
                               [&]()
                               {
                                   entry->mask = softenedExclusionMask(
                                       loadMaskForImage(_context, image_path, image_size), _options);
                               });
                return entry->mask;
            }

        private:
            struct Entry
            {
                std::once_flag once;
                cv::Mat mask;
            };

            const MatchPhotosContext& _context;
            const MatchPhotosOptions& _options;
            std::mutex _mutex;
            QHash<QString, std::shared_ptr<Entry>> _entries;
        };

        double elapsedSeconds(std::chrono::steady_clock::time_point start)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }

        void refreshGuidedGeometryMetadata(const MatchPhotosContext& context,
                                           const MatchPhotosOptions& options,
                                           bool model_estimated,
                                           MatchPhotosMatchRecord* record)
        {
            image_matching::PairMatchData& pair = *record->pairData;
            const int raw_count = static_cast<int>(pair.correspondences.size());
            const int inlier_count = static_cast<int>(
                std::count_if(pair.correspondences.cbegin(),
                              pair.correspondences.cend(),
                              [](const auto& correspondence)
                              {
                                  return image_matching::hasFlag(correspondence.flags,
                                                                 image_matching::MatchRecordFlag::GeometryInlier);
                              }));
            pair.rawMatchCount = static_cast<std::uint32_t>(raw_count);
            pair.geometryInlierCount = static_cast<std::uint32_t>(inlier_count);
            const bool adjacent = areSequenceAdjacent(context, options, record->image0Path, record->image1Path);
            const GeometryQualityMetrics quality = measureGeometryQuality(pair, options, adjacent);
            const GeometryQualityDecision decision = evaluateGeometryQuality(quality, options);
            pair.geometryPassed = model_estimated && decision.passed;

            record->matchCount = raw_count;
            record->geometricInlierCount = inlier_count;
            record->passedGeometry = pair.geometryPassed;
            record->settings[QStringLiteral("geometry_verified")] = true;
            record->settings[QStringLiteral("geometry_cache_reusable")] = true;
            record->settings[QStringLiteral("geometry_raw_matches")] = raw_count;
            record->settings[QStringLiteral("geometric_inliers")] = inlier_count;
            record->settings[QStringLiteral("geometry_inlier_ratio")] =
                raw_count > 0 ? static_cast<double>(inlier_count) / static_cast<double>(raw_count) : 0.0;
            record->settings[QStringLiteral("geometry_passed")] = pair.geometryPassed;
            record->settings[QStringLiteral("geometry_grid_coverage_image0")] = quality.image0GridCoverage;
            record->settings[QStringLiteral("geometry_grid_coverage_image1")] = quality.image1GridCoverage;
            record->settings[QStringLiteral("geometry_adjacent_images")] = quality.adjacentImages;
            record->settings[QStringLiteral("geometry_quality_score")] = decision.score;
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

            refreshGuidedGeometryMetadata(context, options, geometry.modelEstimated, record);
            return geometry.modelEstimated;
        }

        struct GuidedPairResult
        {
            QString outcome;
            std::uint64_t descriptorComparisons = 0;
            double policySeconds = 0.0;
            double descriptorSeconds = 0.0;
            double maskSeconds = 0.0;
            double filterSeconds = 0.0;
            double geometrySeconds = 0.0;
            int addedMatches = 0;
            int improvedPairs = 0;
            int unreliableModels = 0;
            int healthyPairsSkipped = 0;
            int degenerateModels = 0;
            int consistencyRejected = 0;
            bool processed = false;
            bool canceled = false;
        };

        GuidedPairResult processGuidedPair(const MatchPhotosContext& context,
                                           const MatchPhotosOptions& options,
                                           const ObservationLinks& observationLinks,
                                           const GuidedMatchPolicyCache& policyCache,
                                           GuidedMaskCache* maskCache,
                                           MatchPhotosMatchRecord* record,
                                           const std::function<bool()>& shouldCancel)
        {
            GuidedPairResult result;
            if (!record)
            {
                result.outcome = QStringLiteral("跳过（像对记录缺失）");
                return result;
            }
            const auto features0 = context.featureCache->find(record->image0Path);
            const auto features1 = context.featureCache->find(record->image1Path);
            if (!features0 || !features1)
            {
                result.outcome = QStringLiteral("跳过（特征缓存缺失）");
                return result;
            }

            auto phase_start = std::chrono::steady_clock::now();
            const GuidedMatchGeometryChoice geometryChoice =
                chooseGuidedMatchGeometry(context, options, policyCache, *record, *features0, *features1);
            result.policySeconds = elapsedSeconds(phase_start);
            record->settings[QStringLiteral("guided_geometry_source")] = geometryChoice.geometrySource;
            record->settings[QStringLiteral("guided_epipolar_threshold_pixels")] =
                geometryChoice.epipolarThresholdPixels;
            record->settings[QStringLiteral("guided_reference_pose_conflict")] = geometryChoice.referenceConflict;
            record->settings[QStringLiteral("guided_homography_dominant")] = geometryChoice.homographyDominant;
            if (!geometryChoice.eligible)
            {
                record->settings[QStringLiteral("guided_skip_reason")] = geometryChoice.skipReason;
                result.outcome = QStringLiteral("跳过（%1）").arg(geometryChoice.skipReason);
                result.healthyPairsSkipped = geometryChoice.autoSkippedHealthy ? 1 : 0;
                result.degenerateModels = geometryChoice.homographyDominant ? 1 : 0;
                result.unreliableModels =
                    !geometryChoice.autoSkippedHealthy && !geometryChoice.homographyDominant ? 1 : 0;
                return result;
            }

            std::vector<int> existing0;
            std::vector<int> existing1;
            existing0.reserve(record->pairData->correspondences.size());
            existing1.reserve(record->pairData->correspondences.size());
            for (const auto& correspondence : record->pairData->correspondences)
            {
                existing0.push_back(static_cast<int>(correspondence.observation0.featureId));
                existing1.push_back(static_cast<int>(correspondence.observation1.featureId));
            }

            image_matching::SiftGuidedMatchOptions guidedOptions;
            guidedOptions.epipolarThresholdPixels = geometryChoice.epipolarThresholdPixels;
            guidedOptions.maximumDescriptorRatio = std::min(0.95f, options.siftMaximumRatio);
            guidedOptions.maximumAdditionalMatches =
                std::max(256, options.maxTiePointsPerImage > 0 ? options.maxTiePointsPerImage : 5000);
            if (shouldCancel && shouldCancel())
            {
                result.outcome = QStringLiteral("已取消（描述子搜索前）");
                result.canceled = true;
                return result;
            }
            guidedOptions.shouldCancel = shouldCancel;
            phase_start = std::chrono::steady_clock::now();
            image_matching::SiftGuidedMatchResult guidedResult = image_matching::findGuidedSiftMatchesDetailed(
                *features0, *features1, geometryChoice.fundamental, existing0, existing1, guidedOptions);
            result.descriptorSeconds = elapsedSeconds(phase_start);
            result.processed = true;
            result.descriptorComparisons = guidedResult.diagnostics.descriptorComparisons;
            if (guidedResult.canceled)
            {
                result.outcome = QStringLiteral("已取消（描述子搜索中）");
                result.canceled = true;
                return result;
            }
            if (guidedResult.matches.empty())
            {
                result.outcome = QStringLiteral("完成（极线带内无新增互检匹配）");
                return result;
            }

            const bool applyTiepointMask = shouldApplyMasksToTiepoints(options);
            phase_start = std::chrono::steady_clock::now();
            const cv::Mat mask0 =
                applyTiepointMask
                    ? maskCache->find(record->image0Path, cv::Size(features0->imageWidth, features0->imageHeight))
                    : cv::Mat();
            const cv::Mat mask1 =
                applyTiepointMask
                    ? maskCache->find(record->image1Path, cv::Size(features1->imageWidth, features1->imageHeight))
                    : cv::Mat();
            result.maskSeconds = elapsedSeconds(phase_start);
            const std::uint32_t image_index0 = observationLinks.imageIndex(record->image0Path);
            const std::uint32_t image_index1 = observationLinks.imageIndex(record->image1Path);
            phase_start = std::chrono::steady_clock::now();
            int accepted = 0;
            for (const image_matching::SiftGuidedMatch& match : guidedResult.matches)
            {
                const cv::KeyPoint& keypoint0 = features0->keypoints[static_cast<std::size_t>(match.index0)];
                const cv::KeyPoint& keypoint1 = features1->keypoints[static_cast<std::size_t>(match.index1)];
                if (options.guidedRequireMultiViewConsistency &&
                    !passesMultiViewConsistency(
                        observationLinks, image_index0, match.index0, image_index1, match.index1))
                {
                    ++result.consistencyRejected;
                    continue;
                }
                const float maskWeight = applyTiepointMask ? std::min(maskPointWeight(mask0, keypoint0.pt, options),
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
                record->pairData->correspondences.push_back(correspondence);
                ++accepted;
            }
            result.filterSeconds = elapsedSeconds(phase_start);
            if (accepted == 0)
            {
                result.outcome = QStringLiteral("完成（候选被一致性或蒙版约束拒绝）");
                return result;
            }

            const int previousInliers = record->geometricInlierCount;
            phase_start = std::chrono::steady_clock::now();
            applyGuidedGeometry(*features0, *features1, context, options, record);
            const auto beforeCleanup = record->pairData->correspondences.size();
            std::erase_if(record->pairData->correspondences,
                          [](const auto& correspondence)
                          {
                              return image_matching::hasFlag(correspondence.flags,
                                                             image_matching::MatchRecordFlag::Guided) &&
                                     !image_matching::hasFlag(correspondence.flags,
                                                              image_matching::MatchRecordFlag::GeometryInlier);
                          });
            const int removed = static_cast<int>(beforeCleanup - record->pairData->correspondences.size());
            refreshGuidedGeometryMetadata(
                context, options, record->pairData->geometryModel != image_matching::GeometryModel::None, record);
            result.geometrySeconds = elapsedSeconds(phase_start);
            const int retained = accepted - removed;
            record->settings[QStringLiteral("guided_matches_added")] = retained;
            record->settings[QStringLiteral("guided_matches_rejected_by_geometry")] = removed;
            record->settings[QStringLiteral("guided_matching_completed")] = true;
            result.addedMatches = retained;
            result.improvedPairs = record->geometricInlierCount > previousInliers ? 1 : 0;
            result.outcome = QStringLiteral("完成");
            return result;
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
        if (!guidedMatchingEnabled(options.guidedMatchingMode))
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
        int healthyPairsSkipped = 0;
        int degenerateModels = 0;
        int consistencyRejected = 0;
        int completedPairs = 0;
        double policy_seconds = 0.0;
        double descriptor_seconds = 0.0;
        double mask_seconds = 0.0;
        double filter_seconds = 0.0;
        double geometry_seconds = 0.0;
        const int total_pairs = static_cast<int>(matchRecords->size());
        const int worker_count = resolveGuidedMatchingWorkers(total_pairs, std::thread::hardware_concurrency());
        QElapsedTimer stage_timer;
        stage_timer.start();
        if (shouldCancelMatchPhotos(context))
        {
            report.status = MatchPhotosStageStatus::Failed;
            report.message = QStringLiteral("SIFT 几何引导重匹配已取消：已完成 0/%1 对，剩余 %1 对").arg(total_pairs);
            return report;
        }

        const auto graph_start = std::chrono::steady_clock::now();
        const ObservationLinks observationLinks = buildReliableObservationLinks(*matchRecords);
        const double graph_seconds = elapsedSeconds(graph_start);
        const GuidedMatchPolicyCache policy_cache = buildGuidedMatchPolicyCache(context);
        GuidedMaskCache mask_cache(context, options);
        std::atomic_int next_index{0};
        std::mutex report_mutex;
        try
        {
            xjw::common::concurrency::runWorkerGroup(
                static_cast<std::size_t>(worker_count),
                [&](std::stop_token stop_token)
                {
                    while (!stop_token.stop_requested() && !shouldCancelMatchPhotos(context))
                    {
                        const int pair_index = next_index.fetch_add(1);
                        if (pair_index >= total_pairs)
                        {
                            break;
                        }
                        if (stop_token.stop_requested() || shouldCancelMatchPhotos(context))
                        {
                            break;
                        }

                        MatchPhotosMatchRecord& record = (*matchRecords)[static_cast<std::size_t>(pair_index)];
                        const QString pair_name = QStringLiteral("%1 ↔ %2").arg(
                            QFileInfo(record.image0Path).fileName(), QFileInfo(record.image1Path).fileName());
                        QElapsedTimer pair_timer;
                        pair_timer.start();

                        const GuidedPairResult pair_result = processGuidedPair(
                            context,
                            options,
                            observationLinks,
                            policy_cache,
                            &mask_cache,
                            &record,
                            [&context, stop_token]()
                            { return stop_token.stop_requested() || shouldCancelMatchPhotos(context); });

                        {
                            std::lock_guard lock(report_mutex);
                            if (pair_result.canceled)
                            {
                                const QString message =
                                    QStringLiteral("SIFT 引导匹配 %1：%2，耗时 %3 秒，已完成 %4/%5，"
                                                   "剩余 %6 对，CPU worker %7；描述子比较 %8 次")
                                        .arg(pair_name)
                                        .arg(pair_result.outcome)
                                        .arg(pair_timer.elapsed() / 1000.0, 0, 'f', 3)
                                        .arg(completedPairs)
                                        .arg(total_pairs)
                                        .arg(total_pairs - completedPairs)
                                        .arg(worker_count)
                                        .arg(static_cast<qulonglong>(pair_result.descriptorComparisons));
                                LOG_INFO(message);
                                reportMatchPhotosProgress(
                                    context, QStringLiteral("guided_match"), message, completedPairs, total_pairs);
                            }
                            else
                            {
                                processedPairs += pair_result.processed ? 1 : 0;
                                addedMatches += pair_result.addedMatches;
                                improvedPairs += pair_result.improvedPairs;
                                unreliableModels += pair_result.unreliableModels;
                                healthyPairsSkipped += pair_result.healthyPairsSkipped;
                                degenerateModels += pair_result.degenerateModels;
                                consistencyRejected += pair_result.consistencyRejected;
                                policy_seconds += pair_result.policySeconds;
                                descriptor_seconds += pair_result.descriptorSeconds;
                                mask_seconds += pair_result.maskSeconds;
                                filter_seconds += pair_result.filterSeconds;
                                geometry_seconds += pair_result.geometrySeconds;
                                ++completedPairs;
                                const int remaining_pairs = total_pairs - completedPairs;
                                const QString message =
                                    QStringLiteral("SIFT 引导匹配 [完成 %1/%2] %3：%4，耗时 %5 秒，"
                                                   "剩余 %6 对，CPU worker %7；新增内点 %8，"
                                                   "描述子比较 %9 次")
                                        .arg(completedPairs)
                                        .arg(total_pairs)
                                        .arg(pair_name)
                                        .arg(pair_result.outcome)
                                        .arg(pair_timer.elapsed() / 1000.0, 0, 'f', 3)
                                        .arg(remaining_pairs)
                                        .arg(worker_count)
                                        .arg(pair_result.addedMatches)
                                        .arg(static_cast<qulonglong>(pair_result.descriptorComparisons));
                                const int log_interval = std::max(1, total_pairs / 20);
                                if (completedPairs == 1 || completedPairs == total_pairs ||
                                    completedPairs % log_interval == 0)
                                {
                                    LOG_INFO(message);
                                }
                                reportMatchPhotosProgress(
                                    context, QStringLiteral("guided_match"), message, completedPairs, total_pairs);
                            }
                        }

                        if (pair_result.canceled)
                        {
                            break;
                        }
                    }
                });
        }
        catch (const std::exception& error)
        {
            report.status = MatchPhotosStageStatus::Failed;
            report.itemCount = addedMatches;
            report.message = QStringLiteral("SIFT 引导匹配 worker 异常：%1；已完成 %2/%3")
                                 .arg(QString::fromUtf8(error.what()))
                                 .arg(completedPairs)
                                 .arg(total_pairs);
            LOG_ERROR(report.message);
            return report;
        }
        catch (...)
        {
            report.status = MatchPhotosStageStatus::Failed;
            report.itemCount = addedMatches;
            report.message =
                QStringLiteral("SIFT 引导匹配 worker 发生未知异常；已完成 %1/%2").arg(completedPairs).arg(total_pairs);
            LOG_ERROR(report.message);
            return report;
        }

        if (shouldCancelMatchPhotos(context))
        {
            report.status = MatchPhotosStageStatus::Failed;
            report.itemCount = addedMatches;
            report.message = QStringLiteral("SIFT 几何引导重匹配已取消：已完成 %1/%2 对，耗时 %3 秒，"
                                            "剩余 %4 对，CPU worker %5")
                                 .arg(completedPairs)
                                 .arg(total_pairs)
                                 .arg(stage_timer.elapsed() / 1000.0, 0, 'f', 3)
                                 .arg(total_pairs - completedPairs)
                                 .arg(worker_count);
            LOG_INFO(report.message);
            return report;
        }

        report.status = MatchPhotosStageStatus::Completed;
        report.itemCount = addedMatches;
        report.message = QStringLiteral("SIFT 几何引导重匹配完成：检查 %1 对，保留 %2 个新增内点，"
                                        "%3 对内点增加；自动跳过健康像对 %4 对，退化模型 %5 对，"
                                        "不可用几何 %6 对，多视图一致性拒绝 %7 个；"
                                        "CPU worker %8，总耗时 %9 秒；阶段累计秒数 "
                                        "graph=%10 policy=%11 descriptor=%12 mask=%13 filter=%14 geometry=%15")
                             .arg(processedPairs)
                             .arg(addedMatches)
                             .arg(improvedPairs)
                             .arg(healthyPairsSkipped)
                             .arg(degenerateModels)
                             .arg(unreliableModels)
                             .arg(consistencyRejected)
                             .arg(worker_count)
                             .arg(stage_timer.elapsed() / 1000.0, 0, 'f', 3)
                             .arg(graph_seconds, 0, 'f', 3)
                             .arg(policy_seconds, 0, 'f', 3)
                             .arg(descriptor_seconds, 0, 'f', 3)
                             .arg(mask_seconds, 0, 'f', 3)
                             .arg(filter_seconds, 0, 'f', 3)
                             .arg(geometry_seconds, 0, 'f', 3);
        LOG_INFO(report.message);
        return report;
    }

} // namespace xjw::matchphotos
