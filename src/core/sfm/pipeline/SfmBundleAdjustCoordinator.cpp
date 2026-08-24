#include "SfmBundleAdjustCoordinator.h"
#include "HierarchicalBundleAdjuster.h"
#include "IncrementalSfmDetail.h"
#include "SfmCalibrationPreviewSampler.h"
#include "geometry/TriangulationQuality.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "geometry/SimilarityGaugeNormalizer.h"
#include "BundleAdjustAdaptiveCameraModel.h"
#include "Intersection.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"
#include "triangulation/Triangulator.h"

#include "log/Logger.h"

#include "DeterministicOpenCvRansac.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw
{

    using namespace incremental_sfm_detail;

    namespace
    {

        struct AerialCameraPlaneEstimate
        {
            bool valid = false;
            std::array<double, 3> center{{0.0, 0.0, 0.0}};
            std::array<double, 3> normal{{0.0, 0.0, 1.0}};
            double opticalAxisConcentration = 0.0;
            double planarityRatio = 1.0;
            double normalToSpanRmsRatio = 1.0;
            double normalRms = 0.0;
            double spanRms = 0.0;
        };

        AerialCameraPlaneEstimate estimateAerialCameraPlane(const std::vector<FramePinholeCamera>& cameras)
        {
            AerialCameraPlaneEstimate estimate;
            if (cameras.size() < 3)
            {
                return estimate;
            }

            std::array<double, 3> meanAxis{{0.0, 0.0, 0.0}};
            for (const FramePinholeCamera& sourceCamera : cameras)
            {
                const FramePinholeCamera camera = sourceCamera.normalizedForPositiveDepth();
                const auto center = camera.cameraCenter();
                const auto rotation = camera.cameraToWorldRotation();
                for (int axis = 0; axis < 3; ++axis)
                {
                    estimate.center[axis] += center[axis];
                    meanAxis[axis] += rotation[axis * 3 + 2];
                }
            }
            const double count = static_cast<double>(cameras.size());
            for (int axis = 0; axis < 3; ++axis)
            {
                estimate.center[axis] /= count;
                meanAxis[axis] /= count;
            }
            estimate.opticalAxisConcentration =
                std::sqrt(meanAxis[0] * meanAxis[0] + meanAxis[1] * meanAxis[1] + meanAxis[2] * meanAxis[2]);

            cv::Matx33d covariance = cv::Matx33d::zeros();
            for (const FramePinholeCamera& camera : cameras)
            {
                const auto center = camera.cameraCenter();
                const cv::Vec3d delta(
                    center[0] - estimate.center[0], center[1] - estimate.center[1], center[2] - estimate.center[2]);
                covariance += (delta * delta.t()) / count;
            }

            cv::Mat eigenvalues;
            cv::Mat eigenvectors;
            if (!cv::eigen(cv::Mat(covariance), eigenvalues, eigenvectors) || eigenvalues.total() != 3 ||
                eigenvectors.rows != 3 || eigenvectors.cols != 3)
            {
                return estimate;
            }
            const double largest = std::max(0.0, eigenvalues.at<double>(0));
            const double middle = std::max(0.0, eigenvalues.at<double>(1));
            const double smallest = std::max(0.0, eigenvalues.at<double>(2));
            const double trace = largest + middle + smallest;
            if (!std::isfinite(trace) || trace <= 1.0e-12)
            {
                return estimate;
            }

            estimate.normal = {
                {eigenvectors.at<double>(2, 0), eigenvectors.at<double>(2, 1), eigenvectors.at<double>(2, 2)}};
            estimate.planarityRatio = smallest / trace;
            estimate.normalRms = std::sqrt(smallest);
            estimate.spanRms = std::sqrt(trace);
            estimate.normalToSpanRmsRatio = estimate.normalRms / estimate.spanRms;
            estimate.valid = std::isfinite(estimate.opticalAxisConcentration) &&
                             std::isfinite(estimate.planarityRatio) && std::isfinite(estimate.normalToSpanRmsRatio) &&
                             std::isfinite(estimate.normalRms) && std::isfinite(estimate.spanRms);
            return estimate;
        }

        double cameraLayerReferenceDriftRms(const std::vector<FramePinholeCamera>& cameras,
                                            const BACameraPlaneConstraint& constraint)
        {
            if (cameras.empty() || constraint.referenceSignedDistances.size() != cameras.size())
            {
                return std::numeric_limits<double>::infinity();
            }

            double squaredSum = 0.0;
            for (std::size_t index = 0; index < cameras.size(); ++index)
            {
                const auto center = cameras[index].cameraCenter();
                const double signedDistance = constraint.normal[0] * (center[0] - constraint.point[0]) +
                                              constraint.normal[1] * (center[1] - constraint.point[1]) +
                                              constraint.normal[2] * (center[2] - constraint.point[2]);
                const double drift = signedDistance - constraint.referenceSignedDistances[index];
                squaredSum += drift * drift;
            }
            return std::sqrt(squaredSum / static_cast<double>(cameras.size()));
        }

        bool retriangulateTrackInitialPoint(const std::vector<FramePinholeCamera>& cameras, BATrack* track)
        {
            if (!track || track->observations.size() < 2)
            {
                return false;
            }

            const BAObservation* best_left = nullptr;
            const BAObservation* best_right = nullptr;
            double largest_baseline_squared = 0.0;
            for (std::size_t left_index = 0; left_index < track->observations.size(); ++left_index)
            {
                const BAObservation& left = track->observations[left_index];
                if (left.cameraIndex < 0 || left.cameraIndex >= static_cast<int>(cameras.size()))
                {
                    continue;
                }
                const auto left_center = cameras[static_cast<std::size_t>(left.cameraIndex)].cameraCenter();
                for (std::size_t right_index = left_index + 1; right_index < track->observations.size(); ++right_index)
                {
                    const BAObservation& right = track->observations[right_index];
                    if (right.cameraIndex < 0 || right.cameraIndex >= static_cast<int>(cameras.size()))
                    {
                        continue;
                    }
                    const auto right_center = cameras[static_cast<std::size_t>(right.cameraIndex)].cameraCenter();
                    const double dx = right_center[0] - left_center[0];
                    const double dy = right_center[1] - left_center[1];
                    const double dz = right_center[2] - left_center[2];
                    const double baseline_squared = dx * dx + dy * dy + dz * dz;
                    if (baseline_squared > largest_baseline_squared)
                    {
                        largest_baseline_squared = baseline_squared;
                        best_left = &left;
                        best_right = &right;
                    }
                }
            }
            if (!best_left || !best_right || largest_baseline_squared <= 1.0e-20)
            {
                return false;
            }

            const PairIntersectionCandidate candidate =
                triangulatePairWithDirectionFallback(cameras[static_cast<std::size_t>(best_left->cameraIndex)],
                                                     {{best_left->u, best_left->v}},
                                                     cameras[static_cast<std::size_t>(best_right->cameraIndex)],
                                                     {{best_right->u, best_right->v}});
            if (!candidate.valid)
            {
                return false;
            }
            track->initialPoint = candidate.point;
            return true;
        }

    } // namespace

    SfmBundleAdjustCoordinator::SfmBundleAdjustCoordinator(IncrementalSfm& owner) : _owner(owner)
    {
    }

    bool SfmBundleAdjustCoordinator::shouldRefineSharedIntrinsics(bool localOnly,
                                                                  int activeCameraCount,
                                                                  int registeredImageCount,
                                                                  int totalImageCount)
    {
        if (localOnly || totalImageCount < 3 || registeredImageCount < 3 || activeCameraCount != registeredImageCount)
        {
            return false;
        }

        if (registeredImageCount == totalImageCount)
        {
            return true;
        }

        // 大型工程最后少量影像可能正是因为零畸变/焦距种子不准而无法注册。允许在
        // 已注册率达到 98% 且缺失数不超过 2% 时释放共享内参；求解后的内参会同步到
        // 未注册影像，再由最终 PnP 重试接回。小工程仍要求完整注册，避免过早自标定。
        const int missing_image_count = totalImageCount - registeredImageCount;
        const int maximum_missing_count = std::max(1, totalImageCount / 50);
        return totalImageCount >= 50 && missing_image_count > 0 && missing_image_count <= maximum_missing_count &&
               static_cast<long long>(registeredImageCount) * 100LL >= static_cast<long long>(totalImageCount) * 98LL;
    }

    bool SfmBundleAdjustCoordinator::shouldPreserveCameraLayer(bool localOnly,
                                                               bool hasAbsoluteConstraint,
                                                               bool completeRegistration,
                                                               bool refiningSharedIntrinsics)
    {
        return !localOnly && !hasAbsoluteConstraint && completeRegistration && refiningSharedIntrinsics;
    }

    bool SfmBundleAdjustCoordinator::shouldUseLowOrderAerialSelfCalibration(bool localOnly,
                                                                            bool hasAbsoluteConstraint,
                                                                            bool completeRegistration,
                                                                            bool refiningSharedRadialDistortion,
                                                                            int activeCameraCount,
                                                                            double opticalAxisConcentration)
    {
        return !localOnly && !hasAbsoluteConstraint && completeRegistration && refiningSharedRadialDistortion &&
               activeCameraCount >= 20 && std::isfinite(opticalAxisConcentration) && opticalAxisConcentration >= 0.90;
    }

    bool SfmBundleAdjustCoordinator::shouldCorrectUnanchoredAerialDoming(bool localOnly,
                                                                         bool hasAbsoluteConstraint,
                                                                         bool completeRegistration,
                                                                         bool hasTrustedFocalPrior,
                                                                         bool refiningRadialK1Only,
                                                                         int activeCameraCount,
                                                                         double opticalAxisConcentration,
                                                                         double cameraCenterNormalSpanRmsRatio)
    {
        return !localOnly && !hasAbsoluteConstraint && completeRegistration && hasTrustedFocalPrior &&
               refiningRadialK1Only && activeCameraCount >= 20 && std::isfinite(opticalAxisConcentration) &&
               opticalAxisConcentration >= 0.90 && std::isfinite(cameraCenterNormalSpanRmsRatio) &&
               cameraCenterNormalSpanRmsRatio >= 0.0;
    }

    bool SfmBundleAdjustCoordinator::hasIterativeGlobalBaConverged(int completedRoundCount,
                                                                   double pointChangeRate,
                                                                   bool sharedIntrinsicsRefined,
                                                                   double maximumIntrinsicChange)
    {
        if (completedRoundCount < 2 || !std::isfinite(pointChangeRate) || pointChangeRate >= 0.01)
        {
            return false;
        }
        if (!sharedIntrinsicsRefined)
        {
            return true;
        }
        return std::isfinite(maximumIntrinsicChange) && maximumIntrinsicChange < 5.0e-4;
    }

    bool SfmBundleAdjustCoordinator::shouldRunPeriodicGlobalBa(int registeredImageCount,
                                                               int registrationTarget,
                                                               int iterationsSinceGlobalBa,
                                                               int globalBaInterval)
    {
        const int safe_interval = std::max(1, globalBaInterval);
        if (iterationsSinceGlobalBa < safe_interval || registeredImageCount >= registrationTarget)
        {
            return false;
        }

        const int remaining_images = registrationTarget - registeredImageCount;
        const int final_guard = std::max(2, safe_interval / 4);
        return remaining_images > final_guard;
    }

    bool SfmBundleAdjustCoordinator::shouldUseMultiViewOnlyGlobalBa(
        bool localOnly, int activeCameraCount, int totalTrackCount, int twoViewTrackCount, int multiViewTrackCount)
    {
        if (localOnly || activeCameraCount < 96 || totalTrackCount < 250000 || twoViewTrackCount < 0 ||
            multiViewTrackCount < 0 || twoViewTrackCount + multiViewTrackCount != totalTrackCount)
        {
            return false;
        }

        // 两视图点经 Schur 消元后仍会向相机网提供一条有效约束，弱网中过早丢弃会
        // 进一步削弱块间连接。只对超大问题启用缩减，并要求每台相机至少有 64 条
        // 多视轨迹；中等规模工程保留全部轨迹以优先保证整体几何。
        const bool weakNetwork =
            static_cast<long long>(twoViewTrackCount) * 100LL >= static_cast<long long>(totalTrackCount) * 65LL;
        const int minimumMultiViewTracks = std::max(5000, activeCameraCount * 64);
        return weakNetwork && multiViewTrackCount >= minimumMultiViewTracks;
    }

    bool SfmBundleAdjustCoordinator::shouldAcceptTrackConsolidation(std::size_t oldPointCount,
                                                                    std::size_t oldObservationCount,
                                                                    std::size_t oldLongTrackCount,
                                                                    std::size_t newPointCount,
                                                                    std::size_t newObservationCount,
                                                                    std::size_t newLongTrackCount)
    {
        if (newPointCount == 0 || newObservationCount < newPointCount * 2)
        {
            return false;
        }
        if (oldPointCount == 0)
        {
            return true;
        }

        // 合并可能把同一输入组件对应的多个二视图点重新变成一个长轨迹，因此点数
        // 允许下降，但不能以丢失大部分观测为代价。真正需要增加的是跨相机冗余：
        // 每个点超过最小二视观测的部分，以及至少三视的轨迹数量。
        if (newPointCount * 2 < oldPointCount || newObservationCount * 10 < oldObservationCount * 7 ||
            newLongTrackCount < oldLongTrackCount)
        {
            return false;
        }

        const std::size_t oldRedundantObservations =
            oldObservationCount > oldPointCount * 2 ? oldObservationCount - oldPointCount * 2 : 0;
        const std::size_t newRedundantObservations = newObservationCount - newPointCount * 2;
        const bool longTrackGain = oldLongTrackCount == 0 ? newLongTrackCount > 0
                                                          : static_cast<double>(newLongTrackCount) >=
                                                                static_cast<double>(oldLongTrackCount) * 1.10;
        const bool redundancyGain =
            oldRedundantObservations == 0
                ? newRedundantObservations > 0
                : static_cast<double>(newRedundantObservations) >= static_cast<double>(oldRedundantObservations) * 1.25;
        return longTrackGain || redundancyGain;
    }

    int SfmBundleAdjustCoordinator::iterativeGlobalBaRoundLimit(int configuredRounds,
                                                                bool finalRefinement,
                                                                int activeCameraCount)
    {
        const int safe_rounds = std::max(1, configuredRounds);
        if (finalRefinement)
        {
            return safe_rounds;
        }
        return activeCameraCount > 96 ? 1 : std::min(2, safe_rounds);
    }

    SfmAdaptiveCameraModelDiagnosticMergeResult SfmBundleAdjustCoordinator::mergeAdaptiveCameraModelDiagnostics(
        const SfmAdaptiveCameraModelDiagnosticSnapshot& previous,
        const SfmAdaptiveCameraModelDiagnosticSnapshot& current)
    {
        SfmAdaptiveCameraModelDiagnosticMergeResult result;
        result.accumulated = previous;
        if (!current.evaluated)
        {
            return result;
        }

        result.accumulated.evaluated = true;
        result.accumulated.applied = previous.applied || current.applied;
        result.shouldReplaceEvidence = current.applied || !previous.applied;
        if (result.shouldReplaceEvidence)
        {
            result.accumulated.parameterMask = current.parameterMask;
            result.accumulated.modelName = current.modelName;
        }
        return result;
    }

    void
    SfmBundleAdjustCoordinator::refreshCalibrationSeedApplicationCount(const std::vector<FramePinholeCamera>& before,
                                                                       BAResult* result)
    {
        if (!result || before.size() != result->refinedCameras.size())
        {
            return;
        }

        int changedCount = 0;
        for (std::size_t index = 0; index < before.size(); ++index)
        {
            const FramePinholeCamera::Intrinsics initial = before[index].intrinsics();
            const FramePinholeCamera::Intrinsics refined = result->refinedCameras[index].intrinsics();
            const FramePinholeCamera::Distortion initialDistortion = before[index].distortion();
            const FramePinholeCamera::Distortion refinedDistortion = result->refinedCameras[index].distortion();
            const bool changed =
                std::abs(refined.focalX - initial.focalX) > 1.0e-8 * std::max(1.0, std::abs(initial.focalX)) ||
                std::abs(refined.focalY - initial.focalY) > 1.0e-8 * std::max(1.0, std::abs(initial.focalY)) ||
                std::abs(refined.principalX - initial.principalX) > 1.0e-8 ||
                std::abs(refined.principalY - initial.principalY) > 1.0e-8 ||
                std::abs(refinedDistortion.radialK1 - initialDistortion.radialK1) > 1.0e-10 ||
                std::abs(refinedDistortion.radialK2 - initialDistortion.radialK2) > 1.0e-10 ||
                std::abs(refinedDistortion.radialK3 - initialDistortion.radialK3) > 1.0e-10 ||
                std::abs(refinedDistortion.tangentialP1 - initialDistortion.tangentialP1) > 1.0e-10 ||
                std::abs(refinedDistortion.tangentialP2 - initialDistortion.tangentialP2) > 1.0e-10;
            if (changed)
            {
                ++changedCount;
            }
        }
        result->refinedIntrinsicCount = changedCount;
    }

    bool SfmBundleAdjustCoordinator::hasReusableCalibrationSeed(const std::vector<FramePinholeCamera>& current,
                                                                const std::vector<FramePinholeCamera>& stableReferences,
                                                                bool focalEnabled,
                                                                bool radialK1Enabled)
    {
        if (current.empty() || current.size() != stableReferences.size())
        {
            return false;
        }

        for (std::size_t index = 0; index < current.size(); ++index)
        {
            if (focalEnabled)
            {
                const FramePinholeCamera::Intrinsics value = current[index].intrinsics();
                const FramePinholeCamera::Intrinsics reference = stableReferences[index].intrinsics();
                if (std::abs(value.focalX - reference.focalX) > 1.0e-8 * std::max(1.0, std::abs(reference.focalX)) ||
                    std::abs(value.focalY - reference.focalY) > 1.0e-8 * std::max(1.0, std::abs(reference.focalY)))
                {
                    return true;
                }
            }
            if (radialK1Enabled)
            {
                const FramePinholeCamera::Distortion value = current[index].distortion();
                const FramePinholeCamera::Distortion reference = stableReferences[index].distortion();
                if (std::abs(value.radialK1) > 1.0e-8 || std::abs(reference.radialK1) > 1.0e-8)
                {
                    return true;
                }
            }
        }
        return false;
    }

    int SfmBundleAdjustCoordinator::selfCalibrationIterationBudget(int configuredIterations, bool hasReusableSeed)
    {
        const int safe_iterations = std::max(1, configuredIterations);
        return hasReusableSeed ? safe_iterations : std::max(60, safe_iterations);
    }

    std::vector<FramePinholeCamera> SfmBundleAdjustCoordinator::buildPersistentIntrinsicReferences(
        const std::vector<ImageId>& imageIds,
        const std::vector<FramePinholeCamera>& current,
        std::unordered_map<ImageId, FramePinholeCamera>* referencesByImageId)
    {
        if (!referencesByImageId || imageIds.size() != current.size())
        {
            return {};
        }

        std::vector<FramePinholeCamera> references;
        references.reserve(current.size());
        for (std::size_t index = 0; index < imageIds.size(); ++index)
        {
            const auto [iterator, inserted] = referencesByImageId->try_emplace(imageIds[index], current[index]);
            (void)inserted;
            references.push_back(iterator->second);
        }
        return references;
    }

    double
    SfmBundleAdjustCoordinator::maximumCameraIntrinsicChange(const std::vector<FramePinholeCamera>& previous,
                                                             const std::vector<FramePinholeCamera>& current,
                                                             const std::vector<FramePinholeCamera>& stableReferences)
    {
        if (previous.empty() || previous.size() != current.size() || previous.size() != stableReferences.size())
        {
            return std::numeric_limits<double>::infinity();
        }

        double maximumChange = 0.0;
        for (std::size_t index = 0; index < current.size(); ++index)
        {
            const FramePinholeCamera::Intrinsics before = previous[index].intrinsics();
            const FramePinholeCamera::Intrinsics after = current[index].intrinsics();
            const FramePinholeCamera::Intrinsics reference = stableReferences[index].intrinsics();
            const FramePinholeCamera::Distortion beforeDistortion = previous[index].distortion();
            const FramePinholeCamera::Distortion afterDistortion = current[index].distortion();
            const double focalScale = std::max({1.0, std::abs(reference.focalX), std::abs(reference.focalY)});
            const double beforeAspect = before.focalX > 1.0e-12 ? before.focalY / before.focalX : 1.0;
            const double afterAspect = after.focalX > 1.0e-12 ? after.focalY / after.focalX : 1.0;
            const double cameraChange =
                std::max({std::abs(after.focalX - before.focalX) / focalScale,
                          std::abs(after.focalY - before.focalY) / focalScale,
                          std::abs(afterAspect - beforeAspect),
                          std::abs(after.principalX - before.principalX) / focalScale,
                          std::abs(after.principalY - before.principalY) / focalScale,
                          std::abs(afterDistortion.radialK1 - beforeDistortion.radialK1),
                          std::abs(afterDistortion.radialK2 - beforeDistortion.radialK2),
                          std::abs(afterDistortion.radialK3 - beforeDistortion.radialK3),
                          std::abs(afterDistortion.tangentialP1 - beforeDistortion.tangentialP1),
                          std::abs(afterDistortion.tangentialP2 - beforeDistortion.tangentialP2)});
            if (!std::isfinite(cameraChange))
            {
                return std::numeric_limits<double>::infinity();
            }
            maximumChange = std::max(maximumChange, cameraChange);
        }
        return maximumChange;
    }

    void SfmBundleAdjustCoordinator::run(bool localOnly, const std::vector<ImageId>& anchorIds)
    {
        _owner.runBundleAdjust(localOnly, anchorIds);
    }

    void SfmBundleAdjustCoordinator::iterative(bool finalRefinement)
    {
        _owner.iterativeGlobalBA(finalRefinement);
    }

    int SfmBundleAdjustCoordinator::filterNegativeDepthPoints()
    {
        return _owner.filterNegativeDepthPoints();
    }

    bool SfmBundleAdjustCoordinator::consolidateInputTracksForFinalBa()
    {
        if (_owner._finalTrackConsolidationAttempted)
        {
            return false;
        }
        _owner._finalTrackConsolidationAttempted = true;

        if (_owner._inputMultiViewTracks.empty())
        {
            Logger::instance()->info("[SFM] Multiview track consolidation skipped reason=no_retained_input_tracks");
            return false;
        }
        if (_owner._controlNetworkApplied || !_owner._pendingPriorTracks.empty() ||
            !_owner._pendingPriorScaleBars.empty() || !_owner._materializedPriorTracks.empty())
        {
            Logger::instance()->info(
                "[SFM] Multiview track consolidation skipped reason=absolute_or_manual_constraints_present");
            return false;
        }

        std::vector<Track> registeredTracks;
        registeredTracks.reserve(_owner._inputMultiViewTracks.size());
        for (const Track& inputTrack : _owner._inputMultiViewTracks)
        {
            Track registeredTrack;
            registeredTrack.confidence = inputTrack.confidence;
            registeredTrack.source = inputTrack.source;
            registeredTrack.sourceId = inputTrack.sourceId;
            registeredTrack.elements.reserve(inputTrack.elements.size());
            for (const TrackElement& element : inputTrack.elements)
            {
                if (!_owner._reconstruction->isRegistered(element.imageId) ||
                    !_owner._reconstruction->hasCamera(element.imageId) ||
                    !_owner._reconstruction->hasImage(element.imageId))
                {
                    continue;
                }
                const ImageData& image = _owner._reconstruction->image(element.imageId);
                if (element.featureIdx >= image.keypoints.size() || element.featureIdx >= image.point3DIds.size())
                {
                    continue;
                }
                registeredTrack.elements.push_back(element);
            }
            if (registeredTrack.length() >= 2)
            {
                registeredTracks.push_back(std::move(registeredTrack));
            }
        }

        if (registeredTracks.empty())
        {
            Logger::instance()->info(
                "[SFM] Multiview track consolidation skipped reason=no_tracks_on_registered_cameras");
            return false;
        }

        auto measureNetwork = [](const SfmReconstruction& reconstruction)
        {
            std::array<std::size_t, 3> statistics{{0, 0, 0}};
            statistics[0] = reconstruction.numPoints3D();
            for (const auto& [pointId, point] : reconstruction.points3D())
            {
                (void)pointId;
                statistics[1] += point.track.length();
                if (point.track.length() >= 3)
                {
                    ++statistics[2];
                }
            }
            return statistics;
        };

        const std::array<std::size_t, 3> oldNetwork = measureNetwork(*_owner._reconstruction);
        const auto reconstructionSnapshot = std::make_shared<SfmReconstruction>(*_owner._reconstruction);
        for (Point3DId pointId : _owner._reconstruction->allPoint3DIds())
        {
            _owner._reconstruction->deletePoint3D(pointId);
        }

        TriangulatorOptions options = _owner._sfmOptions.triangulatorOptions;
        options.maxReprojError =
            std::max(options.maxReprojError, std::max(4.0, _owner._sfmOptions.filterMaxReprojError * 2.0));
        options.completeMaxReprojError =
            std::max(options.completeMaxReprojError, std::max(12.0, _owner._sfmOptions.filterMaxReprojError * 6.0));
        Triangulator triangulator(
            *_owner._reconstruction, _owner._correspondenceGraph, _owner._sfmOptions.baOptions.numThreads);
        const TriangulationStats triangulation = triangulator.triangulateTracks(registeredTracks, options);
        const std::array<std::size_t, 3> newNetwork = measureNetwork(*_owner._reconstruction);

        const bool accepted = shouldAcceptTrackConsolidation(
            oldNetwork[0], oldNetwork[1], oldNetwork[2], newNetwork[0], newNetwork[1], newNetwork[2]);
        Logger::instance()->infof("[SFM] Multiview track consolidation accepted=%s inputTracks=%zu usableTracks=%zu "
                                  "points=%zu->%zu observations=%zu->%zu longTracks=%zu->%zu "
                                  "createdTwoView=%d createdLong=%d deferredPureTwoView=%d "
                                  "suppressedTwoViewFragments=%d "
                                  "indirectTwoViewCandidates=%d unstableTwoViewCandidates=%d "
                                  "localDepthInconsistentTwoViewPoints=%d "
                                  "rejectedTracks=%d reprojRejected=%d",
                                  accepted ? "true" : "false",
                                  _owner._inputMultiViewTracks.size(),
                                  registeredTracks.size(),
                                  oldNetwork[0],
                                  newNetwork[0],
                                  oldNetwork[1],
                                  newNetwork[1],
                                  oldNetwork[2],
                                  newNetwork[2],
                                  triangulation.createdTwoViewTracks,
                                  triangulation.createdLongTracks,
                                  triangulation.deferredPureTwoViewTracks,
                                  triangulation.suppressedTwoViewFragments,
                                  triangulation.indirectTwoViewCandidates,
                                  triangulation.unstableTwoViewCandidates,
                                  triangulation.localDepthInconsistentTwoViewPoints,
                                  triangulation.noCandidateTracks,
                                  triangulation.reprojObservationRejected);

        if (!accepted)
        {
            _owner._reconstruction = reconstructionSnapshot;
        }
        _owner.invalidateVisibilityCache();
        return accepted;
    }

    void IncrementalSfm::runBundleAdjust(bool localOnly,
                                         const std::vector<ImageId>& anchorIds,
                                         const std::vector<FramePinholeCamera>* stableIntrinsicReferences,
                                         bool allowCalibrationSeedSearch)
    {
        const char* scopeName = localOnly ? "local" : "global";
        const auto reportSkipped = [this, localOnly, scopeName](const std::string& reason)
        {
            Logger::instance()->infof("[BA] skipped scope=%s reason=%s", scopeName, reason.c_str());
            if (!localOnly)
            {
                _lastGlobalBARmsBefore = 0.0;
                _lastGlobalBARmsAfter = 0.0;
                _lastGlobalBATracksTotal = 0;
                _lastGlobalBATracksOptimized = 0;
                _lastGlobalBATracksFiltered = 0;
                _lastGlobalBARefinedIntrinsicCount = 0;
                _lastGlobalBASharedFocalScale = 1.0;
                _lastGlobalBASharedFocalAspectScale = 1.0;
                _lastGlobalBASharedPrincipalOffsetX = 0.0;
                _lastGlobalBASharedPrincipalOffsetY = 0.0;
                _lastGlobalBASharedRadialK1 = 0.0;
                _lastGlobalBASharedRadialK2 = 0.0;
                _lastGlobalBASharedRadialK3 = 0.0;
                _lastGlobalBASharedTangentialP1 = 0.0;
                _lastGlobalBASharedTangentialP2 = 0.0;
                _lastGlobalBARequestedBackend = _sfmOptions.baOptions.backend;
                _lastGlobalBAUsedBackend = BABackend::LegacyCpu;
                _lastGlobalBASolveStatus = BASolveStatus::NotRun;
                _lastGlobalBASolutionUsable = false;
                _lastGlobalBAResultApplied = false;
                _lastGlobalBABackendFallback = false;
                _lastGlobalBAObservationCount = 0;
                _lastGlobalBATotalSeconds = 0.0;
                _lastGlobalBABackendMessage = reason;
                // 迭代全局 BA 的后续轮可能因点网已收敛而跳过；此时保留此前
                // 已完成的自适应模型评估/应用事实，避免把整次精化误报为未执行。
                SfmAdaptiveCameraModelDiagnosticSnapshot previousDiagnostics;
                previousDiagnostics.evaluated = _lastGlobalBAAdaptiveCameraModelFittingEvaluated;
                previousDiagnostics.applied = _lastGlobalBAAdaptiveCameraModelFittingApplied;
                previousDiagnostics.parameterMask = _lastGlobalBAIntrinsicParameterMask;
                previousDiagnostics.modelName = _lastGlobalBAAdaptiveCameraModel;
                const auto mergedDiagnostics =
                    SfmBundleAdjustCoordinator::mergeAdaptiveCameraModelDiagnostics(previousDiagnostics, {});
                _lastGlobalBAAdaptiveCameraModelFittingEvaluated = mergedDiagnostics.accumulated.evaluated;
                _lastGlobalBAAdaptiveCameraModelFittingApplied = mergedDiagnostics.accumulated.applied;
                _lastGlobalBAIntrinsicParameterMask = mergedDiagnostics.accumulated.parameterMask;
                _lastGlobalBAAdaptiveCameraModel = mergedDiagnostics.accumulated.modelName;
                if (!mergedDiagnostics.accumulated.evaluated)
                {
                    _lastGlobalBAIntrinsicParameterMask.fill(false);
                    _lastGlobalBAIntrinsicParameterReliability.fill(0.0);
                    _lastGlobalBAIntrinsicParameterIncrementalInformationScore.fill(0.0);
                    _lastGlobalBAIntrinsicParameterSensitivity.fill(0.0);
                    _lastGlobalBAAdaptiveCameraModel.clear();
                    _lastGlobalBAAdaptiveCameraModelReason.clear();
                    _lastGlobalBACameraModelGeometryStrength = 0.0;
                    _lastGlobalBACameraModelOpticalAxisConcentration = 0.0;
                    _lastGlobalBACameraModelMedianTriangulationAngle = 0.0;
                    _lastGlobalBACameraModelNormalizedRadiusP90 = 0.0;
                    _lastGlobalBACameraModelOccupiedPeripheralSectors = 0;
                    _lastGlobalBACameraModelObservationCount = 0;
                    _lastGlobalBACameraModelMultiViewTrackRatio = 0.0;
                    _lastGlobalBACameraModelObservationSupport = 0.0;
                    _lastGlobalBACameraModelPeripheralCoverage = 0.0;
                    _lastGlobalBACameraModelSectorCoverage = 0.0;
                    _lastGlobalBACameraModelImageAxisBalance = 0.0;
                }
            }
        };

        // 收集参与 BA 的图像 ID
        std::vector<ImageId> baImageIds;
        if (localOnly && !anchorIds.empty())
        {
            // 局部 BA：收集锚定图像及其邻居
            std::unordered_set<ImageId> baSet;
            for (ImageId aid : anchorIds)
            {
                baSet.insert(aid);
                // 按匹配数选取前 N 个已注册邻居
                auto topN =
                    _correspondenceGraph.topConnectedImages(aid, static_cast<size_t>(_sfmOptions.localBANumImages));
                for (auto& [nid, _] : topN)
                {
                    if (_reconstruction->isRegistered(nid))
                        baSet.insert(nid);
                }
            }
            baImageIds.assign(baSet.begin(), baSet.end());
        }
        else
        {
            // 全局 BA：所有已注册图像
            baImageIds = _reconstruction->registeredImageIds();
        }

        std::sort(baImageIds.begin(), baImageIds.end());
        baImageIds.erase(std::unique(baImageIds.begin(), baImageIds.end()), baImageIds.end());

        if (baImageIds.size() < 2)
        {
            reportSkipped("fewer_than_two_registered_cameras");
            return;
        }

        // 局部 BA 把与活动图像共享三维点最多的两台外部相机作为固定边界。
        // 这样既保留局部问题规模，也避免局部块在 7 自由度 gauge 下整体漂移或缩放。
        std::vector<ImageId> fixedBoundaryImageIds;
        if (localOnly)
        {
            const std::unordered_set<ImageId> activeImageIds(baImageIds.begin(), baImageIds.end());
            std::unordered_map<ImageId, int> boundaryObservationCounts;
            for (Point3DId pointId : _reconstruction->allPoint3DIds())
            {
                if (!_reconstruction->hasPoint3D(pointId))
                {
                    continue;
                }
                const ScenePoint3D& point = _reconstruction->point3D(pointId);
                bool touchesActiveImage = false;
                for (const TrackElement& element : point.track.elements)
                {
                    if (activeImageIds.count(element.imageId) > 0)
                    {
                        touchesActiveImage = true;
                        break;
                    }
                }
                if (!touchesActiveImage)
                {
                    continue;
                }
                for (const TrackElement& element : point.track.elements)
                {
                    if (activeImageIds.count(element.imageId) == 0 && _reconstruction->isRegistered(element.imageId))
                    {
                        ++boundaryObservationCounts[element.imageId];
                    }
                }
            }

            std::vector<std::pair<ImageId, int>> rankedBoundaryImages(boundaryObservationCounts.begin(),
                                                                      boundaryObservationCounts.end());
            std::sort(rankedBoundaryImages.begin(),
                      rankedBoundaryImages.end(),
                      [](const auto& left, const auto& right)
                      {
                          if (left.second != right.second)
                          {
                              return left.second > right.second;
                          }
                          return left.first < right.first;
                      });
            for (const auto& [imageId, observationCount] : rankedBoundaryImages)
            {
                if (observationCount <= 0 || fixedBoundaryImageIds.size() >= 2)
                {
                    break;
                }
                fixedBoundaryImageIds.push_back(imageId);
                baImageIds.push_back(imageId);
            }
        }

        // 构造 imageId → BA 内部相机索引的映射
        std::unordered_map<ImageId, int> idToIdx;
        std::vector<FramePinholeCamera> baCameras;
        for (size_t i = 0; i < baImageIds.size(); ++i)
        {
            idToIdx[baImageIds[i]] = static_cast<int>(i);
            baCameras.push_back(_reconstruction->camera(baImageIds[i]));
        }

        if (_sfmOptions.useKnownCameraPoses && !localOnly && !_controlNetworkApplied)
        {
            alignReconstructionToKnownPosePriors(baImageIds, &baCameras);
        }
        if (!localOnly)
        {
            // 控制网绝对定向必须在构造 BA 点之前执行，使相机、普通点和标记点处于同一物方坐标系。
            tryApplyControlNetwork(baImageIds, &baCameras);
        }

        // 收集轨迹（同时记录 trackIdx → Point3DId 的映射，用于回写和过滤）
        std::vector<BATrack> baTracks;
        std::vector<Point3DId> trackToPid; // 与 baTracks 索引对应
        std::unordered_map<std::string, int> marker_track_indices;

        const auto allPtIds = _reconstruction->allPoint3DIds();
        int globalCandidateTrackCount = 0;
        int globalTwoViewTrackCount = 0;
        int globalMultiViewTrackCount = 0;
        int control_constraint_count = 0;
        std::unordered_map<std::string, const control_points::MarkerResidual*> control_residual_by_marker;
        control_residual_by_marker.reserve(_controlNetworkResult.controlResiduals.size());
        for (const control_points::MarkerResidual& residual : _controlNetworkResult.controlResiduals)
        {
            control_residual_by_marker[residual.markerId] = &residual;
        }
        for (Point3DId pid : allPtIds)
        {
            if (!_reconstruction->hasPoint3D(pid))
                continue;
            const ScenePoint3D& pt = _reconstruction->point3D(pid);
            BATrack track;
            track.initialPoint = pt.xyz;

            if (_controlNetworkApplied && pt.track.source == TrackSource::PriorMarker)
            {
                const control_points::PriorTrack* prior = priorTrack(pt.track.sourceId);
                const auto residual = control_residual_by_marker.find(pt.track.sourceId);
                if (prior && prior->role == control_points::MarkerRole::ControlPoint && prior->hasReference &&
                    prior->referenceUsable && residual != control_residual_by_marker.end() && residual->second->inlier)
                {
                    double sigma_sum_squared = 0.0;
                    int sigma_count = 0;
                    for (double sigma : prior->referenceSigma)
                    {
                        if (std::isfinite(sigma) && sigma > 0.0)
                        {
                            sigma_sum_squared += sigma * sigma;
                            ++sigma_count;
                        }
                    }
                    BAControlPointConstraint constraint;
                    constraint.point = prior->referencePoint;
                    constraint.sigmaMeters =
                        sigma_count > 0 ? std::sqrt(sigma_sum_squared / static_cast<double>(sigma_count)) : 1.0;
                    constraint.weight = 1.0;
                    constraint.sourceIndex = static_cast<int>(
                        prior - static_cast<const control_points::PriorTrack*>(_pendingPriorTracks.data()));
                    track.controlPointConstraints.push_back(constraint);
                    ++control_constraint_count;
                }
            }

            for (const auto& elem : pt.track.elements)
            {
                auto idxIt = idToIdx.find(elem.imageId);
                if (idxIt == idToIdx.end())
                    continue;

                const ImageData& img = _reconstruction->image(elem.imageId);
                if (elem.featureIdx >= img.keypoints.size())
                    continue;

                BAObservation obs;
                obs.cameraIndex = idxIt->second;
                obs.u = img.keypoints[elem.featureIdx].x;
                obs.v = img.keypoints[elem.featureIdx].y;
                obs.weight = pt.track.confidence;
                track.observations.push_back(obs);
            }

            // 先收集全部有效轨迹，同时完成全局网络统计；是否丢弃普通两视图点在本轮
            // 扫描结束后统一决定，避免为策略判断再次遍历整个三维点网。
            if (track.observations.size() >= 2)
            {
                if (!localOnly)
                {
                    ++globalCandidateTrackCount;
                    if (track.observations.size() == 2)
                    {
                        ++globalTwoViewTrackCount;
                    }
                    else
                    {
                        ++globalMultiViewTrackCount;
                    }
                }
                if (pt.track.source == TrackSource::PriorMarker && !pt.track.sourceId.empty())
                {
                    marker_track_indices[pt.track.sourceId] = static_cast<int>(baTracks.size());
                }
                baTracks.push_back(std::move(track));
                trackToPid.push_back(pid);
            }
        }

        const bool useMultiViewOnlyGlobalBa =
            SfmBundleAdjustCoordinator::shouldUseMultiViewOnlyGlobalBa(localOnly,
                                                                       static_cast<int>(baCameras.size()),
                                                                       globalCandidateTrackCount,
                                                                       globalTwoViewTrackCount,
                                                                       globalMultiViewTrackCount);
        if (useMultiViewOnlyGlobalBa)
        {
            std::vector<BATrack> filtered_tracks;
            std::vector<Point3DId> filtered_point_ids;
            filtered_tracks.reserve(baTracks.size());
            filtered_point_ids.reserve(trackToPid.size());
            marker_track_indices.clear();
            for (std::size_t index = 0; index < baTracks.size(); ++index)
            {
                const ScenePoint3D& point = _reconstruction->point3D(trackToPid[index]);
                if (baTracks[index].observations.size() < 3 && point.track.source != TrackSource::PriorMarker)
                {
                    continue;
                }
                if (point.track.source == TrackSource::PriorMarker && !point.track.sourceId.empty())
                {
                    marker_track_indices[point.track.sourceId] = static_cast<int>(filtered_tracks.size());
                }
                filtered_tracks.push_back(std::move(baTracks[index]));
                filtered_point_ids.push_back(trackToPid[index]);
            }
            baTracks = std::move(filtered_tracks);
            trackToPid = std::move(filtered_point_ids);
        }
        if (!localOnly)
        {
            Logger::instance()->infof("[BA] network scope=global policy=%s candidateTracks=%d twoView=%d multiView=%d",
                                      useMultiViewOnlyGlobalBa ? "multi_view_only" : "all_tracks",
                                      globalCandidateTrackCount,
                                      globalTwoViewTrackCount,
                                      globalMultiViewTrackCount);
        }

        if (baTracks.empty())
        {
            reportSkipped("no_tracks_with_two_or_more_observations");
            return;
        }

        // 构造本次 BA 选项，并显式消除无绝对约束问题的 7 自由度 gauge。
        BAOptions baOpt = _sfmOptions.baOptions;
        if (stableIntrinsicReferences)
        {
            baOpt.sharedIntrinsicReferenceCameras = *stableIntrinsicReferences;
        }
        BAAdaptiveCameraModelAssessment adaptiveCameraModelAssessment;
        bool adaptiveCameraModelFittingEvaluated = false;
        BAIntrinsicParameterMask effectiveIntrinsicParameterMask{};
        std::string effectiveAdaptiveCameraModel = "fixed";
        if (localOnly)
        {
            // 局部窗口只负责稳定新注册相机，不需要沿用最终全局 BA 的 20 轮预算。
            // 这也避免数百图工程在每个局部窗口输出整段迭代日志，造成“逐图平差”的错觉。
            baOpt.maxIterations = std::min(baOpt.maxIterations, 10);
            baOpt.logIterationProgress = false;
        }
        const bool refineSharedIntrinsics = SfmBundleAdjustCoordinator::shouldRefineSharedIntrinsics(
            localOnly,
            static_cast<int>(baCameras.size()),
            static_cast<int>(_reconstruction->registeredImageIds().size()),
            static_cast<int>(_reconstruction->numImages()));
        const bool keepKnownPoseIntrinsicsFixed =
            _sfmOptions.useKnownCameraPoses && _sfmOptions.keepIntrinsicsFixedInKnownPoseBa;
        if (!refineSharedIntrinsics || keepKnownPoseIntrinsicsFixed)
        {
            // 局部/未完整注册阶段只优化位姿和物点；完整已知位姿路径还要遵守调用方
            // 的固定标定契约。两者都不能把 adaptive 最大模型的残余开关带入 BA。
            baOpt.refineSharedFocalLength = false;
            baOpt.refineSharedFocalAspectRatio = false;
            baOpt.refineSharedPrincipalPoint = false;
            baOpt.refineSharedRadialDistortion = false;
            baOpt.refineSharedHighOrderDistortion = false;
            baOpt.useSharedIntrinsicParameterMask = false;
            baOpt.cameraCalibrationGroupIds.clear();
            baOpt.sharedIntrinsicReferenceCameras.clear();
        }
        else if (_sfmOptions.adaptiveCameraModelFitting)
        {
            adaptiveCameraModelAssessment = assessAdaptiveCameraModel(baCameras, baTracks, &baOpt);
            applyAdaptiveCameraModel(adaptiveCameraModelAssessment, &baOpt);
            if (!baOpt.sharedIntrinsicReferenceCameras.empty() &&
                !restoreInactiveAdaptiveIntrinsics(
                    &baCameras, baOpt.sharedIntrinsicReferenceCameras, baOpt.sharedIntrinsicParameterMask))
            {
                Logger::instance()->warn("[BA] adaptive intrinsic restore skipped: stable reference size mismatch");
            }
            adaptiveCameraModelFittingEvaluated = true;
            effectiveIntrinsicParameterMask = baOpt.sharedIntrinsicParameterMask;
            effectiveAdaptiveCameraModel = adaptiveCameraModelName(effectiveIntrinsicParameterMask);

            std::ostringstream parameterSummary;
            for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
            {
                if (index > 0)
                {
                    parameterSummary << ',';
                }
                const auto parameter = static_cast<BAIntrinsicParameter>(index);
                parameterSummary << baIntrinsicParameterName(parameter) << '='
                                 << adaptiveCameraModelAssessment.reliability[index]
                                 << (effectiveIntrinsicParameterMask[index] ? ":on" : ":off");
            }
            Logger::instance()->infof("[BA] adaptive_camera_model scope=global valid=%s model=%s "
                                      "reason=%s geometry=%.4f opticalAxis=%.4f triAngle=%.3f "
                                      "radiusP90=%.4f peripheralRadius=%.4f distortionScale=%.3f "
                                      "sectors=%d/8 cameras=%d/%d observations=%d "
                                      "multiView=%.4f axisBalance=%.4f absoluteConstraint=%s "
                                      "domingGuard=%s parameters=%s",
                                      adaptiveCameraModelAssessment.valid ? "true" : "false",
                                      effectiveAdaptiveCameraModel.c_str(),
                                      adaptiveCameraModelAssessment.reason.c_str(),
                                      adaptiveCameraModelAssessment.geometryStrength,
                                      adaptiveCameraModelAssessment.opticalAxisConcentration,
                                      adaptiveCameraModelAssessment.medianTriangulationAngleDegrees,
                                      adaptiveCameraModelAssessment.normalizedRadiusP90,
                                      adaptiveCameraModelAssessment.peripheralRadiusThreshold,
                                      adaptiveCameraModelAssessment.lowOrderDistortionScale,
                                      adaptiveCameraModelAssessment.occupiedPeripheralSectors,
                                      adaptiveCameraModelAssessment.activeCameraCount,
                                      adaptiveCameraModelAssessment.cameraCount,
                                      adaptiveCameraModelAssessment.observationCount,
                                      adaptiveCameraModelAssessment.multiViewTrackRatio,
                                      adaptiveCameraModelAssessment.imageAxisBalance,
                                      adaptiveCameraModelAssessment.hasAbsoluteGeometryConstraint ? "true" : "false",
                                      adaptiveCameraModelAssessment.unanchoredParallelAerialGuardApplied ? "true"
                                                                                                         : "false",
                                      parameterSummary.str().c_str());
        }
        const bool seedSharedFocal = sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::FocalLength);
        const bool seedSharedK1 = sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::RadialK1);
        const bool refiningSharedCameraModel =
            refineSharedIntrinsics &&
            (seedSharedFocal || sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::FocalAspectRatio) ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::PrincipalPointX) ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::PrincipalPointY) || seedSharedK1 ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::RadialK2) ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::RadialK3) ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::TangentialP1) ||
             sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::TangentialP2));
        if (refiningSharedCameraModel && baOpt.sharedIntrinsicReferenceCameras.empty())
        {
            baOpt.sharedIntrinsicReferenceCameras = baCameras;
        }
        bool hasRefinedLensSeed = false;
        if (refiningSharedCameraModel)
        {
            hasRefinedLensSeed = SfmBundleAdjustCoordinator::hasReusableCalibrationSeed(
                baCameras, baOpt.sharedIntrinsicReferenceCameras, seedSharedFocal, seedSharedK1);
            // 首轮自标定需要充足预算离开零畸变初值；后续稳定化轮次已有
            // 可复用镜头种子，恢复调用方配置，避免每轮都被强制抬高到 60 次。
            baOpt.maxIterations =
                SfmBundleAdjustCoordinator::selfCalibrationIterationBudget(baOpt.maxIterations, hasRefinedLensSeed);
            if (hasRefinedLensSeed || !allowCalibrationSeedSearch)
            {
                // 后续“重三角化—再平差”轮次已有稳定镜头种子，不重复固定内参预热，
                // 把全部迭代预算用于完整模型收敛。
                baOpt.stageSharedFocalRefinement = false;
            }
        }
        // SfM 协调器会固定旋转/平移规范，并在求解后恢复基线尺度，
        // 因此由调用方管理完整的 Sim(3) gauge，避免 BA 模块再自动固定第二台相机。
        baOpt.gaugePolicy = BAGaugePolicy::CallerManaged;
        int control_scale_bar_count = 0;
        for (std::size_t scale_index = 0; scale_index < _pendingPriorScaleBars.size(); ++scale_index)
        {
            const control_points::PriorScaleBar& scale_bar = _pendingPriorScaleBars[scale_index];
            if (!scale_bar.enabled || scale_bar.role != control_points::ScaleBarRole::Control ||
                !std::isfinite(scale_bar.measuredDistance) || scale_bar.measuredDistance <= 0.0 ||
                !std::isfinite(scale_bar.sigma) || scale_bar.sigma <= 0.0)
            {
                continue;
            }
            const auto first = marker_track_indices.find(scale_bar.firstMarkerId);
            const auto second = marker_track_indices.find(scale_bar.secondMarkerId);
            if (first == marker_track_indices.end() || second == marker_track_indices.end() ||
                first->second == second->second)
            {
                continue;
            }
            BAScaleBarConstraint constraint;
            constraint.trackIndexA = first->second;
            constraint.trackIndexB = second->second;
            constraint.measuredDistanceMeters = scale_bar.measuredDistance;
            constraint.sigmaMeters = scale_bar.sigma;
            constraint.weight = 1.0;
            constraint.sourceIndex = static_cast<int>(scale_index);
            baOpt.scaleBarConstraints.push_back(constraint);
            ++control_scale_bar_count;
        }
        if (_sfmOptions.useKnownCameraPoses && baOpt.cameraPosePriors.empty())
        {
            baOpt.cameraPosePriors = buildCameraPosePriorsFromInputCameras(baImageIds);
            baOpt.refineCameraPose = _sfmOptions.refineKnownCameraPoseWithSoftPrior;
        }
        if (_controlNetworkApplied)
        {
            // 已知位姿先验原本位于 SfM 局部坐标系，绝对定向后必须同步变换。
            for (BACameraPosePrior& prior : baOpt.cameraPosePriors)
            {
                if (!prior.enabled)
                    continue;
                prior.cameraCenter = _controlNetworkTransform.apply(prior.cameraCenter);
                prior.cameraToWorldRotation = _controlNetworkTransform.rotate(prior.cameraToWorldRotation);
                prior.positionSigmaMeters *= _controlNetworkTransform.scale;
            }
        }
        if (control_constraint_count > 0)
        {
            baOpt.enableControlPointConstraints = true;
            baOpt.backend = BABackend::Auto;
        }
        if (control_scale_bar_count > 0)
        {
            baOpt.enableScaleBarConstraints = true;
            baOpt.backend = BABackend::Auto;
        }
        AerialCameraPlaneEstimate cameraPlaneBefore;
        bool cameraPlaneConstraintActive = false;
        bool flattenAerialCameraLayer = false;
        const bool hasCameraLayerAbsoluteConstraint =
            control_constraint_count > 0 || control_scale_bar_count > 0 || !baOpt.cameraPosePriors.empty();
        const bool hasAerialSelfCalibrationAbsoluteConstraint =
            control_constraint_count > 0 || std::any_of(baOpt.cameraPosePriors.begin(),
                                                        baOpt.cameraPosePriors.end(),
                                                        [](const BACameraPosePrior& prior) { return prior.enabled; });
        cameraPlaneBefore = estimateAerialCameraPlane(baCameras);
        const bool refining_trusted_focal_k1_only =
            baOpt.hasTrustedSharedFocalPrior &&
            sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::RadialK1) &&
            !sharedIntrinsicParameterEnabled(baOpt, BAIntrinsicParameter::FocalLength) &&
            enabledIntrinsicParameterCount(baOpt.sharedIntrinsicParameterMask) == 1;
        if (_sfmOptions.correctUnanchoredAerialDoming && cameraPlaneBefore.valid &&
            SfmBundleAdjustCoordinator::shouldCorrectUnanchoredAerialDoming(localOnly,
                                                                            hasAerialSelfCalibrationAbsoluteConstraint,
                                                                            refineSharedIntrinsics,
                                                                            baOpt.hasTrustedSharedFocalPrior,
                                                                            refining_trusted_focal_k1_only,
                                                                            static_cast<int>(baCameras.size()),
                                                                            cameraPlaneBefore.opticalAxisConcentration,
                                                                            cameraPlaneBefore.normalToSpanRmsRatio))
        {
            BACameraPlaneConstraint& constraint = baOpt.cameraPlaneConstraint;
            constraint.enabled = true;
            constraint.normal = cameraPlaneBefore.normal;
            const bool flatten_curved_layer = cameraPlaneBefore.normalToSpanRmsRatio >= 0.025;
            flattenAerialCameraLayer = flatten_curved_layer;
            if (flatten_curved_layer)
            {
                // 明显弯曲时让目标平面通过固定规范相机，避免固定相机自身无法满足
                // 约束；k1 负责吸收原先被外参穹顶解释的径向像差。
                constraint.point = baCameras.front().cameraCenter();
                constraint.referenceSignedDistances.assign(baCameras.size(), 0.0);
            }
            else
            {
                // 首轮修平之后的后续 BA 不能撤掉约束，否则 k1 与外参会再次交换
                // 低频弯曲。对已经平坦的相机层只保持本轮形状，不继续强制压平。
                constraint.point = cameraPlaneBefore.center;
                constraint.referenceSignedDistances.clear();
                constraint.referenceSignedDistances.reserve(baCameras.size());
                for (const FramePinholeCamera& camera : baCameras)
                {
                    const auto center = camera.cameraCenter();
                    constraint.referenceSignedDistances.push_back(
                        constraint.normal[0] * (center[0] - constraint.point[0]) +
                        constraint.normal[1] * (center[1] - constraint.point[1]) +
                        constraint.normal[2] * (center[2] - constraint.point[2]));
                }
            }
            constraint.sigmaMeters =
                std::max(1.0e-6, cameraPlaneBefore.spanRms * _sfmOptions.cameraLayerPreservationSigmaFraction);
            const std::size_t observation_count = std::accumulate(baTracks.begin(),
                                                                  baTracks.end(),
                                                                  std::size_t{0},
                                                                  [](std::size_t total, const BATrack& track)
                                                                  { return total + track.observations.size(); });
            const double observations_per_camera = static_cast<double>(observation_count) /
                                                   static_cast<double>(std::max<std::size_t>(1, baCameras.size()));
            // 一个相机只有一个层约束，却通常有数千个像点残差。按每台相机平均
            // 观测数归一化，避免固定权重在大工程中随影像/连接点数增长而失效。
            constraint.weight = std::max(_sfmOptions.cameraLayerPreservationWeight, observations_per_camera);
            baOpt.cameraPlaneHuberDelta = 0.0;
            baOpt.backend = BABackend::Auto;
            // 大幅改变外参/畸变时，仍基于旧几何立即按点 RMS 标 invalid 会把尚未
            // 重三角化的点网误删。修正轮保留全部 BA 点，随后由统一的重三角化和
            // 几何过滤在新相机模型下重新评估。
            baOpt.enablePointFilter = false;
            cameraPlaneConstraintActive = true;
            Logger::instance()->infof("[BA] aerial_doming_correction scope=global cameras=%zu "
                                      "mode=%s model=fixed_focal+k1 opticalAxis=%.6f "
                                      "normalSpanRmsRatio=%.9f normalRms=%.6f spanRms=%.6f "
                                      "sigma=%.6f weight=%.3f",
                                      baCameras.size(),
                                      flatten_curved_layer ? "flatten" : "preserve",
                                      cameraPlaneBefore.opticalAxisConcentration,
                                      cameraPlaneBefore.normalToSpanRmsRatio,
                                      cameraPlaneBefore.normalRms,
                                      cameraPlaneBefore.spanRms,
                                      constraint.sigmaMeters,
                                      constraint.weight);
        }
        const bool lowOrderAerialSelfCalibration = !_sfmOptions.adaptiveCameraModelFitting && cameraPlaneBefore.valid &&
                                                   SfmBundleAdjustCoordinator::shouldUseLowOrderAerialSelfCalibration(
                                                       localOnly,
                                                       hasAerialSelfCalibrationAbsoluteConstraint,
                                                       refineSharedIntrinsics,
                                                       seedSharedK1,
                                                       static_cast<int>(baCameras.size()),
                                                       cameraPlaneBefore.opticalAxisConcentration);
        if (lowOrderAerialSelfCalibration)
        {
            baOpt.refineSharedHighOrderDistortion = false;
            baOpt.sharedFocalPriorSigma = std::min(baOpt.sharedFocalPriorSigma, 0.04);
            baOpt.sharedRadialK1PriorSigma = std::min(baOpt.sharedRadialK1PriorSigma, 0.05);
            Logger::instance()->infof("[BA] aerial_self_calibration_guard scope=global cameras=%zu "
                                      "opticalAxis=%.6f model=focal+k1 focalPriorSigma=%.6f "
                                      "k1PriorSigma=%.6f",
                                      baCameras.size(),
                                      cameraPlaneBefore.opticalAxisConcentration,
                                      baOpt.sharedFocalPriorSigma,
                                      baOpt.sharedRadialK1PriorSigma);
        }
        if (!cameraPlaneConstraintActive && _sfmOptions.preserveCameraLayerDuringSelfCalibration)
        {
            const bool stableParallelCameraLayer = cameraPlaneBefore.valid &&
                                                   cameraPlaneBefore.opticalAxisConcentration >= 0.90 &&
                                                   cameraPlaneBefore.normalToSpanRmsRatio <= 0.05;
            if (stableParallelCameraLayer &&
                SfmBundleAdjustCoordinator::shouldPreserveCameraLayer(
                    localOnly, hasCameraLayerAbsoluteConstraint, refineSharedIntrinsics, refiningSharedCameraModel))
            {
                BACameraPlaneConstraint& constraint = baOpt.cameraPlaneConstraint;
                constraint.enabled = true;
                constraint.point = cameraPlaneBefore.center;
                constraint.normal = cameraPlaneBefore.normal;
                constraint.referenceSignedDistances.clear();
                constraint.referenceSignedDistances.reserve(baCameras.size());
                for (const FramePinholeCamera& camera : baCameras)
                {
                    const auto center = camera.cameraCenter();
                    constraint.referenceSignedDistances.push_back(
                        constraint.normal[0] * (center[0] - constraint.point[0]) +
                        constraint.normal[1] * (center[1] - constraint.point[1]) +
                        constraint.normal[2] * (center[2] - constraint.point[2]));
                }
                constraint.sigmaMeters =
                    std::max(1.0e-6, cameraPlaneBefore.spanRms * _sfmOptions.cameraLayerPreservationSigmaFraction);
                constraint.weight = _sfmOptions.cameraLayerPreservationWeight;
                // 这是对本轮新增漂移的正则项，不是对相机绝对形状的平面先验。
                // 使用完整二次项，使所有相机相对各自参考偏移得到一致约束。
                baOpt.cameraPlaneHuberDelta = 0.0;
                baOpt.backend = BABackend::Auto;
                cameraPlaneConstraintActive = true;
                Logger::instance()->infof("[BA] camera_layer_preservation scope=global cameras=%zu "
                                          "opticalAxis=%.6f planarityVariance=%.9f "
                                          "normalSpanRmsRatio=%.9f normalRms=%.6f spanRms=%.6f "
                                          "sigma=%.6f weight=%.3f",
                                          baCameras.size(),
                                          cameraPlaneBefore.opticalAxisConcentration,
                                          cameraPlaneBefore.planarityRatio,
                                          cameraPlaneBefore.normalToSpanRmsRatio,
                                          cameraPlaneBefore.normalRms,
                                          cameraPlaneBefore.spanRms,
                                          constraint.sigmaMeters,
                                          constraint.weight);
            }
        }
        const bool hasAbsolutePoseConstraint =
            control_constraint_count > 0 || std::any_of(baOpt.cameraPosePriors.begin(),
                                                        baOpt.cameraPosePriors.end(),
                                                        [](const BACameraPosePrior& prior) { return prior.enabled; });
        const bool hasAbsoluteScaleConstraint = hasAbsolutePoseConstraint || control_scale_bar_count > 0;
        std::optional<std::pair<int, int>> similarityGaugeCameras;

        if (localOnly)
        {
            for (ImageId fixedImageId : fixedBoundaryImageIds)
            {
                const auto index = idToIdx.find(fixedImageId);
                if (index != idToIdx.end())
                {
                    baOpt.fixedCameraIndices.push_back(index->second);
                }
            }
            // 没有外部边界和绝对控制时固定一台相机，消除局部块的旋转和平移规范。
            if (baOpt.fixedCameraIndices.empty() && !hasAbsolutePoseConstraint && !baImageIds.empty())
            {
                baOpt.fixedCameraIndices.push_back(0);
            }
        }
        else if (!baImageIds.empty())
        {
            if (!hasAbsolutePoseConstraint)
            {
                baOpt.fixedCameraIndices = {0};
                if (cameraPlaneConstraintActive && baCameras.size() >= 2)
                {
                    const auto anchor_center = baCameras.front().cameraCenter();
                    int scale_camera_index = -1;
                    double farthest_squared_distance = 0.0;
                    for (int index = 1; index < static_cast<int>(baCameras.size()); ++index)
                    {
                        const auto center = baCameras[static_cast<std::size_t>(index)].cameraCenter();
                        const double dx = center[0] - anchor_center[0];
                        const double dy = center[1] - anchor_center[1];
                        const double dz = center[2] - anchor_center[2];
                        const double squared_distance = dx * dx + dy * dy + dz * dz;
                        if (squared_distance > farthest_squared_distance)
                        {
                            farthest_squared_distance = squared_distance;
                            scale_camera_index = index;
                        }
                    }
                    if (scale_camera_index >= 0 && farthest_squared_distance > 1.0e-20)
                    {
                        baOpt.fixedCameraIndices.push_back(scale_camera_index);
                        if (flattenAerialCameraLayer)
                        {
                            // 自动 gauge 会固定远端相机的完整位姿。保留 PCA 层法向，
                            // 只让该固定光心保持当前法向偏移；为迁就单个锚点旋转整个
                            // 目标平面会把修正方向带离真实相机层。
                            const auto scale_center =
                                baCameras[static_cast<std::size_t>(scale_camera_index)].cameraCenter();
                            BACameraPlaneConstraint& constraint = baOpt.cameraPlaneConstraint;
                            constraint.referenceSignedDistances[static_cast<std::size_t>(scale_camera_index)] =
                                constraint.normal[0] * (scale_center[0] - constraint.point[0]) +
                                constraint.normal[1] * (scale_center[1] - constraint.point[1]) +
                                constraint.normal[2] * (scale_center[2] - constraint.point[2]);
                        }
                        Logger::instance()->infof("[BA] aerial_doming_gauge anchor=0 scaleCamera=%d "
                                                  "baseline=%.6f targetPlaneCompatible=%s",
                                                  scale_camera_index,
                                                  std::sqrt(farthest_squared_distance),
                                                  flattenAerialCameraLayer ? "true" : "preserved_layer");
                    }
                }
            }
        }

        // 单目 BA 固定一台相机后仍有尺度规范。不要固定第二台相机的完整位姿；
        // 记录一条非退化基线，并在求解后对全部相机中心和点做同一 Sim(3) 尺度恢复。
        if (baOpt.refineCameraPose && !hasAbsoluteScaleConstraint && !cameraPlaneConstraintActive &&
            baOpt.fixedCameraIndices.size() == 1)
        {
            const int anchorIndex = baOpt.fixedCameraIndices.front();
            if (anchorIndex >= 0 && anchorIndex < static_cast<int>(baCameras.size()))
            {
                const auto anchorCenter = baCameras[static_cast<std::size_t>(anchorIndex)].cameraCenter();
                for (int candidateIndex = 0; candidateIndex < static_cast<int>(baCameras.size()); ++candidateIndex)
                {
                    if (candidateIndex == anchorIndex)
                    {
                        continue;
                    }
                    const auto candidateCenter = baCameras[static_cast<std::size_t>(candidateIndex)].cameraCenter();
                    const double dx = candidateCenter[0] - anchorCenter[0];
                    const double dy = candidateCenter[1] - anchorCenter[1];
                    const double dz = candidateCenter[2] - anchorCenter[2];
                    if (dx * dx + dy * dy + dz * dz > 1.0e-20)
                    {
                        similarityGaugeCameras = std::make_pair(anchorIndex, candidateIndex);
                        break;
                    }
                }
            }
        }

        // 在进入可能耗时数分钟的大规模求解前输出完整问题规模和自动后端决策。
        // 之前该日志位于 optimizePoints 之后，运行中无法判断相机/观测规模，
        // 也无法区分 CUDA 未编译和问题规模未达到阈值。
        if (baOpt.logIterationProgress)
        {
            const BAProblemStats stats = BundleAdjust::summarizeProblem(baCameras, baTracks);
            const BABackendDecision decision = BundleAdjust::decideBackendForProblem(stats, baOpt);
            Logger::instance()->infof("[BA] problem scope=%s cameras=%d tracks=%d observations=%d threads=%d "
                                      "requested=%s selected=%s reason=%s plaMatrixCudaAvailable=%s "
                                      "cudaMinCameras=%d cudaMinObservations=%d",
                                      scopeName,
                                      stats.cameraCount,
                                      stats.trackCount,
                                      stats.observationCount,
                                      baOpt.numThreads,
                                      BundleAdjust::backendName(baOpt.backend),
                                      BundleAdjust::backendName(decision.backend),
                                      decision.reason.c_str(),
                                      BundleAdjust::isBackendAvailable(BABackend::PlaMatrixCuda) ? "true" : "false",
                                      baOpt.minPlaMatrixGpuCameras,
                                      baOpt.minPlaMatrixGpuObservations);
        }

        // 首次自由网络自标定采用少量场景无关的焦距/低阶径向多起点，避免零畸变
        // 产生的 dome 局部极小值锁住后续联合 BA。候选只改变镜头初值，不使用相机
        // 轨迹平面、地面法向或无人机语义；通过同一 BA 有效轨迹率与 RMS 选择。
        BAResult baResult;
        bool usedCalibrationSeedSearch = false;
        if (allowCalibrationSeedSearch && refiningSharedCameraModel && !hasRefinedLensSeed &&
            (seedSharedFocal || seedSharedK1) && baCameras.size() >= 50 && baTracks.size() >= 1000)
        {
            usedCalibrationSeedSearch = true;
            const std::size_t preview_limit =
                _sfmOptions.selfCalibrationPreviewMaxTracks > 0
                    ? static_cast<std::size_t>(_sfmOptions.selfCalibrationPreviewMaxTracks)
                    : baTracks.size();
            const std::vector<std::size_t> preview_track_indices =
                sfm_calibration_preview::selectTrackIndices(baCameras, baTracks, preview_limit);
            std::vector<BATrack> preview_tracks;
            preview_tracks.reserve(preview_track_indices.size());
            for (const std::size_t track_index : preview_track_indices)
            {
                preview_tracks.push_back(baTracks[track_index]);
            }
            BAOptions preview_options = baOpt;
            std::vector<int> preview_index_by_full_track(baTracks.size(), -1);
            for (std::size_t preview_index = 0; preview_index < preview_track_indices.size(); ++preview_index)
            {
                preview_index_by_full_track[preview_track_indices[preview_index]] = static_cast<int>(preview_index);
            }
            preview_options.fixedTrackIndices.clear();
            for (const int full_index : baOpt.fixedTrackIndices)
            {
                if (full_index >= 0 && full_index < static_cast<int>(preview_index_by_full_track.size()) &&
                    preview_index_by_full_track[static_cast<std::size_t>(full_index)] >= 0)
                {
                    preview_options.fixedTrackIndices.push_back(
                        preview_index_by_full_track[static_cast<std::size_t>(full_index)]);
                }
            }
            preview_options.scaleBarConstraints.clear();
            for (const BAScaleBarConstraint& constraint : baOpt.scaleBarConstraints)
            {
                if (constraint.trackIndexA < 0 || constraint.trackIndexB < 0 ||
                    constraint.trackIndexA >= static_cast<int>(preview_index_by_full_track.size()) ||
                    constraint.trackIndexB >= static_cast<int>(preview_index_by_full_track.size()))
                {
                    continue;
                }
                const int preview_a = preview_index_by_full_track[static_cast<std::size_t>(constraint.trackIndexA)];
                const int preview_b = preview_index_by_full_track[static_cast<std::size_t>(constraint.trackIndexB)];
                if (preview_a < 0 || preview_b < 0)
                {
                    continue;
                }
                BAScaleBarConstraint remapped = constraint;
                remapped.trackIndexA = preview_a;
                remapped.trackIndexB = preview_b;
                preview_options.scaleBarConstraints.push_back(remapped);
            }
            preview_options.enableScaleBarConstraints =
                baOpt.enableScaleBarConstraints && !preview_options.scaleBarConstraints.empty();
            struct CalibrationSeed
            {
                double focalScale = 1.0;
                double radialK1 = 0.0;
            };
            const std::array<CalibrationSeed, 3> seeds{{
                {1.00, 0.00},
                {0.98, -0.04},
                {0.96, -0.08},
            }};
            const int candidateThreads = std::max(1, baOpt.numThreads / static_cast<int>(seeds.size()));
            const int candidateIterations = std::min(12, baOpt.maxIterations);
            std::vector<std::future<BAResult>> candidates;
            candidates.reserve(seeds.size());
            for (const CalibrationSeed seed : seeds)
            {
                candidates.push_back(
                    std::async(std::launch::async,
                               [&, seed]()
                               {
                                   std::vector<FramePinholeCamera> seedCameras = baCameras;
                                   for (FramePinholeCamera& camera : seedCameras)
                                   {
                                       if (seedSharedFocal)
                                       {
                                           const FramePinholeCamera::Intrinsics intrinsics = camera.intrinsics();
                                           camera.setIntrinsics(intrinsics.focalX * seed.focalScale,
                                                                intrinsics.focalY * seed.focalScale,
                                                                intrinsics.principalX,
                                                                intrinsics.principalY);
                                       }
                                       if (seedSharedK1)
                                       {
                                           FramePinholeCamera::Distortion distortion = camera.distortion();
                                           distortion.radialK1 = seed.radialK1;
                                           camera.setDistortion(distortion);
                                       }
                                   }
                                   BAOptions candidateOptions = preview_options;
                                   candidateOptions.numThreads = candidateThreads;
                                   candidateOptions.maxIterations = candidateIterations;
                                   candidateOptions.progressCallback = {};
                                   candidateOptions.logIterationProgress = false;
                                   return BundleAdjust::optimizePoints(seedCameras, preview_tracks, candidateOptions);
                               }));
            }

            double bestScore = std::numeric_limits<double>::infinity();
            int bestIndex = -1;
            for (int index = 0; index < static_cast<int>(candidates.size()); ++index)
            {
                BAResult candidate = candidates[static_cast<size_t>(index)].get();
                const double score = candidate.solutionUsable && std::isfinite(candidate.meanRmsAfter)
                                         ? candidate.meanRmsAfter / std::max(0.10, candidate.validTrackRatio)
                                         : std::numeric_limits<double>::infinity();
                Logger::instance()->infof("[BA] self_calibration_seed index=%d focalScale=%.3f k1=%.3f "
                                          "usable=%s validTrackRatio=%.6f rms=%.6f score=%.6f",
                                          index,
                                          seeds[static_cast<size_t>(index)].focalScale,
                                          seeds[static_cast<size_t>(index)].radialK1,
                                          candidate.solutionUsable ? "true" : "false",
                                          candidate.validTrackRatio,
                                          candidate.meanRmsAfter,
                                          score);
                if (score < bestScore)
                {
                    bestScore = score;
                    bestIndex = index;
                    baResult = std::move(candidate);
                }
            }
            Logger::instance()->infof("[BA] self_calibration_seed selected=%d candidates=%zu "
                                      "previewTracks=%zu/%zu previewIterations=%d threadsPerCandidate=%d",
                                      bestIndex,
                                      seeds.size(),
                                      preview_tracks.size(),
                                      baTracks.size(),
                                      candidateIterations,
                                      candidateThreads);
            if (bestIndex >= 0 && baResult.solutionUsable && baResult.refinedCameras.size() == baCameras.size() &&
                baResult.points.size() == preview_tracks.size())
            {
                std::vector<BATrack> warmTracks = baTracks;
                int retriangulated_track_count = 0;
                if (cameraPlaneConstraintActive)
                {
                    // 预览解已经改变相机层和 k1。全量点若仍携带旧穹顶坐标进入
                    // BA，会产生大面积负深度/高残差并被误判为离群点。先用预览解
                    // 的相机对全部普通轨迹重新交会，再进入全量联合优化。
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : retriangulated_track_count)
                    for (int index = 0; index < static_cast<int>(warmTracks.size()); ++index)
                    {
                        if (std::find(baOpt.fixedTrackIndices.begin(), baOpt.fixedTrackIndices.end(), index) !=
                            baOpt.fixedTrackIndices.end())
                        {
                            continue;
                        }
                        if (retriangulateTrackInitialPoint(baResult.refinedCameras,
                                                           &warmTracks[static_cast<std::size_t>(index)]))
                        {
                            ++retriangulated_track_count;
                        }
                    }
                }
                else
                {
                    for (std::size_t index = 0; index < preview_track_indices.size(); ++index)
                    {
                        const BARefinedPoint& point = baResult.points[index];
                        if (point.valid && std::isfinite(point.point[0]) && std::isfinite(point.point[1]) &&
                            std::isfinite(point.point[2]))
                        {
                            warmTracks[preview_track_indices[index]].initialPoint = point.point;
                        }
                    }
                }
                Logger::instance()->infof("[BA] self_calibration_seed full_refinement seed=%d threads=%d "
                                          "iterations=%d retriangulatedTracks=%d",
                                          bestIndex,
                                          baOpt.numThreads,
                                          baOpt.maxIterations,
                                          retriangulated_track_count);
                baResult = BundleAdjust::optimizePoints(baResult.refinedCameras, warmTracks, baOpt);
            }
            else
            {
                Logger::instance()->warn(
                    "[BA] self_calibration_seed no usable preview; falling back to full refinement");
                baResult = BundleAdjust::optimizePoints(baCameras, baTracks, baOpt);
            }
        }
        else
        {
            baResult = BundleAdjust::optimizePoints(baCameras, baTracks, baOpt);
        }
        if (usedCalibrationSeedSearch && baResult.solutionUsable && baResult.refinedCameras.size() == baCameras.size())
        {
            // 完整精化以最佳候选为 warm start；应用统计必须仍以本轮原始相机为基线，
            // 否则候选已改变内参而末轮数值 no-op 时会误报“未应用”。
            SfmBundleAdjustCoordinator::refreshCalibrationSeedApplicationCount(baCameras, &baResult);
        }
        const double cameraLayerDriftRms =
            baResult.solutionUsable && cameraPlaneConstraintActive
                ? cameraLayerReferenceDriftRms(baResult.refinedCameras, baOpt.cameraPlaneConstraint)
                : std::numeric_limits<double>::infinity();

        bool gaugeNormalizationFailed = false;
        if (baResult.solutionUsable && similarityGaugeCameras.has_value())
        {
            const SimilarityGaugeNormalizationResult gaugeResult =
                normalizeSimilarityGauge(baCameras,
                                         similarityGaugeCameras->first,
                                         similarityGaugeCameras->second,
                                         &baResult.refinedCameras,
                                         &baResult.points);
            if (gaugeResult.applied)
            {
                Logger::instance()->infof(
                    "[BA] similarity gauge normalized scope=%s anchor=%d scaleCamera=%d scale=%.9f",
                    scopeName,
                    similarityGaugeCameras->first,
                    similarityGaugeCameras->second,
                    gaugeResult.scale);
            }
            else
            {
                gaugeNormalizationFailed = true;
                Logger::instance()->warnf("[BA] similarity gauge normalization failed scope=%s reason=%s",
                                          scopeName,
                                          gaugeResult.reason.c_str());
            }
        }

        bool applyBaResult = baResult.solutionUsable && !gaugeNormalizationFailed;
        if (!applyBaResult)
        {
            Logger::instance()->warnf("[BA] 求解结果不可写回: status=%d backend=%s message=%s",
                                      static_cast<int>(baResult.solveStatus),
                                      BundleAdjust::backendName(baResult.usedBackend),
                                      baResult.backendMessage.c_str());
        }
        if (applyBaResult && control_constraint_count == 0 && control_scale_bar_count == 0 &&
            std::isfinite(baResult.meanRmsBefore) && std::isfinite(baResult.meanRmsAfter))
        {
            const double rmsTolerance = std::max(1.0e-9, std::abs(baResult.meanRmsBefore) * 1.0e-6);
            const double acceptedRmsGrowth =
                cameraPlaneConstraintActive ? std::max(0.03, std::abs(baResult.meanRmsBefore) * 0.03) : rmsTolerance;
            if (baResult.meanRmsAfter > baResult.meanRmsBefore + acceptedRmsGrowth)
            {
                applyBaResult = false;
                Logger::instance()->warnf("[BA] rejected scope=%s reason=reprojection_rms_regressed rms=%.9f->%.9f",
                                          scopeName,
                                          baResult.meanRmsBefore,
                                          baResult.meanRmsAfter);
            }
        }
        if (applyBaResult && cameraPlaneConstraintActive)
        {
            const double driftRatio = cameraLayerDriftRms / std::max(1.0e-9, cameraPlaneBefore.spanRms);
            const bool geometryPreserved =
                std::isfinite(driftRatio) && driftRatio <= _sfmOptions.cameraLayerPreservationSigmaFraction;
            Logger::instance()->infof("[BA] camera_layer_result scope=global appliedCandidate=%s "
                                      "referenceDriftRms=%.9f referenceDriftRatio=%.9f limit=%.9f",
                                      geometryPreserved ? "true" : "false",
                                      cameraLayerDriftRms,
                                      driftRatio,
                                      _sfmOptions.cameraLayerPreservationSigmaFraction);
            if (!geometryPreserved)
            {
                applyBaResult = false;
                Logger::instance()->warn("[BA] rejected scope=global reason=camera_layer_reference_drift_exceeded");
            }
        }
        const bool knownPoseGlobalBa =
            _sfmOptions.useKnownCameraPoses && !localOnly && baOpt.refineCameraPose && !baOpt.cameraPosePriors.empty();
        if (knownPoseGlobalBa)
        {
            if (!std::isfinite(baResult.meanRmsAfter) || baResult.meanRmsAfter > _sfmOptions.filterMaxReprojError)
            {
                applyBaResult = false;
            }
            for (size_t i = 0; applyBaResult && i < baImageIds.size() && i < baCameras.size() &&
                               i < baResult.refinedCameras.size() && i < baOpt.cameraPosePriors.size();
                 ++i)
            {
                const int cameraIndex = static_cast<int>(i);
                if (std::find(baOpt.fixedCameraIndices.begin(), baOpt.fixedCameraIndices.end(), cameraIndex) !=
                    baOpt.fixedCameraIndices.end())
                {
                    continue;
                }
                const BACameraPosePrior& prior = baOpt.cameraPosePriors[i];
                if (!prior.enabled)
                {
                    continue;
                }
                const auto beforeCenter = baCameras[i].cameraCenter();
                const auto afterCenter = baResult.refinedCameras[i].cameraCenter();
                const double beforeDistance =
                    std::sqrt((beforeCenter[0] - prior.cameraCenter[0]) * (beforeCenter[0] - prior.cameraCenter[0]) +
                              (beforeCenter[1] - prior.cameraCenter[1]) * (beforeCenter[1] - prior.cameraCenter[1]) +
                              (beforeCenter[2] - prior.cameraCenter[2]) * (beforeCenter[2] - prior.cameraCenter[2]));
                const double afterDistance =
                    std::sqrt((afterCenter[0] - prior.cameraCenter[0]) * (afterCenter[0] - prior.cameraCenter[0]) +
                              (afterCenter[1] - prior.cameraCenter[1]) * (afterCenter[1] - prior.cameraCenter[1]) +
                              (afterCenter[2] - prior.cameraCenter[2]) * (afterCenter[2] - prior.cameraCenter[2]));
                const double tolerance = std::max(1e-3, prior.positionSigmaMeters * 3.0);
                if (afterDistance > std::max(beforeDistance + tolerance, beforeDistance * 2.0 + 1e-3))
                {
                    applyBaResult = false;
                }
            }
            if (!applyBaResult)
            {
                Logger::instance()->info(
                    "[SFM] Known-pose BA result rejected by prior/RMS gate; keeping pre-BA cameras and points");
            }
        }
        Logger::instance()->infof("[BA] result scope=%s cameras=%zu tracks=%zu observations=%d requested=%s used=%s "
                                  "status=%s usable=%s applied=%s fallback=%s rms=%.6f->%.6f "
                                  "seconds(setup=%.3f solve=%.3f post=%.3f total=%.3f assembly=%.3f "
                                  "objective=%.3f trialState=%.3f linear=%.3f) "
                                  "pcg(iterations=%d tolerance=%.3e..%.3e) denseFallbacks=%d message=%s",
                                  scopeName,
                                  baCameras.size(),
                                  baTracks.size(),
                                  baResult.observationCount,
                                  BundleAdjust::backendName(baResult.requestedBackend),
                                  BundleAdjust::backendName(baResult.usedBackend),
                                  BundleAdjust::solveStatusName(baResult.solveStatus),
                                  baResult.solutionUsable ? "true" : "false",
                                  applyBaResult ? "true" : "false",
                                  baResult.backendFallback ? "true" : "false",
                                  baResult.meanRmsBefore,
                                  baResult.meanRmsAfter,
                                  baResult.setupSeconds,
                                  baResult.solveSeconds,
                                  baResult.postprocessSeconds,
                                  baResult.totalSeconds,
                                  baResult.plaMatrixAssemblySeconds,
                                  baResult.plaMatrixObjectiveSeconds,
                                  baResult.plaMatrixTrialStateSeconds,
                                  baResult.plaMatrixLinearSolveSeconds,
                                  baResult.plaMatrixLinearIterations,
                                  baResult.plaMatrixLinearToleranceMinimum,
                                  baResult.plaMatrixLinearToleranceMaximum,
                                  baResult.plaMatrixDenseFallbacks,
                                  baResult.backendMessage.c_str());
        if (baOpt.refineSharedFocalLength || baOpt.refineSharedFocalAspectRatio || baOpt.refineSharedPrincipalPoint ||
            baOpt.refineSharedRadialDistortion)
        {
            Logger::instance()->infof("[BA] intrinsics scope=%s applied=%s cameras=%d groups=%d "
                                      "focalScale=%.8f aspectScale=%.8f principalOffsetPx=(%.4f,%.4f) "
                                      "distortion=(k1=%.8f,k2=%.8f,k3=%.8f,p1=%.8f,p2=%.8f)",
                                      scopeName,
                                      applyBaResult ? "true" : "false",
                                      applyBaResult ? baResult.refinedIntrinsicCount : 0,
                                      applyBaResult ? baResult.refinedCalibrationGroupCount : 0,
                                      applyBaResult ? baResult.refinedSharedFocalScale : 1.0,
                                      applyBaResult ? baResult.refinedSharedFocalAspectScale : 1.0,
                                      applyBaResult ? baResult.refinedSharedPrincipalOffsetX : 0.0,
                                      applyBaResult ? baResult.refinedSharedPrincipalOffsetY : 0.0,
                                      applyBaResult ? baResult.refinedSharedRadialK1 : 0.0,
                                      applyBaResult ? baResult.refinedSharedRadialK2 : 0.0,
                                      applyBaResult ? baResult.refinedSharedRadialK3 : 0.0,
                                      applyBaResult ? baResult.refinedSharedTangentialP1 : 0.0,
                                      applyBaResult ? baResult.refinedSharedTangentialP2 : 0.0);
        }

        // 回写优化后的相机位姿（跳过被 gauge 固定的相机）
        if (applyBaResult)
        {
            for (size_t i = 0; i < baImageIds.size(); ++i)
            {
                if (i < baResult.refinedCameras.size())
                {
                    _reconstruction->camera(baImageIds[i]) = baResult.refinedCameras[i];
                }
            }

            if (!localOnly && refineSharedIntrinsics && baResult.refinedCalibrationGroupCount == 1 &&
                !baResult.refinedCameras.empty())
            {
                // 未注册影像的 PnP 会从预载相机读取内参。单一镜头组完成全局自标定后，
                // 将同一组内参同步过去，避免最终重试继续使用零畸变/旧焦距。
                const FramePinholeCamera& calibratedCamera = baResult.refinedCameras.front();
                const FramePinholeCamera::Intrinsics calibratedIntrinsics = calibratedCamera.intrinsics();
                const FramePinholeCamera::Distortion calibratedDistortion = calibratedCamera.distortion();
                for (auto& [imageId, camera] : _preloadedCameras)
                {
                    (void)imageId;
                    camera.setIntrinsics(calibratedIntrinsics.focalX,
                                         calibratedIntrinsics.focalY,
                                         calibratedIntrinsics.principalX,
                                         calibratedIntrinsics.principalY);
                    camera.setDistortion(calibratedDistortion);
                }
            }
        }

        // 回写优化后的三维点坐标、误差；并进行观测级过滤（参考 COLMAP）
        // BA 标记为 invalid 的点 → 先检查各观测的个别重投影误差，
        // 移除高误差观测后若轨迹仍 >= 2，则保留点但缩短轨迹；否则删除整个点。
        int deletedPts = 0;
        int removedObs = 0;
        for (size_t ti = 0; ti < baTracks.size() && ti < baResult.points.size(); ++ti)
        {
            if (!applyBaResult)
            {
                continue;
            }
            const Point3DId pid = trackToPid[ti];
            if (!_reconstruction->hasPoint3D(pid))
                continue;

            const BARefinedPoint& bp = baResult.points[ti];
            if (!bp.valid)
            {
                // ── 观测级过滤：逐观测检查重投影误差 ──
                auto& pt = _reconstruction->point3D(pid);
                const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
                std::vector<size_t> badObsIndices;

                for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
                {
                    const auto& elem = pt.track.elements[oi];
                    if (!_reconstruction->isRegistered(elem.imageId))
                        continue;
                    if (!_reconstruction->hasCamera(elem.imageId))
                        continue;

                    const FramePinholeCamera& cam = _reconstruction->camera(elem.imageId);
                    const ImageData& imgData = _reconstruction->image(elem.imageId);
                    if (elem.featureIdx >= imgData.keypoints.size())
                        continue;

                    double u_obs = imgData.keypoints[elem.featureIdx].x;
                    double v_obs = imgData.keypoints[elem.featureIdx].y;

                    // 计算重投影
                    double world[3] = {bp.point[0], bp.point[1], bp.point[2]};
                    double uv_proj[2] = {0, 0};
                    bool projected = cam.projectWorldPoint(world, uv_proj);
                    if (!projected)
                    {
                        badObsIndices.push_back(oi);
                        continue;
                    }
                    double du = uv_proj[0] - u_obs;
                    double dv = uv_proj[1] - v_obs;
                    double reproj = std::sqrt(du * du + dv * dv);
                    if (reproj > filterThresh)
                    {
                        badObsIndices.push_back(oi);
                    }
                }

                size_t goodObs = pt.track.elements.size() - badObsIndices.size();
                if (goodObs >= 2 && !badObsIndices.empty())
                {
                    // 保留点，仅移除坏观测
                    for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
                    {
                        const TrackElement elem = pt.track.elements[*it];
                        if (_reconstruction->removeObservation(pid, elem.imageId, elem.featureIdx))
                        {
                            ++removedObs;
                        }
                    }
                    // 更新点坐标和误差为 BA 结果
                    pt.xyz = bp.point;
                    pt.error = bp.rmsAfter;
                }
                else
                {
                    // 彻底删除
                    _reconstruction->deletePoint3D(pid);
                    ++deletedPts;
                }
            }
            else
            {
                // ── 有效点也进行观测级过滤：移除残差特别高的观测 ──
                auto& pt = _reconstruction->point3D(pid);
                const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
                std::vector<size_t> badObsIndices;

                for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
                {
                    const auto& elem = pt.track.elements[oi];
                    if (!_reconstruction->isRegistered(elem.imageId))
                        continue;
                    if (!_reconstruction->hasCamera(elem.imageId))
                        continue;

                    const FramePinholeCamera& cam = _reconstruction->camera(elem.imageId);
                    const ImageData& imgData = _reconstruction->image(elem.imageId);
                    if (elem.featureIdx >= imgData.keypoints.size())
                        continue;

                    double u_obs = imgData.keypoints[elem.featureIdx].x;
                    double v_obs = imgData.keypoints[elem.featureIdx].y;

                    double world[3] = {bp.point[0], bp.point[1], bp.point[2]};
                    double uv_proj[2] = {0, 0};
                    bool projected = cam.projectWorldPoint(world, uv_proj);
                    if (!projected)
                    {
                        badObsIndices.push_back(oi);
                        continue;
                    }
                    double du = uv_proj[0] - u_obs;
                    double dv = uv_proj[1] - v_obs;
                    double reproj = std::sqrt(du * du + dv * dv);
                    if (reproj > filterThresh)
                    {
                        badObsIndices.push_back(oi);
                    }
                }

                if (!badObsIndices.empty())
                {
                    size_t goodObs = pt.track.elements.size() - badObsIndices.size();
                    if (goodObs >= 2)
                    {
                        for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
                        {
                            const TrackElement elem = pt.track.elements[*it];
                            if (_reconstruction->removeObservation(pid, elem.imageId, elem.featureIdx))
                            {
                                ++removedObs;
                            }
                        }
                    }
                    else
                    {
                        _reconstruction->deletePoint3D(pid);
                        ++deletedPts;
                        continue;
                    }
                }

                pt.xyz = bp.point;
                pt.error = bp.rmsAfter;
            }
        }

        Logger::instance()->infof("[BA] deletedPts=%d, removedObs=%d", deletedPts, removedObs);

        // 全局 BA：记录统计供最终结果使用（lastGlobalBA* 成员变量）
        if (!localOnly)
        {
            const double appliedRmsAfter = applyBaResult ? baResult.meanRmsAfter : baResult.meanRmsBefore;
            _lastControlPointConstraintCount = control_constraint_count;
            _lastControlScaleBarConstraintCount = control_scale_bar_count;
            _lastGlobalBARmsBefore = baResult.meanRmsBefore;
            _lastGlobalBARmsAfter = appliedRmsAfter;
            _lastGlobalBATracksTotal = baResult.totalTracks;
            _lastGlobalBATracksOptimized = baResult.optimizedTracks;
            _lastGlobalBATracksFiltered =
                applyBaResult ? static_cast<int>(baTracks.size()) - baResult.optimizedTracks : 0;
            if (_lastGlobalBATracksFiltered < 0)
                _lastGlobalBATracksFiltered = deletedPts;
            _lastGlobalBARefinedIntrinsicCount = applyBaResult ? baResult.refinedIntrinsicCount : 0;
            _lastGlobalBASharedFocalScale = applyBaResult ? baResult.refinedSharedFocalScale : 1.0;
            _lastGlobalBASharedFocalAspectScale = applyBaResult ? baResult.refinedSharedFocalAspectScale : 1.0;
            _lastGlobalBASharedPrincipalOffsetX = applyBaResult ? baResult.refinedSharedPrincipalOffsetX : 0.0;
            _lastGlobalBASharedPrincipalOffsetY = applyBaResult ? baResult.refinedSharedPrincipalOffsetY : 0.0;
            _lastGlobalBASharedRadialK1 = applyBaResult ? baResult.refinedSharedRadialK1 : 0.0;
            _lastGlobalBASharedRadialK2 = applyBaResult ? baResult.refinedSharedRadialK2 : 0.0;
            _lastGlobalBASharedRadialK3 = applyBaResult ? baResult.refinedSharedRadialK3 : 0.0;
            _lastGlobalBASharedTangentialP1 = applyBaResult ? baResult.refinedSharedTangentialP1 : 0.0;
            _lastGlobalBASharedTangentialP2 = applyBaResult ? baResult.refinedSharedTangentialP2 : 0.0;
            _lastGlobalBARequestedBackend = baResult.requestedBackend;
            _lastGlobalBAUsedBackend = baResult.usedBackend;
            _lastGlobalBASolveStatus = baResult.solveStatus;
            _lastGlobalBASolutionUsable = baResult.solutionUsable;
            _lastGlobalBAResultApplied = applyBaResult;
            _lastGlobalBABackendFallback = baResult.backendFallback;
            _lastGlobalBAObservationCount = baResult.observationCount;
            _lastGlobalBATotalSeconds = baResult.totalSeconds;
            _lastGlobalBABackendMessage = useMultiViewOnlyGlobalBa
                                              ? "network=multi_view_only; " + baResult.backendMessage
                                              : baResult.backendMessage;
            const bool adaptiveCameraModelAppliedThisRound =
                adaptiveCameraModelFittingEvaluated && applyBaResult && baResult.refinedIntrinsicCount > 0;
            SfmAdaptiveCameraModelDiagnosticSnapshot previousDiagnostics;
            previousDiagnostics.evaluated = _lastGlobalBAAdaptiveCameraModelFittingEvaluated;
            previousDiagnostics.applied = _lastGlobalBAAdaptiveCameraModelFittingApplied;
            previousDiagnostics.parameterMask = _lastGlobalBAIntrinsicParameterMask;
            previousDiagnostics.modelName = _lastGlobalBAAdaptiveCameraModel;
            SfmAdaptiveCameraModelDiagnosticSnapshot currentDiagnostics;
            currentDiagnostics.evaluated = adaptiveCameraModelFittingEvaluated;
            currentDiagnostics.applied = adaptiveCameraModelAppliedThisRound;
            currentDiagnostics.parameterMask = effectiveIntrinsicParameterMask;
            currentDiagnostics.modelName = effectiveAdaptiveCameraModel;
            const auto mergedDiagnostics = SfmBundleAdjustCoordinator::mergeAdaptiveCameraModelDiagnostics(
                previousDiagnostics, currentDiagnostics);
            _lastGlobalBAAdaptiveCameraModelFittingEvaluated = mergedDiagnostics.accumulated.evaluated;
            _lastGlobalBAAdaptiveCameraModelFittingApplied = mergedDiagnostics.accumulated.applied;
            _lastGlobalBAIntrinsicParameterMask = mergedDiagnostics.accumulated.parameterMask;
            _lastGlobalBAAdaptiveCameraModel = mergedDiagnostics.accumulated.modelName;
            if (mergedDiagnostics.shouldReplaceEvidence)
            {
                _lastGlobalBAIntrinsicParameterReliability = adaptiveCameraModelAssessment.reliability;
                _lastGlobalBAIntrinsicParameterIncrementalInformationScore =
                    adaptiveCameraModelAssessment.incrementalInformationScore;
                _lastGlobalBAIntrinsicParameterSensitivity = adaptiveCameraModelAssessment.sensitivity;
                _lastGlobalBAAdaptiveCameraModelReason = adaptiveCameraModelAssessment.reason;
                _lastGlobalBACameraModelGeometryStrength = adaptiveCameraModelAssessment.geometryStrength;
                _lastGlobalBACameraModelOpticalAxisConcentration =
                    adaptiveCameraModelAssessment.opticalAxisConcentration;
                _lastGlobalBACameraModelMedianTriangulationAngle =
                    adaptiveCameraModelAssessment.medianTriangulationAngleDegrees;
                _lastGlobalBACameraModelNormalizedRadiusP90 = adaptiveCameraModelAssessment.normalizedRadiusP90;
                _lastGlobalBACameraModelOccupiedPeripheralSectors =
                    adaptiveCameraModelAssessment.occupiedPeripheralSectors;
                _lastGlobalBACameraModelObservationCount = adaptiveCameraModelAssessment.observationCount;
                _lastGlobalBACameraModelMultiViewTrackRatio = adaptiveCameraModelAssessment.multiViewTrackRatio;
                _lastGlobalBACameraModelObservationSupport = adaptiveCameraModelAssessment.observationSupport;
                _lastGlobalBACameraModelPeripheralCoverage = adaptiveCameraModelAssessment.peripheralCoverage;
                _lastGlobalBACameraModelSectorCoverage = adaptiveCameraModelAssessment.sectorCoverage;
                _lastGlobalBACameraModelImageAxisBalance = adaptiveCameraModelAssessment.imageAxisBalance;
            }
        }
    }

    // ============================================================
    // 内部：迭代全局 BA 精化（参考 COLMAP IterativeGlobalRefinement）
    // ============================================================

    void IncrementalSfm::iterativeGlobalBA(bool finalRefinement)
    {
        if (finalRefinement)
        {
            SfmBundleAdjustCoordinator(*this).consolidateInputTracksForFinalBa();
        }

        const int registered_count = static_cast<int>(_reconstruction->numRegisteredImages());
        int maxRounds = SfmBundleAdjustCoordinator::iterativeGlobalBaRoundLimit(
            _sfmOptions.iterativeBARounds, finalRefinement, registered_count);
        const int repeat_threshold = std::max(2, _sfmOptions.hierarchicalBATargetBlockSize / 2);
        const bool hierarchical_schedule_active =
            HierarchicalBundleAdjuster::shouldRun(_sfmOptions.enableHierarchicalBA,
                                                  registered_count,
                                                  _sfmOptions.hierarchicalBAMinImages,
                                                  _sfmOptions.baOptions.refineCameraPose) &&
            !_sfmOptions.useKnownCameraPoses && !_controlNetworkApplied && _pendingPriorTracks.empty() &&
            _pendingPriorScaleBars.empty();
        const bool recently_partitioned = hierarchical_schedule_active && _lastHierarchicalBAImageCount > 0 &&
                                          registered_count - _lastHierarchicalBAImageCount < repeat_threshold;

        if (_sfmOptions.filterNegativeDepth && hierarchical_schedule_active && !recently_partitioned)
        {
            const int negative_count = filterNegativeDepthPoints();
            if (negative_count > 0)
            {
                Logger::instance()->infof("[SFM] HierarchicalBA filtered %d negative-depth points", negative_count);
            }
        }

        const HierarchicalBaRunSummary hierarchical_summary =
            hierarchical_schedule_active ? HierarchicalBundleAdjuster(*this).run() : HierarchicalBaRunSummary{};
        if (hierarchical_summary.applied())
        {
            Triangulator hierarchical_tri(*_reconstruction, _correspondenceGraph, _sfmOptions.baOptions.numThreads);
            const int retriangulated = hierarchical_tri.retriangulatePoints(_sfmOptions.filterMaxReprojError);
            hierarchical_tri.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);
            hierarchical_tri.completeTracks(_sfmOptions.triangulatorOptions);
            Logger::instance()->infof("[SFM] HierarchicalBA point-network reconciliation retriangulated=%d points=%zu",
                                      retriangulated,
                                      _reconstruction->numPoints3D());
        }

        if (hierarchical_schedule_active && !finalRefinement)
        {
            if (hierarchical_summary.applied() || recently_partitioned)
            {
                Logger::instance()->info("[SFM] Periodic full global BA replaced by hierarchical block stabilization");
                return;
            }
            // 块求解真实失败时保留原全局 BA 回退，不让调度优化降低鲁棒性。
        }
        if (finalRefinement && (hierarchical_summary.applied() || recently_partitioned))
        {
            // 共享镜头自标定后必须重三角化并再做一次全局求解，否则输出点网仍对应
            // 旧内参。固定内参路径保持单轮，避免大工程产生不必要的重复计算。
            bool refiningSharedCameraModel = false;
            for (std::size_t index = 0; index < kBAIntrinsicParameterCount; ++index)
            {
                refiningSharedCameraModel =
                    refiningSharedCameraModel ||
                    sharedIntrinsicParameterEnabled(_sfmOptions.baOptions, static_cast<BAIntrinsicParameter>(index));
            }
            maxRounds = refiningSharedCameraModel ? std::min(3, maxRounds) : 1;
            Logger::instance()->info(
                refiningSharedCameraModel
                    ? "[SFM] Final global BA uses up to three calibration/retriangulation rounds after hierarchical BA"
                    : "[SFM] Final global BA limited to one seam refinement after hierarchical BA");
        }

        // 各轮可以继续使用上一轮内参作为数值初值，但硬边界与弱先验必须相对整次
        // IncrementalSfm 生命周期的首次影像标定，不能在周期/最终/重试 BA 间重新锚定。
        std::vector<ImageId> iterativeImageIds = _reconstruction->registeredImageIds();
        std::sort(iterativeImageIds.begin(), iterativeImageIds.end());
        std::vector<FramePinholeCamera> currentIntrinsicCameras;
        currentIntrinsicCameras.reserve(iterativeImageIds.size());
        for (const ImageId imageId : iterativeImageIds)
        {
            currentIntrinsicCameras.push_back(_reconstruction->camera(imageId));
        }
        std::vector<FramePinholeCamera> iterativeIntrinsicReferences =
            _sfmOptions.baOptions.sharedIntrinsicReferenceCameras;
        if (iterativeIntrinsicReferences.empty())
        {
            iterativeIntrinsicReferences = SfmBundleAdjustCoordinator::buildPersistentIntrinsicReferences(
                iterativeImageIds, currentIntrinsicCameras, &_stableIntrinsicReferenceByImageId);
        }

        size_t prevNumPoints = _reconstruction->numPoints3D();
        std::vector<FramePinholeCamera> previousRoundIntrinsicCameras;

        for (int round = 0; round < maxRounds; ++round)
        {
            Logger::instance()->infof("[SFM] IterativeGlobalBA mode=%s round %d/%d: numPts=%zu",
                                      finalRefinement ? "final" : "periodic",
                                      round + 1,
                                      maxRounds,
                                      _reconstruction->numPoints3D());

            // (1) 过滤负深度点
            if (_sfmOptions.filterNegativeDepth)
            {
                int nNeg = filterNegativeDepthPoints();
                if (nNeg > 0)
                {
                    Logger::instance()->infof("[SFM]   Filtered %d negative-depth points", nNeg);
                }
            }

            // (2) 执行全局 BA
            runBundleAdjust(false, {}, &iterativeIntrinsicReferences, round == 0);
            currentIntrinsicCameras.clear();
            for (const ImageId imageId : iterativeImageIds)
            {
                currentIntrinsicCameras.push_back(_reconstruction->camera(imageId));
            }

            // (3) 利用 BA 后更新的相机位姿重三角化所有 3D 点（参考 COLMAP Retriangulate）
            Triangulator tri(*_reconstruction, _correspondenceGraph, _sfmOptions.baOptions.numThreads);
            int nRetri = tri.retriangulatePoints(_sfmOptions.filterMaxReprojError);
            if (nRetri > 0)
            {
                Logger::instance()->infof("[SFM]   Retriangulated %d points with updated poses", nRetri);
            }

            // (4) 过滤点质量（重投影误差 + 三角化角）
            tri.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);

            // (5) 补三角化（尝试延伸已有轨迹，含深度检查）。补入的新观测会改变
            // 点的 RMS 和有效交会角，因此必须重算并再次执行同一质量门控。
            const int nCompleted = tri.completeTracks(_sfmOptions.triangulatorOptions);
            if (nCompleted > 0)
            {
                tri.recomputeReprojErrors();
                const int nPostCompleteFiltered =
                    tri.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);
                if (nPostCompleteFiltered > 0)
                {
                    Logger::instance()->infof("[SFM]   Filtered %d points after completing %d observations",
                                              nPostCompleteFiltered,
                                              nCompleted);
                }
            }

            // (6) 收敛判断：3D 点数变化 < 1%
            size_t curNumPoints = _reconstruction->numPoints3D();
            double changeRate =
                (prevNumPoints > 0)
                    ? std::fabs(static_cast<double>(curNumPoints) - static_cast<double>(prevNumPoints)) /
                          static_cast<double>(prevNumPoints)
                    : 1.0;

            const bool sharedIntrinsicsRefined = _lastGlobalBAResultApplied && _lastGlobalBARefinedIntrinsicCount > 0;
            const double maximumIntrinsicChange =
                sharedIntrinsicsRefined
                    ? SfmBundleAdjustCoordinator::maximumCameraIntrinsicChange(
                          previousRoundIntrinsicCameras, currentIntrinsicCameras, iterativeIntrinsicReferences)
                    : std::numeric_limits<double>::infinity();

            Logger::instance()->infof("[SFM]   After round %d: numPts=%zu, pointChange=%.4f, "
                                      "maxIntrinsicChange=%.6f",
                                      round + 1,
                                      curNumPoints,
                                      changeRate,
                                      maximumIntrinsicChange);

            if (SfmBundleAdjustCoordinator::hasIterativeGlobalBaConverged(
                    round + 1, changeRate, sharedIntrinsicsRefined, maximumIntrinsicChange))
            {
                Logger::instance()->info("[SFM]   Converged (point network and shared intrinsics are stable)");
                break;
            }
            prevNumPoints = curNumPoints;
            if (sharedIntrinsicsRefined)
            {
                previousRoundIntrinsicCameras = currentIntrinsicCameras;
            }
        }
    }

    // ============================================================
    // 内部：过滤负深度点（参考 COLMAP FilterObservationsWithNegativeDepth）
    // ============================================================

    int IncrementalSfm::filterNegativeDepthPoints()
    {
        int deletedCount = 0;
        auto allPtIds = _reconstruction->allPoint3DIds();

        for (Point3DId pid : allPtIds)
        {
            if (!_reconstruction->hasPoint3D(pid))
                continue;
            auto& pt = _reconstruction->point3D(pid);
            const auto& xyz = pt.xyz;

            // 检查该点在每个观测相机中的深度
            bool hasNegativeDepth = false;
            std::vector<size_t> badObsIndices;

            for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
            {
                const auto& elem = pt.track.elements[oi];
                if (!_reconstruction->isRegistered(elem.imageId))
                    continue;
                if (!_reconstruction->hasCamera(elem.imageId))
                    continue;

                const FramePinholeCamera& cam = _reconstruction->camera(elem.imageId);
                const double world[3] = {xyz[0], xyz[1], xyz[2]};
                if (!cam.isPointInFront(world))
                {
                    badObsIndices.push_back(oi);
                    hasNegativeDepth = true;
                }
            }

            if (!hasNegativeDepth)
                continue;

            // 如果全部观测都是负深度或移除坏观测后不足 2 个，删除整个点
            size_t goodObs = pt.track.elements.size() - badObsIndices.size();
            if (goodObs < 2)
            {
                _reconstruction->deletePoint3D(pid);
                ++deletedCount;
            }
            else
            {
                // 移除坏观测（从后往前删）
                for (auto it = badObsIndices.rbegin(); it != badObsIndices.rend(); ++it)
                {
                    const auto& elem = pt.track.elements[*it];
                    // 清理 ImageData 中的关联
                    if (_reconstruction->hasImage(elem.imageId))
                    {
                        auto& imgData = _reconstruction->image(elem.imageId);
                        if (elem.featureIdx < imgData.point3DIds.size())
                        {
                            imgData.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                        }
                    }
                    pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
                }
            }
        }

        return deletedCount;
    }

    // ============================================================
    // 内部：可见性缓存管理
    // ============================================================

} // namespace xjw
