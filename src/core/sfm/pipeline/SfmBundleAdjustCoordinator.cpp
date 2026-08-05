#include "SfmBundleAdjustCoordinator.h"
#include "IncrementalSfmDetail.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "geometry/SimilarityGaugeNormalizer.h"
#include "Intersection.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"

#include "log/Logger.h"

#include "DeterministicOpenCvRansac.h"
#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
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

SfmBundleAdjustCoordinator::SfmBundleAdjustCoordinator(IncrementalSfm &owner)
    : _owner(owner)
{
}

bool SfmBundleAdjustCoordinator::shouldRefineSharedIntrinsics(
    bool localOnly,
    int activeCameraCount,
    int registeredImageCount,
    int totalImageCount)
{
    if (localOnly || totalImageCount < 3 || registeredImageCount < 3 ||
        activeCameraCount != registeredImageCount)
    {
        return false;
    }

    // 少量难配准影像不应阻止整个航摄块完成相机自标定。达到 90% 覆盖时，
    // 已注册相机已经形成稳定全局网络；仍要求活动相机覆盖全部已注册影像，
    // 从而禁止局部窗口或残缺 BA 更新共享镜头参数。
    return static_cast<long long>(registeredImageCount) * 10LL >=
           static_cast<long long>(totalImageCount) * 9LL;
}

bool SfmBundleAdjustCoordinator::hasIterativeGlobalBaConverged(
    int completedRoundCount,
    double pointChangeRate,
    bool sharedIntrinsicsRefined,
    double focalScaleChange,
    double radialCoefficientChange)
{
    if (completedRoundCount < 2 || !std::isfinite(pointChangeRate) ||
        pointChangeRate >= 0.01)
    {
        return false;
    }
    if (!sharedIntrinsicsRefined)
    {
        return true;
    }
    return std::isfinite(focalScaleChange) && focalScaleChange < 5.0e-4 &&
           std::isfinite(radialCoefficientChange) &&
           radialCoefficientChange < 5.0e-4;
}

bool SfmBundleAdjustCoordinator::shouldRunPeriodicGlobalBa(
    int registeredImageCount,
    int registrationTarget,
    int iterationsSinceGlobalBa,
    int globalBaInterval)
{
    const int safe_interval = std::max(1, globalBaInterval);
    if (iterationsSinceGlobalBa < safe_interval ||
        registeredImageCount >= registrationTarget)
    {
        return false;
    }

    const int remaining_images = registrationTarget - registeredImageCount;
    const int final_guard = std::max(2, safe_interval / 4);
    return remaining_images > final_guard;
}

bool SfmBundleAdjustCoordinator::shouldUseMultiViewOnlyGlobalBa(
    bool localOnly,
    int activeCameraCount,
    int totalTrackCount,
    int twoViewTrackCount,
    int multiViewTrackCount)
{
    if (localOnly || activeCameraCount < 96 || totalTrackCount < 1000 ||
        twoViewTrackCount < 0 || multiViewTrackCount < 0 ||
        twoViewTrackCount + multiViewTrackCount != totalTrackCount)
    {
        return false;
    }

    // 两视图点只约束一对相机，无法为长航带提供跨影像刚性。只有弱轨迹占比达到
    // 65%，且仍有足够多三视图轨迹覆盖相机块时才缩减全局 BA，避免小工程或本就
    // 缺少强轨迹的工程因过滤后约束不足。完整点云会在 BA 后统一重三角化。
    const bool weakNetwork =
        static_cast<long long>(twoViewTrackCount) * 100LL >=
        static_cast<long long>(totalTrackCount) * 65LL;
    const int minimumMultiViewTracks = std::max(500, activeCameraCount * 4);
    return weakNetwork && multiViewTrackCount >= minimumMultiViewTracks;
}

int SfmBundleAdjustCoordinator::iterativeGlobalBaRoundLimit(
    int configuredRounds,
    bool finalRefinement)
{
    const int safe_rounds = std::max(1, configuredRounds);
    return finalRefinement ? safe_rounds : std::min(2, safe_rounds);
}

void SfmBundleAdjustCoordinator::run(bool localOnly,
                                     const std::vector<ImageId> &anchorIds)
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

void IncrementalSfm::runBundleAdjust(bool localOnly, const std::vector<ImageId> &anchorIds)
{
    const char *scopeName = localOnly ? "local" : "global";
    const auto reportSkipped = [this, localOnly, scopeName](const std::string &reason)
    {
        Logger::instance()->infof("[BA] skipped scope=%s reason=%s",
                                  scopeName,
                                  reason.c_str());
        if (!localOnly)
        {
            _lastGlobalBARmsBefore = 0.0;
            _lastGlobalBARmsAfter = 0.0;
            _lastGlobalBATracksTotal = 0;
            _lastGlobalBATracksOptimized = 0;
            _lastGlobalBATracksFiltered = 0;
            _lastGlobalBARefinedIntrinsicCount = 0;
            _lastGlobalBASharedFocalScale = 1.0;
            _lastGlobalBASharedRadialK1 = 0.0;
            _lastGlobalBASharedRadialK2 = 0.0;
            _lastGlobalBARequestedBackend = _sfmOptions.baOptions.backend;
            _lastGlobalBAUsedBackend = BABackend::LegacyCpu;
            _lastGlobalBASolveStatus = BASolveStatus::NotRun;
            _lastGlobalBASolutionUsable = false;
            _lastGlobalBAResultApplied = false;
            _lastGlobalBABackendFallback = false;
            _lastGlobalBAObservationCount = 0;
            _lastGlobalBATotalSeconds = 0.0;
            _lastGlobalBABackendMessage = reason;
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
            auto topN = _correspondenceGraph.topConnectedImages(aid, static_cast<size_t>(_sfmOptions.localBANumImages));
            for (auto &[nid, _] : topN)
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
            const ScenePoint3D &point = _reconstruction->point3D(pointId);
            bool touchesActiveImage = false;
            for (const TrackElement &element : point.track.elements)
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
            for (const TrackElement &element : point.track.elements)
            {
                if (activeImageIds.count(element.imageId) == 0 &&
                    _reconstruction->isRegistered(element.imageId))
                {
                    ++boundaryObservationCounts[element.imageId];
                }
            }
        }

        std::vector<std::pair<ImageId, int>> rankedBoundaryImages(
            boundaryObservationCounts.begin(), boundaryObservationCounts.end());
        std::sort(rankedBoundaryImages.begin(),
                  rankedBoundaryImages.end(),
                  [](const auto &left, const auto &right)
                  {
                      if (left.second != right.second)
                      {
                          return left.second > right.second;
                      }
                      return left.first < right.first;
                  });
        for (const auto &[imageId, observationCount] : rankedBoundaryImages)
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
    std::vector<Camera> baCameras;
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
    if (!localOnly)
    {
        for (Point3DId pid : allPtIds)
        {
            if (!_reconstruction->hasPoint3D(pid))
            {
                continue;
            }
            const ScenePoint3D &point = _reconstruction->point3D(pid);
            int activeObservationCount = 0;
            for (const TrackElement &element : point.track.elements)
            {
                if (idToIdx.find(element.imageId) != idToIdx.end() &&
                    _reconstruction->hasImage(element.imageId) &&
                    element.featureIdx <
                        _reconstruction->image(element.imageId).keypoints.size())
                {
                    ++activeObservationCount;
                }
            }
            if (activeObservationCount < 2)
            {
                continue;
            }
            ++globalCandidateTrackCount;
            if (activeObservationCount == 2)
            {
                ++globalTwoViewTrackCount;
            }
            else
            {
                ++globalMultiViewTrackCount;
            }
        }
    }
    const bool useMultiViewOnlyGlobalBa =
        SfmBundleAdjustCoordinator::shouldUseMultiViewOnlyGlobalBa(
            localOnly,
            static_cast<int>(baCameras.size()),
            globalCandidateTrackCount,
            globalTwoViewTrackCount,
            globalMultiViewTrackCount);
    if (!localOnly)
    {
        Logger::instance()->infof(
            "[BA] network scope=global policy=%s candidateTracks=%d twoView=%d multiView=%d",
            useMultiViewOnlyGlobalBa ? "multi_view_only" : "all_tracks",
            globalCandidateTrackCount,
            globalTwoViewTrackCount,
            globalMultiViewTrackCount);
    }
    int control_constraint_count = 0;
    for (Point3DId pid : allPtIds)
    {
        if (!_reconstruction->hasPoint3D(pid))
            continue;
        const ScenePoint3D &pt = _reconstruction->point3D(pid);
        BATrack track;
        track.initialPoint = pt.xyz;

        if (_controlNetworkApplied && pt.track.source == TrackSource::PriorMarker)
        {
            const control_points::PriorTrack *prior = priorTrack(pt.track.sourceId);
            const auto residual = std::find_if(
                _controlNetworkResult.controlResiduals.cbegin(),
                _controlNetworkResult.controlResiduals.cend(),
                [&pt](const control_points::MarkerResidual &value)
                {
                    return value.markerId == pt.track.sourceId;
                });
            if (prior && prior->role == control_points::MarkerRole::ControlPoint
                && prior->hasReference && prior->referenceUsable
                && residual != _controlNetworkResult.controlResiduals.cend() && residual->inlier)
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
                constraint.sigmaMeters = sigma_count > 0
                    ? std::sqrt(sigma_sum_squared / static_cast<double>(sigma_count))
                    : 1.0;
                constraint.weight = 1.0;
                constraint.sourceIndex = static_cast<int>(
                    prior - static_cast<const control_points::PriorTrack *>(
                                _pendingPriorTracks.data()));
                track.controlPointConstraints.push_back(constraint);
                ++control_constraint_count;
            }
        }

        for (const auto &elem : pt.track.elements)
        {
            auto idxIt = idToIdx.find(elem.imageId);
            if (idxIt == idToIdx.end())
                continue;

            const ImageData &img = _reconstruction->image(elem.imageId);
            if (elem.featureIdx >= img.keypoints.size())
                continue;

            BAObservation obs;
            obs.cameraIndex = idxIt->second;
            obs.u = img.keypoints[elem.featureIdx].x;
            obs.v = img.keypoints[elem.featureIdx].y;
            obs.weight = pt.track.confidence;
            track.observations.push_back(obs);
        }

        // 至少 2 个观测才有意义
        const bool keepWeakPriorTrack =
            pt.track.source == TrackSource::PriorMarker;
        if (track.observations.size() >= 2 &&
            (!useMultiViewOnlyGlobalBa || track.observations.size() >= 3 ||
             keepWeakPriorTrack))
        {
            if (pt.track.source == TrackSource::PriorMarker && !pt.track.sourceId.empty())
            {
                marker_track_indices[pt.track.sourceId] = static_cast<int>(baTracks.size());
            }
            baTracks.push_back(std::move(track));
            trackToPid.push_back(pid);
        }
    }

    if (baTracks.empty())
    {
        reportSkipped("no_tracks_with_two_or_more_observations");
        return;
    }

    // 构造本次 BA 选项，并显式消除无绝对约束问题的 7 自由度 gauge。
    BAOptions baOpt = _sfmOptions.baOptions;
    if (localOnly)
    {
        // 局部窗口只负责稳定新注册相机，不需要沿用最终全局 BA 的 20 轮预算。
        // 这也避免数百图工程在每个局部窗口输出整段迭代日志，造成“逐图平差”的错觉。
        baOpt.maxIterations = std::min(baOpt.maxIterations, 10);
        baOpt.logIterationProgress = false;
    }
    const bool refineSharedIntrinsics =
        SfmBundleAdjustCoordinator::shouldRefineSharedIntrinsics(
        localOnly,
        static_cast<int>(baCameras.size()),
        static_cast<int>(_reconstruction->registeredImageIds().size()),
        static_cast<int>(_reconstruction->numImages()));
    if (!refineSharedIntrinsics)
    {
        // 局部窗口或尚未完整注册时更新“共享”内参，会把同一镜头组拆成多个主点/焦距。
        // 此阶段仅优化位姿和三维点，待最终全局 BA 再统一释放镜头组内参。
        baOpt.refineSharedFocalLength = false;
        baOpt.refineSharedFocalAspectRatio = false;
        baOpt.refineSharedPrincipalPoint = false;
        baOpt.refineSharedRadialDistortion = false;
    }
    // SfM 协调器会固定旋转/平移规范，并在求解后恢复基线尺度，
    // 因此由调用方管理完整的 Sim(3) gauge，避免 BA 模块再自动固定第二台相机。
    baOpt.gaugePolicy = BAGaugePolicy::CallerManaged;
    int control_scale_bar_count = 0;
    for (std::size_t scale_index = 0; scale_index < _pendingPriorScaleBars.size(); ++scale_index)
    {
        const control_points::PriorScaleBar &scale_bar = _pendingPriorScaleBars[scale_index];
        if (!scale_bar.enabled || scale_bar.role != control_points::ScaleBarRole::Control
            || !std::isfinite(scale_bar.measuredDistance) || scale_bar.measuredDistance <= 0.0
            || !std::isfinite(scale_bar.sigma) || scale_bar.sigma <= 0.0)
        {
            continue;
        }
        const auto first = marker_track_indices.find(scale_bar.firstMarkerId);
        const auto second = marker_track_indices.find(scale_bar.secondMarkerId);
        if (first == marker_track_indices.end() || second == marker_track_indices.end()
            || first->second == second->second)
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
        for (BACameraPosePrior &prior : baOpt.cameraPosePriors)
        {
            if (!prior.enabled) continue;
            prior.cameraCenter = _controlNetworkTransform.apply(prior.cameraCenter);
            prior.cameraToWorldRotation =
                _controlNetworkTransform.rotate(prior.cameraToWorldRotation);
            prior.positionSigmaMeters *= _controlNetworkTransform.scale;
        }
    }
    if (control_constraint_count > 0)
    {
        baOpt.enableControlPointConstraints = true;
        // 当前阶段控制点约束统一走 Ceres CPU；原生 CUDA 约束雅可比在后续任务补齐。
        if (BundleAdjust::isBackendAvailable(BABackend::CeresCpu))
        {
            baOpt.backend = BABackend::CeresCpu;
        }
    }
    if (control_scale_bar_count > 0)
    {
        baOpt.enableScaleBarConstraints = true;
        if (BundleAdjust::isBackendAvailable(BABackend::CeresCpu))
        {
            baOpt.backend = BABackend::CeresCpu;
        }
    }
    const bool hasAbsolutePoseConstraint =
        control_constraint_count > 0 ||
        std::any_of(baOpt.cameraPosePriors.begin(),
                    baOpt.cameraPosePriors.end(),
                    [](const BACameraPosePrior &prior)
                    {
                        return prior.enabled;
                    });
    const bool hasAbsoluteScaleConstraint =
        hasAbsolutePoseConstraint || control_scale_bar_count > 0;
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
        if (baOpt.fixedCameraIndices.empty() &&
            !hasAbsolutePoseConstraint &&
            !baImageIds.empty())
        {
            baOpt.fixedCameraIndices.push_back(0);
        }
    }
    else if (!baImageIds.empty())
    {
        if (!hasAbsolutePoseConstraint)
        {
            baOpt.fixedCameraIndices = {0};
        }
    }

    // 单目 BA 固定一台相机后仍有尺度规范。不要固定第二台相机的完整位姿；
    // 记录一条非退化基线，并在求解后对全部相机中心和点做同一 Sim(3) 尺度恢复。
    if (baOpt.refineCameraPose &&
        !hasAbsoluteScaleConstraint &&
        baOpt.fixedCameraIndices.size() == 1)
    {
        const int anchorIndex = baOpt.fixedCameraIndices.front();
        if (anchorIndex >= 0 && anchorIndex < static_cast<int>(baCameras.size()))
        {
            const auto anchorCenter = baCameras[static_cast<std::size_t>(anchorIndex)].cameraCenter();
            for (int candidateIndex = 0;
                 candidateIndex < static_cast<int>(baCameras.size());
                 ++candidateIndex)
            {
                if (candidateIndex == anchorIndex)
                {
                    continue;
                }
                const auto candidateCenter =
                    baCameras[static_cast<std::size_t>(candidateIndex)].cameraCenter();
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
        Logger::instance()->infof(
            "[BA] problem scope=%s cameras=%d tracks=%d observations=%d threads=%d "
            "requested=%s selected=%s reason=%s ceresCudaAvailable=%s "
            "cudaMinCameras=%d cudaMinObservations=%d",
            scopeName,
            stats.cameraCount,
            stats.trackCount,
            stats.observationCount,
            baOpt.numThreads,
            BundleAdjust::backendName(baOpt.backend),
            BundleAdjust::backendName(decision.backend),
            decision.reason.c_str(),
            BundleAdjust::isBackendAvailable(BABackend::CeresCuda) ? "true" : "false",
            baOpt.minCeresCudaCameras,
            baOpt.minCeresCudaObservations);
    }

    // 执行 BA
    BAResult baResult = BundleAdjust::optimizePoints(baCameras, baTracks, baOpt);

    bool gaugeNormalizationFailed = false;
    if (baResult.solutionUsable && similarityGaugeCameras.has_value())
    {
        const SimilarityGaugeNormalizationResult gaugeResult =
            normalizeSimilarityGauge(
                baCameras,
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
            Logger::instance()->warnf(
                "[BA] similarity gauge normalization failed scope=%s reason=%s",
                scopeName,
                gaugeResult.reason.c_str());
        }
    }

    bool applyBaResult = baResult.solutionUsable && !gaugeNormalizationFailed;
    if (!applyBaResult)
    {
        Logger::instance()->warnf(
            "[BA] 求解结果不可写回: status=%d backend=%s message=%s",
            static_cast<int>(baResult.solveStatus),
            BundleAdjust::backendName(baResult.usedBackend),
            baResult.backendMessage.c_str());
    }
    if (applyBaResult && control_constraint_count == 0 &&
        control_scale_bar_count == 0 &&
        std::isfinite(baResult.meanRmsBefore) &&
        std::isfinite(baResult.meanRmsAfter))
    {
        const double rmsTolerance = std::max(
            1.0e-9,
            std::abs(baResult.meanRmsBefore) * 1.0e-6);
        if (baResult.meanRmsAfter > baResult.meanRmsBefore + rmsTolerance)
        {
            applyBaResult = false;
            Logger::instance()->warnf(
                "[BA] rejected scope=%s reason=reprojection_rms_regressed rms=%.9f->%.9f",
                scopeName,
                baResult.meanRmsBefore,
                baResult.meanRmsAfter);
        }
    }
    const bool knownPoseGlobalBa = _sfmOptions.useKnownCameraPoses && !localOnly &&
                                   baOpt.refineCameraPose && !baOpt.cameraPosePriors.empty();
    if (knownPoseGlobalBa)
    {
        if (!std::isfinite(baResult.meanRmsAfter) ||
            baResult.meanRmsAfter > _sfmOptions.filterMaxReprojError)
        {
            applyBaResult = false;
        }
        for (size_t i = 0; applyBaResult && i < baImageIds.size() &&
                            i < baCameras.size() &&
                            i < baResult.refinedCameras.size() &&
                            i < baOpt.cameraPosePriors.size(); ++i)
        {
            const int cameraIndex = static_cast<int>(i);
            if (std::find(baOpt.fixedCameraIndices.begin(),
                          baOpt.fixedCameraIndices.end(),
                          cameraIndex) != baOpt.fixedCameraIndices.end())
            {
                continue;
            }
            const BACameraPosePrior &prior = baOpt.cameraPosePriors[i];
            if (!prior.enabled)
            {
                continue;
            }
            const auto beforeCenter = baCameras[i].cameraCenter();
            const auto afterCenter = baResult.refinedCameras[i].cameraCenter();
            const double beforeDistance = std::sqrt(
                (beforeCenter[0] - prior.cameraCenter[0]) * (beforeCenter[0] - prior.cameraCenter[0]) +
                (beforeCenter[1] - prior.cameraCenter[1]) * (beforeCenter[1] - prior.cameraCenter[1]) +
                (beforeCenter[2] - prior.cameraCenter[2]) * (beforeCenter[2] - prior.cameraCenter[2]));
            const double afterDistance = std::sqrt(
                (afterCenter[0] - prior.cameraCenter[0]) * (afterCenter[0] - prior.cameraCenter[0]) +
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
    Logger::instance()->infof(
        "[BA] result scope=%s cameras=%zu tracks=%zu observations=%d requested=%s used=%s "
        "status=%s usable=%s applied=%s fallback=%s rms=%.6f->%.6f totalSeconds=%.3f message=%s",
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
        baResult.totalSeconds,
        baResult.backendMessage.c_str());
    if (baOpt.refineSharedFocalLength ||
        baOpt.refineSharedFocalAspectRatio ||
        baOpt.refineSharedPrincipalPoint ||
        baOpt.refineSharedRadialDistortion)
    {
        Logger::instance()->infof(
            "[BA] intrinsics scope=%s applied=%s cameras=%d groups=%d "
            "focalScale=%.8f aspectScale=%.8f principalOffsetPx=(%.4f,%.4f) "
            "radial=(%.8f,%.8f)",
            scopeName,
            applyBaResult ? "true" : "false",
            applyBaResult ? baResult.refinedIntrinsicCount : 0,
            applyBaResult ? baResult.refinedCalibrationGroupCount : 0,
            applyBaResult ? baResult.refinedSharedFocalScale : 1.0,
            applyBaResult ? baResult.refinedSharedFocalAspectScale : 1.0,
            applyBaResult ? baResult.refinedSharedPrincipalOffsetX : 0.0,
            applyBaResult ? baResult.refinedSharedPrincipalOffsetY : 0.0,
            applyBaResult ? baResult.refinedSharedRadialK1 : 0.0,
            applyBaResult ? baResult.refinedSharedRadialK2 : 0.0);
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

        if (!localOnly && refineSharedIntrinsics &&
            baResult.refinedCalibrationGroupCount == 1 &&
            !baResult.refinedCameras.empty())
        {
            // 未注册影像的 PnP 会从预载相机读取内参。单一镜头组完成全局自标定后，
            // 将同一组内参同步过去，避免最终重试继续使用零畸变/旧焦距。
            const Camera &calibratedCamera = baResult.refinedCameras.front();
            const Camera::Intrinsics calibratedIntrinsics = calibratedCamera.intrinsics();
            const Camera::Distortion calibratedDistortion = calibratedCamera.distortion();
            for (auto &[imageId, camera] : _preloadedCameras)
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

        const BARefinedPoint &bp = baResult.points[ti];
        if (!bp.valid)
        {
            // ── 观测级过滤：逐观测检查重投影误差 ──
            auto &pt = _reconstruction->point3D(pid);
            const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
            std::vector<size_t> badObsIndices;

            for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
            {
                const auto &elem = pt.track.elements[oi];
                if (!_reconstruction->isRegistered(elem.imageId))
                    continue;
                if (!_reconstruction->hasCamera(elem.imageId))
                    continue;

                const Camera &cam = _reconstruction->camera(elem.imageId);
                const ImageData &imgData = _reconstruction->image(elem.imageId);
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
                if (reproj > filterThresh * 1.5)
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
                    const auto &elem = pt.track.elements[*it];
                    if (_reconstruction->hasImage(elem.imageId))
                    {
                        auto &imgd = _reconstruction->image(elem.imageId);
                        if (elem.featureIdx < imgd.point3DIds.size())
                        {
                            imgd.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                        }
                    }
                    pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
                    ++removedObs;
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
            auto &pt = _reconstruction->point3D(pid);
            const double filterThresh = _sfmOptions.baOptions.filterMaxReprojError;
            std::vector<size_t> badObsIndices;

            for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
            {
                const auto &elem = pt.track.elements[oi];
                if (!_reconstruction->isRegistered(elem.imageId))
                    continue;
                if (!_reconstruction->hasCamera(elem.imageId))
                    continue;

                const Camera &cam = _reconstruction->camera(elem.imageId);
                const ImageData &imgData = _reconstruction->image(elem.imageId);
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
                // 对有效点使用 2 倍阈值剔除明显错误观测
                if (reproj > filterThresh * 2.0)
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
                        const auto &elem = pt.track.elements[*it];
                        if (_reconstruction->hasImage(elem.imageId))
                        {
                            auto &imgd = _reconstruction->image(elem.imageId);
                            if (elem.featureIdx < imgd.point3DIds.size())
                            {
                                imgd.point3DIds[elem.featureIdx] = kInvalidPoint3DId;
                            }
                        }
                        pt.track.elements.erase(pt.track.elements.begin() + static_cast<long>(*it));
                        ++removedObs;
                    }
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
        const double appliedRmsAfter = applyBaResult
            ? baResult.meanRmsAfter
            : baResult.meanRmsBefore;
        _lastControlPointConstraintCount = control_constraint_count;
        _lastControlScaleBarConstraintCount = control_scale_bar_count;
        _lastGlobalBARmsBefore = baResult.meanRmsBefore;
        _lastGlobalBARmsAfter = appliedRmsAfter;
        _lastGlobalBATracksTotal = baResult.totalTracks;
        _lastGlobalBATracksOptimized = baResult.optimizedTracks;
        _lastGlobalBATracksFiltered = applyBaResult
            ? static_cast<int>(baTracks.size()) - baResult.optimizedTracks
            : 0;
        if (_lastGlobalBATracksFiltered < 0)
            _lastGlobalBATracksFiltered = deletedPts;
        _lastGlobalBARefinedIntrinsicCount =
            applyBaResult ? baResult.refinedIntrinsicCount : 0;
        _lastGlobalBASharedFocalScale =
            applyBaResult ? baResult.refinedSharedFocalScale : 1.0;
        _lastGlobalBASharedFocalAspectScale =
            applyBaResult ? baResult.refinedSharedFocalAspectScale : 1.0;
        _lastGlobalBASharedPrincipalOffsetX =
            applyBaResult ? baResult.refinedSharedPrincipalOffsetX : 0.0;
        _lastGlobalBASharedPrincipalOffsetY =
            applyBaResult ? baResult.refinedSharedPrincipalOffsetY : 0.0;
        _lastGlobalBASharedRadialK1 =
            applyBaResult ? baResult.refinedSharedRadialK1 : 0.0;
        _lastGlobalBASharedRadialK2 =
            applyBaResult ? baResult.refinedSharedRadialK2 : 0.0;
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
    }
}

// ============================================================
// 内部：迭代全局 BA 精化（参考 COLMAP IterativeGlobalRefinement）
// ============================================================

void IncrementalSfm::iterativeGlobalBA(bool finalRefinement)
{
    const int maxRounds = SfmBundleAdjustCoordinator::iterativeGlobalBaRoundLimit(
        _sfmOptions.iterativeBARounds,
        finalRefinement);
    size_t prevNumPoints = _reconstruction->numPoints3D();
    double previousFocalScale = std::numeric_limits<double>::quiet_NaN();
    double previousRadialK1 = std::numeric_limits<double>::quiet_NaN();
    double previousRadialK2 = std::numeric_limits<double>::quiet_NaN();

    for (int round = 0; round < maxRounds; ++round)
    {
        Logger::instance()->infof(
            "[SFM] IterativeGlobalBA mode=%s round %d/%d: numPts=%zu",
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
        runBundleAdjust(false);

        // (3) 利用 BA 后更新的相机位姿重三角化所有 3D 点（参考 COLMAP Retriangulate）
        Triangulator tri(*_reconstruction, _correspondenceGraph);
        int nRetri = tri.retriangulatePoints(_sfmOptions.filterMaxReprojError);
        if (nRetri > 0)
        {
            Logger::instance()->infof("[SFM]   Retriangulated %d points with updated poses", nRetri);
        }

        // (4) 过滤点质量（重投影误差 + 三角化角）
        tri.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);

        // (5) 补三角化（尝试延伸已有轨迹，含深度检查）
        tri.completeTracks(_sfmOptions.triangulatorOptions);

        // (6) 收敛判断：3D 点数变化 < 1%
        size_t curNumPoints = _reconstruction->numPoints3D();
        double changeRate = (prevNumPoints > 0)
                                ? std::fabs(static_cast<double>(curNumPoints) - static_cast<double>(prevNumPoints)) /
                                      static_cast<double>(prevNumPoints)
                                : 1.0;

        const bool sharedIntrinsicsRefined =
            _lastGlobalBAResultApplied && _lastGlobalBARefinedIntrinsicCount > 0;
        const double focalScaleChange =
            sharedIntrinsicsRefined && std::isfinite(previousFocalScale)
                ? std::abs(_lastGlobalBASharedFocalScale - previousFocalScale)
                : std::numeric_limits<double>::infinity();
        const double radialCoefficientChange =
            sharedIntrinsicsRefined && std::isfinite(previousRadialK1) &&
                    std::isfinite(previousRadialK2)
                ? std::max(std::abs(_lastGlobalBASharedRadialK1 - previousRadialK1),
                           std::abs(_lastGlobalBASharedRadialK2 - previousRadialK2))
                : std::numeric_limits<double>::infinity();

        Logger::instance()->infof(
            "[SFM]   After round %d: numPts=%zu, pointChange=%.4f, "
            "focalChange=%.6f, radialChange=%.6f",
            round + 1,
            curNumPoints,
            changeRate,
            focalScaleChange,
            radialCoefficientChange);

        if (SfmBundleAdjustCoordinator::hasIterativeGlobalBaConverged(
                round + 1,
                changeRate,
                sharedIntrinsicsRefined,
                focalScaleChange,
                radialCoefficientChange))
        {
            Logger::instance()->info(
                "[SFM]   Converged (point network and shared intrinsics are stable)");
            break;
        }
        prevNumPoints = curNumPoints;
        if (sharedIntrinsicsRefined)
        {
            previousFocalScale = _lastGlobalBASharedFocalScale;
            previousRadialK1 = _lastGlobalBASharedRadialK1;
            previousRadialK2 = _lastGlobalBASharedRadialK2;
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
        auto &pt = _reconstruction->point3D(pid);
        const auto &xyz = pt.xyz;

        // 检查该点在每个观测相机中的深度
        bool hasNegativeDepth = false;
        std::vector<size_t> badObsIndices;

        for (size_t oi = 0; oi < pt.track.elements.size(); ++oi)
        {
            const auto &elem = pt.track.elements[oi];
            if (!_reconstruction->isRegistered(elem.imageId))
                continue;
            if (!_reconstruction->hasCamera(elem.imageId))
                continue;

            const Camera &cam = _reconstruction->camera(elem.imageId);
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
                const auto &elem = pt.track.elements[*it];
                // 清理 ImageData 中的关联
                if (_reconstruction->hasImage(elem.imageId))
                {
                    auto &imgData = _reconstruction->image(elem.imageId);
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
