/**
 * @file IncrementalSfm.cpp
 * @brief 增量 SfM 的输入装配、主状态机和人工控制网络接入。
 *
 * 未知位姿路径依次执行：对应图构建、初始像对试算、增量 PnP、三角化和 BA；
 * 已知位姿路径委托 KnownPoseReconstructor。FramePinholeCamera 内部始终使用 camera-to-world
 * 旋转和世界系相机中心，OpenCV 的 world-to-camera 约定只在适配层出现。
 */

#include "IncrementalSfm.h"
#include "ImageRegistrationEngine.h"
#include "InitialPairInitializer.h"
#include "IncrementalSfmDetail.h"
#include "KnownPoseReconstructor.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "Intersection.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"

#include "log/Logger.h"

#include "DeterministicOpenCvRansac.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw
{

using namespace incremental_sfm_detail;



IncrementalSfmOptions effectiveSfmOptions(const IncrementalSfmOptions &options)
{
    IncrementalSfmOptions effective = options;
    // 三角化、BA 点过滤和观测级过滤必须使用同一像素阈值，避免同一个点在相邻阶段
    // 被分别判为有效和无效。BAOptions 中的副本仅作为求解器输入，不再独立配置。
    effective.baOptions.filterMaxReprojError = effective.filterMaxReprojError;
    if (effective.executionProfile != SfmExecutionProfile::CoarseEvaluation)
    {
        return effective;
    }

    // 转台闭环种子（首尾影像）在按内点排序时可能落到第 4～6 位，
    // 粗筛保留六个轻量种子，避免只看前三名而丢失能够扩展完整环路的模型。
    effective.maxInitPairCandidates = std::min(effective.maxInitPairCandidates, 6);
    effective.baOptions.maxIterations = std::min(effective.baOptions.maxIterations, 5);
    effective.iterativeBARounds = 1;
    effective.localBAInterval = std::max(effective.localBAInterval, 6);
    effective.globalBAInterval = std::numeric_limits<int>::max();
    effective.baOptions.refineSharedFocalLength = false;
    effective.baOptions.refineSharedFocalAspectRatio = false;
    effective.baOptions.refineSharedPrincipalPoint = false;
    effective.baOptions.refineSharedRadialDistortion = false;
    effective.baOptions.refineSharedHighOrderDistortion = false;
    effective.baOptions.useSharedIntrinsicParameterMask = false;
    effective.baOptions.cameraCalibrationGroupIds.clear();
    effective.baOptions.sharedIntrinsicReferenceCameras.clear();
    effective.baOptions.logIterationProgress = false;
    effective.retryUnregisteredAfterFinalBA = false;
    return effective;
}

// ============================================================
// 构造
// ============================================================

/**
 * @brief 保存规范化选项并创建空重建容器。
 *
 * 构造阶段不读取文件、不运行几何；所有影像、匹配和先验必须在 run() 前加入。
 */
IncrementalSfm::IncrementalSfm(const IncrementalSfmOptions &options)
    : _sfmOptions(effectiveSfmOptions(options)), _reconstruction(std::make_shared<SfmReconstruction>())
{
}

// ============================================================
// 数据输入
// ============================================================

void IncrementalSfm::addImage(ImageId id, const std::string &imagePath, const std::string &cameraPath,
                              const std::vector<FeatureKeypoint> &keypoints)
{
    ImageData data;
    data.id = id;
    data.imagePath = imagePath;
    data.cameraPath = cameraPath;
    data.keypoints = keypoints;
    data.point3DIds.resize(keypoints.size(), kInvalidPoint3DId);

    _reconstruction->addImage(data);
    _correspondenceGraph.addImage(id, keypoints.size());
    _cameraPaths[id] = cameraPath;
}

void IncrementalSfm::addImageWithCamera(ImageId id, const std::string &imagePath, const FramePinholeCamera &camera,
                                        const std::vector<FeatureKeypoint> &keypoints)
{
    ImageData data;
    data.id = id;
    data.imagePath = imagePath;
    data.cameraPath = ""; // 无文件
    data.keypoints = keypoints;
    data.point3DIds.resize(keypoints.size(), kInvalidPoint3DId);

    _reconstruction->addImage(data);
    _correspondenceGraph.addImage(id, keypoints.size());
    _preloadedCameras[id] = camera;
}

void IncrementalSfm::addMatches(ImageId id1, ImageId id2, const std::vector<FeatureMatch> &matches)
{
    _correspondenceGraph.addMatches(id1, id2, matches);
}

void IncrementalSfm::addPriorTrack(const control_points::PriorTrack &track)
{
    if (_priorTracksMaterialized)
    {
        throw std::logic_error("Prior tracks must be added before IncrementalSfm::run");
    }
    _pendingPriorTracks.push_back(track);
}

void IncrementalSfm::addPriorScaleBar(const control_points::PriorScaleBar &scaleBar)
{
    if (_priorTracksMaterialized)
    {
        throw std::logic_error("Prior scale bars must be added before IncrementalSfm::run");
    }
    _pendingPriorScaleBars.push_back(scaleBar);
}

void IncrementalSfm::materializePriorTracks()
{
    if (_priorTracksMaterialized) return;
    _priorTracksMaterialized = true;
    _priorTrackDiagnostics = {};
    _materializedPriorTracks.clear();
    _priorTrackDiagnostics.tracksSubmitted = static_cast<int>(_pendingPriorTracks.size());

    // 人工投影追加为独立的合成关键点，绝不复用附近自动关键点索引。
    // 这样可保留手工观测身份，并把位置冲突作为诊断而不是静默合并。
    std::unordered_set<std::string> marker_ids;
    for (const control_points::PriorTrack &priorTrack : _pendingPriorTracks)
    {
        const auto rejectTrack = [&](const std::string &reason)
        {
            ++_priorTrackDiagnostics.tracksRejected;
            _priorTrackDiagnostics.messages.push_back(priorTrack.markerId + ": " + reason);
        };

        if (priorTrack.markerId.empty() || !marker_ids.insert(priorTrack.markerId).second)
        {
            rejectTrack("empty or duplicate marker id");
            continue;
        }

        std::vector<const control_points::PriorObservation *> accepted_observations;
        std::unordered_set<ImageId> image_ids;
        bool duplicate_image = false;
        for (const control_points::PriorObservation &observation : priorTrack.observations)
        {
            const bool usable_state = control_points::priorObservationParticipates(observation.state);
            const bool finite = std::isfinite(observation.x) && std::isfinite(observation.y)
                && std::isfinite(observation.confidence) && observation.confidence >= 0.0;
            if (!usable_state || observation.stale || !finite
                || !_reconstruction->hasImage(observation.imageId))
            {
                ++_priorTrackDiagnostics.observationsRejected;
                continue;
            }
            if (!image_ids.insert(observation.imageId).second)
            {
                duplicate_image = true;
                break;
            }
            accepted_observations.push_back(&observation);
        }

        if (duplicate_image || accepted_observations.size() < 2)
        {
            rejectTrack(duplicate_image ? "duplicate image observation"
                                        : "fewer than two usable observations");
            continue;
        }

        std::vector<TrackElement> graph_observations;
        graph_observations.reserve(accepted_observations.size());
        for (const control_points::PriorObservation *observation : accepted_observations)
        {
            ImageData &image = _reconstruction->image(observation->imageId);
            const bool conflicts_with_feature = std::any_of(
                image.keypoints.begin(), image.keypoints.end(), [observation](const FeatureKeypoint &keypoint)
                {
                    const double dx = static_cast<double>(keypoint.x) - observation->x;
                    const double dy = static_cast<double>(keypoint.y) - observation->y;
                    return dx * dx + dy * dy <= 0.25;
                });
            if (conflicts_with_feature) ++_priorTrackDiagnostics.observationConflicts;

            const FeatureIdx feature_index = static_cast<FeatureIdx>(image.keypoints.size());
            image.keypoints.push_back({static_cast<float>(observation->x),
                                       static_cast<float>(observation->y)});
            image.point3DIds.push_back(kInvalidPoint3DId);
            _correspondenceGraph.addImage(observation->imageId, image.keypoints.size());
            graph_observations.push_back({observation->imageId, feature_index});
        }

        const float confidence = static_cast<float>(std::clamp(priorTrack.confidence, 0.0, 1.0));
        // 整条 prior 必须原子加入。对应图拒绝时回滚所有合成关键点。
        if (!_correspondenceGraph.addPriorTrack(priorTrack.markerId,
                                                graph_observations,
                                                confidence))
        {
            for (auto it = graph_observations.rbegin(); it != graph_observations.rend(); ++it)
            {
                ImageData &image = _reconstruction->image(it->imageId);
                image.keypoints.pop_back();
                image.point3DIds.pop_back();
                _correspondenceGraph.addImage(it->imageId, image.keypoints.size());
            }
            rejectTrack("correspondence graph rejected synthetic observations");
            continue;
        }

        ++_priorTrackDiagnostics.tracksAccepted;
        _priorTrackDiagnostics.observationsAccepted +=
            static_cast<int>(accepted_observations.size());
        Track materialized_track;
        materialized_track.elements = graph_observations;
        materialized_track.confidence = confidence;
        materialized_track.source = TrackSource::PriorMarker;
        materialized_track.sourceId = priorTrack.markerId;
        _materializedPriorTracks.push_back(std::move(materialized_track));
    }

    Logger::instance()->infof(
        "[SFM] Prior tracks: submitted=%d accepted=%d rejected=%d observations=%d conflicts=%d",
        _priorTrackDiagnostics.tracksSubmitted,
        _priorTrackDiagnostics.tracksAccepted,
        _priorTrackDiagnostics.tracksRejected,
        _priorTrackDiagnostics.observationsAccepted,
        _priorTrackDiagnostics.observationConflicts);
}

void IncrementalSfm::applyPriorTrackDiagnostics(IncrementalSfmResult *result) const
{
    if (!result) return;
    result->priorTracksAccepted = _priorTrackDiagnostics.tracksAccepted;
    result->priorTracksRejected = _priorTrackDiagnostics.tracksRejected;
    result->priorObservationsAccepted = _priorTrackDiagnostics.observationsAccepted;
    result->priorObservationConflicts = _priorTrackDiagnostics.observationConflicts;
    result->priorTrackDiagnostics = _priorTrackDiagnostics.messages;
}

void IncrementalSfm::applyControlNetworkDiagnostics(IncrementalSfmResult *result) const
{
    if (!result) return;
    result->controlNetworkApplied = _controlNetworkApplied;
    result->controlPointConstraintCount = _lastControlPointConstraintCount;
    result->checkPointResidualCount = _controlNetworkResult.checkPointResiduals.size();
    result->controlPointRms = _controlNetworkResult.controlInlierRms;
    double check_sum_squared = 0.0;
    for (const control_points::MarkerResidual &residual : _controlNetworkResult.checkPointResiduals)
    {
        check_sum_squared += residual.total * residual.total;
    }
    result->checkPointRms = _controlNetworkResult.checkPointResiduals.empty()
        ? 0.0
        : std::sqrt(check_sum_squared
                    / static_cast<double>(_controlNetworkResult.checkPointResiduals.size()));
    result->controlScaleBarConstraintCount = _lastControlScaleBarConstraintCount;
    double control_scale_sum_squared = 0.0;
    double check_scale_sum_squared = 0.0;
    int control_scale_count = 0;
    int check_scale_count = 0;
    const auto marker_point = [this](const std::string &marker_id,
                                     std::array<double, 3> *point)
    {
        if (!point) return false;
        for (Point3DId point_id : _reconstruction->allPoint3DIds())
        {
            if (!_reconstruction->hasPoint3D(point_id)) continue;
            const ScenePoint3D &candidate = _reconstruction->point3D(point_id);
            if (candidate.track.source == TrackSource::PriorMarker
                && candidate.track.sourceId == marker_id)
            {
                *point = candidate.xyz;
                return true;
            }
        }
        return false;
    };
    for (const control_points::PriorScaleBar &scale_bar : _pendingPriorScaleBars)
    {
        if (!scale_bar.enabled || !std::isfinite(scale_bar.measuredDistance)
            || scale_bar.measuredDistance <= 0.0)
        {
            continue;
        }
        std::array<double, 3> first{};
        std::array<double, 3> second{};
        if (!marker_point(scale_bar.firstMarkerId, &first)
            || !marker_point(scale_bar.secondMarkerId, &second))
        {
            continue;
        }
        const double dx = first[0] - second[0];
        const double dy = first[1] - second[1];
        const double dz = first[2] - second[2];
        const double residual = std::sqrt(dx * dx + dy * dy + dz * dz)
            - scale_bar.measuredDistance;
        if (scale_bar.role == control_points::ScaleBarRole::Control)
        {
            control_scale_sum_squared += residual * residual;
            ++control_scale_count;
        }
        else
        {
            check_scale_sum_squared += residual * residual;
            ++check_scale_count;
        }
    }
    result->checkScaleBarResidualCount = check_scale_count;
    result->controlScaleBarRms = control_scale_count > 0
        ? std::sqrt(control_scale_sum_squared / static_cast<double>(control_scale_count))
        : 0.0;
    result->checkScaleBarRms = check_scale_count > 0
        ? std::sqrt(check_scale_sum_squared / static_cast<double>(check_scale_count))
        : 0.0;
    result->controlNetworkError = _controlNetworkResult.error;
}

const control_points::PriorTrack *IncrementalSfm::priorTrack(const std::string &markerId) const
{
    const auto it = std::find_if(_pendingPriorTracks.cbegin(),
                                 _pendingPriorTracks.cend(),
                                 [&markerId](const control_points::PriorTrack &track)
                                 {
                                     return track.markerId == markerId;
                                 });
    return it == _pendingPriorTracks.cend() ? nullptr : &*it;
}

bool IncrementalSfm::tryApplyControlNetwork(const std::vector<ImageId> &baImageIds,
                                            std::vector<FramePinholeCamera> *baCameras)
{
    if (_controlNetworkApplied || !baCameras || baImageIds.size() != baCameras->size())
    {
        return _controlNetworkApplied;
    }

    // 只有至少三个可用控制点才能解除自由 SfM 的 Sim(3) 规范自由度。
    // 检查点不参与求解，只在变换应用后报告独立残差。
    control_points::ControlNetworkInput input;
    int usable_control_count = 0;
    for (Point3DId point_id : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(point_id)) continue;
        const ScenePoint3D &point = _reconstruction->point3D(point_id);
        if (point.track.source != TrackSource::PriorMarker) continue;
        const control_points::PriorTrack *prior = priorTrack(point.track.sourceId);
        if (!prior || !prior->hasReference || !prior->referenceUsable) continue;

        control_points::ControlNetworkPoint network_point;
        network_point.markerId = prior->markerId;
        network_point.role = prior->role;
        network_point.estimatedPoint = point.xyz;
        network_point.referencePoint = prior->referencePoint;
        network_point.sigma = prior->referenceSigma;
        input.points.push_back(network_point);
        if (prior->role == control_points::MarkerRole::ControlPoint) ++usable_control_count;
    }
    if (usable_control_count < 3) return false;

    _controlNetworkResult = control_points::solveControlNetwork(input);
    if (!_controlNetworkResult.ok)
    {
        Logger::instance()->warnf("[SFM] Control-network absolute orientation rejected: %s",
                                  _controlNetworkResult.error.c_str());
        return false;
    }

    // 同一个绝对定向必须一致作用于相机中心、camera-to-world 旋转和三维点。
    _controlNetworkTransform = _controlNetworkResult.transform;
    for (ImageId image_id : _reconstruction->registeredImageIds())
    {
        if (!_reconstruction->hasCamera(image_id)) continue;
        FramePinholeCamera &camera = _reconstruction->camera(image_id);
        camera.setPose(_controlNetworkTransform.rotate(camera.cameraToWorldRotation()),
                       _controlNetworkTransform.apply(camera.cameraCenter()));
    }
    for (Point3DId point_id : _reconstruction->allPoint3DIds())
    {
        if (_reconstruction->hasPoint3D(point_id))
        {
            ScenePoint3D &point = _reconstruction->point3D(point_id);
            point.xyz = _controlNetworkTransform.apply(point.xyz);
        }
    }
    for (std::size_t index = 0; index < baImageIds.size(); ++index)
    {
        if (_reconstruction->hasCamera(baImageIds[index]))
        {
            (*baCameras)[index] = _reconstruction->camera(baImageIds[index]);
        }
    }
    _controlNetworkApplied = true;
    Logger::instance()->infof(
        "[SFM] Control-network absolute orientation: controls=%d inliers=%d checks=%d scale=%.9f rms=%.6f",
        usable_control_count,
        _controlNetworkResult.controlInlierCount,
        _controlNetworkResult.checkPointResiduals.size(),
        _controlNetworkTransform.scale,
        _controlNetworkResult.controlInlierRms);
    return true;
}

void IncrementalSfm::tagPriorTrackSource(Track *track) const
{
    if (!track || track->elements.empty()) return;
    std::string source_id;
    for (const TrackElement &element : track->elements)
    {
        const std::string current = _correspondenceGraph.priorTrackId(element.imageId,
                                                                      element.featureIdx);
        if (current.empty() || (!source_id.empty() && current != source_id)) return;
        source_id = current;
    }
    if (!source_id.empty())
    {
        track->source = TrackSource::PriorMarker;
        track->sourceId = source_id;
    }
}

// ============================================================
// 主流程
// ============================================================

IncrementalSfmResult IncrementalSfm::run(SfmProgressCallback progressCb)
{
    IncrementalSfmResult result;
    _inputMultiViewTracks.clear();
    _finalTrackConsolidationAttempted = false;
    _stableIntrinsicReferenceByImageId.clear();
    _isAborted = false;

    const int totalImages = static_cast<int>(_reconstruction->numImages());
    if (totalImages < 2)
    {
        result.summary = "At least 2 images required for SfM";
        return result;
    }

    Logger::instance()->infof(
        "[SFM] Incremental run started: images=%d, useKnownCameraPoses=%s, initMinMatches=%d, "
        "initMinInliers=%d, initMinTriAngle=%.3f, pnpMinInliers=%d, pnpMinTrackLength=%d, pnpMaxReproj=%.3f",
        totalImages,
        _sfmOptions.useKnownCameraPoses ? "true" : "false",
        _sfmOptions.initMinNumMatches,
        _sfmOptions.initMinNumInliers,
        _sfmOptions.initMinTriAngle,
        _sfmOptions.pnpOptions.minNumInliers,
        _sfmOptions.pnpMinTrackLength,
        _sfmOptions.pnpOptions.maxReprojError);

    // ---- 步骤 0：筛选输入多视轨迹并构建对应关系图 ----
    if (!_sfmOptions.useKnownCameraPoses &&
        (_sfmOptions.maxTracksPerImage > 0 || _sfmOptions.maxTracksPerGridCell > 0))
    {
        CorrespondenceTrackThinningOptions thinningOptions;
        thinningOptions.maxTracksPerImage = _sfmOptions.maxTracksPerImage;
        thinningOptions.maxTracksPerGridCell = _sfmOptions.maxTracksPerGridCell;
        thinningOptions.gridColumns = _sfmOptions.trackThinningGridColumns;
        thinningOptions.gridRows = _sfmOptions.trackThinningGridRows;
        CorrespondenceTrackThinningResult thinning =
            thinCorrespondenceTracks(*_reconstruction, &_correspondenceGraph, thinningOptions);
        Logger::instance()->infof(
            "[SFM] Input multiview track thinning: tracks=%d retained=%d pruned=%d "
            "removedMatches=%zu retainedMatches=%zu perImageLimit=%d perCellLimit=%d",
            thinning.inputTrackCount,
            thinning.retainedTrackCount,
            thinning.prunedTrackCount,
            thinning.removedMatchCount,
            thinning.retainedMatchCount,
            _sfmOptions.maxTracksPerImage,
            _sfmOptions.maxTracksPerGridCell);
        _inputMultiViewTracks = std::move(thinning.retainedTracks);
        Logger::instance()->infof(
            "[SFM] Retained %zu complete input tracks for final point-network consolidation",
            _inputMultiViewTracks.size());
    }
    materializePriorTracks();
    applyPriorTrackDiagnostics(&result);
    _correspondenceGraph.buildCorrespondences();
    Logger::instance()->infof("[SFM] Correspondence graph ready: images=%zu, imagePairs=%zu",
                              _correspondenceGraph.numImages(),
                              _correspondenceGraph.numImagePairs());

    if (!reportProgress(0, totalImages, "Building correspondence graph...", progressCb))
        return result;

    // 已知位姿和未知位姿是互斥的两条算法路径，避免部分先验相机被当成固定真值。
    if (_sfmOptions.useKnownCameraPoses)
    {
        KnownPoseReconstructor known_pose_reconstructor(*this);
        IncrementalSfmResult known_pose_result = known_pose_reconstructor.run(progressCb);
        applyPriorTrackDiagnostics(&known_pose_result);
        applyControlNetworkDiagnostics(&known_pose_result);
        return known_pose_result;
    }

    // ---- 步骤 1：选择并初始化初始像对 ----
    if (_sfmOptions.autoSelectInitPair)
    {
        InitialPairInitializer initializer(*this);
        ImageRegistrationEngine registration_engine(*this);
        const auto candidates = initializer.selectCandidates(_sfmOptions.maxInitPairCandidates);
        if (candidates.empty())
        {
            result.summary = "Failed to find a suitable initial image pair";
            return result;
        }

        const bool evaluateMultipleSeeds =
            shouldEvaluateMultipleInitialPairModels(_sfmOptions, totalImages, candidates.size());
        const SfmReconstruction baseReconstruction = *_reconstruction;
        IncrementalSfmResult bestTrialResult;
        double bestScore = -std::numeric_limits<double>::infinity();
        bool anyInitialized = false;

        // 每个初始 pair 都从同一 baseReconstruction 开始，防止前一候选状态泄漏。
        for (size_t ci = 0; ci < candidates.size(); ++ci)
        {
            initializer.resetTrial(baseReconstruction);
            const ImageId initId1 = candidates[ci].first;
            const ImageId initId2 = candidates[ci].second;
            Logger::instance()->infof("[SFM] Trying init pair candidate %zu/%zu: (%u, %u)",
                                      ci + 1,
                                      candidates.size(),
                                      initId1,
                                      initId2);
            if (!initializer.initialize(initId1, initId2))
            {
                Logger::instance()->warnf("[SFM] Candidate (%u, %u) failed: %s",
                                          initId1,
                                          initId2,
                                          _lastErrorMessage.c_str());
                continue;
            }

            anyInitialized = true;
            if (!reportProgress(2, totalImages, "Initialized from pair", progressCb))
            {
                return result;
            }

            IncrementalSfmResult trialResult =
                registration_engine.run(totalImages, progressCb);
            trialResult.selectedInitialImageId1 = initId1;
            trialResult.selectedInitialImageId2 = initId2;
            const double trialScore = scoreInitialPairTrial(trialResult, totalImages);
            Logger::instance()->infof("[SFM] Init pair trial (%u, %u): registered=%d/%d, points=%d, "
                                      "rms=%.4f, score=%.1f",
                                      initId1,
                                      initId2,
                                      trialResult.numRegisteredImages,
                                      totalImages,
                                      trialResult.numPoints3D,
                                      trialResult.meanReprojError,
                                      trialScore);

            if (!evaluateMultipleSeeds)
            {
                return trialResult;
            }

            if (trialScore > bestScore)
            {
                bestScore = trialScore;
                bestTrialResult = trialResult;
            }
            if (trialResult.numRegisteredImages >= totalImages)
            {
                break;
            }
        }

        if (!anyInitialized)
        {
            result.summary = "Failed to initialize from image pair: " + _lastErrorMessage;
            return result;
        }

        if (bestTrialResult.reconstruction)
        {
            _reconstruction = bestTrialResult.reconstruction;
            Logger::instance()->infof("[SFM] Selected best initial-pair trial: registered=%d/%d, points=%d, "
                                      "rms=%.4f, score=%.1f",
                                      bestTrialResult.numRegisteredImages,
                                      totalImages,
                                      bestTrialResult.numPoints3D,
                                      bestTrialResult.meanReprojError,
                                      bestScore);
        }
        return bestTrialResult;
    }

    InitialPairInitializer initializer(*this);
    ImageRegistrationEngine registration_engine(*this);
    if (!initializer.initialize(_sfmOptions.initImageId1, _sfmOptions.initImageId2))
    {
        result.summary = "Failed to initialize from image pair: " + _lastErrorMessage;
        return result;
    }
    if (!reportProgress(2, totalImages, "Initialized from pair", progressCb))
    {
        return result;
    }
    IncrementalSfmResult explicitPairResult =
        registration_engine.run(totalImages, progressCb);
    explicitPairResult.selectedInitialImageId1 = _sfmOptions.initImageId1;
    explicitPairResult.selectedInitialImageId2 = _sfmOptions.initImageId2;
    return explicitPairResult;
}

bool IncrementalSfm::reportProgress(int numRegistered, int numTotal, const std::string &msg, SfmProgressCallback &cb)
{
    if (!cb)
        return true;
    bool continueRun = cb(numRegistered, numTotal, msg);
    if (!continueRun)
        _isAborted = true;
    return continueRun;
}

} // namespace xjw
