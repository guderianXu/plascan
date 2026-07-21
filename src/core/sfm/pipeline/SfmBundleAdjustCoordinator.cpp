#include "SfmBundleAdjustCoordinator.h"
#include "IncrementalSfmDetail.h"
#include "SfmBundleAdjustCoordinator.h"
#include "geometry/OpenCvCameraAdapter.h"
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

void SfmBundleAdjustCoordinator::run(bool localOnly,
                                     const std::vector<ImageId> &anchorIds)
{
    _owner.runBundleAdjust(localOnly, anchorIds);
}

void SfmBundleAdjustCoordinator::iterative()
{
    _owner.iterativeGlobalBA();
}

int SfmBundleAdjustCoordinator::filterNegativeDepthPoints()
{
    return _owner.filterNegativeDepthPoints();
}

void IncrementalSfm::runBundleAdjust(bool localOnly, const std::vector<ImageId> &anchorIds)
{
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

    if (baImageIds.size() < 2)
        return;

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
        if (track.observations.size() >= 2)
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
        return;

    // 构造本次 BA 选项：
    //   - 全局 BA 固定 index=0 的相机（gauge 固定，消除坐标系漂移）
    //   - 局部 BA 不固定（局部块坐标由全局约束）
    BAOptions baOpt = _sfmOptions.baOptions;
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
    if (!localOnly && !baImageIds.empty())
    {
        baOpt.fixedCameraIndices = {0}; // gauge: 第一个相机位姿保持不变
    }

    // 执行 BA
    const BAResult baResult = BundleAdjust::optimizePoints(baCameras, baTracks, baOpt);
    if (baOpt.logIterationProgress)
    {
        const BAProblemStats stats = BundleAdjust::summarizeProblem(baCameras, baTracks);
        Logger::instance()->infof(
            "[BA] problem cameras=%d tracks=%d observations=%d threads=%d backend=%s reason=%s",
            stats.cameraCount,
            stats.trackCount,
            stats.observationCount,
            baOpt.numThreads,
            BundleAdjust::backendName(baResult.usedBackend),
            baResult.backendSelectionReason.c_str());
    }

    bool applyBaResult = true;
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
        _lastControlPointConstraintCount = control_constraint_count;
        _lastControlScaleBarConstraintCount = control_scale_bar_count;
        _lastGlobalBARmsBefore = baResult.meanRmsBefore;
        _lastGlobalBARmsAfter = baResult.meanRmsAfter;
        _lastGlobalBATracksTotal = baResult.totalTracks;
        _lastGlobalBATracksOptimized = baResult.optimizedTracks;
        _lastGlobalBATracksFiltered = (int)baTracks.size() - baResult.optimizedTracks;
        if (_lastGlobalBATracksFiltered < 0)
            _lastGlobalBATracksFiltered = deletedPts;
        _lastGlobalBARefinedIntrinsicCount = baResult.refinedIntrinsicCount;
        _lastGlobalBASharedFocalScale = baResult.refinedSharedFocalScale;
    }
}

// ============================================================
// 内部：迭代全局 BA 精化（参考 COLMAP IterativeGlobalRefinement）
// ============================================================

void IncrementalSfm::iterativeGlobalBA()
{
    const int maxRounds = std::max(1, _sfmOptions.iterativeBARounds);
    size_t prevNumPoints = _reconstruction->numPoints3D();

    for (int round = 0; round < maxRounds; ++round)
    {
        Logger::instance()->infof("[SFM] IterativeGlobalBA round %d/%d: numPts=%zu", round + 1, maxRounds,
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

        Logger::instance()->infof("[SFM]   After round %d: numPts=%zu, changeRate=%.4f", round + 1, curNumPoints,
                      changeRate);

        if (round >= 1 && changeRate < 0.01)
        {
            Logger::instance()->info("[SFM]   Converged (changeRate < 1%)");
            break;
        }
        prevNumPoints = curNumPoints;
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
            double cameraPoint[3] = {0.0, 0.0, 0.0};
            cam.worldToCamera(world, cameraPoint);

            if (cameraPoint[2] < 0.0)
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
