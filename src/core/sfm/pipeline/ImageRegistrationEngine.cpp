#include "ImageRegistrationEngine.h"
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

ImageRegistrationEngine::ImageRegistrationEngine(IncrementalSfm &owner)
    : _owner(owner)
{
}

IncrementalSfmResult ImageRegistrationEngine::run(int totalImages,
                                                  SfmProgressCallback progressCb)
{
    return _owner.runRegistrationFromCurrentInitialization(totalImages, std::move(progressCb));
}

IncrementalSfmResult IncrementalSfm::runRegistrationFromCurrentInitialization(
    int totalImages,
    SfmProgressCallback progressCb)
{
    IncrementalSfmResult result;
    applyPriorTrackDiagnostics(&result);
    rebuildVisibilityCache();

    // ---- 步骤 2：逐帧注册循环 ----
    int regCount = static_cast<int>(_reconstruction->numRegisteredImages());
    const int registrationTarget = _sfmOptions.maxRegisteredImages > 0
        ? std::clamp(_sfmOptions.maxRegisteredImages, 2, totalImages)
        : totalImages;
    int iterSinceLastLocalBA = 0;
    int iterSinceLastGlobalBA = 0;

    while (regCount < registrationTarget && !_isAborted)
    {
        ImageId nextId = selectNextImage();
        if (nextId == kInvalidImageId)
        {
            if (!_deferredFailedImages.empty())
            {
                Logger::instance()->infof(
                    "[SFM] Registration pass exhausted %zu deferred candidate(s); retrying with current model",
                    _deferredFailedImages.size());
                _deferredFailedImages.clear();
                continue;
            }

            Logger::instance()->warnf(
                "[SFM] Registration stopped: no selectable next image (registered=%d/%d, points=%zu, "
                "deferred=%zu, permanentlyFailed=%zu, minPnPInliers=%d)",
                regCount,
                totalImages,
                _reconstruction->numPoints3D(),
                _deferredFailedImages.size(),
                _permanentlyFailedImages.size(),
                _sfmOptions.pnpOptions.minNumInliers);
            break; // 无更多可注册图像
        }

        if (!registerImage(nextId))
        {
            int &failCount = _registerFailCount[nextId];
            ++failCount;
            Logger::instance()->warnf("[SFM] Image %u registration failed: %s",
                                      nextId,
                                      _lastErrorMessage.empty() ? "unknown error" : _lastErrorMessage.c_str());
            // 区分暂时失败（无足够 3D-2D 点）和永久失败（相机文件缺失等）
            // 超过 5 次仍无法注册视为永久失败，不再尝试
            const int kMaxRetries = 5;
            if (failCount >= kMaxRetries)
            {
                _permanentlyFailedImages.insert(nextId);
                _deferredFailedImages.erase(nextId);
                Logger::instance()->warnf(
                    "[SFM] Image %u permanently skipped after %d failed registration attempts",
                    nextId, failCount);
            }
            else
            {
                // 这类失败通常是当前 3D 模型还不足或外点比例过高。
                // 先把该图像放到本轮末尾，允许其它候选先注册并扩展模型；
                // 若没有其它候选可用，再进入下一轮重试。
                _deferredFailedImages.insert(nextId);
                Logger::instance()->infof(
                    "[SFM] Image %u registration attempt %d/%d failed, deferred until the model grows",
                    nextId, failCount, kMaxRetries);
            }
            continue;
        }
        // 注册成功，清除失败计数
        _registerFailCount.erase(nextId);
        // 新图像注册后可见三维点集合发生变化，之前暂缓的候选需要重新参与排序。
        _deferredFailedImages.clear();

        regCount = static_cast<int>(_reconstruction->numRegisteredImages());

        // 三角化新注册图像
        std::vector<Point3DId> previousPointIds = _reconstruction->image(nextId).point3DIds;
        Triangulator triangulator(*_reconstruction, _correspondenceGraph);
        triangulator.triangulateImage(nextId, _sfmOptions.triangulatorOptions);
        updateVisibilityCacheForImage(nextId, previousPointIds);

        ++iterSinceLastLocalBA;
        ++iterSinceLastGlobalBA;

        // 局部 BA
        if (iterSinceLastLocalBA >= _sfmOptions.localBAInterval)
        {
            SfmBundleAdjustCoordinator(*this).run(true, {nextId});
            triangulator.filterPoints(_sfmOptions.filterMaxReprojError, _sfmOptions.filterMinTriAngle);
            invalidateVisibilityCache();
            iterSinceLastLocalBA = 0;
        }

        // 全局 BA（使用迭代精化策略）
        if (iterSinceLastGlobalBA >= _sfmOptions.globalBAInterval)
        {
            if (SfmBundleAdjustCoordinator::shouldRunPeriodicGlobalBa(
                    regCount,
                    registrationTarget,
                    iterSinceLastGlobalBA,
                    _sfmOptions.globalBAInterval))
            {
                SfmBundleAdjustCoordinator(*this).iterative(false);
                invalidateVisibilityCache();
            }
            else
            {
                Logger::instance()->infof(
                    "[SFM] Deferred periodic global BA to final refinement: "
                    "registered=%d target=%d remaining=%d interval=%d",
                    regCount,
                    registrationTarget,
                    std::max(0, registrationTarget - regCount),
                    _sfmOptions.globalBAInterval);
            }
            iterSinceLastGlobalBA = 0;
        }

        std::ostringstream msg;
        msg << "Registered " << regCount << "/" << totalImages << " images";
        if (registrationTarget < totalImages)
        {
            msg << " (focal probe target " << registrationTarget << ")";
        }
        msg << ", " << _reconstruction->numPoints3D() << " 3D points";

        if (!reportProgress(regCount, registrationTarget, msg.str(), progressCb))
            return result;
    }

    // ---- 步骤 3：最终全局 BA 和清理（迭代精化） ----
    if (!_isAborted && _reconstruction->numRegisteredImages() >= 2)
    {
        SfmBundleAdjustCoordinator(*this).iterative(true);
        retryUnregisteredImagesAfterFinalBA(totalImages);

        // 过滤轨迹长度过短的不可靠三维点
        Triangulator finalTri(*_reconstruction, _correspondenceGraph);
        if (_sfmOptions.filterMinTrackLen > 1)
        {
            int nShort = finalTri.filterShortTracks(_sfmOptions.filterMinTrackLen);
            Logger::instance()->infof("[SFM] Filtered %d short-track points (minTrackLen=%d)", nShort,
                                      _sfmOptions.filterMinTrackLen);
        }
    }

    // ---- 步骤 4：用最新相机位姿重算重投影误差（确保统计精确） ----
    if (!_isAborted && _reconstruction->numRegisteredImages() >= 2)
    {
        Triangulator finalReprojTri(*_reconstruction, _correspondenceGraph);
        finalReprojTri.recomputeReprojErrors();
    }

    // ---- 步骤 5：组装结果 ----
    result.success = _reconstruction->numRegisteredImages() >= 2;
    result.numRegisteredImages = static_cast<int>(_reconstruction->numRegisteredImages());
    result.numPoints3D = static_cast<int>(_reconstruction->numPoints3D());
    result.meanReprojError = _reconstruction->meanReprojError();
    result.reconstruction = _reconstruction;
    result.summary = _reconstruction->summary();
    result.hierarchicalBAPlannedBlocks = _lastHierarchicalBAPlannedBlocks;
    result.hierarchicalBAAppliedBlocks = _lastHierarchicalBAAppliedBlocks;
    result.hierarchicalBAUpdatedCameras = _lastHierarchicalBAUpdatedCameras;
    result.hierarchicalBATotalSeconds = _lastHierarchicalBATotalSeconds;

    if (!_permanentlyFailedImages.empty())
    {
        Logger::instance()->warnf("[SFM] %zu image(s) permanently failed to register",
                                  _permanentlyFailedImages.size());
    }

    // 填充最后一轮全局 BA 统计
    result.baRmsBefore = _lastGlobalBARmsBefore;
    result.baRmsAfter = _lastGlobalBARmsAfter;
    result.baTracksTotal = _lastGlobalBATracksTotal;
    result.baTracksOptimized = _lastGlobalBATracksOptimized;
    result.baTracksFiltered = _lastGlobalBATracksFiltered;
    result.baRefinedIntrinsicCount = _lastGlobalBARefinedIntrinsicCount;
    result.baSharedFocalScale = _lastGlobalBASharedFocalScale;
    result.baSharedFocalAspectScale = _lastGlobalBASharedFocalAspectScale;
    result.baSharedPrincipalOffsetX = _lastGlobalBASharedPrincipalOffsetX;
    result.baSharedPrincipalOffsetY = _lastGlobalBASharedPrincipalOffsetY;
    result.baSharedRadialK1 = _lastGlobalBASharedRadialK1;
    result.baSharedRadialK2 = _lastGlobalBASharedRadialK2;
    result.baSharedRadialK3 = _lastGlobalBASharedRadialK3;
    result.baSharedTangentialP1 = _lastGlobalBASharedTangentialP1;
    result.baSharedTangentialP2 = _lastGlobalBASharedTangentialP2;
    result.baRequestedBackend = _lastGlobalBARequestedBackend;
    result.baUsedBackend = _lastGlobalBAUsedBackend;
    result.baSolveStatus = _lastGlobalBASolveStatus;
    result.baSolutionUsable = _lastGlobalBASolutionUsable;
    result.baResultApplied = _lastGlobalBAResultApplied;
    result.baBackendFallback = _lastGlobalBABackendFallback;
    result.baObservationCount = _lastGlobalBAObservationCount;
    result.baTotalSeconds = _lastGlobalBATotalSeconds;
    result.baBackendMessage = _lastGlobalBABackendMessage;
    applyControlNetworkDiagnostics(&result);

    return result;
}

int IncrementalSfm::retryUnregisteredImagesAfterFinalBA(int totalImages)
{
    if (!_sfmOptions.retryUnregisteredAfterFinalBA || !_reconstruction ||
        _isAborted || _reconstruction->numRegisteredImages() < 3 ||
        static_cast<int>(_reconstruction->numRegisteredImages()) >= totalImages)
    {
        return 0;
    }

    int registeredByRetry = 0;
    const int maxPasses = std::clamp(_sfmOptions.maxFinalRegistrationRetryPasses, 1, 3);
    for (int pass = 0; pass < maxPasses && !_isAborted; ++pass)
    {
        std::vector<ImageId> candidates;
        const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
        for (ImageId imageId = 0; imageId < imageCount; ++imageId)
        {
            if (!_reconstruction->isRegistered(imageId) && hasRegisteredSequenceNeighbor(imageId))
            {
                candidates.push_back(imageId);
            }
        }
        if (candidates.empty())
        {
            break;
        }

        _registerFailCount.clear();
        _deferredFailedImages.clear();
        _permanentlyFailedImages.clear();
        rebuildVisibilityCache();
        std::stable_sort(candidates.begin(), candidates.end(), [&](ImageId lhs, ImageId rhs)
        {
            return _visibilityCache[lhs] > _visibilityCache[rhs];
        });

        int registeredThisPass = 0;
        for (ImageId imageId : candidates)
        {
            if (!registerImage(imageId))
            {
                Logger::instance()->warnf(
                    "[SFM] Final-BA retry rejected image %u: %s",
                    imageId,
                    _lastErrorMessage.empty() ? "unknown error" : _lastErrorMessage.c_str());
                continue;
            }

            std::vector<Point3DId> previousPointIds = _reconstruction->image(imageId).point3DIds;
            Triangulator triangulator(*_reconstruction, _correspondenceGraph);
            triangulator.triangulateImage(imageId, _sfmOptions.triangulatorOptions);
            updateVisibilityCacheForImage(imageId, previousPointIds);
            ++registeredThisPass;
            ++registeredByRetry;
            Logger::instance()->infof(
                "[SFM] Final-BA retry registered image %u (%zu/%d)",
                imageId,
                _reconstruction->numRegisteredImages(),
                totalImages);
        }

        if (registeredThisPass == 0)
        {
            break;
        }

        SfmBundleAdjustCoordinator(*this).iterative(true);
        invalidateVisibilityCache();
        if (static_cast<int>(_reconstruction->numRegisteredImages()) >= totalImages)
        {
            break;
        }
    }
    return registeredByRetry;
}

void IncrementalSfm::resetForInitialPairTrial(const SfmReconstruction &baseReconstruction)
{
    // 多初始像对评估时，每个 seed 必须重置到同一份输入影像/匹配，
    // 否则前一个 seed 产生的相机、三维点和 point3DIds 会污染后续试跑。
    _reconstruction = std::make_shared<SfmReconstruction>(baseReconstruction);
    _lastErrorMessage.clear();
    _isAborted = false;
    _lastGlobalBARmsBefore = 0.0;
    _lastGlobalBARmsAfter = 0.0;
    _lastGlobalBATracksTotal = 0;
    _lastGlobalBATracksOptimized = 0;
    _lastGlobalBATracksFiltered = 0;
    _lastGlobalBARefinedIntrinsicCount = 0;
    _lastGlobalBASharedFocalScale = 1.0;
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
    _lastGlobalBABackendMessage.clear();
    _lastHierarchicalBAImageCount = 0;
    _lastHierarchicalBAPlannedBlocks = 0;
    _lastHierarchicalBAAppliedBlocks = 0;
    _lastHierarchicalBAUpdatedCameras = 0;
    _lastHierarchicalBATotalSeconds = 0.0;
    _finalTrackConsolidationAttempted = false;
    _controlNetworkApplied = false;
    _controlNetworkResult = {};
    _controlNetworkTransform = {};
    _lastControlPointConstraintCount = 0;
    _lastControlScaleBarConstraintCount = 0;
    _visibilityCache.clear();
    _visibilityCacheDirty = true;
    _registerFailCount.clear();
    _deferredFailedImages.clear();
    _permanentlyFailedImages.clear();
}


ImageId IncrementalSfm::selectNextImage() const
{
    // 使用增量维护的可见性缓存
    if (_visibilityCacheDirty)
    {
        const_cast<IncrementalSfm *>(this)->rebuildVisibilityCache();
    }

    ImageId bestId = kInvalidImageId;
    size_t bestVisible = 0;

    for (const auto &[id, count] : _visibilityCache)
    {
        if (_reconstruction->isRegistered(id))
            continue;
        if (_deferredFailedImages.count(id))
            continue;
        if (_permanentlyFailedImages.count(id))
            continue;
        if (_sfmOptions.enforceSequencePoseConsistency &&
            _reconstruction->numRegisteredImages() >= 3 &&
            !hasRegisteredSequenceNeighbor(id))
        {
            continue;
        }
        if (count > bestVisible)
        {
            bestVisible = count;
            bestId = id;
        }
    }

    // 至少需要若干可见三维点才值得注册
    if (bestVisible < static_cast<size_t>(_sfmOptions.pnpOptions.minNumInliers))
    {
        Logger::instance()->warnf(
            "[SFM] selectNextImage rejected all candidates: bestId=%u, bestVisible=%zu, minPnPInliers=%d, "
            "registered=%zu/%zu, points=%zu, deferred=%zu, permanentlyFailed=%zu",
            bestId,
            bestVisible,
            _sfmOptions.pnpOptions.minNumInliers,
            _reconstruction->numRegisteredImages(),
            _reconstruction->numImages(),
            _reconstruction->numPoints3D(),
            _deferredFailedImages.size(),
            _permanentlyFailedImages.size());
        return kInvalidImageId;
    }

    Logger::instance()->infof("[SFM] selectNextImage: id=%u, visible3D=%zu, minPnPInliers=%d, deferred=%zu",
                              bestId,
                              bestVisible,
                              _sfmOptions.pnpOptions.minNumInliers,
                              _deferredFailedImages.size());
    return bestId;
}

// ============================================================
// 内部：注册图像（PnP）
// ============================================================

bool IncrementalSfm::registerImage(ImageId imageId)
{
    // 加载相机内参
    Camera cam;
    if (!getCamera(imageId, cam))
    {
        _lastErrorMessage = "getCamera(" + std::to_string(imageId) + ") failed";
        return false;
    }

    const ImageData &img = _reconstruction->image(imageId);

    // 收集 3D-2D 对应关系
    std::vector<std::array<double, 3>> worldPts;
    std::vector<std::array<double, 2>> imagePts;

    auto neighbors = _correspondenceGraph.connectedImages(imageId);
    const int minTrackLengthForPnp =
        effectivePnpMinTrackLength(_sfmOptions, _reconstruction->numRegisteredImages());
    std::unordered_set<Point3DId> addedPoints;
    std::unordered_set<FeatureIdx> addedFeatures;

    for (ImageId nid : neighbors)
    {
        if (!_reconstruction->isRegistered(nid))
        {
            continue;
        }
        const auto &matches = _correspondenceGraph.matchesBetween(imageId, nid);
        const ImageData &nimg = _reconstruction->image(nid);

        for (const auto &m : matches)
        {
            const FeatureIdx myFeat = (imageId < nid) ? m.idx1 : m.idx2;
            const FeatureIdx nFeat = (imageId < nid) ? m.idx2 : m.idx1;
            if (nFeat >= nimg.point3DIds.size())
            {
                continue;
            }
            const Point3DId p3dId = nimg.point3DIds[nFeat];
            if (!pointUsableForPnp(*_reconstruction, p3dId, minTrackLengthForPnp) ||
                addedPoints.count(p3dId) != 0 || addedFeatures.count(myFeat) != 0 ||
                myFeat >= img.keypoints.size())
            {
                continue;
            }

            addedPoints.insert(p3dId);
            addedFeatures.insert(myFeat);
            worldPts.push_back(_reconstruction->point3D(p3dId).xyz);
            imagePts.push_back({static_cast<double>(img.keypoints[myFeat].x),
                                static_cast<double>(img.keypoints[myFeat].y)});
        }
    }

    if (static_cast<int>(worldPts.size()) < _sfmOptions.pnpOptions.minNumInliers)
    {
        std::ostringstream oss;
        oss << "not enough 2D-3D observations for PnP: observations=" << worldPts.size()
            << ", minPnPInliers=" << _sfmOptions.pnpOptions.minNumInliers
            << ", minTrackLength=" << minTrackLengthForPnp
            << ", connectedNeighbors=" << neighbors.size();
        _lastErrorMessage = oss.str();
        return false;
    }

    // 先执行不带运动模型的标准 PnP。照片序列外推对非等速航带只是弱先验，
    // 若一开始就用它预过滤观测，会把原本可由全局 PnP 注册的相机排除。
    PnpOptions pnpOptions = _sfmOptions.pnpOptions;
    pnpOptions.ransacSeed = opencv_compat::stableRansacSeed(
        imageId,
        static_cast<std::uint32_t>(_reconstruction->numRegisteredImages()),
        static_cast<std::uint32_t>(worldPts.size()));
    PnpResult pnpResult = PnpSolver::solveWithCamera(worldPts, imagePts, cam, pnpOptions);
    bool usedSequenceRecovery = false;

    // 标准 PnP 失败后，才使用相邻序号相机插值/外推的初值进行一次确定性恢复。
    // 恢复仍必须通过真实 3D-2D 重投影内点门槛，不会直接接受外推位姿。
    Camera sequenceGuessCamera = cam;
    if (!pnpResult.success && makeSequenceInitialPoseGuess(imageId, &sequenceGuessCamera))
    {
        PnpOptions recoveryOptions = pnpOptions;
        recoveryOptions.useInitialPose = true;
        recoveryOptions.initialCameraToWorldRotation = sequenceGuessCamera.cameraToWorldRotation();
        recoveryOptions.initialCameraCenter = sequenceGuessCamera.cameraCenter();

        ImageId previousImageId = kInvalidImageId;
        ImageId nextImageId = kInvalidImageId;
        int previousSteps = 0;
        int nextSteps = 0;
        const bool hasPrevious = findRegisteredSequenceNeighbor(imageId,
                                                                 -1,
                                                                 &previousImageId,
                                                                 &previousSteps);
        const bool hasNext = findRegisteredSequenceNeighbor(imageId,
                                                             1,
                                                             &nextImageId,
                                                             &nextSteps);
        const bool hasDirectPrevious = hasPrevious && previousSteps == 1;
        const bool hasDirectNext = hasNext && nextSteps == 1;
        const bool isDirectlyBracketed = hasDirectPrevious && hasDirectNext;
        const bool useOneSidedRecovery = _sfmOptions.allowOneSidedSequencePoseRecovery &&
            (hasDirectPrevious != hasDirectNext);
        const int sequenceMinInliers = isDirectlyBracketed
            ? _sfmOptions.bracketedSequencePnpMinInliers
            : _sfmOptions.oneSidedSequencePnpMinInliers;
        const double sequenceMinInlierRatio = isDirectlyBracketed
            ? _sfmOptions.bracketedSequencePnpMinInlierRatio
            : _sfmOptions.oneSidedSequencePnpMinInlierRatio;
        if (isDirectlyBracketed || useOneSidedRecovery)
        {
            recoveryOptions.useInitialPosePrefilter = true;
            recoveryOptions.initialPosePrefilterMaxReprojError = std::max(
                recoveryOptions.initialPosePrefilterMaxReprojError,
                isDirectlyBracketed
                    ? recoveryOptions.maxReprojError * 12.0
                    : _sfmOptions.oneSidedSequencePosePrefilterMaxReprojError);
            recoveryOptions.initialPosePrefilterMinCandidates = std::max(
                recoveryOptions.minNumInliers,
                sequenceMinInliers);
        }
        if (_sfmOptions.allowBracketedSequencePnpRelaxation &&
            (isDirectlyBracketed || useOneSidedRecovery))
        {
            // 两侧紧邻位姿为当前相机提供了强序列约束。此时允许大量候选 2D-3D
            // 对应中的低比例真实内点通过，但仍要求足够绝对内点并在随后检查相机距离。
            recoveryOptions.allowRelaxedInlierRatio = true;
            recoveryOptions.relaxedMinInlierRatio =
                std::clamp(sequenceMinInlierRatio,
                           0.0,
                           recoveryOptions.minInlierRatio);
            recoveryOptions.relaxedMinNumInliers =
                std::max(recoveryOptions.minNumInliers,
                         sequenceMinInliers);
        }
        pnpResult = PnpSolver::solveWithCamera(
            worldPts, imagePts, cam, recoveryOptions);
        usedSequenceRecovery = true;
    }
    if (!pnpResult.success)
    {
        std::ostringstream oss;
        oss << "PnP failed: observations=" << worldPts.size()
            << ", inliers=" << pnpResult.numInliers
            << ", inlierRatio=" << pnpResult.inlierRatio
            << ", minPnPInliers=" << _sfmOptions.pnpOptions.minNumInliers
            << ", minTrackLength=" << minTrackLengthForPnp
            << ", minInlierRatio=" << _sfmOptions.pnpOptions.minInlierRatio
            << ", sequenceRecovery=" << (usedSequenceRecovery ? "true" : "false")
            << ", posePrefilter=" << (pnpResult.usedInitialPosePrefilter ? "true" : "false")
            << ", prefilterCandidates=" << pnpResult.prefilterCandidateCount
            << ", maxReprojError=" << _sfmOptions.pnpOptions.maxReprojError;
        _lastErrorMessage = oss.str();
        return false;
    }

    Logger::instance()->infof("[SFM] Image %u PnP success: observations=%zu, inliers=%d, inlierRatio=%.3f, "
                              "minTrackLength=%d, sequenceRecovery=%s, posePrefilter=%s, "
                              "prefilterCandidates=%d",
                              imageId,
                              worldPts.size(),
                              pnpResult.numInliers,
                              pnpResult.inlierRatio,
                              minTrackLengthForPnp,
                              usedSequenceRecovery ? "true" : "false",
                              pnpResult.usedInitialPosePrefilter ? "true" : "false",
                              pnpResult.prefilterCandidateCount);

    // 应用 PnP 结果更新相机外参
    cam.setPose(pnpResult.R, pnpResult.C);
    std::string sequenceConsistencyReason;
    if (!validateSequencePoseConsistency(imageId, cam, &sequenceConsistencyReason))
    {
        _lastErrorMessage = "sequence distance check failed: " + sequenceConsistencyReason;
        Logger::instance()->warnf("[SFM] Image %u sequence distance rejected: %s",
                                  imageId,
                                  sequenceConsistencyReason.c_str());
        return false;
    }

    // 注册到重建
    _reconstruction->registerImage(imageId, cam);

    return true;
}

bool IncrementalSfm::findRegisteredSequenceNeighbor(ImageId imageId,
                                                    int direction,
                                                    ImageId *neighborOut,
                                                    int *stepsOut) const
{
    if (!neighborOut || !stepsOut || !_reconstruction || direction == 0)
    {
        return false;
    }

    *neighborOut = kInvalidImageId;
    *stepsOut = 0;

    const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
    if (imageCount < 2 || imageId >= imageCount)
    {
        return false;
    }

    auto hasRegisteredCamera = [&](ImageId candidate)
    {
        return candidate < imageCount &&
               _reconstruction->isRegistered(candidate) &&
               _reconstruction->hasCamera(candidate);
    };

    const int signedImageCount = static_cast<int>(imageCount);
    const int start = static_cast<int>(imageId);
    const int stepDirection = direction < 0 ? -1 : 1;
    for (ImageId step = 1; step < imageCount; ++step)
    {
        int candidateIndex = start + stepDirection * static_cast<int>(step);
        if (candidateIndex < 0 || candidateIndex >= signedImageCount)
        {
            if (!_sfmOptions.sequenceLoopClosure)
            {
                break;
            }
            candidateIndex = (candidateIndex % signedImageCount + signedImageCount) % signedImageCount;
        }

        const ImageId candidate = static_cast<ImageId>(candidateIndex);
        if (hasRegisteredCamera(candidate))
        {
            *neighborOut = candidate;
            *stepsOut = static_cast<int>(step);
            return true;
        }
    }

    return false;
}

bool IncrementalSfm::hasRegisteredSequenceNeighbor(ImageId imageId) const
{
    if (!_sfmOptions.enforceSequencePoseConsistency || !_reconstruction)
    {
        return true;
    }

    const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
    if (imageCount == 0)
    {
        return false;
    }

    std::vector<ImageId> neighbors;
    if (imageId > 0)
    {
        neighbors.push_back(imageId - 1);
    }
    if (imageId + 1 < imageCount)
    {
        neighbors.push_back(imageId + 1);
    }
    if (_sfmOptions.sequenceLoopClosure && imageCount > 2)
    {
        if (imageId == 0)
        {
            neighbors.push_back(imageCount - 1);
        }
        else if (imageId + 1 == imageCount)
        {
            neighbors.push_back(0);
        }
    }

    for (ImageId neighborId : neighbors)
    {
        if (_reconstruction->isRegistered(neighborId) && _reconstruction->hasCamera(neighborId))
        {
            return true;
        }
    }
    return false;
}

double IncrementalSfm::registeredSequenceAdjacentDistanceMedian(ImageId excludedImageId) const
{
    if (!_reconstruction)
    {
        return 0.0;
    }

    const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
    if (imageCount < 2)
    {
        return 0.0;
    }

    std::vector<double> distances;
    distances.reserve(static_cast<std::size_t>(imageCount));
    auto addDistance = [&](ImageId a, ImageId b)
    {
        if (a == excludedImageId || b == excludedImageId ||
            !_reconstruction->isRegistered(a) || !_reconstruction->isRegistered(b) ||
            !_reconstruction->hasCamera(a) || !_reconstruction->hasCamera(b))
        {
            return;
        }
        const double distance = distance3d(_reconstruction->camera(a).cameraCenter(),
                                           _reconstruction->camera(b).cameraCenter());
        if (std::isfinite(distance) && distance > 1e-9)
        {
            distances.push_back(distance);
        }
    };

    for (ImageId imageId = 0; imageId + 1 < imageCount; ++imageId)
    {
        addDistance(imageId, imageId + 1);
    }
    if (_sfmOptions.sequenceLoopClosure && imageCount > 2)
    {
        addDistance(imageCount - 1, 0);
    }

    if (distances.size() < static_cast<std::size_t>(std::max(1, _sfmOptions.sequencePoseConsistencyMinSamples)))
    {
        return 0.0;
    }
    return percentile(distances, 0.5);
}

bool IncrementalSfm::validateSequencePoseConsistency(ImageId imageId,
                                                     const Camera &candidateCamera,
                                                     std::string *reason) const
{
    if (!_sfmOptions.enforceSequencePoseConsistency || !_reconstruction ||
        _reconstruction->numRegisteredImages() < 3)
    {
        return true;
    }

    const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
    std::vector<ImageId> neighbors;
    if (imageId > 0)
    {
        neighbors.push_back(imageId - 1);
    }
    if (imageId + 1 < imageCount)
    {
        neighbors.push_back(imageId + 1);
    }
    if (_sfmOptions.sequenceLoopClosure && imageCount > 2)
    {
        if (imageId == 0)
        {
            neighbors.push_back(imageCount - 1);
        }
        else if (imageId + 1 == imageCount)
        {
            neighbors.push_back(0);
        }
    }

    std::vector<ImageId> registeredNeighbors;
    for (ImageId neighborId : neighbors)
    {
        if (_reconstruction->isRegistered(neighborId) && _reconstruction->hasCamera(neighborId))
        {
            registeredNeighbors.push_back(neighborId);
        }
    }

    if (registeredNeighbors.empty())
    {
        if (reason)
        {
            *reason = "no registered sequence neighbor";
        }
        return false;
    }

    const double medianDistance = registeredSequenceAdjacentDistanceMedian(imageId);
    if (!(medianDistance > 0.0) || !std::isfinite(medianDistance))
    {
        return true;
    }

    const double minDistance = medianDistance * std::max(0.0, _sfmOptions.sequenceAdjacentDistanceMinFactor);
    const double maxDistance = medianDistance * std::max(_sfmOptions.sequenceAdjacentDistanceMinFactor,
                                                         _sfmOptions.sequenceAdjacentDistanceMaxFactor);
    const auto candidateCenter = candidateCamera.cameraCenter();
    for (ImageId neighborId : registeredNeighbors)
    {
        const double distance = distance3d(candidateCenter, _reconstruction->camera(neighborId).cameraCenter());
        if (!std::isfinite(distance) || distance < minDistance || distance > maxDistance)
        {
            if (reason)
            {
                std::ostringstream oss;
                oss << "image=" << imageId
                    << ", neighbor=" << neighborId
                    << ", distance=" << distance
                    << ", median=" << medianDistance
                    << ", allowed=[" << minDistance << "," << maxDistance << "]";
                *reason = oss.str();
            }
            return false;
        }
    }
    return true;
}

bool IncrementalSfm::makeSequenceInitialPoseGuess(ImageId imageId, Camera *guessCamera) const
{
    if (!guessCamera || !_sfmOptions.useSequencePoseRecovery || !_reconstruction ||
        _reconstruction->numRegisteredImages() < 2)
    {
        return false;
    }

    const ImageId imageCount = static_cast<ImageId>(_reconstruction->numImages());
    if (imageCount < 3)
    {
        return false;
    }

    auto hasRegisteredCamera = [&](ImageId id)
    {
        return id < imageCount && _reconstruction->isRegistered(id) && _reconstruction->hasCamera(id);
    };
    auto centerAdd = [](const std::array<double, 3> &a, const std::array<double, 3> &b)
    {
        return std::array<double, 3>{{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
    };
    auto centerSub = [](const std::array<double, 3> &a, const std::array<double, 3> &b)
    {
        return std::array<double, 3>{{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
    };
    auto centerScale = [](const std::array<double, 3> &a, double scale)
    {
        return std::array<double, 3>{{a[0] * scale, a[1] * scale, a[2] * scale}};
    };

    ImageId prev = kInvalidImageId;
    ImageId next = kInvalidImageId;
    int prevSteps = 0;
    int nextSteps = 0;
    const bool prevRegistered = findRegisteredSequenceNeighbor(imageId, -1, &prev, &prevSteps);
    const bool nextRegistered = findRegisteredSequenceNeighbor(imageId, 1, &next, &nextSteps);
    const bool hasDirectPrevious = prevRegistered && prevSteps == 1;
    const bool hasDirectNext = nextRegistered && nextSteps == 1;

    std::array<double, 3> center{};
    std::array<double, 9> rotation{};
    if (hasDirectPrevious && hasDirectNext)
    {
        const double denom = static_cast<double>(std::max(1, prevSteps + nextSteps));
        const double t = static_cast<double>(prevSteps) / denom;
        const auto prevCenter = _reconstruction->camera(prev).cameraCenter();
        const auto nextCenter = _reconstruction->camera(next).cameraCenter();
        // 连续缺口按序列步长插值中心，并对 camera-to-world 旋转执行 SLERP；
        // 直接复制一侧旋转会给环拍相邻帧引入一个完整帧间角度的 PnP 初值偏差。
        center = centerAdd(centerScale(prevCenter, 1.0 - t), centerScale(nextCenter, t));
        rotation = interpolateCameraRotation(
            _reconstruction->camera(prev).cameraToWorldRotation(),
            _reconstruction->camera(next).cameraToWorldRotation(),
            t);
    }
    else if (hasDirectPrevious)
    {
        ImageId prev2 = kInvalidImageId;
        int prev2Steps = 0;
        if (!findRegisteredSequenceNeighbor(prev, -1, &prev2, &prev2Steps) || !hasRegisteredCamera(prev2))
        {
            return false;
        }
        const auto prevCenter = _reconstruction->camera(prev).cameraCenter();
        const auto delta = centerSub(prevCenter, _reconstruction->camera(prev2).cameraCenter());
        const double scale = static_cast<double>(std::max(1, prevSteps)) /
                             static_cast<double>(std::max(1, prev2Steps));
        center = centerAdd(prevCenter, centerScale(delta, scale));
        rotation = interpolateCameraRotation(
            _reconstruction->camera(prev2).cameraToWorldRotation(),
            _reconstruction->camera(prev).cameraToWorldRotation(),
            1.0 + scale);
    }
    else if (hasDirectNext)
    {
        ImageId next2 = kInvalidImageId;
        int next2Steps = 0;
        if (!findRegisteredSequenceNeighbor(next, 1, &next2, &next2Steps) || !hasRegisteredCamera(next2))
        {
            return false;
        }
        const auto nextCenter = _reconstruction->camera(next).cameraCenter();
        const auto delta = centerSub(nextCenter, _reconstruction->camera(next2).cameraCenter());
        const double scale = static_cast<double>(std::max(1, nextSteps)) /
                             static_cast<double>(std::max(1, next2Steps));
        center = centerAdd(nextCenter, centerScale(delta, scale));
        rotation = interpolateCameraRotation(
            _reconstruction->camera(next2).cameraToWorldRotation(),
            _reconstruction->camera(next).cameraToWorldRotation(),
            1.0 + scale);
    }
    else if (prevRegistered && nextRegistered)
    {
        // 当前影像位于已注册区间内部的多帧缺口时，才使用跨缺口插值。
        // 闭环前沿的另一侧可能要绕行数百帧，不能因其“存在”就压制直接邻居外推。
        const double denom = static_cast<double>(std::max(1, prevSteps + nextSteps));
        const double t = static_cast<double>(prevSteps) / denom;
        const auto prevCenter = _reconstruction->camera(prev).cameraCenter();
        const auto nextCenter = _reconstruction->camera(next).cameraCenter();
        center = centerAdd(centerScale(prevCenter, 1.0 - t), centerScale(nextCenter, t));
        rotation = interpolateCameraRotation(
            _reconstruction->camera(prev).cameraToWorldRotation(),
            _reconstruction->camera(next).cameraToWorldRotation(),
            t);
    }
    else
    {
        return false;
    }

    guessCamera->setPose(rotation, center);
    return true;
}

// ============================================================
// 内部：光束法平差
// ============================================================


void IncrementalSfm::rebuildVisibilityCache()
{
    _visibilityCache.clear();
    auto allIds = _reconstruction->allImageIds();

    for (ImageId id : allIds)
    {
        if (_reconstruction->isRegistered(id))
        {
            continue;
        }

        size_t numVisible = 0;
        auto neighbors = _correspondenceGraph.connectedImages(id);
        const int minTrackLengthForPnp =
            effectivePnpMinTrackLength(_sfmOptions, _reconstruction->numRegisteredImages());
        for (ImageId nid : neighbors)
        {
            if (!_reconstruction->isRegistered(nid))
            {
                continue;
            }
            const auto &matches = _correspondenceGraph.matchesBetween(id, nid);
            const ImageData &nimg = _reconstruction->image(nid);

            for (const auto &m : matches)
            {
                FeatureIdx nfeat = (id < nid) ? m.idx2 : m.idx1;
                if (nfeat < nimg.point3DIds.size() &&
                    pointUsableForPnp(*_reconstruction, nimg.point3DIds[nfeat], minTrackLengthForPnp))
                {
                    ++numVisible;
                }
            }
        }

        _visibilityCache[id] = numVisible;
    }
    _visibilityCacheDirty = false;
}

void IncrementalSfm::invalidateVisibilityCache()
{
    _visibilityCacheDirty = true;
}

void IncrementalSfm::updateVisibilityCacheForImage(ImageId imageId, const std::vector<Point3DId> &previousPointIds)
{
    if (_visibilityCacheDirty || !_reconstruction->hasImage(imageId))
    {
        return;
    }

    const auto &currentPointIds = _reconstruction->image(imageId).point3DIds;
    const size_t numFeatures = std::min(previousPointIds.size(), currentPointIds.size());

    for (size_t featureIndex = 0; featureIndex < numFeatures; ++featureIndex)
    {
        const Point3DId previousId = previousPointIds[featureIndex];
        const Point3DId currentId = currentPointIds[featureIndex];
        if (currentId == kInvalidPoint3DId || currentId == previousId || !_reconstruction->hasPoint3D(currentId))
        {
            continue;
        }

        const ScenePoint3D &point = _reconstruction->point3D(currentId);
        const int minTrackLengthForPnp =
            effectivePnpMinTrackLength(_sfmOptions, _reconstruction->numRegisteredImages());
        if (!pointUsableForPnp(*_reconstruction, currentId, minTrackLengthForPnp))
        {
            continue;
        }
        for (const auto &trackElem : point.track.elements)
        {
            auto correspondences = _correspondenceGraph.findCorrespondences(trackElem.imageId, trackElem.featureIdx);
            for (const auto &corr : correspondences)
            {
                if (_reconstruction->isRegistered(corr.imageId))
                {
                    continue;
                }
                ++_visibilityCache[corr.imageId];
            }
        }
    }
}

// ============================================================
// 内部：进度报告
// ============================================================


} // namespace xjw
