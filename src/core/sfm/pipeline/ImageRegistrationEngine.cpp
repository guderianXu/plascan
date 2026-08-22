#include "ImageRegistrationEngine.h"
#include "IncrementalSfmDetail.h"
#include "SfmBundleAdjustCoordinator.h"
#include "geometry/OpenCvCameraAdapter.h"
#include "Intersection.h"
#include "tracks/CorrespondenceTrackThinner.h"
#include "tracks/MultiViewTrackBuilder.h"
#include "concurrency/SafeWorkerGroup.h"

#include "log/Logger.h"

#include "DeterministicOpenCvRansac.h"
#include "OpenCvCompat.h"
#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
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

std::vector<ImageId> ImageRegistrationEngine::findParallelAerialPoseOutliers(
    const SfmReconstruction &reconstruction,
    double minimumAxisConcentration,
    double minimumAngularDeviationDegrees)
{
    const std::vector<ImageId> image_ids = reconstruction.registeredImageIds();
    if (image_ids.size() < 12)
    {
        return {};
    }

    std::vector<std::array<double, 3>> axes;
    std::vector<ImageId> valid_ids;
    axes.reserve(image_ids.size());
    valid_ids.reserve(image_ids.size());
    std::array<double, 3> mean_axis{{0.0, 0.0, 0.0}};
    for (const ImageId image_id : image_ids)
    {
        if (!reconstruction.hasCamera(image_id))
        {
            continue;
        }
        const auto rotation = reconstruction.camera(image_id)
                                  .normalizedForPositiveDepth()
                                  .cameraToWorldRotation();
        std::array<double, 3> axis{{rotation[2], rotation[5], rotation[8]}};
        const double norm = std::sqrt(
            axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        if (!(norm > 1.0e-12) || !std::isfinite(norm))
        {
            continue;
        }
        for (int component = 0; component < 3; ++component)
        {
            axis[component] /= norm;
            mean_axis[component] += axis[component];
        }
        axes.push_back(axis);
        valid_ids.push_back(image_id);
    }
    if (axes.size() < 12)
    {
        return {};
    }

    const double mean_norm = std::sqrt(
        mean_axis[0] * mean_axis[0] +
        mean_axis[1] * mean_axis[1] +
        mean_axis[2] * mean_axis[2]);
    const double concentration = mean_norm / static_cast<double>(axes.size());
    if (!std::isfinite(concentration) ||
        concentration < std::clamp(minimumAxisConcentration, 0.0, 1.0) ||
        !(mean_norm > 1.0e-12))
    {
        return {};
    }
    for (double &component : mean_axis)
    {
        component /= mean_norm;
    }

    constexpr double radians_to_degrees = 57.2957795130823208768;
    std::vector<double> angular_deviations;
    angular_deviations.reserve(axes.size());
    for (const auto &axis : axes)
    {
        const double dot = std::clamp(
            axis[0] * mean_axis[0] +
                axis[1] * mean_axis[1] +
                axis[2] * mean_axis[2],
            -1.0,
            1.0);
        angular_deviations.push_back(std::acos(dot) * radians_to_degrees);
    }
    const double median_deviation = percentile(angular_deviations, 0.5);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(angular_deviations.size());
    for (const double deviation : angular_deviations)
    {
        absolute_deviations.push_back(std::fabs(deviation - median_deviation));
    }
    const double mad = percentile(absolute_deviations, 0.5);
    const double threshold = std::max(
        std::max(1.0, minimumAngularDeviationDegrees),
        median_deviation + 6.0 * std::max(0.5, mad));

    std::unordered_map<ImageId, double> deviation_by_id;
    deviation_by_id.reserve(valid_ids.size());
    std::unordered_set<ImageId> outlier_ids;
    for (std::size_t index = 0; index < valid_ids.size(); ++index)
    {
        deviation_by_id.emplace(valid_ids[index], angular_deviations[index]);
        if (angular_deviations[index] > threshold)
        {
            outlier_ids.insert(valid_ids[index]);
        }
    }

    // 强离群相机经常不是孤立点，而是一小段内部姿态平滑、整体方向错误的刚性分支。
    // 只摘除超过强阈值的中心帧，会让它们重新 PnP 时继续以两侧的错误肩部为锚。
    // 采用滞回阈值沿输入序列扩展，直到回到主航摄方向，再整体摘除并从稳定边界恢复。
    const double branch_expansion_threshold = std::max(
        12.0,
        median_deviation + 3.0 * std::max(0.5, mad));
    const std::vector<ImageId> strong_outliers(outlier_ids.begin(), outlier_ids.end());
    for (const ImageId seed_id : strong_outliers)
    {
        ImageId candidate_id = seed_id;
        while (candidate_id > 0)
        {
            --candidate_id;
            const auto deviation_it = deviation_by_id.find(candidate_id);
            if (deviation_it == deviation_by_id.end() ||
                deviation_it->second <= branch_expansion_threshold)
            {
                break;
            }
            outlier_ids.insert(candidate_id);
        }

        candidate_id = seed_id;
        while (candidate_id + 1 < reconstruction.numImages())
        {
            ++candidate_id;
            const auto deviation_it = deviation_by_id.find(candidate_id);
            if (deviation_it == deviation_by_id.end() ||
                deviation_it->second <= branch_expansion_threshold)
            {
                break;
            }
            outlier_ids.insert(candidate_id);
        }
    }

    std::vector<ImageId> outliers(outlier_ids.begin(), outlier_ids.end());
    std::sort(outliers.begin(), outliers.end());
    return outliers;
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
        Triangulator triangulator(*_reconstruction,
                                  _correspondenceGraph,
                                  _sfmOptions.baOptions.numThreads);
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
        repairParallelAerialPoseOutliersAfterFinalBA();

        // 过滤轨迹长度过短的不可靠三维点
        Triangulator finalTri(*_reconstruction,
                              _correspondenceGraph,
                              _sfmOptions.baOptions.numThreads);
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
        Triangulator finalReprojTri(*_reconstruction,
                                    _correspondenceGraph,
                                    _sfmOptions.baOptions.numThreads);
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
    result.aerialPoseOutliersDetected = _lastAerialPoseOutliersDetected;
    result.aerialPoseOutliersRepaired = _lastAerialPoseOutliersRepaired;

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
    result.baAdaptiveCameraModelFittingEvaluated =
        _lastGlobalBAAdaptiveCameraModelFittingEvaluated;
    result.baAdaptiveCameraModelFittingApplied =
        _lastGlobalBAAdaptiveCameraModelFittingApplied;
    result.baIntrinsicParameterMask = _lastGlobalBAIntrinsicParameterMask;
    result.baIntrinsicParameterReliability =
        _lastGlobalBAIntrinsicParameterReliability;
    result.baIntrinsicParameterIncrementalInformationScore =
        _lastGlobalBAIntrinsicParameterIncrementalInformationScore;
    result.baIntrinsicParameterSensitivity =
        _lastGlobalBAIntrinsicParameterSensitivity;
    result.baAdaptiveCameraModel = _lastGlobalBAAdaptiveCameraModel;
    result.baAdaptiveCameraModelReason = _lastGlobalBAAdaptiveCameraModelReason;
    result.baCameraModelGeometryStrength =
        _lastGlobalBACameraModelGeometryStrength;
    result.baCameraModelOpticalAxisConcentration =
        _lastGlobalBACameraModelOpticalAxisConcentration;
    result.baCameraModelMedianTriangulationAngle =
        _lastGlobalBACameraModelMedianTriangulationAngle;
    result.baCameraModelNormalizedRadiusP90 =
        _lastGlobalBACameraModelNormalizedRadiusP90;
    result.baCameraModelOccupiedPeripheralSectors =
        _lastGlobalBACameraModelOccupiedPeripheralSectors;
    result.baCameraModelObservationCount =
        _lastGlobalBACameraModelObservationCount;
    result.baCameraModelMultiViewTrackRatio =
        _lastGlobalBACameraModelMultiViewTrackRatio;
    result.baCameraModelObservationSupport =
        _lastGlobalBACameraModelObservationSupport;
    result.baCameraModelPeripheralCoverage =
        _lastGlobalBACameraModelPeripheralCoverage;
    result.baCameraModelSectorCoverage =
        _lastGlobalBACameraModelSectorCoverage;
    result.baCameraModelImageAxisBalance =
        _lastGlobalBACameraModelImageAxisBalance;
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
            Triangulator triangulator(*_reconstruction,
                                      _correspondenceGraph,
                                      _sfmOptions.baOptions.numThreads);
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

int IncrementalSfm::repairParallelAerialPoseOutliersAfterFinalBA()
{
    _lastAerialPoseOutliersDetected = 0;
    _lastAerialPoseOutliersRepaired = 0;
    if (!_sfmOptions.repairParallelAerialPoseOutliers ||
        !_sfmOptions.correctUnanchoredAerialDoming ||
        !_sfmOptions.useSequencePoseRecovery ||
        _sfmOptions.sequenceLoopClosure ||
        !_reconstruction || _isAborted || _controlNetworkApplied ||
        _reconstruction->numRegisteredImages() < 32)
    {
        return 0;
    }

    const std::vector<ImageId> outliers =
        ImageRegistrationEngine::findParallelAerialPoseOutliers(*_reconstruction);
    const std::size_t maximumOutliers = std::max<std::size_t>(
        2,
        static_cast<std::size_t>(std::ceil(
            static_cast<double>(_reconstruction->numRegisteredImages()) * 0.05)));
    if (outliers.empty())
    {
        return 0;
    }
    if (outliers.size() > maximumOutliers)
    {
        Logger::instance()->warnf(
            "[SFM] Parallel-aerial pose audit found %zu/%zu outliers; "
            "repair skipped because this is not a small isolated branch",
            outliers.size(),
            _reconstruction->numRegisteredImages());
        return 0;
    }

    _lastAerialPoseOutliersDetected = static_cast<int>(outliers.size());
    std::ostringstream imageList;
    for (std::size_t index = 0; index < outliers.size(); ++index)
    {
        if (index > 0)
        {
            imageList << ',';
        }
        imageList << outliers[index];
        _reconstruction->deregisterImage(outliers[index]);
    }
    Logger::instance()->warnf(
        "[SFM] Parallel-aerial pose audit isolated %zu camera(s): [%s]",
        outliers.size(),
        imageList.str().c_str());

    // 删除仅由错误分支支撑的三维点，再让稳定主网单独收敛。仍被主网至少两幅
    // 影像观测的点会保留，供随后受序列先验约束的 PnP 使用。
    Triangulator stableTriangulator(*_reconstruction,
                                    _correspondenceGraph,
                                    _sfmOptions.baOptions.numThreads);
    const int removedBranchPoints = stableTriangulator.filterShortTracks(2);
    Logger::instance()->infof(
        "[SFM] Parallel-aerial pose repair removed %d branch-only point(s)",
        removedBranchPoints);
    SfmBundleAdjustCoordinator(*this).iterative(true);
    invalidateVisibilityCache();

    std::unordered_set<ImageId> pending(outliers.begin(), outliers.end());
    for (std::size_t pass = 0; pass < outliers.size() && !pending.empty() && !_isAborted; ++pass)
    {
        rebuildVisibilityCache();
        auto directNeighborCount = [&](ImageId imageId)
        {
            int count = 0;
            if (imageId > 0 && _reconstruction->isRegistered(imageId - 1))
            {
                ++count;
            }
            if (imageId + 1 < _reconstruction->numImages() &&
                _reconstruction->isRegistered(imageId + 1))
            {
                ++count;
            }
            return count;
        };

        int registeredThisPass = 0;
        std::unordered_set<ImageId> attemptedThisPass;
        while (!_isAborted)
        {
            std::vector<ImageId> candidates;
            for (const ImageId imageId : pending)
            {
                if (attemptedThisPass.count(imageId) == 0 &&
                    directNeighborCount(imageId) > 0)
                {
                    candidates.push_back(imageId);
                }
            }
            if (candidates.empty())
            {
                break;
            }
            std::stable_sort(candidates.begin(), candidates.end(), [&](ImageId lhs, ImageId rhs)
            {
                const int lhsNeighbors = directNeighborCount(lhs);
                const int rhsNeighbors = directNeighborCount(rhs);
                if (lhsNeighbors != rhsNeighbors)
                {
                    return lhsNeighbors > rhsNeighbors;
                }
                return _visibilityCache[lhs] > _visibilityCache[rhs];
            });

            const ImageId imageId = candidates.front();
            attemptedThisPass.insert(imageId);
            if (!registerImage(imageId, true, true))
            {
                continue;
            }

            const std::vector<Point3DId> previousPointIds =
                _reconstruction->image(imageId).point3DIds;
            Triangulator triangulator(*_reconstruction,
                                      _correspondenceGraph,
                                      _sfmOptions.baOptions.numThreads);
            triangulator.triangulateImage(imageId, _sfmOptions.triangulatorOptions);
            updateVisibilityCacheForImage(imageId, previousPointIds);
            pending.erase(imageId);
            ++registeredThisPass;
            Logger::instance()->infof(
                "[SFM] Parallel-aerial pose repair registered image %u (%zu remaining)",
                imageId,
                pending.size());
        }

        if (registeredThisPass == 0)
        {
            break;
        }
        SfmBundleAdjustCoordinator(*this).iterative(true);
        invalidateVisibilityCache();
    }

    _lastAerialPoseOutliersRepaired =
        _lastAerialPoseOutliersDetected - static_cast<int>(pending.size());
    Logger::instance()->infof(
        "[SFM] Parallel-aerial pose repair completed: detected=%d repaired=%d rejected=%zu",
        _lastAerialPoseOutliersDetected,
        _lastAerialPoseOutliersRepaired,
        pending.size());
    return _lastAerialPoseOutliersRepaired;
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
    _lastGlobalBABackendMessage.clear();
    _lastGlobalBAAdaptiveCameraModelFittingEvaluated = false;
    _lastGlobalBAAdaptiveCameraModelFittingApplied = false;
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
    _lastHierarchicalBAImageCount = 0;
    _lastHierarchicalBAAttemptImageCount = 0;
    _lastHierarchicalBAPlannedBlocks = 0;
    _lastHierarchicalBAAppliedBlocks = 0;
    _lastHierarchicalBAUpdatedCameras = 0;
    _lastHierarchicalBATotalSeconds = 0.0;
    _lastAerialPoseOutliersDetected = 0;
    _lastAerialPoseOutliersRepaired = 0;
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

bool IncrementalSfm::registerImage(ImageId imageId,
                                   bool preferSequencePrior,
                                   bool forceSequenceConsistency)
{
    // 加载相机内参
    FramePinholeCamera cam;
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
    std::vector<PnpCorrespondenceProposal> proposals;

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
                myFeat >= img.keypoints.size())
            {
                continue;
            }
            const ScenePoint3D &point = _reconstruction->point3D(p3dId);
            proposals.push_back({myFeat,
                                 p3dId,
                                 1,
                                 point.track.length(),
                                 std::isfinite(m.score) ? static_cast<double>(m.score) : 0.0,
                                 std::isfinite(point.error)
                                     ? point.error
                                     : std::numeric_limits<double>::infinity()});
        }
    }

    const std::vector<PnpCorrespondenceProposal> selected_proposals =
        selectUniquePnpCorrespondences(proposals);
    worldPts.reserve(selected_proposals.size());
    imagePts.reserve(selected_proposals.size());
    for (const PnpCorrespondenceProposal &proposal : selected_proposals)
    {
        worldPts.push_back(_reconstruction->point3D(proposal.pointId).xyz);
        imagePts.push_back({static_cast<double>(img.keypoints[proposal.featureIdx].x),
                            static_cast<double>(img.keypoints[proposal.featureIdx].y)});
    }

    const int configured_min_inliers = _sfmOptions.pnpOptions.minNumInliers;
    const int strict_min_inliers = std::clamp(
        _sfmOptions.pnpOptions.strictSmallSupportMinInliers,
        4,
        std::max(4, configured_min_inliers));
    const bool strict_small_support =
        _sfmOptions.pnpOptions.allowStrictSmallSupportRecovery &&
        static_cast<int>(worldPts.size()) < configured_min_inliers &&
        static_cast<int>(worldPts.size()) >= strict_min_inliers;
    if (static_cast<int>(worldPts.size()) < configured_min_inliers &&
        !strict_small_support)
    {
        std::ostringstream oss;
        oss << "not enough 2D-3D observations for PnP: observations=" << worldPts.size()
            << ", minPnPInliers=" << configured_min_inliers
            << ", strictRecoveryMin=" << strict_min_inliers
            << ", minTrackLength=" << minTrackLengthForPnp
            << ", connectedNeighbors=" << neighbors.size()
            << ", rawProposals=" << proposals.size();
        _lastErrorMessage = oss.str();
        return false;
    }

    PnpOptions pnpOptions = _sfmOptions.pnpOptions;
    if (strict_small_support)
    {
        pnpOptions.minNumInliers = strict_min_inliers;
        pnpOptions.minInlierRatio = std::max(
            pnpOptions.minInlierRatio,
            std::clamp(pnpOptions.strictSmallSupportMinInlierRatio, 0.0, 1.0));
        pnpOptions.smallSampleMinInlierRatio = std::max(
            pnpOptions.smallSampleMinInlierRatio,
            pnpOptions.minInlierRatio);
        pnpOptions.allowRelaxedInlierRatio = false;
    }
    pnpOptions.ransacSeed = opencv_compat::stableRansacSeed(
        imageId,
        static_cast<std::uint32_t>(_reconstruction->numRegisteredImages()),
        static_cast<std::uint32_t>(worldPts.size()));
    bool usedSequenceRecovery = false;
    auto solveSequenceRecovery = [&]()
    {
        FramePinholeCamera sequenceGuessCamera = cam;
        if (!makeSequenceInitialPoseGuess(imageId, &sequenceGuessCamera))
        {
            return PnpResult{};
        }
        usedSequenceRecovery = true;
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
        if (!strict_small_support && _sfmOptions.allowBracketedSequencePnpRelaxation &&
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
        return PnpSolver::solveWithCamera(worldPts, imagePts, cam, recoveryOptions);
    };

    PnpResult pnpResult;
    if (preferSequencePrior)
    {
        // 最终离群修复时主网已经稳定。先用序列预测过滤远距离重复纹理形成的
        // 伪 2D-3D 对应；若预测不适用于真实转弯，再退回标准全局 PnP。
        pnpResult = solveSequenceRecovery();
        if (!pnpResult.success)
        {
            pnpResult = PnpSolver::solveWithCamera(worldPts, imagePts, cam, pnpOptions);
        }
    }
    else
    {
        // 常规增量阶段仍先执行不带运动模型的标准 PnP，避免序列外推妨碍
        // 非等速航带或真实转弯。标准解失败后才进入确定性序列恢复。
        pnpResult = PnpSolver::solveWithCamera(worldPts, imagePts, cam, pnpOptions);
        if (!pnpResult.success)
        {
            pnpResult = solveSequenceRecovery();
        }
    }
    PnpInlierSpatialSupport spatial_support;
    bool strict_spatial_support_accepted = true;
    if (pnpResult.success && strict_small_support)
    {
        const std::optional<CameraImageSize> image_size = cam.imageSize();
        double maximum_x = std::max(0.0, cam.principalX() * 2.0 + 1.0);
        double maximum_y = std::max(0.0, cam.principalY() * 2.0 + 1.0);
        for (const FeatureKeypoint &keypoint : img.keypoints)
        {
            if (std::isfinite(keypoint.x))
            {
                maximum_x = std::max(maximum_x, static_cast<double>(keypoint.x) + 1.0);
            }
            if (std::isfinite(keypoint.y))
            {
                maximum_y = std::max(maximum_y, static_cast<double>(keypoint.y) + 1.0);
            }
        }
        const int width = image_size && image_size->samples > 0
            ? image_size->samples
            : static_cast<int>(std::ceil(maximum_x));
        const int height = image_size && image_size->lines > 0
            ? image_size->lines
            : static_cast<int>(std::ceil(maximum_y));
        spatial_support = measurePnpInlierSpatialSupport(
            imagePts, pnpResult.inlierMask, width, height);
        strict_spatial_support_accepted =
            spatial_support.occupiedCells >=
                std::max(1, pnpOptions.strictSmallSupportMinGridCells) &&
            spatial_support.occupiedRows >= 2 &&
            spatial_support.occupiedColumns >= 2;
    }
    if (!pnpResult.success || !strict_spatial_support_accepted)
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
            << ", strictSmallSupport=" << (strict_small_support ? "true" : "false")
            << ", gridCells=" << spatial_support.occupiedCells
            << ", gridRows=" << spatial_support.occupiedRows
            << ", gridColumns=" << spatial_support.occupiedColumns
            << ", maxReprojError=" << _sfmOptions.pnpOptions.maxReprojError;
        _lastErrorMessage = oss.str();
        return false;
    }

    Logger::instance()->infof("[SFM] Image %u PnP success: observations=%zu, inliers=%d, inlierRatio=%.3f, "
                              "minTrackLength=%d, sequenceRecovery=%s, posePrefilter=%s, "
                              "prefilterCandidates=%d, rawProposals=%zu, strictSmallSupport=%s, "
                              "gridCells=%d",
                              imageId,
                              worldPts.size(),
                              pnpResult.numInliers,
                              pnpResult.inlierRatio,
                              minTrackLengthForPnp,
                              usedSequenceRecovery ? "true" : "false",
                              pnpResult.usedInitialPosePrefilter ? "true" : "false",
                              pnpResult.prefilterCandidateCount,
                              proposals.size(),
                              strict_small_support ? "true" : "false",
                              spatial_support.occupiedCells);

    // 应用 PnP 结果更新相机外参
    cam.setPose(pnpResult.R, pnpResult.C);
    std::string sequenceConsistencyReason;
    if (!validateSequencePoseConsistency(
            imageId, cam, &sequenceConsistencyReason, forceSequenceConsistency))
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
                                                     const FramePinholeCamera &candidateCamera,
                                                     std::string *reason,
                                                     bool force) const
{
    if ((!_sfmOptions.enforceSequencePoseConsistency && !force) || !_reconstruction ||
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

    if (force)
    {
        const auto candidateRotation = candidateCamera.normalizedForPositiveDepth()
                                           .cameraToWorldRotation();
        const std::array<double, 3> candidateAxis{{
            candidateRotation[2], candidateRotation[5], candidateRotation[8]}};
        constexpr double radiansToDegrees = 57.2957795130823208768;
        constexpr double maximumNeighborAxisAngleDegrees = 15.0;
        std::array<double, 3> main_axis{{0.0, 0.0, 0.0}};
        int main_axis_samples = 0;
        for (const ImageId registered_id : _reconstruction->registeredImageIds())
        {
            if (!_reconstruction->hasCamera(registered_id))
            {
                continue;
            }
            const auto rotation = _reconstruction->camera(registered_id)
                                      .normalizedForPositiveDepth()
                                      .cameraToWorldRotation();
            main_axis[0] += rotation[2];
            main_axis[1] += rotation[5];
            main_axis[2] += rotation[8];
            ++main_axis_samples;
        }
        const double main_axis_norm = std::sqrt(
            main_axis[0] * main_axis[0] +
            main_axis[1] * main_axis[1] +
            main_axis[2] * main_axis[2]);
        if (main_axis_samples >= 12 && main_axis_norm > 1.0e-12 &&
            std::isfinite(main_axis_norm))
        {
            for (double &component : main_axis)
            {
                component /= main_axis_norm;
            }
            const double dot = std::clamp(
                candidateAxis[0] * main_axis[0] +
                    candidateAxis[1] * main_axis[1] +
                    candidateAxis[2] * main_axis[2],
                -1.0,
                1.0);
            const double angle = std::acos(dot) * radiansToDegrees;
            if (!std::isfinite(angle) || angle > maximumNeighborAxisAngleDegrees)
            {
                if (reason)
                {
                    std::ostringstream oss;
                    oss << "image=" << imageId
                        << ", mainOpticalAxisAngle=" << angle
                        << ", maximum=" << maximumNeighborAxisAngleDegrees;
                    *reason = oss.str();
                }
                return false;
            }
        }
        for (const ImageId neighborId : registeredNeighbors)
        {
            const auto neighborRotation = _reconstruction->camera(neighborId)
                                              .normalizedForPositiveDepth()
                                              .cameraToWorldRotation();
            const double dot = std::clamp(
                candidateAxis[0] * neighborRotation[2] +
                    candidateAxis[1] * neighborRotation[5] +
                    candidateAxis[2] * neighborRotation[8],
                -1.0,
                1.0);
            const double angle = std::acos(dot) * radiansToDegrees;
            if (!std::isfinite(angle) || angle > maximumNeighborAxisAngleDegrees)
            {
                if (reason)
                {
                    std::ostringstream oss;
                    oss << "image=" << imageId
                        << ", neighbor=" << neighborId
                        << ", opticalAxisAngle=" << angle
                        << ", maximum=" << maximumNeighborAxisAngleDegrees;
                    *reason = oss.str();
                }
                return false;
            }
        }
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

bool IncrementalSfm::makeSequenceInitialPoseGuess(ImageId imageId, FramePinholeCamera *guessCamera) const
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
    const auto visibilityStarted = std::chrono::steady_clock::now();
    _visibilityCache.clear();
    const auto allIds = _reconstruction->allImageIds();
    std::vector<ImageId> pendingIds;
    pendingIds.reserve(allIds.size());
    for (ImageId id : allIds)
    {
        if (!_reconstruction->isRegistered(id))
        {
            pendingIds.push_back(id);
        }
    }

    const int minTrackLengthForPnp = effectivePnpMinTrackLength(
        _sfmOptions, _reconstruction->numRegisteredImages());
    std::vector<std::size_t> visibleCounts(pendingIds.size(), 0);
    const int threadCount = _sfmOptions.baOptions.numThreads > 0
        ? _sfmOptions.baOptions.numThreads
        : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    common::concurrency::parallelForIndices(
        pendingIds.size(), static_cast<std::size_t>(threadCount), [&](std::size_t pendingIndex)
        {
        const ImageId id = pendingIds[pendingIndex];
        size_t numVisible = 0;
        const auto neighbors = _correspondenceGraph.connectedImages(id);
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

        visibleCounts[pendingIndex] = numVisible;
        });

    for (std::size_t pendingIndex = 0;
         pendingIndex < pendingIds.size();
         ++pendingIndex)
    {
        _visibilityCache[pendingIds[pendingIndex]] =
            visibleCounts[pendingIndex];
    }
    _visibilityCacheDirty = false;
    if (pendingIds.size() >= 32)
    {
        const double elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - visibilityStarted).count();
        Logger::instance()->infof(
            "[SFM] visibility_cache rebuilt pending=%zu threads=%d seconds=%.3f",
            pendingIds.size(),
            threadCount,
            elapsedSeconds);
    }
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
