#include "KnownPoseReconstructor.h"
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
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace xjw
{

using namespace incremental_sfm_detail;

KnownPoseReconstructor::KnownPoseReconstructor(IncrementalSfm &owner)
    : _owner(owner)
{
}

IncrementalSfmResult KnownPoseReconstructor::run(SfmProgressCallback progressCb)
{
    return _owner.runKnownCameraPoseReconstruction(std::move(progressCb));
}

IncrementalSfmResult IncrementalSfm::runKnownCameraPoseReconstruction(SfmProgressCallback progressCb)
{
    IncrementalSfmResult result;
    applyPriorTrackDiagnostics(&result);
    const int totalImages = static_cast<int>(_reconstruction->numImages());

    auto imageIds = _reconstruction->allImageIds();
    std::sort(imageIds.begin(), imageIds.end());

    // 阶段 1：已知位姿路径不执行初始对/PnP，直接把每台可信 Camera 注册到同一
    // 重建中。任何缺失相机都使该路径失败，不能混入任意估计位姿。
    int registeredCount = 0;
    for (ImageId imageId : imageIds)
    {
        Camera camera;
        if (!getCamera(imageId, camera))
        {
            result.summary = "Failed to load known camera pose for image " + std::to_string(imageId);
            return result;
        }

        _reconstruction->registerImage(imageId, camera);
        ++registeredCount;

        if (!reportProgress(registeredCount, totalImages, "Registered known camera pose", progressCb))
        {
            result.summary = "Known camera pose reconstruction aborted";
            return result;
        }
    }

    // 阶段 2：根据真实基线/交会角分布选择阈值。窄基线可有限放宽，但不能通过
    // 影像序号假设相机必然成圆来伪造几何。
    const auto triangulationPolicy = resolveKnownPoseTriangulationPolicy(_reconstruction,
                                                                         _correspondenceGraph,
                                                                         imageIds,
                                                                         _sfmOptions);
    if (triangulationPolicy.adapted)
    {
        Logger::instance()->infof(
            "[SFM] Known-pose narrow-baseline adaptation: minTriAngle %.3f -> %.3f deg "
            "(valid=%d defaultAccepted=%d adaptedAccepted=%d)",
            _sfmOptions.triangulatorOptions.minTriAngle,
            triangulationPolicy.chosenMinTriAngle,
            triangulationPolicy.validCandidates,
            triangulationPolicy.acceptedWithDefault,
            triangulationPolicy.acceptedWithAdapted);
    }

    Triangulator triangulator(*_reconstruction,
                              _correspondenceGraph,
                              _sfmOptions.baOptions.numThreads);
    int createdPoints = 0;
    int continuedObservations = 0;
    int completedObservations = 0;
    int inputMultiViewTrackCount = 0;
    int createdLongTrackCount = 0;
    int longTrackTwoViewOnlyCount = 0;

    // 阶段 3：先按已知相机几何过滤 pairwise 边，再合并多视轨迹。若先合并后
    // 过滤，一条坏边可能把两个独立物点组件错误地粘在一起。
    MultiViewTrackBuilder trackBuilder;
    float maxKeypointX = 0.0f;
    float maxKeypointY = 0.0f;
    for (ImageId imageId : imageIds)
    {
        const ImageData &image = _reconstruction->image(imageId);
        trackBuilder.setImageKeypoints(imageId, image.keypoints);
        for (const FeatureKeypoint &keypoint : image.keypoints)
        {
            if (std::isfinite(keypoint.x))
            {
                maxKeypointX = std::max(maxKeypointX, keypoint.x);
            }
            if (std::isfinite(keypoint.y))
            {
                maxKeypointY = std::max(maxKeypointY, keypoint.y);
            }
        }
    }

    int indexedPairCount = 0;
    int indexedMatchCount = 0;
    int rawIndexedMatchCount = 0;
    int geometryRejectedMatchCount = 0;
    for (ImageId imageId : imageIds)
    {
        const std::vector<ImageId> connected = _correspondenceGraph.connectedImages(imageId);
        for (ImageId otherImageId : connected)
        {
            if (otherImageId <= imageId)
            {
                continue;
            }

            const auto &matches = _correspondenceGraph.matchesBetween(imageId, otherImageId);
            if (matches.empty())
            {
                continue;
            }

            std::vector<MultiViewTrackBuilder::MatchIndexPair> indexedMatches;
            indexedMatches.reserve(matches.size());
            for (const FeatureMatch &match : matches)
            {
                if (match.idx1 == kInvalidFeatureIdx || match.idx2 == kInvalidFeatureIdx)
                {
                    continue;
                }
                ++rawIndexedMatchCount;
                if (!knownPoseMatchPassesGeometry(*_reconstruction,
                                                  imageId,
                                                  otherImageId,
                                                  match,
                                                  triangulationPolicy.triangulatorOptions))
                {
                    ++geometryRejectedMatchCount;
                    continue;
                }
                indexedMatches.emplace_back(match.idx1, match.idx2, match.score);
            }

            if (!indexedMatches.empty())
            {
                trackBuilder.addMatchPair(imageId, otherImageId, indexedMatches);
                ++indexedPairCount;
                indexedMatchCount += static_cast<int>(indexedMatches.size());
            }
        }
    }

    MultiViewTrackBuilder::BuildOptions trackBuildOptions;
    trackBuildOptions.enableQualityThinning =
        _sfmOptions.maxTracksPerImage > 0 ||
        _sfmOptions.maxTracksPerGridCell > 0;
    trackBuildOptions.maxTracksPerImage = _sfmOptions.maxTracksPerImage;
    trackBuildOptions.maxTracksPerGridCell = _sfmOptions.maxTracksPerGridCell;
    trackBuildOptions.gridColumns = _sfmOptions.trackThinningGridColumns;
    trackBuildOptions.gridRows = _sfmOptions.trackThinningGridRows;
    trackBuildOptions.imageWidth = std::max(1.0f, maxKeypointX + 1.0f);
    trackBuildOptions.imageHeight = std::max(1.0f, maxKeypointY + 1.0f);

    MultiViewTrackBuildResult multiViewTracks = trackBuilder.build(trackBuildOptions);

    // 人工先验轨迹拥有独立身份。自动组件中若包含同一人工观测，先删除该自动轨迹，
    // 再追加 materializedPriorTracks，避免一个 2D 特征同时归属两个三维点。
    multiViewTracks.tracks.erase(
        std::remove_if(multiViewTracks.tracks.begin(), multiViewTracks.tracks.end(), [this](const Track &track)
        {
            return std::any_of(track.elements.begin(), track.elements.end(), [this](const TrackElement &element)
            {
                return !_correspondenceGraph.priorTrackId(element.imageId, element.featureIdx).empty();
            });
        }),
        multiViewTracks.tracks.end());
    multiViewTracks.tracks.insert(multiViewTracks.tracks.end(),
                                  _materializedPriorTracks.begin(),
                                  _materializedPriorTracks.end());
    for (Track &track : multiViewTracks.tracks) tagPriorTrackSource(&track);
    if (!multiViewTracks.tracks.empty())
    {
        std::ostringstream trackLengthHistogram;
        int emittedTrackLengthBins = 0;
        for (const auto &entry : multiViewTracks.trackLengthHistogram)
        {
            if (emittedTrackLengthBins > 0)
            {
                trackLengthHistogram << ",";
            }
            trackLengthHistogram << entry.first << ":" << entry.second;
            ++emittedTrackLengthBins;
            if (emittedTrackLengthBins >= 8)
            {
                break;
            }
        }

        const auto trackStats = triangulator.triangulateTracks(multiViewTracks.tracks,
                                                               triangulationPolicy.triangulatorOptions);
        createdPoints += trackStats.numCreated;
        continuedObservations += trackStats.numContinued;
        inputMultiViewTrackCount = trackStats.inputLongTracks;
        createdLongTrackCount = trackStats.createdLongTracks;
        longTrackTwoViewOnlyCount = trackStats.longTrackTwoViewOnly;
        Logger::instance()->infof(
            "[SFM] Known-pose multiview tracks: pairs=%d matches=%d rawMatches=%d "
            "geometryRejected=%d components=%d accepted=%d rejectedConflict=%d "
            "rejectedConflictEdges=%d qualityPruned=%d hist=%s created=%d addedObservations=%d",
            indexedPairCount,
            indexedMatchCount,
            rawIndexedMatchCount,
            geometryRejectedMatchCount,
            multiViewTracks.totalComponents,
            multiViewTracks.acceptedComponents,
            multiViewTracks.rejectedConflictComponents,
            multiViewTracks.rejectedConflictEdges,
            multiViewTracks.prunedByQualityThinning,
            trackLengthHistogram.str().c_str(),
            trackStats.numCreated,
            trackStats.numContinued);
        Logger::instance()->infof(
            "[SFM] Known-pose triangulation stats: inputTracks=%d inputLong=%d unusable=%d "
            "noCandidate=%d createdTwoView=%d createdLong=%d seedTests=%d seedRejected=%d "
            "depthRejected=%d reprojRejected=%d longTwoView=%d rejectedExtraSamples=%d "
            "rejectedExtraAvg=%.3f rejectedExtraMax=%.3f rejectedExtraHist=<=5:%d,<=10:%d,<=25:%d,>25:%d",
            trackStats.inputTracks,
            trackStats.inputLongTracks,
            trackStats.unusableTracks,
            trackStats.noCandidateTracks,
            trackStats.createdTwoViewTracks,
            trackStats.createdLongTracks,
            trackStats.seedPairTests,
            trackStats.seedPairRejected,
            trackStats.depthObservationRejected,
            trackStats.reprojObservationRejected,
            trackStats.longTrackTwoViewOnly,
            trackStats.longTrackRejectedExtraSamples,
            trackStats.longTrackRejectedExtraSamples > 0
                ? trackStats.longTrackRejectedExtraErrorSum /
                      static_cast<double>(trackStats.longTrackRejectedExtraSamples)
                : 0.0,
            trackStats.longTrackRejectedExtraErrorMax,
            trackStats.longTrackRejectedExtraLe5,
            trackStats.longTrackRejectedExtraLe10,
            trackStats.longTrackRejectedExtraLe25,
            trackStats.longTrackRejectedExtraGt25);
    }

    // 多视轨迹完全无法成点时才回退逐图双视三角化。只要已有多视结果就不混入
    // 大量弱双视点，以免后续 BA 的短轨迹占比失控。
    if (createdPoints == 0)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose multiview triangulation produced no points; falling back to pairwise triangulation");
        for (ImageId imageId : imageIds)
        {
            const auto stats = triangulator.triangulateImage(imageId, triangulationPolicy.triangulatorOptions);
            createdPoints += stats.numCreated;
            continuedObservations += stats.numContinued;
        }
    }

    completedObservations = triangulator.completeTracks(triangulationPolicy.triangulatorOptions);
    triangulator.retriangulatePoints(triangulationPolicy.triangulatorOptions.maxReprojError);
    triangulator.recomputeReprojErrors();
    const int filteredPoints = triangulator.filterPoints(_sfmOptions.filterMaxReprojError,
                                                         triangulationPolicy.filterMinTriAngle);
    const int shortTrackFiltered = triangulator.filterShortTracks(_sfmOptions.filterMinTrackLen);
    triangulator.recomputeReprojErrors();

    int baRetriangulated = 0;
    int baCompletedObservations = 0;
    int baFilteredPoints = 0;
    int baShortTrackFiltered = 0;
    // 阶段 4：先以当前点提供 PnP 微调，再调用统一 BA 协调器；BA 后必须重三角化、
    // 补观测和再次过滤，因为相机位姿更新会改变全部交会几何。
    if (_reconstruction->numRegisteredImages() >= 2 && _reconstruction->numPoints3D() > 0)
    {
        Logger::instance()->infof("[SFM] Known-pose global BA: tracks=%zu cameras=%zu",
                                  _reconstruction->numPoints3D(),
                                  _reconstruction->numRegisteredImages());
        if (!reportProgress(totalImages, totalImages, "Running known-pose bundle adjustment...", progressCb))
        {
            result.summary = "Known camera pose reconstruction aborted before bundle adjustment";
            return result;
        }

        refineKnownCameraPosesWithPnp();
        SfmBundleAdjustCoordinator(*this).run(false);

        Triangulator baTriangulator(*_reconstruction,
                                    _correspondenceGraph,
                                    _sfmOptions.baOptions.numThreads);
        baRetriangulated = baTriangulator.retriangulatePoints(triangulationPolicy.triangulatorOptions.maxReprojError);
        baCompletedObservations = baTriangulator.completeTracks(triangulationPolicy.triangulatorOptions);
        baFilteredPoints = baTriangulator.filterPoints(_sfmOptions.filterMaxReprojError,
                                                       triangulationPolicy.filterMinTriAngle);
        baShortTrackFiltered = baTriangulator.filterShortTracks(_sfmOptions.filterMinTrackLen);
        baTriangulator.recomputeReprojErrors();
    }

    result.numRegisteredImages = static_cast<int>(_reconstruction->numRegisteredImages());
    result.numPoints3D = static_cast<int>(_reconstruction->numPoints3D());
    result.meanReprojError = _reconstruction->meanReprojError();
    result.reconstruction = _reconstruction;
    result.summary = _reconstruction->summary();
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

    int finalLongTrackCount = 0;
    int finalTwoViewTrackCount = 0;
    for (Point3DId pointId : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(pointId))
        {
            continue;
        }

        const size_t trackLength = _reconstruction->point3D(pointId).track.length();
        if (trackLength >= 3)
        {
            ++finalLongTrackCount;
        }
        else if (trackLength == 2)
        {
            ++finalTwoViewTrackCount;
        }
    }

    const int finalTrackCount = finalLongTrackCount + finalTwoViewTrackCount;
    const double finalTwoViewTrackRatio = finalTrackCount > 0
        ? static_cast<double>(finalTwoViewTrackCount) / static_cast<double>(finalTrackCount)
        : 0.0;
    // 阶段 5：已知位姿下“注册全部相机”本身不是成功标准。若输入存在充足长轨迹，
    // 输出却几乎全为双视点，通常说明相机坐标约定或匹配身份错误，必须拒绝。
    const bool hasEnoughMultiViewInputForQualityGate =
        inputMultiViewTrackCount >= kKnownPoseMinLongInputTracksForQualityGate &&
        result.numPoints3D >= kKnownPoseMinPointsForTrackRatioGate;
    const bool hasOnlyTwoViewOutputFromMultiViewInput =
        result.numRegisteredImages >= 3 &&
        inputMultiViewTrackCount > 0 &&
        result.numPoints3D > 0 &&
        finalLongTrackCount == 0;
    const bool hasTooLittleMultiViewOutputFromMultiViewInput =
        result.numRegisteredImages >= 3 &&
        hasEnoughMultiViewInputForQualityGate &&
        finalTwoViewTrackRatio >= kKnownPoseMaxTwoViewTrackRatio;
    result.success = result.numRegisteredImages >= 2 &&
        result.numPoints3D > 0 &&
        !hasOnlyTwoViewOutputFromMultiViewInput &&
        !hasTooLittleMultiViewOutputFromMultiViewInput;

    Logger::instance()->infof(
        "[SFM] Known-pose reconstruction: registered=%d/%d created=%d continued=%d completed=%d "
        "filtered=%d shortTrackFiltered=%d baTracks=%d baOptimized=%d baRms=%.4f->%.4f "
        "baRetriangulated=%d baCompleted=%d baFiltered=%d baShortTrackFiltered=%d points=%d "
        "finalLongTracks=%d finalTwoViewTracks=%d finalTwoViewRatio=%.4f",
        result.numRegisteredImages,
        totalImages,
        createdPoints,
        continuedObservations,
        completedObservations,
        filteredPoints,
        shortTrackFiltered,
        result.baTracksTotal,
        result.baTracksOptimized,
        result.baRmsBefore,
        result.baRmsAfter,
        baRetriangulated,
        baCompletedObservations,
        baFilteredPoints,
        baShortTrackFiltered,
        result.numPoints3D,
        finalLongTrackCount,
        finalTwoViewTrackCount,
        finalTwoViewTrackRatio);

    if (hasOnlyTwoViewOutputFromMultiViewInput)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose reconstruction rejected all-two-view output: inputLongTracks=%d "
            "createdLongTracks=%d longTrackTwoViewOnly=%d finalTwoViewTracks=%d",
            inputMultiViewTrackCount,
            createdLongTrackCount,
            longTrackTwoViewOnlyCount,
            finalTwoViewTrackCount);
        result.summary =
            "Known camera pose reconstruction produced only two-view tracks despite multi-view matches; "
            "check camera poses, match consistency, or rerun SfM/BA with adjusted cameras.";
    }
    else if (hasTooLittleMultiViewOutputFromMultiViewInput)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose reconstruction rejected weak multi-view output: inputLongTracks=%d "
            "createdLongTracks=%d longTrackTwoViewOnly=%d finalLongTracks=%d finalTwoViewTracks=%d "
            "finalTwoViewRatio=%.4f threshold=%.4f",
            inputMultiViewTrackCount,
            createdLongTrackCount,
            longTrackTwoViewOnlyCount,
            finalLongTrackCount,
            finalTwoViewTrackCount,
            finalTwoViewTrackRatio,
            kKnownPoseMaxTwoViewTrackRatio);
        result.summary =
            "Known camera pose reconstruction produced mostly two-view tracks despite multi-view matches; "
            "check camera poses, match consistency, or rerun SfM/BA with adjusted cameras.";
    }

    if (!result.success)
    {
        if (result.summary.empty())
        {
            result.summary = "Known camera pose reconstruction produced insufficient sparse points";
        }
    }

    return result;
}

// ============================================================
// 内部：加载相机
// ============================================================

bool IncrementalSfm::loadCamera(const std::string &cameraPath, Camera &cam) const
{
    return cam.loadFromFile(cameraPath);
}

bool IncrementalSfm::getCamera(ImageId imageId, Camera &cam) const
{
    // 优先使用预设相机对象
    auto pit = _preloadedCameras.find(imageId);
    if (pit != _preloadedCameras.end())
    {
        cam = pit->second;
        return true;
    }
    // 其次从 .tsai 文件加载
    auto cit = _cameraPaths.find(imageId);
    if (cit != _cameraPaths.end() && !cit->second.empty())
    {
        return loadCamera(cit->second, cam);
    }
    return false;
}

std::vector<BACameraPosePrior> IncrementalSfm::buildCameraPosePriorsFromInputCameras(
    const std::vector<ImageId> &imageIds) const
{
    std::vector<Camera> inputCameras;
    inputCameras.reserve(imageIds.size());
    std::vector<std::array<double, 3>> inputCenters;
    inputCenters.reserve(imageIds.size());
    for (ImageId imageId : imageIds)
    {
        Camera inputCamera;
        if (getCamera(imageId, inputCamera))
        {
            inputCenters.push_back(inputCamera.cameraCenter());
        }
        inputCameras.push_back(inputCamera);
    }

    const double inputExtent = centerExtent(inputCenters);
    const double adaptivePositionSigmaMeters = std::max(
        0.25,
        inputExtent * 0.01 * std::max(1e-6, _sfmOptions.knownPosePriorPositionSigmaScale));

    std::vector<BACameraPosePrior> priors;
    priors.reserve(imageIds.size());
    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        BACameraPosePrior prior;
        const Camera &inputCamera = inputCameras[i];
        if (inputCamera.isValid())
        {
            prior.enabled = true;
            prior.cameraToWorldRotation = inputCamera.cameraToWorldRotation();
            prior.cameraCenter = inputCamera.cameraCenter();
            prior.positionSigmaMeters = std::max(1e-6, adaptivePositionSigmaMeters);
            prior.rotationSigmaDegrees =
                std::max(1e-6, _sfmOptions.knownPosePriorRotationSigmaDegrees);
        }
        priors.push_back(prior);
    }
    return priors;
}

void IncrementalSfm::alignReconstructionToKnownPosePriors(const std::vector<ImageId> &imageIds,
                                                          std::vector<Camera> *baCameras)
{
    if (!baCameras || imageIds.size() != baCameras->size() || imageIds.size() < 3)
    {
        return;
    }

    std::vector<std::array<double, 3>> currentCenters;
    std::vector<std::array<double, 3>> inputCenters;
    currentCenters.reserve(imageIds.size());
    inputCenters.reserve(imageIds.size());
    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        Camera inputCamera;
        if (!getCamera(imageIds[i], inputCamera))
        {
            continue;
        }
        currentCenters.push_back((*baCameras)[i].cameraCenter());
        inputCenters.push_back(inputCamera.cameraCenter());
    }

    if (currentCenters.size() < 3)
    {
        return;
    }

    double directSquaredError = 0.0;
    for (size_t i = 0; i < currentCenters.size(); ++i)
    {
        const double dx = currentCenters[i][0] - inputCenters[i][0];
        const double dy = currentCenters[i][1] - inputCenters[i][1];
        const double dz = currentCenters[i][2] - inputCenters[i][2];
        directSquaredError += dx * dx + dy * dy + dz * dz;
    }
    const double directRmse = std::sqrt(
        directSquaredError / static_cast<double>(currentCenters.size()));
    const double inputExtent = centerExtent(inputCenters);
    const double noOpTolerance = std::max(1e-8, inputExtent * 1e-6);
    if (directRmse <= noOpTolerance)
    {
        Logger::instance()->infof(
            "[SFM] Known-pose Sim3 alignment skipped: camera centers already aligned "
            "(rmse=%.9f tolerance=%.9f)",
            directRmse,
            noOpTolerance);
        return;
    }

    // 仅凭共线相机中心无法确定绕基线方向的旋转。强行套用该 Sim(3) 会保持
    // 重投影误差不变，却把相机姿态整体转离输入先验，随后软先验 BA 会发生振荡。
    double maxTriangleArea2 = 0.0;
    for (size_t i = 0; i < inputCenters.size(); ++i)
    {
        for (size_t j = i + 1; j < inputCenters.size(); ++j)
        {
            const std::array<double, 3> first{{
                inputCenters[j][0] - inputCenters[i][0],
                inputCenters[j][1] - inputCenters[i][1],
                inputCenters[j][2] - inputCenters[i][2],
            }};
            for (size_t k = j + 1; k < inputCenters.size(); ++k)
            {
                const std::array<double, 3> second{{
                    inputCenters[k][0] - inputCenters[i][0],
                    inputCenters[k][1] - inputCenters[i][1],
                    inputCenters[k][2] - inputCenters[i][2],
                }};
                const std::array<double, 3> cross{{
                    first[1] * second[2] - first[2] * second[1],
                    first[2] * second[0] - first[0] * second[2],
                    first[0] * second[1] - first[1] * second[0],
                }};
                maxTriangleArea2 = std::max(
                    maxTriangleArea2,
                    std::sqrt(cross[0] * cross[0] +
                              cross[1] * cross[1] +
                              cross[2] * cross[2]));
            }
        }
    }
    const double layoutAreaTolerance =
        std::max(1e-12, inputExtent * inputExtent * 1e-6);
    if (maxTriangleArea2 <= layoutAreaTolerance)
    {
        Logger::instance()->warnf(
            "[SFM] Known-pose Sim3 alignment skipped: camera-center layout is collinear "
            "(area2=%.9e tolerance=%.9e)",
            maxTriangleArea2,
            layoutAreaTolerance);
        return;
    }

    const SimilarityTransform3d transform = estimateRobustCameraCenterSimilarity(currentCenters, inputCenters);
    if (!transform.valid || transform.inlierCount < 3)
    {
        Logger::instance()->warnf("[SFM] Known-pose Sim3 alignment skipped: insufficient robust camera-center inliers");
        return;
    }
    if (!(transform.rmse + noOpTolerance < directRmse))
    {
        Logger::instance()->infof(
            "[SFM] Known-pose Sim3 alignment skipped: no meaningful center improvement "
            "(before=%.9f after=%.9f tolerance=%.9f)",
            directRmse,
            transform.rmse,
            noOpTolerance);
        return;
    }

    for (ImageId imageId : _reconstruction->registeredImageIds())
    {
        if (!_reconstruction->hasCamera(imageId))
        {
            continue;
        }
        Camera &camera = _reconstruction->camera(imageId);
        camera.setPose(multiplyRotation(transform.rotation, camera.cameraToWorldRotation()),
                       transformPoint(transform, camera.cameraCenter()));
    }

    for (Point3DId pointId : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(pointId))
        {
            continue;
        }
        _reconstruction->point3D(pointId).xyz =
            transformPoint(transform, _reconstruction->point3D(pointId).xyz);
    }

    for (size_t i = 0; i < imageIds.size(); ++i)
    {
        if (_reconstruction->hasCamera(imageIds[i]))
        {
            (*baCameras)[i] = _reconstruction->camera(imageIds[i]);
        }
    }

    Logger::instance()->infof(
        "[SFM] Known-pose Sim3/RANSAC alignment before BA: cameras=%zu inliers=%d scale=%.9f rmse=%.6f",
        currentCenters.size(),
        transform.inlierCount,
        transform.scale,
        transform.rmse);
}

void IncrementalSfm::refineKnownCameraPosesWithPnp()
{
    if (!_sfmOptions.useKnownCameraPoses ||
        !_sfmOptions.refineKnownCameraPoseWithSoftPrior ||
        !_reconstruction)
    {
        return;
    }

    const std::vector<ImageId> imageIds = _reconstruction->registeredImageIds();

    // 输入相机和相机中心范围在整轮 PnP 中保持不变。一次加载可避免 .tsai
    // 路径在逐影像循环中被重复打开，并将原来的 O(N^2) 相机读取降为 O(N)。
    std::unordered_map<ImageId, Camera> inputCameras;
    inputCameras.reserve(imageIds.size());
    std::vector<std::array<double, 3>> inputCenters;
    inputCenters.reserve(imageIds.size());
    for (ImageId imageId : imageIds)
    {
        Camera inputCamera;
        if (getCamera(imageId, inputCamera))
        {
            inputCenters.push_back(inputCamera.cameraCenter());
            inputCameras.emplace(imageId, std::move(inputCamera));
        }
    }
    const double inputExtent = centerExtent(inputCenters);
    const double maxAcceptedMove = std::max(0.5, inputExtent * 0.05);

    struct PnpCorrespondences
    {
        std::vector<std::array<double, 3>> worldPoints;
        std::vector<std::array<double, 2>> imagePoints;
    };

    // 倒排一次全部轨迹，替代“每张影像扫描全部三维点”的 O(N * points)
    // 数据收集。轨迹理论上每幅影像只有一个观测，这里仍保留首个有效观测语义。
    std::unordered_map<ImageId, PnpCorrespondences> correspondencesByImage;
    correspondencesByImage.reserve(imageIds.size());
    for (ImageId imageId : imageIds)
    {
        if (_reconstruction->hasCamera(imageId) && _reconstruction->hasImage(imageId))
        {
            correspondencesByImage.try_emplace(imageId);
        }
    }

    for (Point3DId pointId : _reconstruction->allPoint3DIds())
    {
        if (!_reconstruction->hasPoint3D(pointId))
        {
            continue;
        }

        const ScenePoint3D &point = _reconstruction->point3D(pointId);
        for (std::size_t elementIndex = 0; elementIndex < point.track.elements.size(); ++elementIndex)
        {
            const TrackElement &element = point.track.elements[elementIndex];
            const auto correspondencesIt = correspondencesByImage.find(element.imageId);
            if (correspondencesIt == correspondencesByImage.end())
            {
                continue;
            }

            const ImageData &image = _reconstruction->image(element.imageId);
            const bool alreadyIndexed = std::any_of(
                point.track.elements.begin(),
                point.track.elements.begin() + static_cast<std::ptrdiff_t>(elementIndex),
                [&](const TrackElement &previous)
                {
                    return previous.imageId == element.imageId &&
                        previous.featureIdx < image.keypoints.size();
                });
            if (element.featureIdx >= image.keypoints.size() || alreadyIndexed)
            {
                continue;
            }

            const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
            PnpCorrespondences &correspondences = correspondencesIt->second;
            correspondences.worldPoints.push_back(point.xyz);
            correspondences.imagePoints.push_back(
                {{static_cast<double>(keypoint.x), static_cast<double>(keypoint.y)}});
        }
    }

    for (ImageId imageId : imageIds)
    {
        const auto correspondencesIt = correspondencesByImage.find(imageId);
        if (correspondencesIt == correspondencesByImage.end())
        {
            continue;
        }

        const std::vector<std::array<double, 3>> &worldPoints = correspondencesIt->second.worldPoints;
        const std::vector<std::array<double, 2>> &imagePoints = correspondencesIt->second.imagePoints;

        if (static_cast<int>(worldPoints.size()) < std::max(6, _sfmOptions.pnpOptions.minNumInliers))
        {
            continue;
        }

        PnpOptions pnpOptions = _sfmOptions.pnpOptions;
        pnpOptions.maxReprojError = std::max(pnpOptions.maxReprojError, _sfmOptions.filterMaxReprojError * 2.0);
        pnpOptions.minNumInliers = std::min(pnpOptions.minNumInliers, static_cast<int>(worldPoints.size()));
        pnpOptions.minInlierRatio = std::min(pnpOptions.minInlierRatio, 0.10);

        const Camera before = _reconstruction->camera(imageId);
        const PnpResult pnp = PnpSolver::solveWithCamera(worldPoints, imagePoints, before, pnpOptions);
        if (!pnp.success)
        {
            continue;
        }

        Camera candidate = before;
        candidate.setPose(pnp.R, pnp.C);
        const auto inputCameraIt = inputCameras.find(imageId);
        if (inputCameraIt != inputCameras.end() &&
            pointDistance(candidate.cameraCenter(), inputCameraIt->second.cameraCenter()) > maxAcceptedMove)
        {
            Logger::instance()->warnf(
                "[SFM] Known-pose PnP refinement rejected for image %u: move from prior %.3f > %.3f",
                imageId,
                pointDistance(candidate.cameraCenter(), inputCameraIt->second.cameraCenter()),
                maxAcceptedMove);
            continue;
        }

        _reconstruction->camera(imageId) = candidate;
        Logger::instance()->infof("[SFM] Known-pose PnP refinement accepted: image=%u inliers=%d/%zu",
                                  imageId,
                                  pnp.numInliers,
                                  worldPoints.size());
    }
}

// ============================================================
// 内部：选择初始像对
// ============================================================


} // namespace xjw
