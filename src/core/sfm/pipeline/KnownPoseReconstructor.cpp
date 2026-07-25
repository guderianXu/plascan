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

    Triangulator triangulator(*_reconstruction, _correspondenceGraph);
    int createdPoints = 0;
    int continuedObservations = 0;
    int completedObservations = 0;
    int inputMultiViewTrackCount = 0;
    int createdLongTrackCount = 0;
    int longTrackTwoViewOnlyCount = 0;

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

        Triangulator baTriangulator(*_reconstruction, _correspondenceGraph);
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

    const SimilarityTransform3d transform = estimateRobustCameraCenterSimilarity(currentCenters, inputCenters);
    if (!transform.valid || transform.inlierCount < 3)
    {
        Logger::instance()->warnf("[SFM] Known-pose Sim3 alignment skipped: insufficient robust camera-center inliers");
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
    for (ImageId imageId : imageIds)
    {
        if (!_reconstruction->hasCamera(imageId) || !_reconstruction->hasImage(imageId))
        {
            continue;
        }

        std::vector<std::array<double, 3>> worldPoints;
        std::vector<std::array<double, 2>> imagePoints;
        const ImageData &image = _reconstruction->image(imageId);
        for (Point3DId pointId : _reconstruction->allPoint3DIds())
        {
            if (!_reconstruction->hasPoint3D(pointId))
            {
                continue;
            }

            const ScenePoint3D &point = _reconstruction->point3D(pointId);
            for (const TrackElement &element : point.track.elements)
            {
                if (element.imageId != imageId || element.featureIdx >= image.keypoints.size())
                {
                    continue;
                }

                const FeatureKeypoint &keypoint = image.keypoints[element.featureIdx];
                worldPoints.push_back(point.xyz);
                imagePoints.push_back({{static_cast<double>(keypoint.x), static_cast<double>(keypoint.y)}});
                break;
            }
        }

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

        Camera inputCamera;
        const double inputExtent = [&]()
        {
            std::vector<std::array<double, 3>> centers;
            for (ImageId id : imageIds)
            {
                Camera cam;
                if (getCamera(id, cam))
                {
                    centers.push_back(cam.cameraCenter());
                }
            }
            return centerExtent(centers);
        }();
        const double maxAcceptedMove = std::max(0.5, inputExtent * 0.05);
        Camera candidate = before;
        candidate.setPose(pnp.R, pnp.C);
        if (getCamera(imageId, inputCamera) &&
            pointDistance(candidate.cameraCenter(), inputCamera.cameraCenter()) > maxAcceptedMove)
        {
            Logger::instance()->warnf(
                "[SFM] Known-pose PnP refinement rejected for image %u: move from prior %.3f > %.3f",
                imageId,
                pointDistance(candidate.cameraCenter(), inputCamera.cameraCenter()),
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
