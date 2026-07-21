#include "reconstruction/SfmAttemptRunner.h"
#include "reconstruction/MarkerPriorLoader.h"

#include "io/PathIO.h"
#include "project/ProjectCameraIO.h"
#include "project/ProjectCommonUtils.h"
#include "project/ProjectMetadata.h"
#include "pipeline/IncrementalSfm.h"

#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSize>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace xjw::aerial_triangulation
{
namespace
{

quint64 pairKey(ImageId imageA, ImageId imageB)
{
    const quint64 first = std::min(imageA, imageB);
    const quint64 second = std::max(imageA, imageB);
    return (first << 32U) | second;
}

bool fail(const QString &message, QString *errorMessage)
{
    if (errorMessage)
    {
        *errorMessage = message;
    }
    return false;
}

int selectedImageIndex(const QString &path, const QStringList &selectedImages)
{
    for (int index = 0; index < selectedImages.size(); ++index)
    {
        if (xjw::common::project::pathTokenMatchesImage(path, selectedImages.at(index)))
        {
            return index;
        }
    }
    return -1;
}

struct ParsedObservation
{
    ImageId imageId = kInvalidImageId;
    FeatureIdx featureIndex = kInvalidFeatureIdx;
};

struct SfmQualityPreset
{
    int initMinMatches = 25;
    int initMinInliers = 5;
    int localBaInterval = 3;
    int globalBaInterval = 10;
};

SfmQualityPreset presetForQuality(int quality)
{
    switch (std::clamp(quality, 0, 3))
    {
    case 0:
        return {15, 5, 5, 15};
    case 1:
        return {20, 5, 4, 12};
    case 2:
        return {25, 5, 3, 10};
    case 3:
    default:
        return {30, 5, 3, 10};
    }
}

bool cameraMetadataHasUsablePose(const QJsonObject &cameraObject)
{
    return !cameraObject.isEmpty() &&
           !cameraObject.value(QStringLiteral("pose_initialized_as_identity")).toBool(false);
}

Camera cameraWithIdentityPose(Camera camera)
{
    camera.setPose({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0},
                   {0.0, 0.0, 0.0});
    return camera;
}

QSize imageSizeForInput(const QString &imagePath,
                        const std::vector<FeatureKeypoint> &keypoints)
{
    QImageReader reader(imagePath);
    const QSize decodedSize = reader.size();
    if (decodedSize.isValid())
    {
        return decodedSize;
    }

    float maxX = 0.0f;
    float maxY = 0.0f;
    for (const FeatureKeypoint &keypoint : keypoints)
    {
        maxX = std::max(maxX, keypoint.x);
        maxY = std::max(maxY, keypoint.y);
    }
    if (maxX > 0.0f && maxY > 0.0f)
    {
        return QSize(std::max(2, static_cast<int>(std::ceil(maxX)) + 1),
                     std::max(2, static_cast<int>(std::ceil(maxY)) + 1));
    }
    return QSize(1920, 1080);
}

void configureSfmOptions(const PreparedAerialTriangulationInput &input,
                         IncrementalSfmOptions *options)
{
    const SfmQualityPreset preset = presetForQuality(input.quality);
    options->initMinNumMatches = preset.initMinMatches;
    options->initMinNumInliers = preset.initMinInliers;
    options->localBAInterval = preset.localBaInterval;
    options->globalBAInterval = preset.globalBaInterval;
    options->baOptions.cancelFlag = input.cancelFlag;
    options->baOptions.numThreads = std::max(1, input.threads);

    if (input.useInitialPairHint)
    {
        options->autoSelectInitPair = false;
        options->initImageId1 = input.initialImageId1;
        options->initImageId2 = input.initialImageId2;
    }

    if (input.device.trimmed().compare(QStringLiteral("cpu"), Qt::CaseInsensitive) != 0)
    {
        options->baOptions.backend = BABackend::Auto;
        options->baOptions.minNativeCudaCameras = 50;
        options->baOptions.minNativeCudaObservations = 500000;
        options->baOptions.nativeCudaMaxPcgIterations = 100;
        options->baOptions.nativeCudaPcgTolerance = 1e-4;
        options->baOptions.minCeresCudaObservations = 500000;
        options->baOptions.minCeresCpuObservations = 50000;
        options->baOptions.enableBackendQualityGate = true;
        options->baOptions.maxAcceptedRmsGrowth = 1.25;
        options->baOptions.minAcceptedValidTrackRatio = 0.60;
        options->baOptions.compareAutoBackendWithLegacy = true;
        options->baOptions.allowBackendFallback = true;
    }

    if (input.quality >= 2)
    {
        options->filterMaxReprojError = 1.5;
        options->filterMinTriAngle = 2.0;
        options->iterativeBARounds = 4;
    }

    options->enforceSequencePoseConsistency = input.enforceSequencePoseConsistency;
    options->sequenceLoopClosure = input.sequenceLoopClosure;
    if (input.adaptiveCameraModelFitting)
    {
        options->baOptions.refineSharedFocalLength = true;
        options->baOptions.minSharedFocalScale = 0.65;
        options->baOptions.maxSharedFocalScale = 1.55;
        options->baOptions.maxSharedFocalStepScale = 1.12;
        options->baOptions.maxSharedFocalIterations = 6;
    }
}

} // namespace

SfmAttemptExecutionResult SfmAttemptRunner::run(
    const PreparedAerialTriangulationInput &input) const
{
    SfmAttemptExecutionResult execution;
    if (!readTiePointGraph(input.tiePointPath,
                           input.images,
                           &execution.graph,
                           &execution.result.errorMessage))
    {
        execution.result.summary = execution.result.errorMessage;
        return execution;
    }

    if (input.cancelFlag && input.cancelFlag->load())
    {
        execution.result.errorMessage = QStringLiteral("用户取消");
        execution.result.summary = execution.result.errorMessage;
        return execution;
    }

    IncrementalSfmOptions sfmOptions;
    configureSfmOptions(input, &sfmOptions);

    const bool hasCompleteCameraFiles = input.cameraPaths.size() == input.images.size() &&
        std::all_of(input.cameraPaths.cbegin(), input.cameraPaths.cend(), [](const QString &path)
        {
            return !path.trimmed().isEmpty() && QFileInfo::exists(path);
        });

    QMap<QString, QJsonObject> projectCameraByPath;
    QSet<QString> projectPosePaths;
    if ((input.useProjectCameraIntrinsics || input.useProjectCameraPoses) &&
        !input.projectMeta.isEmpty())
    {
        const QMap<QString, QJsonObject> imageMetadata =
            xjw::common::project::projectImageMetaByPath(input.projectMeta, true);
        for (auto it = imageMetadata.cbegin(); it != imageMetadata.cend(); ++it)
        {
            const QJsonObject cameraObject = it.value().value(QStringLiteral("camera")).toObject();
            Camera camera;
            if (!cameraObject.isEmpty() &&
                xjw::common::project::cameraFromJson(cameraObject, &camera) &&
                camera.isValid())
            {
                projectCameraByPath.insert(it.key(), cameraObject);
                if (input.useProjectCameraPoses && cameraMetadataHasUsablePose(cameraObject))
                {
                    projectPosePaths.insert(it.key());
                }
            }
        }
    }

    bool hasCompleteProjectPoseCameras = !input.images.isEmpty();
    for (const QString &imagePath : input.images)
    {
        if (!projectPosePaths.contains(xjw::common::project::normalizePath(imagePath)))
        {
            hasCompleteProjectPoseCameras = false;
            break;
        }
    }
    sfmOptions.useKnownCameraPoses = hasCompleteCameraFiles || hasCompleteProjectPoseCameras;
    if (sfmOptions.useKnownCameraPoses)
    {
        sfmOptions.baOptions.refineSharedFocalLength = false;
    }
    else
    {
        sfmOptions.pnpOptions.allowRelaxedInlierRatio = true;
        sfmOptions.pnpOptions.minInlierRatio =
            std::max(sfmOptions.pnpOptions.minInlierRatio, 0.25);
        sfmOptions.pnpOptions.minNumInliers =
            std::max(sfmOptions.pnpOptions.minNumInliers, 12);
        sfmOptions.pnpOptions.relaxedMinInlierRatio =
            std::min(sfmOptions.pnpOptions.relaxedMinInlierRatio, 0.05);
        sfmOptions.pnpOptions.relaxedMinNumInliers =
            std::max(sfmOptions.pnpOptions.relaxedMinNumInliers, 24);
    }

    IncrementalSfm sfm(sfmOptions);
    QMap<QString, ImageId> imageIdByPath;
    for (int index = 0; index < input.images.size(); ++index)
    {
        const ImageId imageId = static_cast<ImageId>(index);
        const QString &imagePath = input.images.at(index);
        imageIdByPath.insert(xjw::common::project::normalizePath(imagePath), imageId);
        const std::vector<FeatureKeypoint> keypoints =
            execution.graph.keypointsByImage.value(imageId);

        if (hasCompleteCameraFiles)
        {
            sfm.addImage(imageId,
                         xjw::common::io::toUtf8Path(imagePath),
                         xjw::common::io::toUtf8Path(input.cameraPaths.at(index)),
                         keypoints);
            continue;
        }

        const QString normalizedPath = xjw::common::project::normalizePath(imagePath);
        const auto projectCamera = projectCameraByPath.constFind(normalizedPath);
        if (input.useProjectCameraIntrinsics && projectCamera != projectCameraByPath.cend())
        {
            Camera camera;
            if (xjw::common::project::cameraFromJson(projectCamera.value(), &camera) &&
                camera.isValid())
            {
                if (!input.useProjectCameraPoses)
                {
                    // 重置对齐时只复用内参，外参重新由相对定向/增量注册估计。
                    camera = cameraWithIdentityPose(camera);
                }
                sfm.addImageWithCamera(imageId,
                                       xjw::common::io::toUtf8Path(imagePath),
                                       camera,
                                       keypoints);
                continue;
            }
        }

        const QSize imageSize = imageSizeForInput(imagePath, keypoints);
        const double focal = std::max(imageSize.width(), imageSize.height()) *
            std::max(0.1, input.estimatedFocalScale);
        Camera camera;
        camera.setIntrinsics(focal,
                             focal,
                             imageSize.width() * 0.5,
                             imageSize.height() * 0.5);
        sfm.addImageWithCamera(imageId,
                               xjw::common::io::toUtf8Path(imagePath),
                               camera,
                               keypoints);
    }

    const MarkerPriorLoadResult markerPriors = MarkerPriorLoader::load(
        input.markerSetPath, input.projectMeta, imageIdByPath);
    if (!markerPriors.ok)
    {
        execution.result.errorMessage = markerPriors.errorMessage;
        execution.result.summary = markerPriors.errorMessage;
        return execution;
    }
    for (const control_points::PriorTrack &track : markerPriors.tracks)
    {
        sfm.addPriorTrack(track);
    }
    for (const control_points::PriorScaleBar &scaleBar : markerPriors.scaleBars)
    {
        sfm.addPriorScaleBar(scaleBar);
    }

    for (const PreparedTiePointMatchPair &pair : execution.graph.matchPairs)
    {
        sfm.addMatches(pair.imageA, pair.imageB, pair.matches);
    }

    const IncrementalSfmResult sfmResult = sfm.run(
        [&input](int registered, int total, const std::string &message)
        {
            if (input.cancelFlag && input.cancelFlag->load())
            {
                return false;
            }
            if (input.progressFn)
            {
                const int percent = total > 0
                    ? std::clamp(static_cast<int>(100.0 * registered / total), 0, 100)
                    : 0;
                input.progressFn(QString::fromStdString(message), percent);
            }
            return true;
        });

    execution.reconstruction = sfmResult.reconstruction;
    execution.result.success = sfmResult.success;
    execution.result.numRegisteredImages = sfmResult.numRegisteredImages;
    execution.result.numPoints3D = sfmResult.numPoints3D;
    execution.result.meanReprojError = sfmResult.meanReprojError;
    execution.result.baRmsBefore = sfmResult.baRmsBefore;
    execution.result.baRmsAfter = sfmResult.baRmsAfter;
    execution.result.baTracksTotal = sfmResult.baTracksTotal;
    execution.result.baTracksOptimized = sfmResult.baTracksOptimized;
    execution.result.baTracksFiltered = sfmResult.baTracksFiltered;
    execution.result.summary = QString::fromStdString(sfmResult.summary);

    QJsonObject diagnostics;
    diagnostics.insert(QStringLiteral("selected_initial_pair"),
                       QJsonArray{static_cast<int>(sfmResult.selectedInitialImageId1),
                                  static_cast<int>(sfmResult.selectedInitialImageId2)});
    diagnostics.insert(QStringLiteral("prior_tracks_accepted"), sfmResult.priorTracksAccepted);
    diagnostics.insert(QStringLiteral("prior_tracks_rejected"), sfmResult.priorTracksRejected);
    diagnostics.insert(QStringLiteral("marker_prior_tracks_loaded"),
                       static_cast<int>(markerPriors.tracks.size()));
    diagnostics.insert(QStringLiteral("marker_prior_scale_bars_loaded"),
                       static_cast<int>(markerPriors.scaleBars.size()));
    diagnostics.insert(QStringLiteral("control_network_applied"), sfmResult.controlNetworkApplied);
    diagnostics.insert(QStringLiteral("control_point_constraints"),
                       sfmResult.controlPointConstraintCount);
    execution.result.sfmDiagnostics = diagnostics;

    if (!sfmResult.success)
    {
        execution.result.errorMessage = execution.result.summary.isEmpty()
            ? QStringLiteral("SfM 重建失败")
            : execution.result.summary;
        return execution;
    }

    if (execution.reconstruction)
    {
        for (int index = 0; index < input.images.size(); ++index)
        {
            const ImageId imageId = static_cast<ImageId>(index);
            if (execution.reconstruction->isRegistered(imageId))
            {
                execution.result.pendingCamUpdates.insert(
                    xjw::common::project::normalizePath(input.images.at(index)),
                    xjw::common::project::cameraToJson(
                        execution.reconstruction->camera(imageId)));
            }
        }
    }
    return execution;
}

bool SfmAttemptRunner::readTiePointGraph(const QString &tiePointPath,
                                         const QStringList &selectedImages,
                                         PreparedTiePointGraph *graph,
                                         QString *errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!graph)
    {
        return fail(QStringLiteral("连接点图输出参数为空"), errorMessage);
    }
    *graph = {};

    QFile file(tiePointPath);
    if (!file.open(QIODevice::ReadOnly))
    {
        return fail(QStringLiteral("无法读取连接点文件: %1").arg(tiePointPath), errorMessage);
    }
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll());
    if (!document.isObject())
    {
        return fail(QStringLiteral("连接点文件不是有效 JSON 对象: %1").arg(tiePointPath), errorMessage);
    }

    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("format")).toString() != QLatin1String("plascan_tie_points") ||
        root.value(QStringLiteral("format_version")).toInt() != 1)
    {
        return fail(QStringLiteral("不支持的连接点文件格式或版本"), errorMessage);
    }
    if (selectedImages.size() < 2)
    {
        return fail(QStringLiteral("SfM 至少需要两张影像"), errorMessage);
    }

    QMap<int, ImageId> selectedIdByPersistedId;
    QSet<int> coveredSelectedIds;
    for (const QJsonValue &value : root.value(QStringLiteral("images")).toArray())
    {
        const QJsonObject imageObject = value.toObject();
        const int persistedId = imageObject.value(QStringLiteral("image_id")).toInt(-1);
        const int selectedId = selectedImageIndex(
            imageObject.value(QStringLiteral("path")).toString(), selectedImages);
        if (persistedId >= 0 && selectedId >= 0)
        {
            selectedIdByPersistedId.insert(persistedId, static_cast<ImageId>(selectedId));
            coveredSelectedIds.insert(selectedId);
        }
    }
    if (coveredSelectedIds.size() != selectedImages.size())
    {
        return fail(QStringLiteral("连接点文件的影像集合与当前空三影像集合不一致"), errorMessage);
    }

    graph->imagePaths.reserve(selectedImages.size());
    for (const QString &path : selectedImages)
    {
        graph->imagePaths.append(xjw::common::project::normalizePath(path));
    }

    QMap<ImageId, QMap<qulonglong, FeatureIdx>> compactIndexByOriginal;
    std::map<quint64, std::size_t> pairPosition;
    std::map<quint64, std::set<std::pair<FeatureIdx, FeatureIdx>>> pairObservations;

    for (const QJsonValue &trackValue : root.value(QStringLiteral("tracks")).toArray())
    {
        const QJsonObject trackObject = trackValue.toObject();
        const float confidence = static_cast<float>(
            std::clamp(trackObject.value(QStringLiteral("confidence")).toDouble(1.0), 0.0, 1.0));
        std::vector<ParsedObservation> observations;

        for (const QJsonValue &observationValue :
             trackObject.value(QStringLiteral("observations")).toArray())
        {
            const QJsonObject observationObject = observationValue.toObject();
            const int persistedImageId = observationObject.value(QStringLiteral("image_id")).toInt(-1);
            const qint64 originalFeatureIndex =
                observationObject.value(QStringLiteral("feature_idx")).toInteger(-1);
            const QJsonArray xy = observationObject.value(QStringLiteral("xy")).toArray();
            if (!selectedIdByPersistedId.contains(persistedImageId) ||
                originalFeatureIndex < 0 || xy.size() < 2)
            {
                continue;
            }

            const double x = xy.at(0).toDouble(std::numeric_limits<double>::quiet_NaN());
            const double y = xy.at(1).toDouble(std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(x) || !std::isfinite(y))
            {
                continue;
            }

            const ImageId imageId = selectedIdByPersistedId.value(persistedImageId);
            QMap<qulonglong, FeatureIdx> &indexMap = compactIndexByOriginal[imageId];
            const qulonglong originalKey = static_cast<qulonglong>(originalFeatureIndex);
            FeatureIdx compactIndex = indexMap.value(originalKey, kInvalidFeatureIdx);
            if (compactIndex == kInvalidFeatureIdx)
            {
                compactIndex = static_cast<FeatureIdx>(graph->keypointsByImage[imageId].size());
                indexMap.insert(originalKey, compactIndex);
                graph->keypointsByImage[imageId].push_back(
                    {static_cast<float>(x), static_cast<float>(y)});
            }
            observations.push_back({imageId, compactIndex});
        }

        std::sort(observations.begin(), observations.end(), [](const ParsedObservation &left,
                                                               const ParsedObservation &right)
        {
            return left.imageId < right.imageId;
        });
        observations.erase(std::unique(observations.begin(), observations.end(),
                                       [](const ParsedObservation &left,
                                          const ParsedObservation &right)
        {
            return left.imageId == right.imageId;
        }), observations.end());
        if (observations.size() < 2)
        {
            continue;
        }

        ++graph->trackCount;
        for (std::size_t first = 0; first + 1 < observations.size(); ++first)
        {
            for (std::size_t second = first + 1; second < observations.size(); ++second)
            {
                const ParsedObservation &observationA = observations[first];
                const ParsedObservation &observationB = observations[second];
                const quint64 key = pairKey(observationA.imageId, observationB.imageId);
                const std::pair<FeatureIdx, FeatureIdx> featurePair{
                    observationA.featureIndex, observationB.featureIndex};
                if (!pairObservations[key].insert(featurePair).second)
                {
                    continue;
                }

                auto position = pairPosition.find(key);
                if (position == pairPosition.end())
                {
                    PreparedTiePointMatchPair pair;
                    pair.imageA = observationA.imageId;
                    pair.imageB = observationB.imageId;
                    graph->matchPairs.push_back(std::move(pair));
                    position = pairPosition.emplace(key, graph->matchPairs.size() - 1).first;
                }
                graph->matchPairs[position->second].matches.push_back(
                    {observationA.featureIndex, observationB.featureIndex, confidence});
            }
        }
    }

    if (graph->trackCount <= 0 || graph->matchPairs.empty())
    {
        return fail(QStringLiteral("连接点文件中没有可用于 SfM 的多视图轨迹"), errorMessage);
    }
    return true;
}

} // namespace xjw::aerial_triangulation
