#include "ProjectBundleAdjustExecution.h"

#include "project/ProjectChunkStore.h"
#include "project/PortableProjectFormat.h"
#include "project/ProjectIO.h"
#include "ProjectReferenceTerrainBa.h"
#include "io/MarkerSetStore.h"

#include <QFileInfo>
#include <QJsonArray>

namespace xjw::gui::project {

namespace {

QJsonObject loadBundleAdjustMeta(const QJsonObject &coreData, const QString &plascanPath)
{
    QJsonObject meta = coreData;

    ProjectChunkStore chunkStore(plascanPath);
    QString errorMessage;
    if (!chunkStore.ensureLayout(&errorMessage))
    {
        return meta;
    }

    QJsonObject resultsObject;
    chunkStore.readDefaultChunkSection(
        QString::fromLatin1(
            xjw::common::project::PortableProjectFormat::
                ProjectResultsSection),
        &resultsObject,
        &errorMessage);
    if (!errorMessage.isEmpty())
    {
        return meta;
    }

    for (auto it = resultsObject.constBegin(); it != resultsObject.constEnd(); ++it)
    {
        if (it.key() != QLatin1String("images"))
        {
            meta.insert(it.key(), it.value());
        }
    }
    return meta;
}

} // namespace

BundleAdjustExecutionResult runBundleAdjustExecution(const QJsonObject &coreData,
                                                     const QString &plascanPath,
                                                     const QStringList &selectedImages,
                                                     int minMatches,
                                                     xjw::gui::BaServiceOptions options)
{
    BundleAdjustExecutionResult result;

    const QJsonObject meta = loadBundleAdjustMeta(coreData, plascanPath);

    xjw::core::project::MarkerBaInput markerInput;
    control_points::MarkerSet markerSet;
    const QString markerSetPath = xjw::common::project::ProjectIO::markerSetPath(plascanPath);
    if (QFileInfo::exists(markerSetPath))
    {
        const control_points::MarkerSetIoResult loaded =
            control_points::MarkerSetStore(markerSetPath).load();
        if (!loaded.ok)
        {
            result.serviceResult.errorMessage = QStringLiteral("读取标记 sidecar 失败: %1")
                                                    .arg(loaded.error);
            return result;
        }
        markerSet = loaded.markerSet;
        markerInput.markerSet = &markerSet;
        for (const QJsonValue &value : meta.value(QStringLiteral("images")).toArray())
        {
            const QJsonObject image = value.toObject();
            const QString imageId = image.value(QStringLiteral("image_uuid")).toString().trimmed();
            const QString imagePath = image.value(QStringLiteral("path")).toString().trimmed();
            if (!imageId.isEmpty() && !imagePath.isEmpty())
            {
                markerInput.imagePathById.insert(imageId, imagePath);
            }
        }
    }

    BaInputBuildResult baInput;
    result.buildStatus = buildBaInputFromMeta(
        meta, selectedImages, minMatches, &baInput,
        markerInput.markerSet ? &markerInput : nullptr);
    if (result.buildStatus != BaInputBuildStatus::Ok)
    {
        return result;
    }

    result.beforeCamMeta = baInput.beforeCamMeta;
    options.imagePathByIndex = baInput.imagePathByIndex;
    options.beforeCamMeta = baInput.beforeCamMeta;
    if (options.enablePlanetaryLaserRangeConstraints)
    {
        QString aliasError;
        if (!xjw::gui::mergePlanetaryLaserProjectImageAliases(
                meta,
                baInput.imagePathByIndex,
                &options.planetaryLaserImageAliasesByCameraIndex,
                &aliasError))
        {
            result.serviceResult.errorMessage = aliasError;
            return result;
        }
    }
    if (baInput.surveyControlTrackCount > 0 || baInput.markerControlPointConstraintCount > 0)
    {
        options.baOpt.enableControlPointConstraints = true;
        options.baOpt.backend = xjw::BABackend::Auto;
    }
    if (!baInput.scaleBarConstraints.empty())
    {
        options.baOpt.enableScaleBarConstraints = true;
        options.baOpt.scaleBarConstraints = baInput.scaleBarConstraints;
    }
    for (const BaInputBuildResult::MarkerTrackBinding &binding : baInput.markerTrackBindings)
    {
        xjw::gui::BaServiceOptions::MarkerTrackQualityInput quality;
        quality.markerId = binding.markerId;
        quality.role = binding.role;
        quality.trackIndex = binding.trackIndex;
        quality.referencePoint = binding.referencePoint;
        quality.sigma = binding.sigma;
        quality.usedAsConstraint = binding.usedAsConstraint;
        options.markerTrackQualityInputs.push_back(quality);
    }
    for (const BaInputBuildResult::MarkerScaleBarBinding &binding : baInput.markerScaleBarBindings)
    {
        xjw::gui::BaServiceOptions::MarkerScaleBarQualityInput quality;
        quality.scaleBarId = binding.scaleBarId;
        quality.role = binding.role;
        quality.trackIndexA = binding.trackIndexA;
        quality.trackIndexB = binding.trackIndexB;
        quality.measuredDistance = binding.measuredDistance;
        options.markerScaleBarQualityInputs.push_back(quality);
    }

    const ReferenceTerrainBaApplyResult terrainPriorResult =
        applyReferenceTerrainPriorToBundleAdjust(&baInput.tracks, &options);
    if (!terrainPriorResult.success)
    {
        result.serviceResult.errorMessage = terrainPriorResult.errorMessage;
        return result;
    }

    result.serviceResult = xjw::gui::BundleAdjustService::run(baInput.cameras,
                                                              baInput.tracks,
                                                              options);
    return result;
}

} // namespace xjw::gui::project
