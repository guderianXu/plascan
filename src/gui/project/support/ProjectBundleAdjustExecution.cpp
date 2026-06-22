#include "ProjectBundleAdjustExecution.h"

#include "PlascanArchive.h"
#include "ProjectFilesManager.h"
#include "ProjectReferenceTerrainBa.h"

#include <QJsonDocument>

namespace xjw::gui::project {

namespace {

QJsonObject loadBundleAdjustMeta(const QJsonObject &coreData, const QString &plascanPath)
{
    QJsonObject meta = coreData;

    PlascanArchive archive(plascanPath);
    if (!archive.isValid())
    {
        return meta;
    }

    QString errorMessage;
    QByteArray resultsData = archive.readEntry(ProjectFilesManager::kArchiveResultsFile, &errorMessage);
    if (resultsData.isEmpty())
    {
        resultsData = archive.readEntry(QStringLiteral("project_files.json"), &errorMessage);
    }
    if (resultsData.isEmpty())
    {
        return meta;
    }

    const QJsonObject resultsObject = QJsonDocument::fromJson(resultsData).object();
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

    BaInputBuildResult baInput;
    result.buildStatus = buildBaInputFromMeta(meta, selectedImages, minMatches, &baInput);
    if (result.buildStatus != BaInputBuildStatus::Ok)
    {
        return result;
    }

    result.beforeCamMeta = baInput.beforeCamMeta;
    options.imagePathByIndex = baInput.imagePathByIndex;
    options.beforeCamMeta = baInput.beforeCamMeta;
    if (baInput.surveyControlTrackCount > 0)
    {
        options.baOpt.enableControlPointConstraints = true;
    }
    if (!baInput.scaleBarConstraints.empty())
    {
        options.baOpt.enableScaleBarConstraints = true;
        options.baOpt.scaleBarConstraints = baInput.scaleBarConstraints;
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
