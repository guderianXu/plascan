#include "ProjectBundleAdjustWorkflow.h"

#include "ProjectData.h"

#include <QDir>

namespace xjw::gui::project {

BundleAdjustCommitResult commitBundleAdjustPreview(ProjectData *projectData,
                                                   const QMap<QString, QJsonObject> &cameraMetaByImage,
                                                   const QJsonObject &baResult)
{
    BundleAdjustCommitResult result;
    if (!projectData)
    {
        result.errorMessage = QStringLiteral("项目未就绪");
        return result;
    }

    if (cameraMetaByImage.isEmpty())
    {
        result.errorMessage = QStringLiteral("没有可应用的平差相机结果");
        return result;
    }

    QString errorMessage;
    if (!projectData->setImageCameras(cameraMetaByImage, &result.updatedCameraCount, &errorMessage))
    {
        result.errorMessage = QStringLiteral("写回相机参数失败: %1").arg(errorMessage);
        return result;
    }

    QJsonObject compactBaResult = baResult;
    compactBaResult.remove(QStringLiteral("point_preview"));
    QString saveWarning;
    if (!projectData->appendBundleAdjustResult(compactBaResult, &saveWarning))
    {
        result.warningMessage = QStringLiteral("保存平差结果失败: %1").arg(saveWarning);
    }

    result.success = true;
    return result;
}

BundleAdjustArtifactsResult finalizeBundleAdjustArtifacts(const QString &assetsDir,
                                                          const QJsonObject &baResult,
                                                          const QStringList &images,
                                                          const QString &reportOutputDir,
                                                          const QString &reportSource,
                                                          const QMap<QString, QJsonObject> &beforeCameras,
                                                          const QMap<QString, QJsonObject> &afterCameras,
                                                          const QString &sparseCloudOutputDir,
                                                          bool useDedicatedFileName)
{
    BundleAdjustArtifactsResult result;

    if (!assetsDir.isEmpty())
    {
        const QJsonObject report = buildBundleAdjustWorkflowReport(baResult,
                                                                   images,
                                                                   reportOutputDir,
                                                                   reportSource,
                                                                   beforeCameras,
                                                                   afterCameras);
        result.reportSaved = writeLatestAndAppendHistoryReport(QDir(assetsDir).filePath(QStringLiteral("reports")),
                                                               QStringLiteral("at_report.json"),
                                                               QStringLiteral("at_report_history.json"),
                                                               report);
        if (!result.reportSaved)
        {
            result.reportWarning = QStringLiteral("保存项目报告失败");
        }
    }

    result.sparseCloudExport = exportBundleAdjustSparseCloud(baResult,
                                                             images,
                                                             sparseCloudOutputDir,
                                                             useDedicatedFileName);
    return result;
}

} // namespace xjw::gui::project