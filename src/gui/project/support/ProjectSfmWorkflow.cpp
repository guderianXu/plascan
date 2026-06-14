#include "ProjectSfmWorkflow.h"

#include "ProjectData.h"
#include "ProjectMetadataOperations.h"
#include "SFMService.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::gui::project {

namespace {

QMap<QString, QJsonObject> filterSfmCameraUpdates(const QMap<QString, QJsonObject> &pendingUpdates,
                                                  const QSet<QString> &targetImages,
                                                  const QSet<QString> &existingImages,
                                                  bool overwriteExisting)
{
    QMap<QString, QJsonObject> filteredUpdates;
    for (auto it = pendingUpdates.constBegin(); it != pendingUpdates.constEnd(); ++it)
    {
        const QString normalizedPath = QDir::cleanPath(QFileInfo(it.key()).absoluteFilePath());
        if (!targetImages.contains(normalizedPath))
        {
            continue;
        }
        if (!overwriteExisting && existingImages.contains(normalizedPath))
        {
            continue;
        }
        filteredUpdates.insert(normalizedPath, it.value());
    }
    return filteredUpdates;
}

} // namespace

InitPoseFinalizeResult finalizeInitializedCameraPoses(ProjectData *projectData,
                                                     const xjw::gui::SFMServiceResult &result,
                                                     const QSet<QString> &targetImages,
                                                     const QSet<QString> &existingImages,
                                                     bool overwriteExisting,
                                                     const QStringList &allImages,
                                                     const QString &outputDir)
{
    InitPoseFinalizeResult finalizeResult;
    if (!projectData)
    {
        finalizeResult.errorMessage = QStringLiteral("项目未就绪");
        return finalizeResult;
    }

    const QMap<QString, QJsonObject> filteredUpdates = filterSfmCameraUpdates(result.pendingCamUpdates,
                                                                              targetImages,
                                                                              existingImages,
                                                                              overwriteExisting);
    if (!filteredUpdates.isEmpty())
    {
        QString errorMessage;
        if (!projectData->setImageCameras(filteredUpdates, &finalizeResult.updatedCameraCount, &errorMessage))
        {
            finalizeResult.errorMessage = QStringLiteral("SFM 已完成，但回写相机结果失败: %1").arg(errorMessage);
            return finalizeResult;
        }
    }

    if (!result.sparseCloudPath.isEmpty())
    {
        appendAtResult(projectData,
                       result.sparseCloudPath,
                       result.numPoints3D,
                       allImages,
                       outputDir,
                       result.resultRecordExtra);
    }

    finalizeResult.success = true;
    return finalizeResult;
}

} // namespace xjw::gui::project
