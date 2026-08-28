#include "ProjectSessionFacade.h"

#include "ProjectTiePointResultService.h"
#include "project/ProjectSessionModel.h"

namespace xjw::gui::project
{

ProjectSessionFacade::ProjectSessionFacade(ProjectData *projectData)
    : _projectData(projectData)
{
}

bool ProjectSessionFacade::isDirty() const
{
    return _projectData && _projectData->hasProject() && _projectData->isDirty();
}

QString ProjectSessionFacade::projectPath() const
{
    return _projectData ? _projectData->currentProjectPath() : QString();
}

QString ProjectSessionFacade::activeChunkId() const
{
    return _projectData ? _projectData->activeChunkId() : QString();
}

QJsonObject ProjectSessionFacade::metadata() const
{
    if (!_projectData)
    {
        return {};
    }
    return ProjectTiePointResultService::metadataWithCurrentOnly(
        _projectData->metadataIncludingResults(), projectPath());
}

QJsonObject ProjectSessionFacade::coreMetadata() const
{
    return _projectData ? _projectData->coreFilesMeta() : QJsonObject();
}

QStringList ProjectSessionFacade::imagesByCategory(const QString &category) const
{
    return _projectData ? _projectData->getImagesByCategory(category) : QStringList();
}

QStringList ProjectSessionFacade::allImages() const
{
    return _projectData ? _projectData->getAllImages() : QStringList();
}

QString ProjectSessionFacade::matchFile(const QString &firstImage,
                                        const QString &secondImage) const
{
    return _projectData ? _projectData->findMatchFile(firstImage, secondImage) : QString();
}

QJsonObject ProjectSessionFacade::loadUiSettings() const
{
    return _projectData ? _projectData->loadUiSettings() : QJsonObject();
}

void ProjectSessionFacade::saveUiSettings(const QJsonObject &settings) const
{
    if (_projectData)
    {
        _projectData->saveUiSettings(settings);
    }
}

void ProjectSessionFacade::markWorkspaceDirty() const
{
    if (_projectData)
    {
        _projectData->markWorkspaceDirty();
    }
}

void ProjectSessionFacade::discardTemporaryMetadata() const
{
    if (_projectData)
    {
        _projectData->clearTemporaryMetadata();
    }
}

bool ProjectSessionFacade::requireProjectData(int *updatedCount,
                                              int *clearedCount,
                                              QString *errorMessage) const
{
    if (_projectData)
    {
        return true;
    }
    if (updatedCount)
    {
        *updatedCount = 0;
    }
    if (clearedCount)
    {
        *clearedCount = 0;
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("ProjectData 未初始化");
    }
    return false;
}

bool ProjectSessionFacade::setImageCameras(
    const QMap<QString, QJsonObject> &cameras,
    int *updatedCount,
    QString *errorMessage) const
{
    return requireProjectData(updatedCount, nullptr, errorMessage)
        && _projectData->setImageCameras(cameras, updatedCount, errorMessage);
}

bool ProjectSessionFacade::replaceImageCameras(
    const QStringList &targetImagePaths,
    const QMap<QString, QJsonObject> &cameras,
    int *updatedCount,
    int *clearedCount,
    QString *errorMessage) const
{
    return requireProjectData(updatedCount, clearedCount, errorMessage)
        && _projectData->replaceImageCameras(
            targetImagePaths, cameras, updatedCount, clearedCount, errorMessage);
}

bool ProjectSessionFacade::clearImageCameras(const QStringList &imagePaths,
                                             int *updatedCount,
                                             QString *errorMessage) const
{
    return requireProjectData(updatedCount, nullptr, errorMessage)
        && _projectData->clearImageCameras(imagePaths, updatedCount, errorMessage);
}

bool ProjectSessionFacade::appendIntersectionResult(
    const QJsonObject &result,
    QString *errorMessage) const
{
    return _projectData && _projectData->appendIntersectionResult(result, errorMessage);
}

QJsonArray ProjectSessionFacade::intersectionResults() const
{
    return _projectData ? _projectData->getIntersectionResults() : QJsonArray();
}

} // namespace xjw::gui::project
