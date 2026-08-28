#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project
{

class ProjectSessionFacade
{
public:
    explicit ProjectSessionFacade(ProjectData *projectData = nullptr);

    bool isDirty() const;
    QString projectPath() const;
    QString activeChunkId() const;
    QJsonObject metadata() const;
    QJsonObject coreMetadata() const;
    QStringList imagesByCategory(const QString &category) const;
    QStringList allImages() const;
    QString matchFile(const QString &firstImage, const QString &secondImage) const;

    QJsonObject loadUiSettings() const;
    void saveUiSettings(const QJsonObject &settings) const;
    void markWorkspaceDirty() const;
    void discardTemporaryMetadata() const;

    bool setImageCameras(const QMap<QString, QJsonObject> &cameras,
                         int *updatedCount,
                         QString *errorMessage) const;
    bool replaceImageCameras(const QStringList &targetImagePaths,
                             const QMap<QString, QJsonObject> &cameras,
                             int *updatedCount,
                             int *clearedCount,
                             QString *errorMessage) const;
    bool clearImageCameras(const QStringList &imagePaths,
                           int *updatedCount,
                           QString *errorMessage) const;

    bool appendIntersectionResult(const QJsonObject &result,
                                  QString *errorMessage) const;
    QJsonArray intersectionResults() const;

private:
    bool requireProjectData(int *updatedCount,
                            int *clearedCount,
                            QString *errorMessage) const;

    ProjectData *_projectData = nullptr;
};

} // namespace xjw::gui::project
