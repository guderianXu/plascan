#pragma once

#include <QObject>
#include <QJsonObject>

class QWidget;
class ProjectData;
class ProjectManager;

class ProjectModelManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectModelManager(ProjectManager *owner,
                                 ProjectData *projectData,
                                 QWidget *parentWidget,
                                 QObject *parent = nullptr);

    void startGenerateModelAsync();
    void startMeshReconstructionAsync(const QJsonObject &settings);
    void startTextureMappingAsync(const QJsonObject &settings);

signals:
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);

private:
    bool ensureProjectOpen(const QString &message,
                           const QString &title) const;
    void finalizeModelGenerationSuccess(const QJsonObject &terrainResult,
                                        const QString &sourceCloudPath,
                                        bool sourceIsDense);

    ProjectManager *m_owner = nullptr;
    ProjectData *m_projectData = nullptr;
    QWidget *m_parentWidget = nullptr;
};
