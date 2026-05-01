#pragma once

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>
#include <QStringList>

#include "ProjectReconstructionManager.h"

class ProjectCameraSetupManager;
class ProjectTerrainProductsManager;

class ProjectTaskDispatcher : public QObject
{
    Q_OBJECT

public:
    explicit ProjectTaskDispatcher(ProjectCameraSetupManager *cameraManager,
                                   ProjectTerrainProductsManager *terrainManager,
                                   ProjectReconstructionManager *reconstructionManager,
                                   QObject *parent = nullptr);

    bool importCameraForImage(const QString &imagePath) const;
    bool importCamerasByFilenameBatch() const;
    bool initializeCamerasFromExifOrDefault(const QJsonObject &settings) const;
    bool initializeCamerasFromIntrinsics(const QJsonObject &settings) const;
    bool initializeCameraPosesWithSFM(const QJsonObject &settings) const;

    void startStereoAndPoint2DemAsync(const QStringList &images,
                                      const QString &outputDir,
                                      int threads,
                                      bool genPointCloud,
                                      double demResolution,
                                      const QString &demType,
                                      const QString &tSrs) const;
    void startFullDemPipelineAsync(const QStringList &images,
                                   const QString &outputDir,
                                   const QJsonObject &pipelineSettings) const;
    void startDemFromDenseCloudAsync(const QString &denseCloudPath,
                                     const QString &outputDir,
                                     double demResolution,
                                     const QString &demType) const;
    void startMapProjectAsync(const QStringList &images,
                              const QString &demPath,
                              const QString &outputPath,
                              double resolution) const;

    void startReconstructionTask(ProjectReconstructionManager::Task task,
                                 const QJsonObject &settings = QJsonObject()) const;

    QJsonArray getAvailableAtResults() const;
    void cancelMvs() const;

private:
    ProjectCameraSetupManager *_cameraManager = nullptr;
    ProjectTerrainProductsManager *_terrainManager = nullptr;
    ProjectReconstructionManager *_reconstructionManager = nullptr;
};
