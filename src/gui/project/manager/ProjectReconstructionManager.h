#pragma once

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>

class QWidget;
class ProjectData;
class ProjectManager;
class ProjectSparseReconstructionManager;
class ProjectDenseReconstructionManager;
class ProjectModelManager;

class ProjectReconstructionManager : public QObject
{
    Q_OBJECT

public:
    enum class Task
    {
        GenerateModel,
        MeshReconstruction,
        TextureMapping,
        Triangulation,
        SparseOutlierRemoval,
        SparseLocalOptimization,
        SparseRefine,
        EstimateDepthMaps,
        FuseDepthMaps,
        GenerateDenseCloud,
        RefineDenseCloud
    };

    explicit ProjectReconstructionManager(ProjectManager *owner,
                                          ProjectData *projectData,
                                          QWidget *parentWidget,
                                          QObject *parent = nullptr);

    QJsonArray getAvailableAtResults() const;
    void startTask(Task task, const QJsonObject &settings = QJsonObject());
    void cancelMvs();

signals:
    void mvsProgressChanged(const QString &stage, int percent);
    void mvsProgressFinished(bool success);
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);
    void atProgressChanged(const QString &stage, int percent);
    void atProgressFinished(bool success);

private:
    ProjectSparseReconstructionManager *m_sparseManager = nullptr;
    ProjectDenseReconstructionManager *m_denseManager = nullptr;
    ProjectModelManager *m_modelManager = nullptr;
};
