#pragma once

#include <QJsonObject>
#include <QObject>
#include <QString>

class ProjectData;
class ProjectDenseReconstructionManager;
class ProjectManager;
class ProjectModelManager;
class QWidget;

class ProjectModelGenerationWorkflow : public QObject
{
    Q_OBJECT

public:
    explicit ProjectModelGenerationWorkflow(ProjectManager *owner,
                                            ProjectData *project_data,
                                            QWidget *parent_widget,
                                            ProjectDenseReconstructionManager *dense_manager,
                                            ProjectModelManager *model_manager,
                                            QObject *parent = nullptr);

    void start(const QJsonObject &settings);
    bool isRunning() const;
    bool isPreparingDenseCloud() const;
    bool consumeInternalMvsFinished();

signals:
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);

private:
    enum class Stage
    {
        Idle,
        DenseCloud,
        Model
    };

    void startModelStage(const QJsonObject &settings);
    void startDenseStage(const QJsonObject &dense_settings,
                         const QJsonObject &model_settings);
    bool belongsToActiveProject() const;
    void finish(bool success);

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    ProjectDenseReconstructionManager *_denseManager = nullptr;
    ProjectModelManager *_modelManager = nullptr;
    Stage _stage = Stage::Idle;
    QString _projectPath;
    QString _expectedDenseOutputDir;
    QJsonObject _pendingModelSettings;
    bool _internalMvsFinishPending = false;
};
