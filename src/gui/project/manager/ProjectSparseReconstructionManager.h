#pragma once

#include <QObject>
#include <QJsonObject>

#include "TriangulationService.h"

class QWidget;
class ProjectData;
class ProjectManager;

namespace xjw::gui::project {
enum class SparsePointWorkflowKind;
}

class ProjectSparseReconstructionManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectSparseReconstructionManager(ProjectManager *owner,
                                                ProjectData *projectData,
                                                QWidget *parentWidget,
                                                QObject *parent = nullptr);

    void startTriangulationAsync(const QJsonObject &settings);
    void startSparseCloudOutlierRemovalAsync(const QJsonObject &settings);
    void startSparseCloudLocalOptimAsync(const QJsonObject &settings);
    void startSparseCloudRefineAsync(const QJsonObject &settings);

signals:
    void atProgressChanged(const QString &stage, int percent);
    void atProgressFinished(bool success);
    // 当前正式连接点成果被替换后发出，供三维视图立即切换到新点云。
    void tiePointResultReady(const QString &sparseCloudPath,
                             const QString &sidecarPath);

private:
    void finalizeTriangulationSuccess(const xjw::core::project::TriangulationServiceResult &result,
                                      const QStringList &selectedImages,
                                      const xjw::core::project::TriangulationServiceOptions &options);
    void startSparsePointWorkflow(xjw::gui::project::SparsePointWorkflowKind kind,
                                  const QJsonObject &settings);

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
};
