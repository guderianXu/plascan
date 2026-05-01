#pragma once

#include <QObject>
#include <QJsonArray>
#include <QJsonObject>

#include "ProjectTriangulationService.h"

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

    QJsonArray getAvailableAtResults() const;
    void startTriangulationAsync(const QJsonObject &settings);
    void startSparseCloudOutlierRemovalAsync(const QJsonObject &settings);
    void startSparseCloudLocalOptimAsync(const QJsonObject &settings);
    void startSparseCloudRefineAsync(const QJsonObject &settings);

signals:
    void atProgressChanged(const QString &stage, int percent);
    void atProgressFinished(bool success);

private:
    bool ensureProjectOpen(const QString &message,
                           const QString &title) const;
    void finalizeTriangulationSuccess(const xjw::gui::project::TriangulationServiceResult &result,
                                      const QStringList &selectedImages,
                                      const xjw::gui::project::TriangulationServiceOptions &options,
                                      int replaceIndex);
    void startSparsePointWorkflow(xjw::gui::project::SparsePointWorkflowKind kind,
                                  const QJsonObject &settings);

    ProjectManager *m_owner = nullptr;
    ProjectData *m_projectData = nullptr;
    QWidget *m_parentWidget = nullptr;
};
