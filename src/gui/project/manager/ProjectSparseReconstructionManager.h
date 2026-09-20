#pragma once

#include <QObject>
#include <QFuture>
#include <QJsonObject>
#include <QVector>

#include "project/tasks/ProjectTaskContext.h"
#include "TriangulationService.h"

#include <functional>

class QWidget;
class ProjectUiMessageAdapter;

namespace xjw::gui::project
{
    enum class SparsePointWorkflowKind;
    class ProjectSession;
    class ProjectTaskOrchestrator;
} // namespace xjw::gui::project

class ProjectSparseReconstructionManager : public QObject
{
    Q_OBJECT

public:
    ~ProjectSparseReconstructionManager() override = default;

signals:
    void atProgressChanged(const QString& stage, int percent);
    void atProgressFinished(bool success);
    // 当前正式连接点成果被替换后发出，供三维视图立即切换到新点云。
    void tiePointResultReady(const QString& sparseCloudPath, const QString& sidecarPath);

private:
    friend class xjw::gui::project::ProjectTaskOrchestrator;

    using TiePointResultWriter = std::function<bool(const xjw::gui::project::ProjectTaskContext& expected,
                                                    const QString& sparseCloudPath,
                                                    int sparsePointCount,
                                                    const QStringList& selectedImages,
                                                    const QString& outputDir,
                                                    const QJsonObject& extraRecord)>;

    explicit ProjectSparseReconstructionManager(xjw::gui::project::ProjectSession* session,
                                                ProjectUiMessageAdapter* messages,
                                                QWidget* parentWidget,
                                                TiePointResultWriter tiePointResultWriter = {},
                                                QObject* parent = nullptr);

    void startTriangulationAsync(const QJsonObject& settings, const xjw::gui::project::ProjectTaskContext& taskContext);
    void startSparseCloudOutlierRemovalAsync(const QJsonObject& settings,
                                             const xjw::gui::project::ProjectTaskContext& taskContext);
    void startSparseCloudLocalOptimAsync(const QJsonObject& settings,
                                         const xjw::gui::project::ProjectTaskContext& taskContext);
    void startSparseCloudRefineAsync(const QJsonObject& settings,
                                     const xjw::gui::project::ProjectTaskContext& taskContext);

    void setTiePointResultWriter(TiePointResultWriter tiePointResultWriter);
    void waitForActiveTask();
    bool reserveTask();
    void releaseTaskReservation();
    bool hasRunningTask() const noexcept;
    void trackFutureForTesting(QFuture<void> future);
    void reapFinishedFutures();
    void trackFuture(QFuture<void> future);
    void finalizeTriangulationSuccess(const xjw::core::project::TriangulationServiceResult& result,
                                      const QStringList& selectedImages,
                                      const xjw::core::project::TriangulationServiceOptions& options,
                                      const xjw::gui::project::ProjectTaskContext& taskContext);
    void startSparsePointWorkflow(xjw::gui::project::SparsePointWorkflowKind kind,
                                  const QJsonObject& settings,
                                  const xjw::gui::project::ProjectTaskContext& taskContext);

    xjw::gui::project::ProjectSession *_session = nullptr;
    ProjectUiMessageAdapter *_messages = nullptr;
    QWidget *_parentWidget = nullptr;
    TiePointResultWriter _tiePointResultWriter;
    QVector<QFuture<void>> _activeFutures;
    bool _taskReservation = false;
};
