#pragma once

#include "project_workflows/ProjectResourceCleanup.h"
#include "tasks/GuiTaskRunner.h"
#include "project/support/ProjectSessionContext.h"

#include <QObject>
#include <QFuture>
#include <QPointer>
#include <QString>
#include <QStringList>

#include <memory>

class ProjectData;
class QWidget;

class ProjectUiMessageAdapter;

namespace xjw::gui::project
{

class ProjectSession;

class ProjectResourceCleanupExecutor
{
public:
    virtual ~ProjectResourceCleanupExecutor() = default;

    virtual xjw::core::project::PreparedResourceCleanup
    prepare(ProjectData* projectData,
            const QString& section,
            const QStringList& resourcePaths) = 0;

    virtual xjw::core::project::ResourceCleanupResult
    execute(const xjw::core::project::PreparedResourceCleanup& prepared) = 0;

    virtual bool finalize(ProjectData* projectData,
                          const xjw::core::project::PreparedResourceCleanup& prepared,
                          const xjw::core::project::ResourceCleanupResult& result) = 0;
};

class ProjectResourceCleanupCoordinator final : public QObject
{
    Q_OBJECT

public:
    ProjectResourceCleanupCoordinator(ProjectSession* session,
                                      ProjectUiMessageAdapter* messages,
                                      QObject* parent = nullptr);
    ProjectResourceCleanupCoordinator(
        ProjectSession* session,
        ProjectUiMessageAdapter* messages,
        std::shared_ptr<ProjectResourceCleanupExecutor> executor,
        QObject* parent = nullptr);
    ~ProjectResourceCleanupCoordinator() override;

    ProjectSession* session() const noexcept
    {
        return _session;
    }
    bool isRunning() const;
    bool rejectLifecycleChange(const QString& operation) const;
    // Returns false when the request was rejected or cancelled before the
    // cleanup transaction started. Existing callers may ignore the result.
    bool deleteGeneratedData(const QString& section,
                             const QStringList& resourcePaths,
                             QWidget* requestWidget = nullptr);
    void waitForFinished();

signals:
    void progressChanged(const QString& taskId, int value, int maximum);
    void finished(const QString& taskId);
    void tiePointDeletionFinished();

private:
    void handleOutcome(
        xjw::gui::tasks::TaskOutcome<xjw::core::project::ResourceCleanupResult> outcome);
    void complete(const xjw::core::project::ResourceCleanupResult& result,
                  bool presentResult,
                  bool emitFinished);
    void presentResult(const xjw::core::project::ResourceCleanupResult& result) const;
    void restoreRequestWidget();

    ProjectSession* _session = nullptr;
    ProjectUiMessageAdapter* _messages = nullptr;
    std::shared_ptr<ProjectResourceCleanupExecutor> _executor;
    QFuture<void> _future;
    QPointer<ProjectData> _projectData;
    QPointer<QWidget> _requestWidget;
    ProjectSessionContext _sessionContext;
    xjw::core::project::PreparedResourceCleanup _prepared;
    std::shared_ptr<xjw::core::project::ResourceCleanupResult> _executionResult;
    QString _section;
    QString _taskId;
    bool _running = false;
    bool _completionHandled = false;
    bool _requiresExecution = false;
    bool _asyncTaskStarted = false;
    QString _operationName = QStringLiteral("资源清理");
};

} // namespace xjw::gui::project
