#include "ProjectResourceCleanupCoordinator.h"

#include "ProjectSession.h"
#include "ProjectTiePointResultService.h"
#include "ProjectUiMessageAdapter.h"
#include "project/ProjectSessionModel.h"

#include <QWidget>

namespace xjw::gui::project
{
namespace
{

class CoreProjectResourceCleanupExecutor final : public ProjectResourceCleanupExecutor
{
public:
    xjw::core::project::PreparedResourceCleanup
    prepare(ProjectData* projectData,
            const QString& section,
            const QStringList& resourcePaths) override
    {
        return xjw::core::project::ProjectResourceCleanupService::prepareGeneratedDataCleanup(
            projectData, section, resourcePaths);
    }

    xjw::core::project::ResourceCleanupResult
    execute(const xjw::core::project::PreparedResourceCleanup& prepared) override
    {
        return xjw::core::project::ProjectResourceCleanupService::executePreparedCleanup(prepared);
    }

    bool finalize(ProjectData* projectData,
                  const xjw::core::project::PreparedResourceCleanup& prepared,
                  const xjw::core::project::ResourceCleanupResult& result) override
    {
        return xjw::core::project::ProjectResourceCleanupService::finalizePreparedCleanup(
            projectData, prepared, result);
    }
};

QString cleanupDialogTitle(const QString& section, int selectedCount)
{
    if (section == QStringLiteral("连接点"))
    {
        return QStringLiteral("移除连接点");
    }
    return selectedCount == 1 ? QStringLiteral("删除数据") : QStringLiteral("删除数据");
}

QString cleanupConfirmationText(const QString& section, int selectedCount)
{
    if (section == QStringLiteral("连接点"))
    {
        return QStringLiteral("确定移除当前连接点及其关联生成文件吗？此操作不可撤销。");
    }
    if (selectedCount == 1)
    {
        return QStringLiteral("确定删除所选%1数据及其关联生成文件吗？此操作不可撤销。").arg(section);
    }
    return QStringLiteral("确定删除所选 %1 项%2数据及其关联生成文件吗？此操作不可撤销。")
        .arg(selectedCount)
        .arg(section);
}

} // namespace

ProjectResourceCleanupCoordinator::ProjectResourceCleanupCoordinator(ProjectSession* session,
                                                                     ProjectUiMessageAdapter* messages,
                                                                     QObject* parent)
    : ProjectResourceCleanupCoordinator(
          session, messages, std::make_shared<CoreProjectResourceCleanupExecutor>(), parent)
{
}

ProjectResourceCleanupCoordinator::ProjectResourceCleanupCoordinator(
    ProjectSession* session,
    ProjectUiMessageAdapter* messages,
    std::shared_ptr<ProjectResourceCleanupExecutor> executor,
    QObject* parent)
    : QObject(parent),
      _session(session),
      _messages(messages),
      _executor(std::move(executor))
{
    if (!_executor)
    {
        _executor = std::make_shared<CoreProjectResourceCleanupExecutor>();
    }
}

ProjectResourceCleanupCoordinator::~ProjectResourceCleanupCoordinator()
{
    waitForFinished();
}

bool ProjectResourceCleanupCoordinator::isRunning() const
{
    return _running;
}

bool ProjectResourceCleanupCoordinator::rejectLifecycleChange(const QString& operation) const
{
    if (!_running && (!_session || !_session->isBusy()))
    {
        return false;
    }

    if (_messages)
    {
        _messages->information(nullptr,
                               QStringLiteral("资源清理进行中"),
                               QStringLiteral("资源清理任务完成前无法%1，请稍候。").arg(operation));
    }
    return true;
}

bool ProjectResourceCleanupCoordinator::deleteGeneratedData(const QString& section,
                                                            const QStringList& resourcePaths,
                                                            QWidget* requestWidget)
{
    if (!_session || !_session->data() || resourcePaths.isEmpty())
    {
        return false;
    }
    if (rejectLifecycleChange(QStringLiteral("删除数据")))
    {
        return false;
    }
    if (section == QStringLiteral("照片"))
    {
        if (_messages)
        {
            _messages->information(nullptr,
                                   QStringLiteral("删除数据"),
                                   QStringLiteral("照片分组不支持删除数据，请使用移除引用。"));
        }
        return false;
    }

    const int selectedCount = resourcePaths.size();
    const QString title = cleanupDialogTitle(section, selectedCount);
    if (_messages &&
        _messages->question(nullptr,
                            title,
                            cleanupConfirmationText(section, selectedCount),
                            UiAnswer::No) != UiAnswer::Yes)
    {
        return false;
    }
    if (rejectLifecycleChange(QStringLiteral("删除数据")))
    {
        return false;
    }

    if (!_session->tryBeginOperation(_operationName))
    {
        if (_messages)
        {
            _messages->information(nullptr,
                                   QStringLiteral("资源清理进行中"),
                                   QStringLiteral("当前项目操作完成前无法删除数据，请稍候。"));
        }
        return false;
    }

    if (section == QStringLiteral("连接点"))
    {
        const auto result = ProjectTiePointResultService::deleteAll(_session->data());
        if (!result.success)
        {
            if (_messages)
            {
                _messages->warning(nullptr,
                                   title,
                                   QStringLiteral("删除失败：%1").arg(result.errorMessage));
            }
            _session->endOperation(_operationName);
            return false;
        }

        if (_messages)
        {
            _messages->information(nullptr, title, QStringLiteral("已移除连接点及其关联生成文件。"));
        }
        emit tiePointDeletionFinished();
        _session->endOperation(_operationName);
        return true;
    }

    _running = true;
    _completionHandled = false;
    _asyncTaskStarted = false;
    _sessionContext = _session->context();
    _projectData = _session->data();
    _requestWidget = requestWidget;
    _section = section;
    _taskId = QStringLiteral("resource_cleanup");

    if (_requestWidget)
    {
        _requestWidget->setEnabled(false);
    }

    _prepared = _executor->prepare(_projectData.data(), section, resourcePaths);
    _requiresExecution = _prepared.requiresExecution();
    if (!_requiresExecution)
    {
        complete(_prepared.preparationResult(), true, false);
        return true;
    }

    _executionResult =
        std::make_shared<xjw::core::project::ResourceCleanupResult>(_prepared.preparationResult());
    emit progressChanged(_taskId, 0, 0);

    const auto prepared = _prepared;
    const auto executor = _executor;
    const auto executionResult = _executionResult;
    _future = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [prepared, executor, executionResult]()
        {
            const auto result = executor->execute(prepared);
            *executionResult = result;
            return result;
        },
        [](ProjectResourceCleanupCoordinator* self,
           xjw::gui::tasks::TaskOutcome<xjw::core::project::ResourceCleanupResult> outcome) mutable
        {
            self->handleOutcome(std::move(outcome));
        });
    _asyncTaskStarted = true;
    return true;
}

void ProjectResourceCleanupCoordinator::waitForFinished()
{
    if (!_running)
    {
        return;
    }

    _future.waitForFinished();
    if (_completionHandled)
    {
        return;
    }

    const auto result = _executionResult
                            ? *_executionResult
                            : _prepared.preparationResult();
    complete(result, false, false);
}

void ProjectResourceCleanupCoordinator::handleOutcome(
    xjw::gui::tasks::TaskOutcome<xjw::core::project::ResourceCleanupResult> outcome)
{
    if (_completionHandled)
    {
        return;
    }

    auto result = _prepared.preparationResult();
    if (outcome.succeeded())
    {
        result = std::move(*outcome.value);
    }
    else
    {
        result.success = false;
        result.errorMessage = outcome.errorMessage.isEmpty()
                                  ? QStringLiteral("资源清理任务已取消")
                                  : outcome.errorMessage;
    }
    complete(result, true, true);
}

void ProjectResourceCleanupCoordinator::complete(
    const xjw::core::project::ResourceCleanupResult& result,
    bool present,
    bool emitFinished)
{
    if (!_running || _completionHandled)
    {
        return;
    }

    _completionHandled = true;
    const bool currentSession = _session && _session->isCurrent(_sessionContext);

    // Finalization closes the core transaction and releases ProjectData's cleanup
    // generation. It must run even when the UI session went stale; only the
    // user-facing result is gated by the captured session context.
    const bool finalized = !_requiresExecution ||
                           (_projectData && _executor &&
                            _executor->finalize(_projectData.data(), _prepared, result));

    restoreRequestWidget();
    _running = false;
    if (_session)
    {
        _session->endOperation(_operationName);
    }
    if (_asyncTaskStarted && emitFinished)
    {
        emit finished(_taskId);
    }

    if (present && currentSession && finalized)
    {
        presentResult(result);
    }
}

void ProjectResourceCleanupCoordinator::presentResult(
    const xjw::core::project::ResourceCleanupResult& cleanupResult) const
{
    if (!_messages)
    {
        return;
    }
    if (cleanupResult.unsupportedSection)
    {
        _messages->warning(nullptr,
                           QStringLiteral("删除数据"),
                           QStringLiteral("当前分组暂不支持删除数据：%1").arg(_section));
        return;
    }
    if (!cleanupResult.success && !cleanupResult.errorMessage.isEmpty())
    {
        _messages->warning(nullptr,
                           QStringLiteral("删除数据"),
                           QStringLiteral("删除失败：%1").arg(cleanupResult.errorMessage));
        return;
    }
    if (cleanupResult.noMatchedRecords)
    {
        _messages->information(nullptr,
                               QStringLiteral("删除数据"),
                               QStringLiteral("未找到可删除的%1数据记录。").arg(_section));
        return;
    }
    if (cleanupResult.failedPaths.isEmpty() && cleanupResult.errorMessage.isEmpty())
    {
        _messages->information(nullptr,
                               QStringLiteral("删除数据"),
                               QStringLiteral("已删除 %1 项%2数据。")
                                   .arg(cleanupResult.removedCount)
                                   .arg(_section));
        return;
    }

    QString detail = cleanupResult.errorMessage;
    if (!cleanupResult.failedPaths.isEmpty())
    {
        if (!detail.isEmpty())
        {
            detail += QLatin1Char('\n');
        }
        detail += cleanupResult.failedPaths.join(QStringLiteral("\n"));
    }
    _messages->warning(
        nullptr,
        QStringLiteral("删除数据"),
        QStringLiteral("已移除 %1 项%2数据记录，但部分物理清理将在后续重试：\n%3")
            .arg(cleanupResult.removedCount)
            .arg(_section)
            .arg(detail));
}

void ProjectResourceCleanupCoordinator::restoreRequestWidget()
{
    if (_requestWidget)
    {
        _requestWidget->setEnabled(true);
    }
    _requestWidget = nullptr;
}

} // namespace xjw::gui::project
