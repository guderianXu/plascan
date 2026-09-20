#include "ProjectLifecycleService.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "ProjectResourceCleanupCoordinator.h"
#include "ProjectSession.h"
#include "ProjectUiMessageAdapter.h"
#include "project/ProjectSessionModel.h"

#include <QDir>
#include <QFileInfo>
#include <QPointer>

ProjectLifecycleService::ProjectLifecycleService(ProjectData* projectData,
                                                 xjw::gui::project::ProjectSession* session,
                                                 ProjectUiMessageAdapter* messages,
                                                 xjw::gui::project::ProjectResourceCleanupCoordinator* cleanup,
                                                 QWidget* parentWidget,
                                                 QObject* parent)
    : QObject(parent), _projectData(projectData), _session(session), _messages(messages), _cleanup(cleanup),
      _parentWidget(parentWidget)
{
    if (!_projectData)
    {
        return;
    }

    connect(_projectData,
            &ProjectData::projectSaveCompleted,
            this,
            [this](bool success, const QString& errorMessage)
            {
                if (!_saveInProgress)
                {
                    return;
                }
                const bool currentSession = isCurrent(_saveSessionContext);
                _saveInProgress = false;
                endOperation(QStringLiteral("保存项目"));
                if (!currentSession)
                {
                    return;
                }
                if (!success && !errorMessage.isEmpty() && _messages)
                {
                    _messages->critical(
                        _parentWidget, QStringLiteral("错误"), QStringLiteral("保存项目失败: %1").arg(errorMessage));
                }
                emit saveFinished(success);
                if (success)
                {
                    emit projectModified();
                }
            });
    connect(_projectData,
            &ProjectData::portableProjectExportCompleted,
            this,
            [this](bool success, const QString& outputPath, const QString& errorMessage)
            {
                const bool currentSession = isCurrent(_portableExportSessionContext);
                _portableExportInProgress = false;
                endOperation(QStringLiteral("导出项目"));
                if (!currentSession)
                {
                    return;
                }
                if (!success)
                {
                    reportError(QStringLiteral("导出项目"),
                                errorMessage.isEmpty() ? QStringLiteral("便携项目导出失败") : errorMessage);
                }
                else
                {
                    if (_messages)
                    {
                        _messages->information(_parentWidget,
                                               QStringLiteral("导出项目"),
                                               QStringLiteral("便携项目已导出: %1").arg(outputPath));
                    }
                    emit portableExportFinished(true, outputPath, QStringLiteral("便携项目已导出"));
                }
            });
}

ProjectLifecycleService::~ProjectLifecycleService()
{
    if (!_pendingDrainOperation.isEmpty())
    {
        const QString operation = std::move(_pendingDrainOperation);
        _pendingDrainOperation.clear();
        endOperation(operation);
    }
}

void ProjectLifecycleService::setDirectoryAccessors(
    std::function<QString(const QString& key)> getLastDir,
    std::function<void(const QString& key, const QString& dir)> saveLastDir)
{
    _getLastDir = std::move(getLastDir);
    _saveLastDir = std::move(saveLastDir);
}

void ProjectLifecycleService::setSessionDrainHandler(std::function<bool(std::function<void()>)> handler)
{
    _sessionDrainHandler = std::move(handler);
}

void ProjectLifecycleService::setSessionFutureTracker(std::function<void(QFuture<void>)> tracker)
{
    _sessionFutureTracker = std::move(tracker);
}

bool ProjectLifecycleService::isOpenInProgress() const
{
    return _openInProgress;
}

bool ProjectLifecycleService::isPortableExportInProgress() const
{
    return _portableExportInProgress;
}

bool ProjectLifecycleService::isBusy() const
{
    return _openInProgress || _saveInProgress || _portableExportInProgress || (_session && _session->isBusy());
}

void ProjectLifecycleService::createNewProject()
{
    if (!_projectData || !_messages || rejectLifecycleChange(QStringLiteral("新建项目")))
    {
        return;
    }

    const UiDialogResult selected =
        _messages->selectSaveFile(_parentWidget,
                                  QStringLiteral("创建新项目"),
                                  readLastDir(QStringLiteral("project")),
                                  QStringLiteral("PlaScan项目 (*.plascan)"),
                                  QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot,
                                  QStringLiteral("plascan"));
    if (!selected.accepted || selected.text.trimmed().isEmpty())
    {
        return;
    }

    QString projectPath = QDir::cleanPath(QFileInfo(selected.text).absoluteFilePath());
    if (!projectPath.endsWith(QStringLiteral(".plascan"), Qt::CaseInsensitive))
    {
        projectPath += QStringLiteral(".plascan");
    }
    writeLastDir(QStringLiteral("project"), QFileInfo(projectPath).absolutePath());

    if (!beginOperation(QStringLiteral("新建项目")))
    {
        return;
    }

    const QString operation = QStringLiteral("新建项目");
    continueAfterSessionDrain(
        operation,
        [operation, projectPath](ProjectLifecycleService* self)
        {
            if (!self->_projectData->createProject(projectPath, QFileInfo(projectPath).baseName()))
            {
                self->endOperation(operation);
                self->reportError(operation, QStringLiteral("创建项目失败"));
                return;
            }

            self->endOperation(operation);
            emit self->projectCreated(projectPath);
            emit self->projectOpened(self->_projectData);
            LOG_INFO(QStringLiteral("项目已创建: %1").arg(projectPath));
        });
}

void ProjectLifecycleService::openProject()
{
    if (!_messages)
    {
        return;
    }
    const UiDialogResult selected =
        _messages->selectOpenFile(_parentWidget,
                                  QStringLiteral("打开项目"),
                                  readLastDir(QStringLiteral("project")),
                                  QStringLiteral("PlaScan项目 (*.plascan)"),
                                  QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot);
    if (!selected.accepted || selected.text.trimmed().isEmpty())
    {
        return;
    }
    writeLastDir(QStringLiteral("project"), QFileInfo(selected.text).absolutePath());
    openProjectFromPath(selected.text);
}

void ProjectLifecycleService::openProjectFromPath(const QString& requestedPath)
{
    if (!_projectData || requestedPath.trimmed().isEmpty())
    {
        return;
    }
    if (rejectLifecycleChange(QStringLiteral("打开项目")))
    {
        return;
    }
    if (_openInProgress)
    {
        if (_messages)
        {
            _messages->warning(_parentWidget, QStringLiteral("提示"), QStringLiteral("正在打开项目，请稍候。"));
        }
        return;
    }

    const QString projectPath = QDir::cleanPath(QFileInfo(requestedPath).absoluteFilePath());
    if (!beginOperation(QStringLiteral("打开项目")))
    {
        return;
    }
    const QString operation = QStringLiteral("打开项目");
    continueAfterSessionDrain(
        operation,
        [this, operation, projectPath](ProjectLifecycleService* self)
        {
            self->_openInProgress = true;
            self->_openSessionContext =
                self->_session ? self->_session->context() : xjw::gui::project::ProjectSessionContext{};
            emit projectOpenStarted(projectPath);
            emit projectOpenProgressChanged(QStringLiteral("正在读取项目文件..."), 10);

            xjw::gui::tasks::runGuardedWithOutcome(
                self,
                [projectPath]() { return ProjectData::loadProjectOpenSnapshot(projectPath); },
                [operation, projectPath, requestContext = self->_openSessionContext](
                    ProjectLifecycleService* service, xjw::gui::tasks::TaskOutcome<ProjectOpenSnapshot> outcome)
                {
                    if (service->_session && !service->_session->isCurrent(requestContext))
                    {
                        service->showOpenCancelled(QStringLiteral("项目会话已变化，已取消打开项目。"));
                        return;
                    }
                    emit service->projectOpenProgressChanged(QStringLiteral("正在初始化项目界面..."), 75);
                    if (!outcome.succeeded())
                    {
                        service->showOpenError(outcome.errorMessage);
                        return;
                    }

                    ProjectOpenSnapshot snapshot = std::move(*outcome.value);
                    if (!snapshot.success)
                    {
                        service->showOpenError(snapshot.errorMessage.isEmpty() ? QStringLiteral("读取项目文件失败")
                                                                               : snapshot.errorMessage);
                        return;
                    }

                    QString error;
                    if (!service->_projectData->openProjectFromSnapshot(snapshot, &error))
                    {
                        service->showOpenError(error.isEmpty() ? QStringLiteral("应用项目数据失败") : error);
                        return;
                    }

                    const auto context =
                        service->_session ? service->_session->context() : xjw::gui::project::ProjectSessionContext{};
                    emit service->projectOpenProgressChanged(QStringLiteral("正在启动结果数据后台加载..."), 95);
                    service->loadProjectResultsAsync(projectPath, context);
                    service->_openInProgress = false;
                    service->endOperation(operation);
                    emit service->projectOpened(service->_projectData);
                    emit service->projectOpenFinished(true, QStringLiteral("项目已打开"));
                    LOG_INFO(QStringLiteral("项目已打开: %1").arg(projectPath));
                });
        });
}

void ProjectLifecycleService::saveProject()
{
    if (!_projectData || !_projectData->hasProject())
    {
        return;
    }
    if (_saveInProgress)
    {
        if (_messages)
        {
            _messages->information(_parentWidget, QStringLiteral("保存项目"), QStringLiteral("项目正在保存，请稍候。"));
        }
        return;
    }
    if (rejectLifecycleChange(QStringLiteral("保存项目")))
    {
        return;
    }
    if (!beginOperation(QStringLiteral("保存项目")))
    {
        return;
    }
    emit saveStarted();
    _saveInProgress = true;
    _saveSessionContext = _session ? _session->context() : xjw::gui::project::ProjectSessionContext{};
    _projectData->saveProjectAsync();
}

void ProjectLifecycleService::exportPortableProject()
{
    if (!_projectData || !_projectData->hasProject() || !_messages)
    {
        return;
    }

    const UiDialogResult selected =
        _messages->selectSaveFile(_parentWidget,
                                  QStringLiteral("导出便携项目"),
                                  readLastDir(QStringLiteral("project_export")),
                                  QStringLiteral("便携项目 ZIP (*.zip)"),
                                  QDir::AllEntries | QDir::Hidden | QDir::AllDirs | QDir::NoDotAndDotDot,
                                  {});
    if (!selected.accepted || selected.text.trimmed().isEmpty())
    {
        return;
    }
    QString outputPath = QDir::cleanPath(QFileInfo(selected.text).absoluteFilePath());
    if (!outputPath.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive))
    {
        outputPath += QStringLiteral(".zip");
    }
    writeLastDir(QStringLiteral("project_export"), QFileInfo(outputPath).absolutePath());

    QString error;
    if (!startPortableExport(outputPath, &error))
    {
        reportError(QStringLiteral("导出项目"), error.isEmpty() ? QStringLiteral("无法启动便携项目导出") : error);
    }
}

bool ProjectLifecycleService::startPortableExport(const QString& requestedOutputPath, QString* errorMessage)
{
    if (errorMessage)
    {
        errorMessage->clear();
    }
    if (!_projectData || !_projectData->hasProject())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("请先打开项目");
        }
        return false;
    }
    if (requestedOutputPath.trimmed().isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("导出路径为空");
        }
        return false;
    }
    if (rejectLifecycleChange(QStringLiteral("导出项目")))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("当前项目操作完成前无法导出项目，请稍候。");
        }
        return false;
    }
    if (_portableExportInProgress)
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("便携项目导出正在进行，请稍候。");
        }
        return false;
    }
    if (!beginOperation(QStringLiteral("导出项目")))
    {
        if (errorMessage)
        {
            *errorMessage = QStringLiteral("当前项目操作完成前无法导出项目，请稍候。");
        }
        return false;
    }

    QString outputPath = QDir::cleanPath(QFileInfo(requestedOutputPath).absoluteFilePath());
    if (!outputPath.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive))
    {
        outputPath += QStringLiteral(".zip");
    }
    _portableExportSessionContext = _session ? _session->context() : xjw::gui::project::ProjectSessionContext{};
    QString error;
    _portableExportInProgress = _projectData->exportPortableProjectAsync(outputPath, &error);
    if (!_portableExportInProgress)
    {
        endOperation(QStringLiteral("导出项目"));
        if (errorMessage)
        {
            *errorMessage = error.isEmpty() ? QStringLiteral("无法启动便携项目导出") : error;
        }
        return false;
    }
    return true;
}

void ProjectLifecycleService::closeProject()
{
    if (!_projectData || !_projectData->hasProject() || rejectLifecycleChange(QStringLiteral("关闭项目")))
    {
        return;
    }
    if (!beginOperation(QStringLiteral("关闭项目")))
    {
        return;
    }
    const QString operation = QStringLiteral("关闭项目");
    continueAfterSessionDrain(operation,
                              [operation](ProjectLifecycleService* self)
                              {
                                  QString error;
                                  if (!self->_projectData->closeProject(&error))
                                  {
                                      self->endOperation(operation);
                                      self->reportError(operation,
                                                        error.isEmpty() ? QStringLiteral("无法安全关闭项目") : error);
                                      return;
                                  }
                                  // ProjectData emits projectClosed only after it has released the project
                                  // lock. ProjectSession advances the generation from that signal, so a
                                  // failed close leaves the previous context valid.
                                  self->endOperation(operation);
                                  emit self->projectClosed();
                              });
}

void ProjectLifecycleService::createChunk()
{
    if (!_projectData || !_projectData->hasProject() || !_messages ||
        rejectLifecycleChange(QStringLiteral("新建 Chunk")))
    {
        return;
    }
    const UiDialogResult result =
        _messages->getText(_parentWidget, QStringLiteral("新建 Chunk"), QStringLiteral("名称："), QString());
    if (!result.accepted)
    {
        return;
    }
    const QString operation = QStringLiteral("新建 Chunk");
    if (!beginOperation(operation))
    {
        return;
    }
    const QString chunk_name = result.text.trimmed();
    continueAfterSessionDrain(operation,
                              [chunk_name, operation](ProjectLifecycleService* self)
                              {
                                  QString error;
                                  if (!self->_projectData->createChunk(chunk_name, nullptr, &error) && self->_messages)
                                  {
                                      self->_messages->critical(
                                          self->_parentWidget, QStringLiteral("新建 Chunk 失败"), error);
                                  }
                                  self->endOperation(operation);
                              });
}

void ProjectLifecycleService::renameChunk(const QString& chunkId)
{
    if (!_projectData || chunkId.trimmed().isEmpty() || !_messages ||
        rejectLifecycleChange(QStringLiteral("重命名 Chunk")))
    {
        return;
    }
    const QString normalized_chunk_id = chunkId.trimmed();
    const bool renaming_active_chunk = _projectData->activeChunkId() == normalized_chunk_id;
    QString current_name;
    for (const QJsonValue& value : _projectData->chunks())
    {
        const QJsonObject chunk = value.toObject();
        if (chunk.value(QStringLiteral("id")).toString() == normalized_chunk_id)
        {
            current_name = chunk.value(QStringLiteral("name")).toString();
            break;
        }
    }
    const UiDialogResult result =
        _messages->getText(_parentWidget, QStringLiteral("重命名 Chunk"), QStringLiteral("名称："), current_name);
    if (!result.accepted)
    {
        return;
    }
    const QString new_name = result.text.trimmed();
    const auto rename_chunk = [new_name, normalized_chunk_id](ProjectLifecycleService* self)
    {
        QString error;
        if (!self->_projectData->renameChunk(normalized_chunk_id, new_name, &error) && self->_messages)
        {
            self->_messages->critical(self->_parentWidget, QStringLiteral("重命名 Chunk 失败"), error);
        }
    };
    if (!renaming_active_chunk)
    {
        rename_chunk(this);
        return;
    }

    const QString operation = QStringLiteral("重命名 Chunk");
    if (!beginOperation(operation))
    {
        return;
    }
    continueAfterSessionDrain(operation,
                              [operation, rename_chunk](ProjectLifecycleService* self)
                              {
                                  rename_chunk(self);
                                  self->endOperation(operation);
                              });
}

void ProjectLifecycleService::removeChunk(const QString& chunkId)
{
    if (!_projectData || chunkId.trimmed().isEmpty() || !_messages ||
        rejectLifecycleChange(QStringLiteral("删除 Chunk")))
    {
        return;
    }
    const QString normalized_chunk_id = chunkId.trimmed();
    const bool removing_active_chunk = _projectData->activeChunkId() == normalized_chunk_id;
    QString chunk_name;
    for (const QJsonValue& value : _projectData->chunks())
    {
        const QJsonObject chunk = value.toObject();
        if (chunk.value(QStringLiteral("id")).toString() == normalized_chunk_id)
        {
            chunk_name = chunk.value(QStringLiteral("name")).toString();
            break;
        }
    }
    const UiAnswer answer = _messages->question(
        _parentWidget,
        QStringLiteral("删除 Chunk"),
        QStringLiteral("确定删除“%1”吗？该 Chunk 的影像、处理结果和数字目录都会被删除，此操作不可撤销。")
            .arg(chunk_name),
        UiAnswer::No);
    if (answer != UiAnswer::Yes)
    {
        return;
    }
    const auto remove_chunk = [normalized_chunk_id](ProjectLifecycleService* self)
    {
        QString error;
        if (!self->_projectData->removeChunk(normalized_chunk_id, &error) && self->_messages)
        {
            self->_messages->critical(self->_parentWidget, QStringLiteral("删除 Chunk 失败"), error);
        }
    };
    if (!removing_active_chunk)
    {
        remove_chunk(this);
        return;
    }

    const QString operation = QStringLiteral("删除 Chunk");
    if (!beginOperation(operation))
    {
        return;
    }
    continueAfterSessionDrain(operation,
                              [operation, remove_chunk](ProjectLifecycleService* self)
                              {
                                  remove_chunk(self);
                                  self->endOperation(operation);
                              });
}

void ProjectLifecycleService::switchChunk(const QString& chunkId)
{
    const QString normalized_chunk_id = chunkId.trimmed();
    if (!_projectData || normalized_chunk_id.isEmpty() || rejectLifecycleChange(QStringLiteral("切换 Chunk")) ||
        _projectData->activeChunkId() == normalized_chunk_id)
    {
        return;
    }
    const QString operation = QStringLiteral("切换 Chunk");
    if (!beginOperation(operation))
    {
        return;
    }
    continueAfterSessionDrain(operation,
                              [normalized_chunk_id, operation](ProjectLifecycleService* self)
                              {
                                  QString error;
                                  if (!self->_projectData->switchChunk(normalized_chunk_id, &error) && self->_messages)
                                  {
                                      self->_messages->critical(
                                          self->_parentWidget, QStringLiteral("切换 Chunk 失败"), error);
                                  }
                                  self->endOperation(operation);
                              });
}

QString ProjectLifecycleService::readLastDir(const QString& key) const
{
    if (_getLastDir)
    {
        const QString value = _getLastDir(key);
        if (!value.isEmpty())
        {
            return value;
        }
    }
    return QDir::homePath();
}

void ProjectLifecycleService::writeLastDir(const QString& key, const QString& directory) const
{
    if (_saveLastDir)
    {
        _saveLastDir(key, directory);
    }
}

bool ProjectLifecycleService::rejectLifecycleChange(const QString& operation) const
{
    if (_portableExportInProgress)
    {
        if (_messages)
        {
            _messages->information(_parentWidget,
                                   QStringLiteral("导出项目进行中"),
                                   QStringLiteral("便携项目导出完成前无法%1，请稍候。").arg(operation));
        }
        return true;
    }
    if (_openInProgress || _saveInProgress)
    {
        if (_messages)
        {
            _messages->information(_parentWidget,
                                   QStringLiteral("项目操作进行中"),
                                   QStringLiteral("当前项目操作完成前无法%1，请稍候。").arg(operation));
        }
        return true;
    }
    if (_session && _session->isBusy())
    {
        if (_messages)
        {
            _messages->information(_parentWidget,
                                   QStringLiteral("项目操作进行中"),
                                   QStringLiteral("当前项目操作完成前无法%1，请稍候。").arg(operation));
        }
        return true;
    }
    return _cleanup && _cleanup->rejectLifecycleChange(operation);
}

bool ProjectLifecycleService::beginOperation(const QString& operation)
{
    if (!_session)
    {
        return true;
    }
    if (_session->tryBeginOperation(operation))
    {
        return true;
    }
    if (_messages)
    {
        _messages->information(_parentWidget,
                               QStringLiteral("项目操作进行中"),
                               QStringLiteral("当前项目操作完成前无法%1，请稍候。").arg(operation));
    }
    return false;
}

void ProjectLifecycleService::endOperation(const QString& operation)
{
    if (_session)
    {
        _session->endOperation(operation);
    }
}

bool ProjectLifecycleService::continueAfterSessionDrain(const QString& operation,
                                                        std::function<void(ProjectLifecycleService*)> continuation)
{
    if (!continuation)
    {
        endOperation(operation);
        reportError(operation, QStringLiteral("项目生命周期操作缺少执行步骤。"));
        return false;
    }

    _pendingDrainOperation = operation;
    const QPointer<ProjectLifecycleService> guarded_this(this);
    auto guarded_continuation = [guarded_this, continuation = std::move(continuation)]() mutable
    {
        if (!guarded_this)
        {
            return;
        }
        guarded_this->_pendingDrainOperation.clear();
        continuation(guarded_this.data());
    };

    if (!_sessionDrainHandler)
    {
        guarded_continuation();
        return true;
    }
    if (_sessionDrainHandler(std::move(guarded_continuation)))
    {
        return true;
    }

    _pendingDrainOperation.clear();
    endOperation(operation);
    reportError(operation, QStringLiteral("无法等待当前项目后台任务安全结束，操作已取消。"));
    return false;
}

void ProjectLifecycleService::reportError(const QString& operation, const QString& message)
{
    emit lifecycleError(operation, message);
    if (_messages)
    {
        _messages->critical(_parentWidget, operation, message);
    }
    if (operation == QLatin1String("导出项目"))
    {
        emit portableExportFinished(false, QString(), message);
    }
}

void ProjectLifecycleService::loadProjectResultsAsync(const QString& projectPath,
                                                      const xjw::gui::project::ProjectSessionContext& context)
{
    if (!_projectData || projectPath.trimmed().isEmpty())
    {
        return;
    }
    QFuture<void> future = xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [projectPath]() { return ProjectData::loadProjectResultsSnapshot(projectPath); },
        [projectPath, context](ProjectLifecycleService* self,
                               xjw::gui::tasks::TaskOutcome<ProjectResultsSnapshot> outcome)
        {
            if (!self || !self->_projectData || (self->_session && !self->_session->isCurrent(context)))
            {
                return;
            }
            if (!outcome.succeeded())
            {
                LOG_WARN(QStringLiteral("项目结果数据后台加载失败: %1").arg(outcome.errorMessage));
                return;
            }
            ProjectResultsSnapshot snapshot = std::move(*outcome.value);
            QString error;
            if (!self->_projectData->applyResultsSnapshot(snapshot, &error))
            {
                LOG_WARN(QStringLiteral("项目结果数据后台加载失败: %1").arg(error));
                return;
            }
            if (snapshot.hasResults)
            {
                LOG_INFO(QStringLiteral("项目结果数据已后台加载: %1").arg(projectPath));
            }
        });
    if (_sessionFutureTracker)
    {
        _sessionFutureTracker(std::move(future));
    }
}

void ProjectLifecycleService::showOpenError(const QString& message)
{
    _openInProgress = false;
    endOperation(QStringLiteral("打开项目"));
    const QString detail = message.isEmpty() ? QStringLiteral("打开项目失败") : message;
    emit projectOpenFinished(false, detail);
    reportError(QStringLiteral("打开项目"), detail);
}

void ProjectLifecycleService::showOpenCancelled(const QString& message)
{
    _openInProgress = false;
    endOperation(QStringLiteral("打开项目"));
    emit projectOpenFinished(false, message);
}

bool ProjectLifecycleService::isCurrent(const xjw::gui::project::ProjectSessionContext& context) const
{
    return !_session || _session->isCurrent(context);
}
