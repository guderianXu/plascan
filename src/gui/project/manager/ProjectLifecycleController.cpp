#include "ProjectLifecycleController.h"

#include "GuiTaskRunner.h"
#include "Logger.h"
#include "ProjectData.h"
#include "ProjectUiCommands.h"

#include <QDir>
#include <QFileInfo>
#include <QMessageBox>

ProjectLifecycleController::ProjectLifecycleController(ProjectData *projectData,
                                                       ProjectUiCommands *uiCommands,
                                                       QWidget *parentWidget,
                                                       QObject *parent)
    : QObject(parent)
    , _projectData(projectData)
    , _uiCommands(uiCommands)
    , _parentWidget(parentWidget)
{
    if (_projectData)
    {
        connect(_projectData,
                &ProjectData::projectSaveCompleted,
                this,
                [this](bool success, const QString &errorMessage)
                {
                    if (!success && !errorMessage.isEmpty())
                    {
                        QMessageBox::critical(_parentWidget,
                                              QStringLiteral("错误"),
                                              QStringLiteral("保存项目失败: %1").arg(errorMessage));
                    }
                    emit saveFinished(success);
                });
    }
}

void ProjectLifecycleController::createNewProject()
{
    QString projectPath;
    if (_uiCommands && _uiCommands->createNewProject(&projectPath))
    {
        emit projectCreated(projectPath);
        LOG_INFO(QStringLiteral("项目已创建: %1").arg(projectPath));
    }
}

void ProjectLifecycleController::openProject()
{
    QString projectPath;
    if (_uiCommands && _uiCommands->selectProjectByDialog(&projectPath))
    {
        openProjectFromPath(projectPath);
    }
}

void ProjectLifecycleController::openProjectFromPath(const QString &requestedPath)
{
    if (requestedPath.trimmed().isEmpty())
    {
        return;
    }

    const QString projectPath = QDir::cleanPath(QFileInfo(requestedPath).absoluteFilePath());
    if (_openInProgress)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("提示"),
                             QStringLiteral("正在打开项目，请稍候。"));
        return;
    }

    _openInProgress = true;
    emit projectOpenStarted(projectPath);
    emit projectOpenProgressChanged(QStringLiteral("正在读取项目文件..."), 10);

    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [projectPath]()
        {
            return ProjectData::loadProjectOpenSnapshot(projectPath);
        },
        [projectPath](ProjectLifecycleController *self,
                      xjw::gui::tasks::TaskOutcome<ProjectOpenSnapshot> outcome)
        {
            emit self->projectOpenProgressChanged(QStringLiteral("正在初始化项目界面..."), 75);
            if (!outcome.succeeded())
            {
                self->showOpenError(outcome.errorMessage);
                return;
            }

            ProjectOpenSnapshot snapshot = std::move(*outcome.value);
            if (!snapshot.success)
            {
                self->showOpenError(snapshot.errorMessage.isEmpty()
                                        ? QStringLiteral("读取项目文件失败")
                                        : snapshot.errorMessage);
                return;
            }

            QString error;
            if (!self->_projectData || !self->_projectData->openProjectFromSnapshot(snapshot, &error))
            {
                self->showOpenError(error.isEmpty() ? QStringLiteral("应用项目数据失败") : error);
                return;
            }

            emit self->projectOpenProgressChanged(QStringLiteral("正在启动结果数据后台加载..."), 95);
            if (!snapshot.resultsLoaded)
            {
                self->loadProjectResultsAsync(projectPath);
            }

            self->_openInProgress = false;
            emit self->projectOpenFinished(true, QStringLiteral("项目已打开"));
            LOG_INFO(QStringLiteral("项目已打开: %1").arg(projectPath));
        });
}

void ProjectLifecycleController::saveProject()
{
    if (!_projectData)
    {
        return;
    }

    emit saveStarted();
    _projectData->saveProjectAsync();
}

void ProjectLifecycleController::closeProject()
{
    if (_uiCommands)
    {
        _uiCommands->closeProject();
    }
}

void ProjectLifecycleController::loadProjectResultsAsync(const QString &projectPath)
{
    if (!_projectData || projectPath.trimmed().isEmpty())
    {
        return;
    }

    const QString expectedProjectPath = _projectData->currentProjectPath();
    const QString expectedChunkId = _projectData->activeChunkId();
    xjw::gui::tasks::runGuardedWithOutcome(
        this,
        [projectPath]()
        {
            return ProjectData::loadProjectResultsSnapshot(projectPath);
        },
        [projectPath, expectedProjectPath, expectedChunkId](
            ProjectLifecycleController *self,
            xjw::gui::tasks::TaskOutcome<ProjectResultsSnapshot> outcome)
        {
            if (!self->_projectData
                || self->_projectData->currentProjectPath() != expectedProjectPath
                || self->_projectData->activeChunkId() != expectedChunkId)
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
}

void ProjectLifecycleController::showOpenError(const QString &message)
{
    _openInProgress = false;
    emit projectOpenFinished(false, message);
    QMessageBox::critical(_parentWidget,
                          QStringLiteral("错误"),
                          QStringLiteral("打开项目失败: %1").arg(message));
}
