#include "ProjectLifecyclePresenter.h"

#include "ProjectManager.h"

#include <QFileInfo>
#include <QMainWindow>
#include <QProgressDialog>
#include <QStatusBar>

#include <algorithm>

ProjectLifecyclePresenter::ProjectLifecyclePresenter(ProjectManager *projectManager,
                                                     QMainWindow *window,
                                                     QStatusBar *statusBar,
                                                     QObject *parent)
    : QObject(parent)
    , _projectManager(projectManager)
    , _window(window)
    , _statusBar(statusBar)
{
    connect(_projectManager, &ProjectManager::saveStarted,
            this, &ProjectLifecyclePresenter::showSaveProgress);
    connect(_projectManager, &ProjectManager::saveFinished,
            this, &ProjectLifecyclePresenter::finishSaveProgress);
    connect(_projectManager, &ProjectManager::projectOpenStarted,
            this, &ProjectLifecyclePresenter::showOpenProgress);
    connect(_projectManager, &ProjectManager::projectOpenProgressChanged,
            this, &ProjectLifecyclePresenter::updateOpenProgress);
    connect(_projectManager, &ProjectManager::projectOpenFinished,
            this, &ProjectLifecyclePresenter::finishOpenProgress);
    connect(_projectManager, &ProjectManager::metadataDirtyChanged,
            this, &ProjectLifecyclePresenter::updateWindowTitle);
}

bool ProjectLifecyclePresenter::isCloseSavePending() const
{
    return _closeSavePending;
}

void ProjectLifecyclePresenter::requestCloseAfterSave()
{
    _closeSavePending = true;
    _projectManager->saveProject();
}

void ProjectLifecyclePresenter::showOpenProgress(const QString &projectPath)
{
    if (!_openProgress)
    {
        _openProgress = new QProgressDialog(tr("正在打开项目..."), QString(), 0, 100, _window);
        _openProgress->setWindowModality(Qt::ApplicationModal);
        _openProgress->setCancelButton(nullptr);
        _openProgress->setMinimumDuration(0);
        _openProgress->setAutoClose(false);
        _openProgress->setAutoReset(false);
    }
    _openProgress->setLabelText(tr("正在打开项目：%1").arg(QFileInfo(projectPath).fileName()));
    _openProgress->setValue(0);
    _openProgress->show();
}

void ProjectLifecyclePresenter::updateOpenProgress(const QString &message, int percent)
{
    if (!_openProgress)
    {
        return;
    }
    _openProgress->setLabelText(message.isEmpty() ? tr("正在打开项目...") : message);
    _openProgress->setValue(std::clamp(percent, 0, 100));
}

void ProjectLifecyclePresenter::finishOpenProgress(bool success, const QString &message)
{
    if (_openProgress)
    {
        _openProgress->hide();
        _openProgress->deleteLater();
        _openProgress = nullptr;
    }
    _statusBar->showMessage(success ? message : tr("打开项目失败"), success ? 3000 : 5000);
}

void ProjectLifecyclePresenter::showSaveProgress()
{
    if (!_saveProgress)
    {
        _saveProgress = new QProgressDialog(tr("正在保存项目..."), QString(), 0, 0, _window);
        _saveProgress->setWindowModality(Qt::ApplicationModal);
        _saveProgress->setCancelButton(nullptr);
        _saveProgress->setMinimumDuration(0);
    }
    _saveProgress->show();
}

void ProjectLifecyclePresenter::finishSaveProgress(bool success)
{
    if (_saveProgress)
    {
        _saveProgress->hide();
        _saveProgress->deleteLater();
        _saveProgress = nullptr;
    }
    _statusBar->showMessage(success ? tr("保存完成") : tr("保存失败"), success ? 3000 : 5000);

    if (_closeSavePending)
    {
        _closeSavePending = false;
        if (success)
        {
            emit closeAfterSaveRequested();
        }
    }
}

void ProjectLifecyclePresenter::updateWindowTitle(bool dirty)
{
    const QString projectPath = _projectManager ? _projectManager->currentProjectPath() : QString{};
    if (projectPath.trimmed().isEmpty())
    {
        _window->setWindowTitle(QStringLiteral("PlaScan"));
        return;
    }

    QString name = QFileInfo(projectPath).baseName();
    if (name.isEmpty())
    {
        name = QFileInfo(projectPath).fileName();
    }
    _window->setWindowTitle(dirty ? QStringLiteral("PlaScan - %1*").arg(name)
                                  : QStringLiteral("PlaScan - %1").arg(name));
}
