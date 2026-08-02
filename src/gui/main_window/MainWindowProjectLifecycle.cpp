#include "MainWindow.h"

#include "ui_MainWindow.h"

#include <QApplication>
#include <QAction>
#include <QSplitter>
#include <QDockWidget>
#include <QToolBar>
#include <QMenuBar>
#include <QStatusBar>
#include <QFileInfo>
#include <QDesktopServices>
#include <QUrl>
#include <QJsonObject>
#include <QMessageBox>
#include <QSaveFile>
#include <QTabWidget>
#include <QTextStream>
#include <QCloseEvent>
#include <QTimer>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QScopedValueRollback>
#include <QWidgetAction>

#include <algorithm>
#include <utility>

#include "CanvasWidget.h"
#include "ImageViewRotationSettings.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
#include "LogPanel.h"
#include "MainMenu.h"
#include "MenuWorkflowController.h"
#include "ReconstructionWorkflowController.h"
#include "tie_points/CleanTiePointsDialog.h"
#include "tie_points/CreateTiePointsDialog.h"
#include "tie_points/MatchPairSelectorDialog.h"
#include "MatchPhotosTask.h"
#include "camera/ForwardIntersectionCheckDialog.h"
#include "camera/ForwardIntersectionResultsDialog.h"
#include "HenuBrandWidget.h"
#include "ProjectManager.h"
#include "project/ProjectIO.h"
#include "ProjectData.h"
#include "ProjectDashboardWidget.h"
#include "PhotoStripWidget.h"
#include "AppConfigManager.h"
#include "DataTreeWidget.h"
#include "ReferencePanelWidget.h"
#include "SelectionPropertiesWidget.h"
#include "ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"
#include "WorkspaceCenterWidget.h"
#include "WorkspacePanelController.h"
#include "ProjectUiHydrator.h"
#include "TiePointWorkflowController.h"
#include "ProjectLifecyclePresenter.h"
#include "camera/CameraModel3DDialog.h"
#include "tie_points/ThinTiePointsDialog.h"
#include "LayerRenderer.h"
#include "Logger.h"
#include "ModelDropSupport.h"
#include "MarkerWorkspaceController.h"
#include "MarkerReferencePanel.h"
#include "MarkerFocusMeasurementDialog.h"
#include "DetectMarkersDialog.h"
#include "MarkerDetectionReviewDialog.h"
#include "PrintMarkersDialog.h"


void MainWindow::onClearRecentRequested()
{
    if (!_config)
    {
        return;
    }
    auto btn = QMessageBox::question(this, tr("清空最近打开"),
        tr("确定要清空最近打开的项目列表吗？"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (btn == QMessageBox::Yes)
    {
        _config->recentProjects()->clearRecentProjects();
        if (_mainMenu)
        {
            _mainMenu->setRecentProjects(_config->recentProjects()->recentProjects());
        }
        statusBar()->showMessage(tr("已清空最近打开列表"), 3000);
    }
}

// ============================================================
//  项目生命周期
// ============================================================

void MainWindow::onProjectOpened(const QString &plascanPath)
{
    if (_mainMenu && _mainMenu->saveAction())
    {
        _mainMenu->saveAction()->setEnabled(true);
    }

    if (_dataTree)
    {
        _dataTree->clearTransientResources();
    }

    QFileInfo fi(plascanPath);
    QString name = fi.baseName();
    if (name.isEmpty())
    {
        name = fi.fileName();
    }
    setWindowTitle(QStringLiteral("PlaScan - %1").arg(name));
    statusBar()->showMessage(tr("已打开项目：%1").arg(plascanPath), 4000);

    if (_canvas)
    {
        _canvas->setProperty("currentProjectPath", plascanPath);
    }
    if (_markerWorkspaceController)
    {
        QString marker_error;
        if (!_markerWorkspaceController->openProject(&marker_error))
        {
            QMessageBox::warning(this,
                                 QStringLiteral("加载标记点"),
                                 QStringLiteral("标记点数据未加载：%1").arg(marker_error));
        }
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->showModelView();
    }
    if (_dataTree)
    {
        _dataTree->setProjectPath(plascanPath);
        if (_projectData)
        {
            _dataTree->setChunkContext(
                _projectData->chunks(),
                _projectData->activeChunkId());
        }
    }
    if (_photoStrip)
    {
        _photoStrip->setProjectPath(plascanPath);
    }
    if (_projectManager)
    {
        scheduleProjectMetadataRefresh(_projectManager->coreProjectMeta());
    }

    if (_config && _mainMenu)
    {
        _config->recentProjects()->addRecentProject(plascanPath);
        _mainMenu->setRecentProjects(_config->recentProjects()->recentProjects());
    }

    if (!_projectManager)
    {
        return;
    }

    const QJsonObject ui = _projectManager->loadUiSettings();
    applyUiSettings(ui);
    persistCurrentUiSettings();
}

void MainWindow::openMarkerFocusMeasurement(const QString &markerId,
                                            const QString &preferredImagePath)
{
    if (!_markerWorkspaceController || !_projectData || markerId.isEmpty())
    {
        return;
    }

    QString anchor_path;
    try
    {
        const auto &marker = _markerWorkspaceController->markerSet().marker(markerId);
        auto paths_equal = [](const QString &left, const QString &right)
        {
#ifdef Q_OS_WIN
            return QDir::cleanPath(left).compare(QDir::cleanPath(right), Qt::CaseInsensitive) == 0;
#else
            return QDir::cleanPath(left) == QDir::cleanPath(right);
#endif
        };
        for (const auto &projection : marker.projections)
        {
            if (!preferredImagePath.isEmpty()
                && paths_equal(projection.imagePathSnapshot, preferredImagePath)
                && projection.state != xjw::control_points::ProjectionState::Blocked
                && projection.state != xjw::control_points::ProjectionState::Disabled)
            {
                anchor_path = projection.imagePathSnapshot;
                break;
            }
        }
        if (anchor_path.isEmpty())
        {
            for (const auto &projection : marker.projections)
            {
                if (projection.state != xjw::control_points::ProjectionState::Blocked
                    && projection.state != xjw::control_points::ProjectionState::Disabled)
                {
                    anchor_path = projection.imagePathSnapshot;
                    break;
                }
            }
        }
    }
    catch (const std::exception &exception)
    {
        QMessageBox::warning(this,
                             QStringLiteral("聚焦标记量测"),
                             QString::fromUtf8(exception.what()));
        return;
    }

    if (anchor_path.isEmpty())
    {
        QMessageBox::information(this,
                                 QStringLiteral("聚焦标记量测"),
                                 QStringLiteral("该标记尚无可用投影，请先在照片中放置一次。"));
        return;
    }

    auto *dialog = new xjw::gui::markers::MarkerFocusMeasurementDialog(this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if (!dialog->setContext(_markerWorkspaceController,
                            _projectData,
                            markerId,
                            anchor_path))
    {
        dialog->deleteLater();
        QMessageBox::information(this,
                                 QStringLiteral("聚焦标记量测"),
                                 QStringLiteral("没有可供量测的其他项目照片。"));
        return;
    }
    dialog->show();
    dialog->raise();
    dialog->activateWindow();
}

void MainWindow::scheduleProjectMetadataRefresh(const QJsonObject &meta)
{
    if (_projectUiHydrator)
    {
        _projectUiHydrator->schedule(meta);
    }
}

void MainWindow::onProjectClosed()
{
    if (_projectUiHydrator)
    {
        _projectUiHydrator->cancel();
    }
    _imageViewRotations = QJsonObject{};
    if (_markerWorkspaceController)
    {
        _markerWorkspaceController->closeProject();
    }
    if (_mainMenu)
    {
        if (_mainMenu->saveAction())
        {
            _mainMenu->saveAction()->setEnabled(false);
        }
        _mainMenu->setImageDisplayReady(false);
        _mainMenu->setDepthOverlayAvailable(false);
    }
    if (_canvas)
    {
        _canvas->setProperty("currentProjectPath", QString());
    }
    if (_dataTree)
    {
        _dataTree->clearProject();
    }
    if (_photoStrip)
    {
        _photoStrip->setProjectPath(QString());
        _photoStrip->clearPhotos();
    }
    if (_dashboard)
    {
        _dashboard->clear();
    }
    if (_referencePanel)
    {
        _referencePanel->loadFromJson(QJsonObject{});
    }
    if (_selectionProperties)
    {
        _selectionProperties->clearSelection();
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->clearProjectView();
    }
    _lastSelectedImage.clear();
    setWindowTitle(QStringLiteral("PlaScan"));
    statusBar()->showMessage(tr("项目已关闭"), 3000);
}

// ============================================================
//  applyUiSettings — 恢复项目范围内的 UI 设置
// ============================================================

void MainWindow::applyUiSettings(const QJsonObject &ui)
{
    QScopedValueRollback<bool> applyingRollback(_applyingUiSettings, true);
    _imageViewRotations = ui.value(QStringLiteral("image_view_rotations")).toObject();

    // 匹配观测显示设置独立于具体算法；关键点从 `.pimatch` 分片按需加载。
    if (_menuWorkflowController)
    {
        _menuWorkflowController->applySavedFeatureDisplayOptions(ui);
    }

    const QJsonObject settings = ui;
    const bool restoredDockState = restoreProjectDockState(settings);
    if (_workspacePanels)
    {
        if (restoredDockState)
        {
            _workspacePanels->syncActions();
        }
        else
        {
            _workspacePanels->applyVisibility(settings);
        }
    }

    if (settings.contains(QStringLiteral("log_display_level")) && _log)
    {
        int lvl = settings.value(QStringLiteral("log_display_level")).toInt(static_cast<int>(Logger::Info));
        _log->setDisplayLevel(static_cast<Logger::Level>(lvl));
    }

    if (_logDock && !_logDock->isHidden() && _log)
    {
        _log->loadFromLogFile();
    }

    if (_mainMenu)
    {
        if (settings.contains(QStringLiteral("henu_brand_visible")) && _mainMenu->toggleHenanUniversityBrandAction())
        {
            const bool on = settings.value(QStringLiteral("henu_brand_visible")).toBool();
            {
                const QSignalBlocker blocker(_mainMenu->toggleHenanUniversityBrandAction());
                _mainMenu->toggleHenanUniversityBrandAction()->setChecked(on);
            }
            setHenanUniversityBrandVisible(on);
        }
    }

    if ((settings.contains(QStringLiteral("active_image_id"))
         || settings.contains(QStringLiteral("active_image_path")))
        && _canvas)
    {
        const QString stateKey =
            settings.value(QStringLiteral("active_image_id")).toString();
        QString imagePath = projectImagePathForStateKey(stateKey);
        if (imagePath.isEmpty())
        {
            imagePath = projectImagePathForStateKey(
                settings.value(QStringLiteral("active_image_path"))
                    .toString());
        }
        if (!imagePath.isEmpty() && QFileInfo::exists(imagePath))
        {
            const auto session = _projectManager
                ? _projectManager->currentSessionContext()
                : xjw::gui::project::ProjectSessionContext{};
            QTimer::singleShot(100, this, [this, imagePath, session]()
            {
                if (!_projectManager || !_projectManager->isCurrentSession(session))
                {
                    return;
                }
                if (isProjectPhotoPath(imagePath))
                {
                    selectPhoto(imagePath, true);
                    return;
                }
                if (_workspaceCenter)
                {
                    _workspaceCenter->showImageView(imagePath);
                }
                _lastSelectedImage = imagePath;
            });
        }
    }
}

// ============================================================
//  closeEvent — 退出时保存/提示
// ============================================================

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (_projectLifecyclePresenter && _projectLifecyclePresenter->isCloseSavePending())
    {
        event->ignore();
        return;
    }

    const bool hasProject = _projectManager
        && !_projectManager->currentProjectPath().trimmed().isEmpty();
    if (hasProject)
    {
        persistCurrentUiSettings();
    }

    if (hasProject && _projectManager->isDirty())
    {
        auto btn = QMessageBox::warning(this, tr("未保存的更改"),
            tr("当前项目有未保存的更改。是否保存？"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);

        if (btn == QMessageBox::Cancel)
        {
            event->ignore();
            return;
        }

        if (btn == QMessageBox::Save)
        {
            _projectLifecyclePresenter->requestCloseAfterSave();
            event->ignore();
            return;
        }
        else if (btn == QMessageBox::Discard)
        {
            _projectManager->discardTemporaryMeta();
        }
    }

    if (_config)
    {
        _config->windowState()->save(this);
    }

    event->accept();
    QMainWindow::closeEvent(event);
}
