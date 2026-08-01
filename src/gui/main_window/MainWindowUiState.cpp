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
#include <QJsonArray>
#include <QJsonObject>
#include <QMessageBox>
#include <QProgressDialog>
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
#include "TaskStatusWidget.h"
#include "ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"
#include "WorkspaceCenterWidget.h"
#include "WorkspacePanelController.h"
#include "ProjectUiHydrator.h"
#include "TiePointWorkflowController.h"
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

namespace
{
constexpr int ProjectDockLayoutVersion = 2;
}

QJsonObject MainWindow::currentProjectMeta() const
{
    return _projectManager ? _projectManager->currentMeta() : QJsonObject{};
}

bool MainWindow::isProjectPhotoPath(const QString &imagePath) const
{
    if (imagePath.isEmpty())
    {
        return false;
    }

    const QFileInfo targetInfo(imagePath);
    const QString targetPath = QDir::cleanPath(imagePath);
    const QString targetAbsPath = targetInfo.exists()
        ? QDir::cleanPath(targetInfo.absoluteFilePath())
        : QString();
    const QString projectDirPath = _projectManager
        ? QFileInfo(_projectManager->currentProjectPath()).absolutePath()
        : QString();
    const QDir projectDir(projectDirPath);
    const QJsonArray images = xjw::common::project::projectImageEntries(currentProjectMeta());

    auto matchesTarget = [&targetPath, &targetAbsPath](const QString &candidatePath)
    {
        if (candidatePath.isEmpty())
        {
            return false;
        }

        const QString cleanCandidate = QDir::cleanPath(candidatePath);
        return targetPath == cleanCandidate || targetAbsPath == cleanCandidate;
    };

    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        const QString entryPath = image.value(QStringLiteral("path")).toString();
        if (entryPath.isEmpty())
        {
            continue;
        }

        const QFileInfo entryInfo(entryPath);
        const QString entryCleanPath = QDir::cleanPath(entryPath);
        const QString entryAbsPath = entryInfo.exists()
            ? QDir::cleanPath(entryInfo.absoluteFilePath())
            : QString();
        const QString projectResolvedPath =
            (!projectDirPath.isEmpty() && entryInfo.isRelative())
                ? QDir::cleanPath(projectDir.absoluteFilePath(entryPath))
                : QString();

        if (matchesTarget(entryCleanPath)
            || matchesTarget(entryAbsPath)
            || matchesTarget(projectResolvedPath))
        {
            return true;
        }
    }

    return false;
}

void MainWindow::selectPhoto(const QString &imagePath, bool openImage)
{
    if (imagePath.isEmpty())
    {
        return;
    }

    _lastSelectedImage = imagePath;
    if (_selectionProperties)
    {
        _selectionProperties->showPhotoProperties(currentProjectMeta(), imagePath);
    }
    if (_photoStrip)
    {
        _photoStrip->setCurrentPhoto(imagePath);
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->highlightCameraForImage(imagePath);
        if (openImage)
        {
            _workspaceCenter->showImageView(imagePath);
        }
    }
    saveUiSetting(QJsonObject{
        {QStringLiteral("active_image_id"),
         projectImageStateKey(imagePath)},
        {QStringLiteral("active_image_path"), QString()}
    });
}

QString MainWindow::projectImageStateKey(const QString &imagePath) const
{
    const QString requested =
        QDir::cleanPath(QFileInfo(imagePath).absoluteFilePath());
    const QJsonArray images = _projectManager
        ? _projectManager->coreProjectMeta()
              .value(QStringLiteral("images"))
              .toArray()
        : QJsonArray{};
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        const QString candidate =
            QDir::cleanPath(
                QFileInfo(image.value(QStringLiteral("path")).toString())
                    .absoluteFilePath());
#ifdef Q_OS_WIN
        const bool samePath =
            candidate.compare(requested, Qt::CaseInsensitive) == 0;
#else
        const bool samePath = candidate == requested;
#endif
        if (!samePath)
        {
            continue;
        }
        const QString imageId =
            image.value(QStringLiteral("image_uuid")).toString().trimmed();
        return imageId.isEmpty()
            ? imagePath
            : QStringLiteral("image:%1").arg(imageId);
    }
    return imagePath;
}

QString MainWindow::projectImagePathForStateKey(
    const QString &stateKey) const
{
    const QJsonArray images = _projectManager
        ? _projectManager->coreProjectMeta()
              .value(QStringLiteral("images"))
              .toArray()
        : QJsonArray{};
    const QString imageId = stateKey.startsWith(QStringLiteral("image:"))
        ? stateKey.mid(6)
        : QString();
    QString basenameMatch;
    bool basenameUnique = true;
    const QString legacyName = QFileInfo(stateKey).fileName();
    for (const QJsonValue &value : images)
    {
        const QJsonObject image = value.toObject();
        const QString path =
            image.value(QStringLiteral("path")).toString();
        if (!imageId.isEmpty()
            && image.value(QStringLiteral("image_uuid")).toString()
                == imageId)
        {
            return path;
        }
        if (imageId.isEmpty() && path == stateKey)
        {
            return path;
        }
        if (imageId.isEmpty()
            && !legacyName.isEmpty()
            && QFileInfo(path).fileName() == legacyName)
        {
            if (basenameMatch.isEmpty())
            {
                basenameMatch = path;
            }
            else
            {
                basenameUnique = false;
            }
        }
    }
    return basenameUnique ? basenameMatch : QString();
}

void MainWindow::selectResource(const QString &section, const QString &resourcePath)
{
    if (_selectionProperties)
    {
        _selectionProperties->showResourceProperties(currentProjectMeta(), section, resourcePath);
    }
    if (_photoStrip)
    {
        _photoStrip->setCurrentPhoto(QString());
    }
    if (_workspaceCenter)
    {
        _workspaceCenter->clearHighlightedCamera();
    }
}

QJsonObject MainWindow::currentUiSettingsSnapshot() const
{
    QJsonObject settings = _workspacePanels
        ? _workspacePanels->visibilitySnapshot()
        : QJsonObject{};
    settings[QStringLiteral("bottom_panel")] = currentBottomPanelKey();
    settings[QStringLiteral("dock_layout_version")] = ProjectDockLayoutVersion;
    settings[QStringLiteral("dock_state")] = QString::fromLatin1(saveState().toBase64());

    if (_mainMenu && _mainMenu->toggleHenanUniversityBrandAction())
    {
        settings[QStringLiteral("henu_brand_visible")] =
            _mainMenu->toggleHenanUniversityBrandAction()->isChecked();
    }
    else if (_henuBrandAction)
    {
        settings[QStringLiteral("henu_brand_visible")] = _henuBrandAction->isVisible();
    }
    settings[QStringLiteral("image_view_rotations")] = _imageViewRotations;

    return settings;
}

void MainWindow::restoreDefaultProjectDockLayout()
{
    if (!_workspaceDock || !_propertiesDock || !_photosDock)
    {
        return;
    }

    addDockWidget(Qt::LeftDockWidgetArea, _workspaceDock);
    addDockWidget(Qt::LeftDockWidgetArea, _propertiesDock);
    splitDockWidget(_workspaceDock, _propertiesDock, Qt::Vertical);
    addDockWidget(Qt::BottomDockWidgetArea, _photosDock);

    if (_workspacePanels)
    {
        _workspacePanels->ensureRequiredProjectPanelsVisible();
    }
    else
    {
        _workspaceDock->setVisible(true);
        _propertiesDock->setVisible(true);
        _photosDock->setVisible(true);
        _workspaceDock->raise();
        _propertiesDock->raise();
        _photosDock->raise();
    }

    resizeDocks({_workspaceDock}, {320}, Qt::Horizontal);
    resizeDocks({_workspaceDock, _propertiesDock}, {560, 190}, Qt::Vertical);
    resizeDocks({_photosDock}, {120}, Qt::Vertical);
}

bool MainWindow::restoreProjectDockState(const QJsonObject &settings)
{
    const int layoutVersion = settings.value(QStringLiteral("dock_layout_version")).toInt(0);
    if (layoutVersion != ProjectDockLayoutVersion)
    {
        restoreDefaultProjectDockLayout();
        return false;
    }

    const QString encoded = settings.value(QStringLiteral("dock_state")).toString();
    if (encoded.isEmpty())
    {
        restoreDefaultProjectDockLayout();
        return false;
    }

    const QByteArray state = QByteArray::fromBase64(encoded.toLatin1());
    if (state.isEmpty())
    {
        restoreDefaultProjectDockLayout();
        return false;
    }

    if (!restoreState(state))
    {
        restoreDefaultProjectDockLayout();
        return false;
    }
    return true;
}

void MainWindow::persistCurrentUiSettings()
{
    saveUiSetting(currentUiSettingsSnapshot());
}

void MainWindow::saveUiSetting(const QJsonObject &partial)
{
    if (_applyingUiSettings
        || !_projectManager
        || _projectManager->currentProjectPath().trimmed().isEmpty())
    {
        return;
    }
    _projectManager->saveUiSettings(partial);
}

QString MainWindow::currentBottomPanelKey() const
{
    if (_logDock && _logDock->isVisible())
    {
        return QStringLiteral("log");
    }
    if (_photosDock && _photosDock->isVisible())
    {
        return QStringLiteral("photos");
    }
    return QStringLiteral("none");
}

// Interest-point panel removed: onIpBtnClicked is a no-op now.

// ============================================================
//  菜单动作响应
// ============================================================

// Interest-point info UI removed: related slots are no-ops / deleted.

// ============================================================
//  菜单联动持久化槽
// ============================================================

void MainWindow::onLogDisplayLevelChanged(int lvl)
{
    QJsonObject s;
    s[QStringLiteral("log_display_level")] = lvl;
    saveUiSetting(s);
}

// ============================================================
//  项目管理响应槽
// ============================================================

// Ipfind/Ipmatch finish handlers removed (controllers moved/removed)
