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

// ============================================================
//  构造 / 析构
// ============================================================
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , _ui(new Ui::MainWindow)
{
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::Window;
    flags |= Qt::WindowMinimizeButtonHint;
    flags |= Qt::WindowMaximizeButtonHint;
    flags |= Qt::WindowCloseButtonHint;
    flags |= Qt::WindowSystemMenuHint;
    flags &= ~Qt::FramelessWindowHint;
    setWindowFlags(flags);
    setWindowTitle(QStringLiteral("PlaScan"));
    _config   = new AppConfigManager(this);

    setupUi();
    setupSelectionPanels();
    _mainMenu = new MainMenu(this);
    setupHenanUniversityBrand();
    _config->windowState()->load(this);

    if (windowState().testFlag(Qt::WindowFullScreen))
    {
        setWindowState((windowState() & ~Qt::WindowFullScreen) | Qt::WindowMaximized);
    }

    setupLogDock();
    setupMenuConnections();
    setupProjectManager();

    setAcceptDrops(true);
    statusBar()->showMessage(tr("就绪"));
}

MainWindow::~MainWindow()
{
    delete _ui;
}

void MainWindow::openProjectFromPath(const QString &projectPath)
{
    if (projectPath.trimmed().isEmpty() || !_projectManager)
    {
        return;
    }

    persistCurrentUiSettings();
    _projectManager->openProjectFromPath(projectPath);
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    if (event && event->mimeData()
        && !xjw::gui::main_window::firstStandaloneModelFile(event->mimeData()->urls()).isEmpty())
    {
        event->acceptProposedAction();
        return;
    }

    QMainWindow::dragEnterEvent(event);
}

void MainWindow::dropEvent(QDropEvent *event)
{
    if (!event || !event->mimeData())
    {
        QMainWindow::dropEvent(event);
        return;
    }

    const QString modelPath =
        xjw::gui::main_window::firstStandaloneModelFile(event->mimeData()->urls());
    if (modelPath.isEmpty())
    {
        QMainWindow::dropEvent(event);
        return;
    }

    if (_workspaceCenter)
    {
        _workspaceCenter->showModelFile(modelPath);
        statusBar()->showMessage(tr("已加载三维模型：%1").arg(QFileInfo(modelPath).fileName()), 5000);
    }
    if (_dataTree)
    {
        _dataTree->addTransientModel(modelPath);
    }
    if (_leftTabs && _dataTree)
    {
        _leftTabs->setCurrentWidget(_dataTree);
    }
    event->acceptProposedAction();
}
