#include "MainWindow.h"

#include "ui_MainWindow.h"
#include "AppConfigManager.h"
#include "DataTreeWidget.h"
#include "MainMenu.h"
#include "ModelDropSupport.h"
#include "ProjectManager.h"
#include "WorkspaceCenterWidget.h"

#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileInfo>
#include <QMimeData>
#include <QStatusBar>
#include <QTabWidget>

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
    if (_projectManager)
    {
        _projectManager->waitForResourceCleanup();
    }
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
