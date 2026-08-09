#include "MainWindow.h"

#include "ui_MainWindow.h"
#include "HenuBrandWidget.h"
#include "Logger.h"
#include "MainMenu.h"
#include "PhotoStripWidget.h"
#include "SelectionPropertiesWidget.h"
#include "WorkspaceCenterWidget.h"

#include <QAction>
#include <QDockWidget>
#include <QSizePolicy>
#include <QSplitter>
#include <QTabWidget>
#include <QToolBar>
#include <QWidgetAction>

namespace
{
constexpr int SelectionPropertiesMinHeight = 80;
constexpr int PhotosDockMinHeight = 90;
constexpr int DockMinWidth = 160;
constexpr int DockMinHeight = 80;

void configureMovableDock(QDockWidget *dock)
{
    if (!dock)
    {
        return;
    }

    dock->setAllowedAreas(Qt::AllDockWidgetAreas);
    dock->setFeatures(QDockWidget::DockWidgetClosable
                      | QDockWidget::DockWidgetMovable
                      | QDockWidget::DockWidgetFloatable);
    dock->setMinimumSize(DockMinWidth, DockMinHeight);
    dock->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

} // namespace
void MainWindow::setupUi()
{
    _ui->setupUi(this);

    _mainSplitter = _ui->mainSplitter;
    _leftTabs = _ui->leftTabs;
    _dashboard = _ui->dashboardWidget;
    _dataTree = _ui->dataTree;
    _referencePanel = _ui->referencePanel;
    _workspaceCenter = _ui->workspaceCenter;
    _canvas       = _workspaceCenter->canvas();
    if (centralWidget())
    {
        centralWidget()->setMinimumSize(0, 0);
        centralWidget()->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    _mainSplitter->setMinimumSize(0, 0);
    _mainSplitter->setStretchFactor(1, 1);
    setDockOptions(QMainWindow::AnimatedDocks
                   | QMainWindow::AllowNestedDocks
                   | QMainWindow::AllowTabbedDocks
                   | QMainWindow::GroupedDragging);
    // 左下角归左侧 Dock 区域，使“工作区 + 资源属性”占据完整左列。
    // 照片 Dock 因而只位于中央视图下方，与三维/二维视图组成右列。
    setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);

    _log = _ui->logPanel;
    _logDock = _ui->logDock;
    configureMovableDock(_logDock);
    _logDock->setVisible(false);

    LOG_INFO("%s", qUtf8Printable(tr("日志面板已就绪")));

}

void MainWindow::setupSelectionPanels()
{
    if (!_mainSplitter || !_leftTabs || !_workspaceCenter || _workspaceDock)
    {
        return;
    }

    const int leftIndex = _mainSplitter->indexOf(_leftTabs);
    const int workspaceIndex = _mainSplitter->indexOf(_workspaceCenter);
    if (leftIndex < 0 || workspaceIndex < 0)
    {
        return;
    }

    _leftTabs->setParent(nullptr);
    _leftTabs->setMinimumSize(160, 80);
    _leftTabs->setMaximumWidth(QWIDGETSIZE_MAX);
    _leftTabs->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    _workspaceCenter->setMinimumSize(240, 160);
    _workspaceCenter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    const int currentWorkspaceIndex = _mainSplitter->indexOf(_workspaceCenter);
    if (currentWorkspaceIndex >= 0)
    {
        _mainSplitter->setCollapsible(currentWorkspaceIndex, false);
        _mainSplitter->setStretchFactor(currentWorkspaceIndex, 1);
        _mainSplitter->setSizes({960});
    }

    _workspaceDock = new QDockWidget(tr("工作区"), this);
    _workspaceDock->setObjectName(QStringLiteral("workspaceDock"));
    configureMovableDock(_workspaceDock);
    _workspaceDock->setWidget(_leftTabs);

    _selectionProperties = new SelectionPropertiesWidget(this);
    _selectionProperties->setObjectName(QStringLiteral("selectionProperties"));
    _selectionProperties->setMinimumHeight(SelectionPropertiesMinHeight);
    _selectionProperties->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    _propertiesDock = new QDockWidget(tr("资源属性"), this);
    _propertiesDock->setObjectName(QStringLiteral("propertiesDock"));
    configureMovableDock(_propertiesDock);
    _propertiesDock->setWidget(_selectionProperties);

    _photoStrip = new PhotoStripWidget(this);
    _photoStrip->setMinimumHeight(PhotosDockMinHeight);
    _photoStrip->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    _photosDock = new QDockWidget(tr("照片"), this);
    _photosDock->setObjectName(QStringLiteral("photosDock"));
    configureMovableDock(_photosDock);
    _photosDock->setWidget(_photoStrip);

    restoreDefaultProjectDockLayout();
}

// ============================================================
//  setupLogDock — 日志面板 Dock 标题栏与菜单状态同步
// ============================================================

void MainWindow::setupLogDock()
{
    if (!_logDock)
    {
        return;
    }

    _logDock->setTitleBarWidget(nullptr);
}

void MainWindow::setupHenanUniversityBrand()
{
    if (!_mainMenu || _henuBrandAction)
    {
        return;
    }

    QToolBar *toolBar = _mainMenu->toolBar();
    if (!toolBar)
    {
        return;
    }

    _henuBrandWidget = new HenuBrandWidget(toolBar);
    _henuBrandAction = new QWidgetAction(toolBar);
    _henuBrandAction->setObjectName(QStringLiteral("henuBrandToolbarAction"));
    _henuBrandAction->setDefaultWidget(_henuBrandWidget);

    QAction *firstAction = toolBar->actions().isEmpty() ? nullptr : toolBar->actions().first();
    toolBar->insertAction(firstAction, _henuBrandAction);
    toolBar->insertSeparator(firstAction);
}

void MainWindow::setHenanUniversityBrandVisible(bool visible)
{
    if (_henuBrandAction)
    {
        _henuBrandAction->setVisible(visible);
    }
    if (_henuBrandWidget)
    {
        _henuBrandWidget->setVisible(visible);
    }
}

// ============================================================
//  setupMenuConnections — 菜单/工具栏信号连接
// ============================================================
