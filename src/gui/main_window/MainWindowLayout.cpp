#include "MainWindow.h"

#include "ui_MainWindow.h"
#include "HenuBrandWidget.h"
#include "Logger.h"
#include "LogPanel.h"
#include "MainMenu.h"
#include "PhotoStripWidget.h"
#include "ProjectDashboardWidget.h"
#include "SelectionPropertiesWidget.h"
#include "WorkPanelWidget.h"
#include "WorkspaceCenterWidget.h"
#include "WorkspacePanelController.h"

#include <QAction>
#include <QDockWidget>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSplitter>
#include <QTabWidget>
#include <QToolBar>
#include <QWidgetAction>

namespace
{
    constexpr int SelectionPropertiesMinHeight = 80;
    constexpr int WorkDockMinHeight = 90;
    constexpr int PhotosDockMinHeight = 90;
    constexpr int DockMinWidth = 160;
    constexpr int DockMinHeight = 80;

    void configureMovableDock(QDockWidget* dock)
    {
        if (!dock)
        {
            return;
        }

        dock->setAllowedAreas(Qt::AllDockWidgetAreas);
        dock->setFeatures(QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetMovable |
                          QDockWidget::DockWidgetFloatable);
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
    _canvas = _workspaceCenter->canvas();
    if (centralWidget())
    {
        centralWidget()->setMinimumSize(0, 0);
        centralWidget()->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    }
    _mainSplitter->setMinimumSize(0, 0);
    _mainSplitter->setStretchFactor(1, 1);
    setDockOptions(QMainWindow::AnimatedDocks | QMainWindow::AllowNestedDocks | QMainWindow::AllowTabbedDocks |
                   QMainWindow::GroupedDragging);
    // 左下角归左侧 Dock 区域，使“工作区 + 资源属性”占据完整左列。
    // 工作、照片和控制台组成中央视图下方的一组标签页。
    setCorner(Qt::BottomLeftCorner, Qt::LeftDockWidgetArea);
    setTabPosition(Qt::BottomDockWidgetArea, QTabWidget::South);

    _log = _ui->logPanel;
    _logDock = _ui->logDock;
    configureMovableDock(_logDock);
    _logDock->setVisible(false);

    connect(_log,
            &LogPanel::unreadCountsChanged,
            this,
            [this](int warningCount, int errorCount)
            {
                QString title = tr("控制台");
                if (errorCount > 0 || warningCount > 0)
                {
                    QStringList counts;
                    if (errorCount > 0)
                    {
                        counts << tr("%1 错误").arg(errorCount);
                    }
                    if (warningCount > 0)
                    {
                        counts << tr("%1 警告").arg(warningCount);
                    }
                    title += QStringLiteral(" · ") + counts.join(QStringLiteral(" / "));
                }
                _logDock->setWindowTitle(title);
            });

    LOG_INFO("%s", qUtf8Printable(tr("控制台已就绪")));
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
    connect(_selectionProperties,
            &SelectionPropertiesWidget::selectionStateChanged,
            this,
            [this](bool hasSelection)
            {
                _propertiesDockVisibleOutsideReference = hasSelection;
                const bool referenceTabActive =
                    _leftTabs && _referencePanel && _leftTabs->currentWidget() == _referencePanel;
                const bool visible = hasSelection && !referenceTabActive;
                if (_workspacePanels)
                {
                    _workspacePanels->setPanelVisible(WorkspacePanelId::Properties, visible);
                }
                else if (_propertiesDock)
                {
                    _propertiesDock->setVisible(visible);
                    if (visible)
                    {
                        _propertiesDock->raise();
                    }
                }
            });

    _workPanel = new WorkPanelWidget(this);
    _workPanel->setMinimumHeight(WorkDockMinHeight);
    _workPanel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    connect(_dashboard, &ProjectDashboardWidget::taskSnapshotsChanged, _workPanel, &WorkPanelWidget::setTaskSnapshots);
    connect(_workPanel,
            &WorkPanelWidget::logRangeRequested,
            this,
            [this](qulonglong firstSequence,
                   qulonglong lastSequence,
                   const QString &taskId)
            {
                if (_logDock && _log)
                {
                    _logDock->show();
                    _logDock->raise();
                    _log->focusLogRange(firstSequence, lastSequence, taskId);
                }
            });

    _workDock = new QDockWidget(tr("工作"), this);
    _workDock->setObjectName(QStringLiteral("workDock"));
    configureMovableDock(_workDock);
    _workDock->setWidget(_workPanel);

    _photoStrip = new PhotoStripWidget(this);
    _photoStrip->setMinimumHeight(PhotosDockMinHeight);
    _photoStrip->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    _photosDock = new QDockWidget(tr("照片"), this);
    _photosDock->setObjectName(QStringLiteral("photosDock"));
    configureMovableDock(_photosDock);
    _photosDock->setWidget(_photoStrip);

    restoreDefaultProjectDockLayout();
    connect(_leftTabs, &QTabWidget::currentChanged, this, [this](int) { updatePropertiesDockForCurrentTab(); });
    updatePropertiesDockForCurrentTab();
}

void MainWindow::updatePropertiesDockForCurrentTab()
{
    if (!_leftTabs || !_propertiesDock || !_referencePanel)
    {
        return;
    }

    const bool referenceTabActive = _leftTabs->currentWidget() == _referencePanel;
    QAction* propertiesAction = _mainMenu ? _mainMenu->togglePropertiesAction() : nullptr;
    if (referenceTabActive)
    {
        if (!_propertiesDockSuppressed)
        {
            _propertiesDockVisibleOutsideReference =
                propertiesAction ? propertiesAction->isChecked() : !_propertiesDock->isHidden();
            _propertiesDockSuppressed = true;
        }
        const QSignalBlocker dockBlocker(_propertiesDock);
        _propertiesDock->hide();
        if (propertiesAction)
        {
            propertiesAction->setEnabled(false);
        }
        return;
    }

    if (propertiesAction)
    {
        propertiesAction->setEnabled(true);
    }
    if (!_propertiesDockSuppressed)
    {
        return;
    }

    const QSignalBlocker dockBlocker(_propertiesDock);
    _propertiesDock->setVisible(_propertiesDockVisibleOutsideReference);
    _propertiesDockSuppressed = false;
}

// ============================================================
//  setupLogDock — 控制台 Dock 标题栏与菜单状态同步
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

    QToolBar* toolBar = _mainMenu->toolBar();
    if (!toolBar)
    {
        return;
    }

    _henuBrandWidget = new HenuBrandWidget(toolBar);
    _henuBrandAction = new QWidgetAction(toolBar);
    _henuBrandAction->setObjectName(QStringLiteral("henuBrandToolbarAction"));
    _henuBrandAction->setDefaultWidget(_henuBrandWidget);

    QAction* firstAction = toolBar->actions().isEmpty() ? nullptr : toolBar->actions().first();
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
