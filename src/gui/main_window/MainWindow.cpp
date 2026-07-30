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

#include "AlgorithmCompat.h"
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
constexpr int SelectionPropertiesMinHeight = 80;
constexpr int PhotosDockMinHeight = 90;
constexpr int DockMinWidth = 160;
constexpr int DockMinHeight = 80;
constexpr int ProjectDockLayoutVersion = 2;

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

xjw::matchphotos::MatchPhotosProfile tiePointProfileFromAccuracy(const QString &accuracy)
{
    if (accuracy == QStringLiteral("highest") || accuracy == QStringLiteral("high"))
    {
        return xjw::matchphotos::MatchPhotosProfile::HighAccuracy;
    }
    if (accuracy == QStringLiteral("lowest") || accuracy == QStringLiteral("low"))
    {
        return xjw::matchphotos::MatchPhotosProfile::Fast;
    }
    return xjw::matchphotos::MatchPhotosProfile::Auto;
}

xjw::matchphotos::PairSelectionPreset pairPresetFromAccuracy(const QString &accuracy)
{
    if (accuracy == QStringLiteral("highest") || accuracy == QStringLiteral("high"))
    {
        return xjw::matchphotos::PairSelectionPreset::HighAccuracy;
    }
    if (accuracy == QStringLiteral("lowest") || accuracy == QStringLiteral("low"))
    {
        return xjw::matchphotos::PairSelectionPreset::Fast;
    }
    return xjw::matchphotos::PairSelectionPreset::Auto;
}

int maxImageDimFromAccuracy(const QString &accuracy)
{
    if (accuracy == QStringLiteral("highest"))
    {
        return 4096;
    }
    if (accuracy == QStringLiteral("high"))
    {
        return 3072;
    }
    if (accuracy == QStringLiteral("low"))
    {
        return 1600;
    }
    if (accuracy == QStringLiteral("lowest"))
    {
        return 1200;
    }
    return 2048;
}

} // namespace

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

// ============================================================
//  setupUi — 创建核心布局组件
// ============================================================

void MainWindow::setupUi()
{
    _ui->setupUi(this);

    _mainSplitter = _ui->mainSplitter;
    _leftTabs = _ui->leftTabs;
    _dashboard = _ui->dashboardWidget;
    _dataTree = _ui->dataTree;
    _referencePanel = _ui->referencePanel;
    _markerReferencePanel = new xjw::gui::markers::MarkerReferencePanel(_leftTabs);
    _leftTabs->addTab(_markerReferencePanel, QStringLiteral("标记"));
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

void MainWindow::setupMenuConnections()
{
    if (!_mainMenu)
    {
        return;
    }

    _workspacePanels = new WorkspacePanelController(this);
    _workspacePanels->registerDock(WorkspacePanelId::Workspace,
                                   _mainMenu->toggleWorkspaceAction(),
                                   _workspaceDock);
    _workspacePanels->registerDock(WorkspacePanelId::Properties,
                                   _mainMenu->togglePropertiesAction(),
                                   _propertiesDock);
    _workspacePanels->registerDock(WorkspacePanelId::Photos,
                                   _mainMenu->togglePhotosAction(),
                                   _photosDock);
    _workspacePanels->registerDock(WorkspacePanelId::Log,
                                   _mainMenu->toggleLogAction(),
                                   _logDock);
    _workspacePanels->registerToolBar(WorkspacePanelId::MainToolbar,
                                      _mainMenu->toggleMainToolbarAction(),
                                      _mainMenu->toolBar());
    _mainMenu->setManagedWindowActions(
        _workspacePanels->actions(WorkspacePanelKind::Dock),
        _workspacePanels->actions(WorkspacePanelKind::ToolBar));
    connect(_workspacePanels,
            &WorkspacePanelController::visibilitySettingChanged,
            this,
            [this](WorkspacePanelId id, bool visible)
            {
                const QString settingKey =
                    workspacePanelDescriptor(id).settingKey;
                QJsonObject settings{{settingKey, visible}};
                if (id == WorkspacePanelId::Log)
                {
                    settings[QStringLiteral("bottom_panel")] = currentBottomPanelKey();
                    if (visible && _log)
                    {
                        _log->loadFromLogFile();
                    }
                }
                saveUiSetting(settings);
            });
    if (QAction *restoreLayoutAction =
            _mainMenu->restoreDefaultWindowLayoutAction())
    {
        connect(restoreLayoutAction, &QAction::triggered, this, [this]()
        {
            restoreDefaultProjectDockLayout();
            if (_workspacePanels)
            {
                _workspacePanels->restoreDefaultVisibility();
            }
            persistCurrentUiSettings();
        });
    }

    if (_workspaceCenter)
    {
        auto updateContextualToolbar = [this](WorkspaceCenterWidget::ViewMode mode)
        {
            const bool showModelTools = mode == WorkspaceCenterWidget::ViewMode::Model;
            const bool showImageTools = mode == WorkspaceCenterWidget::ViewMode::Image;
            _mainMenu->setContextualToolbarVisibility(showModelTools, showImageTools);
            _mainMenu->setImageDisplayReady(_canvas && _canvas->hasDisplayImage());
        };
        connect(_workspaceCenter,
                &WorkspaceCenterWidget::viewModeChanged,
                this,
                updateContextualToolbar);
        updateContextualToolbar(_workspaceCenter->currentViewMode());
    }

    if (_mainMenu->zoomInAction())
    {
        connect(_mainMenu->zoomInAction(), &QAction::triggered, this, [this]()
        {
            if (!_workspaceCenter)
            {
                return;
            }
            if (_workspaceCenter->currentViewMode() == WorkspaceCenterWidget::ViewMode::Image)
            {
                _canvas->zoomIn();
            }
            else if (_workspaceCenter->currentViewMode() == WorkspaceCenterWidget::ViewMode::Model)
            {
                _workspaceCenter->modelView()->zoomIn();
            }
        });
    }
    if (_mainMenu->zoomOutAction())
    {
        connect(_mainMenu->zoomOutAction(), &QAction::triggered, this, [this]()
        {
            if (!_workspaceCenter)
            {
                return;
            }
            if (_workspaceCenter->currentViewMode() == WorkspaceCenterWidget::ViewMode::Image)
            {
                _canvas->zoomOut();
            }
            else if (_workspaceCenter->currentViewMode() == WorkspaceCenterWidget::ViewMode::Model)
            {
                _workspaceCenter->modelView()->zoomOut();
            }
        });
    }
    if (_mainMenu->toggleFullScreenAction())
    {
        connect(_mainMenu->toggleFullScreenAction(), &QAction::triggered, this, [this]()
        {
            if (isFullScreen())
            {
                setWindowState(_windowStateBeforeFullScreen & ~Qt::WindowFullScreen);
                return;
            }
            _windowStateBeforeFullScreen = windowState();
            showFullScreen();
        });
    }
    if (_mainMenu->resetViewAction())
    {
        connect(_mainMenu->resetViewAction(), &QAction::triggered, this, [this]()
        {
            if (_workspaceCenter)
            {
                _workspaceCenter->resetActiveView();
            }
        });
    }
    if (_mainMenu->rotateImageLeftAction())
    {
        connect(_mainMenu->rotateImageLeftAction(),
                &QAction::triggered, _canvas, &CanvasWidget::rotateLeft);
    }
    if (_mainMenu->rotateImageRightAction())
    {
        connect(_mainMenu->rotateImageRightAction(),
                &QAction::triggered, _canvas, &CanvasWidget::rotateRight);
    }
    if (_mainMenu->showFeaturePointsAction())
    {
        connect(_mainMenu->showFeaturePointsAction(), &QAction::toggled,
                _canvas, &CanvasWidget::setShowInterestPoints);
    }
    if (_mainMenu->showFeatureResidualsAction())
    {
        connect(_mainMenu->showFeatureResidualsAction(), &QAction::toggled,
                _canvas, &CanvasWidget::setShowFeatureResiduals);
    }
    if (_mainMenu->showMaskOverlayAction())
    {
        connect(_mainMenu->showMaskOverlayAction(), &QAction::toggled,
                _canvas, &CanvasWidget::setShowMaskOverlay);
    }
    if (_mainMenu->showDepthOverlayAction())
    {
        _mainMenu->showDepthOverlayAction()->setEnabled(false);
        connect(_mainMenu->showDepthOverlayAction(),
                &QAction::toggled,
                _canvas,
                &CanvasWidget::setDepthOverlayEnabled);
    }
    const auto connect_depth_level = [this](QAction *action,
                                             xjw::gui::views::DepthOverlayLevel level)
    {
        if (!action)
        {
            return;
        }
        connect(action, &QAction::triggered, this, [this, action, level]()
        {
            if (action->isChecked())
            {
                _canvas->setDepthOverlayLevel(level);
            }
        });
    };
    connect_depth_level(_mainMenu->depthOverlayAllLevelsAction(),
                        xjw::gui::views::DepthOverlayLevel::Final);
    connect_depth_level(_mainMenu->depthOverlayLevel1Action(),
                        xjw::gui::views::DepthOverlayLevel::Level1);
    connect_depth_level(_mainMenu->depthOverlayLevel2Action(),
                        xjw::gui::views::DepthOverlayLevel::Level2);
    connect_depth_level(_mainMenu->depthOverlayLevel3Action(),
                        xjw::gui::views::DepthOverlayLevel::Level3);
    if (_mainMenu->showDepthIntensityAction())
    {
        connect(_mainMenu->showDepthIntensityAction(),
                &QAction::toggled,
                _canvas,
                &CanvasWidget::setDepthIntensityVisible);
    }
    if (_canvas)
    {
        connect(_canvas, &CanvasWidget::displayImageReadyChanged, this, [this](bool ready)
        {
            _mainMenu->setImageDisplayReady(ready);
        });
        connect(_canvas, &CanvasWidget::interestPointsVisibilityChanged, this, [this](bool visible)
        {
            if (QAction *action = _mainMenu->showFeaturePointsAction())
            {
                const QSignalBlocker blocker(action);
                action->setChecked(visible);
            }
        });
        connect(_canvas, &CanvasWidget::featureResidualVisibilityChanged, this, [this](bool visible)
        {
            if (QAction *action = _mainMenu->showFeatureResidualsAction())
            {
                const QSignalBlocker blocker(action);
                action->setChecked(visible);
            }
        });
        connect(_canvas, &CanvasWidget::maskOverlayVisibilityChanged, this, [this](bool visible)
        {
            if (QAction *action = _mainMenu->showMaskOverlayAction())
            {
                const QSignalBlocker blocker(action);
                action->setChecked(visible);
            }
        });
        connect(_canvas, &CanvasWidget::depthOverlayAvailabilityChanged, this, [this](bool available)
        {
            _mainMenu->setDepthOverlayAvailable(available);
        });
        connect(_canvas,
                &CanvasWidget::depthOverlayLevelsAvailabilityChanged,
                this,
                [this](bool finalAvailable,
                       bool level1Available,
                       bool level2Available,
                       bool level3Available,
                       const QString &finalReason,
                       const QString &level1Reason,
                       const QString &level2Reason,
                       const QString &level3Reason)
        {
            _mainMenu->setDepthOverlayLevelsAvailable(finalAvailable,
                                                      level1Available,
                                                      level2Available,
                                                      level3Available,
                                                      finalReason,
                                                      level1Reason,
                                                      level2Reason,
                                                      level3Reason);
        });
        connect(_canvas, &CanvasWidget::depthOverlayError, this, [](const QString &message)
        {
            LOG_WARN(QStringLiteral("[DepthOverlay] %1").arg(message));
        });
    }

    if (_mainMenu->minimizeAction())
    {
        connect(_mainMenu->minimizeAction(), &QAction::triggered, this, &QWidget::showMinimized);
    }

    if (_mainMenu->toggleGizmoAction() && _workspaceCenter && _workspaceCenter->modelView())
    {
        connect(_mainMenu->toggleGizmoAction(), &QAction::toggled,
                _workspaceCenter->modelView(), &CameraSceneWidget::setShowGizmo);
    }
    if (_mainMenu->toggleLocalAxesAction() && _workspaceCenter && _workspaceCenter->modelView())
    {
        connect(_mainMenu->toggleLocalAxesAction(), &QAction::toggled,
                _workspaceCenter->modelView(), &CameraSceneWidget::setShowCameraLocalAxes);
    }
    if (_mainMenu->toggleCamerasAction() && _workspaceCenter && _workspaceCenter->modelView())
    {
        connect(_mainMenu->toggleCamerasAction(), &QAction::toggled,
                _workspaceCenter->modelView(), &CameraSceneWidget::setShowCameras);
    }
    if (_workspaceCenter && _workspaceCenter->modelView())
    {
        auto *modelView = _workspaceCenter->modelView();
        if (_mainMenu->tiePointColorModeAction())
        {
            connect(_mainMenu->tiePointColorModeAction(),
                    &QAction::triggered,
                    this,
                    [modelView](bool checked)
            {
                if (checked)
                {
                    modelView->setTiePointColorMode(
                        CameraSceneWidget::TiePointColorMode::Color);
                }
            });
        }
        if (_mainMenu->tiePointElevationModeAction())
        {
            connect(_mainMenu->tiePointElevationModeAction(),
                    &QAction::triggered,
                    this,
                    [modelView](bool checked)
            {
                if (checked)
                {
                    modelView->setTiePointColorMode(
                        CameraSceneWidget::TiePointColorMode::Elevation);
                }
            });
        }
        if (_mainMenu->tiePointImageCountModeAction())
        {
            connect(_mainMenu->tiePointImageCountModeAction(),
                    &QAction::triggered,
                    this,
                    [modelView](bool checked)
            {
                if (checked)
                {
                    modelView->setTiePointColorMode(
                        CameraSceneWidget::TiePointColorMode::ImageCount);
                }
            });
        }
        const auto connectModelMode =
            [this, modelView](QAction *action,
                              CameraSceneWidget::ModelColorMode mode)
        {
            if (!action)
            {
                return;
            }
            connect(action, &QAction::triggered, this, [modelView, mode](bool checked)
            {
                if (checked)
                {
                    modelView->setModelColorMode(mode);
                }
            });
        };
        connectModelMode(_mainMenu->modelTextureModeAction(),
                         CameraSceneWidget::ModelColorMode::Texture);
        connectModelMode(_mainMenu->modelShadedModeAction(),
                         CameraSceneWidget::ModelColorMode::Shaded);
        connectModelMode(_mainMenu->modelSolidModeAction(),
                         CameraSceneWidget::ModelColorMode::Solid);
        connectModelMode(_mainMenu->modelWireframeModeAction(),
                         CameraSceneWidget::ModelColorMode::Wireframe);
        connectModelMode(_mainMenu->modelElevationModeAction(),
                         CameraSceneWidget::ModelColorMode::Elevation);
        connectModelMode(_mainMenu->modelConfidenceModeAction(),
                         CameraSceneWidget::ModelColorMode::Confidence);
        connectModelMode(_mainMenu->modelAssignedImageModeAction(),
                         CameraSceneWidget::ModelColorMode::AssignedImage);
        if (_mainMenu->toggleCameraThumbnailsAction())
        {
            connect(_mainMenu->toggleCameraThumbnailsAction(), &QAction::toggled,
                    modelView, &CameraSceneWidget::setShowCameraThumbnails);
        }
        if (_mainMenu->toggleCameraImagesAction())
        {
            connect(_mainMenu->toggleCameraImagesAction(), &QAction::toggled,
                    modelView, &CameraSceneWidget::setShowCameraImage);
        }
        if (_mainMenu->showCameraImagesInForegroundAction())
        {
            connect(_mainMenu->showCameraImagesInForegroundAction(), &QAction::toggled,
                    this, [modelView](bool checked)
            {
                if (checked)
                {
                    modelView->setCameraImageDisplayLayer(CameraSceneWidget::CameraImageDisplayLayer::Foreground);
                }
            });
        }
        if (_mainMenu->showCameraImagesInBackgroundAction())
        {
            connect(_mainMenu->showCameraImagesInBackgroundAction(), &QAction::toggled,
                    this, [modelView](bool checked)
            {
                if (checked)
                {
                    modelView->setCameraImageDisplayLayer(CameraSceneWidget::CameraImageDisplayLayer::Background);
                }
            });
        }
        if (_mainMenu->lockCameraImageAction())
        {
            connect(_mainMenu->lockCameraImageAction(), &QAction::toggled,
                    modelView, &CameraSceneWidget::setCameraImageLocked);
        }
    }
    if (_mainMenu->toggleHenanUniversityBrandAction())
    {
        auto *action = _mainMenu->toggleHenanUniversityBrandAction();
        setHenanUniversityBrandVisible(action->isChecked());
        connect(action, &QAction::toggled, this, &MainWindow::setHenanUniversityBrandVisible);
        connect(action, &QAction::toggled, this, [this](bool on)
        {
            saveUiSetting(QJsonObject{{QStringLiteral("henu_brand_visible"), on}});
        });
    }

    if (_mainMenu->manualPointCloudPruneAction())
    {
        connect(_mainMenu->manualPointCloudPruneAction(), &QAction::triggered,
                this, &MainWindow::onManualPointCloudPrune);
    }

    if (_workspaceCenter && _workspaceCenter->modelView())
    {
        auto *modelView = _workspaceCenter->modelView();
        connect(modelView, &CameraSceneWidget::manualPruneApplied, this,
                [this](int removedCount, int remainingCount)
        {
            statusBar()->showMessage(tr("已剔除 %1 个点，剩余 %2 个点")
                                         .arg(removedCount)
                                         .arg(remainingCount),
                                     4500);
        });
        connect(modelView, &CameraSceneWidget::manualPruneUndone, this,
            [this](int restoredCount)
        {
            statusBar()->showMessage(tr("已撤销删除，当前 %1 个点").arg(restoredCount), 4500);
        });
        connect(modelView, &CameraSceneWidget::manualPruneSaved, this,
                [this](const QString &path, int remainingCount)
        {
            statusBar()->showMessage(tr("点云已保存：%1（当前 %2 点）")
                                         .arg(path)
                                         .arg(remainingCount),
                                     5000);
        });
        connect(modelView, &CameraSceneWidget::manualPruneSaveFailed, this,
                [this](const QString &errorMessage)
        {
            QMessageBox::warning(this,
                                 tr("手动点云剔除"),
                                 tr("删除已生效，但保存失败：%1").arg(errorMessage));
        });
    }
}

// ============================================================
//  setupProjectManager — 创建 ProjectManager 并连接所有信号
// ============================================================
void MainWindow::setupProjectManager()
{
    _projectData = new ProjectData(this);
    _projectManager = new ProjectManager(_projectData, this);
    _projectManager->setObjectName(QStringLiteral("ProjectManager"));
    _tiePointWorkflowController = new TiePointWorkflowController(_projectManager, this);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressStarted,
            this,
            &MainWindow::showSgProgress);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressUpdated,
            this,
            &MainWindow::updateSgProgress);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressFinished,
            this,
            &MainWindow::hideSgProgress);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::statusMessageRequested,
            this,
            [this](const QString &message, int timeoutMs)
    {
        statusBar()->showMessage(message, timeoutMs);
    });
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::warningRequested,
            this,
            [this](const QString &title, const QString &message)
    {
        QMessageBox::warning(this, title, message);
    });
    connect(this,
            &MainWindow::sgCancelRequested,
            _tiePointWorkflowController,
            &TiePointWorkflowController::cancel);
    _projectUiHydrator = new ProjectUiHydrator(this);
    _projectUiHydrator->setStages({
        [this](const QJsonObject &meta)
        {
            if (_dashboard)
            {
                _dashboard->loadFromJson(meta);
            }
            if (_referencePanel)
            {
                _referencePanel->loadFromJson(meta);
            }
        },
        [this](const QJsonObject &meta)
        {
            if (_dataTree)
            {
                _dataTree->loadFromJson(meta);
            }
        },
        [this](const QJsonObject &meta)
        {
            if (_workspaceCenter)
            {
                _workspaceCenter->setProjectMeta(meta);
            }
        },
        [this](const QJsonObject &meta)
        {
            if (_photoStrip)
            {
                if (_projectManager)
                {
                    _photoStrip->setProjectPath(_projectManager->currentProjectPath());
                }
                _photoStrip->loadFromJson(meta);
            }
        }
    });
    _markerWorkspaceController = new xjw::gui::markers::MarkerWorkspaceController(
        _canvas, _projectData, this);
    _markerReferencePanel->setController(_markerWorkspaceController);
    connect(_markerWorkspaceController,
            &xjw::gui::markers::MarkerWorkspaceController::persistenceError,
            this,
            [this](const QString &message)
    {
        QMessageBox::warning(this, QStringLiteral("标记点"), message);
    });
    connect(_markerWorkspaceController,
            &xjw::gui::markers::MarkerWorkspaceController::focusMeasurementRequested,
            this,
            &MainWindow::openMarkerFocusMeasurement);
    connect(_markerReferencePanel,
            &xjw::gui::markers::MarkerReferencePanel::focusMeasurementRequested,
            this,
            [this](const QString &markerId)
    {
        openMarkerFocusMeasurement(markerId, _canvas ? _canvas->currentImagePath() : QString());
    });
    if (_mainMenu && _mainMenu->detectMarkersAction())
    {
        connect(_mainMenu->detectMarkersAction(), &QAction::triggered, this, [this]()
        {
            if (!_projectData || !_projectData->hasProject())
            {
                QMessageBox::information(this,
                                         QStringLiteral("检测标靶"),
                                         QStringLiteral("请先创建或打开包含照片的项目"));
                return;
            }
            xjw::gui::markers::DetectMarkersDialog dialog(this);
            if (!dialog.setContext(_markerWorkspaceController, _projectData))
            {
                QMessageBox::information(this,
                                         QStringLiteral("检测标靶"),
                                         QStringLiteral("项目中没有可检测的照片"));
                return;
            }
            dialog.exec();
        });
    }
    if (_mainMenu && _mainMenu->reviewMarkerDetectionsAction())
    {
        QAction *review_action = _mainMenu->reviewMarkerDetectionsAction();
        review_action->setEnabled(false);
        connect(_markerWorkspaceController,
                &xjw::gui::markers::MarkerWorkspaceController::detectionReviewChanged,
                this,
                [review_action](int count)
        {
            review_action->setEnabled(count > 0);
            review_action->setText(count > 0
                                       ? QStringLiteral("复核检测候选 (%1)...").arg(count)
                                       : QStringLiteral("复核检测候选..."));
        });
        connect(review_action, &QAction::triggered, this, [this]()
        {
            if (!_markerWorkspaceController
                || _markerWorkspaceController->detectionReviewQueue().entries.isEmpty())
            {
                return;
            }
            xjw::gui::markers::MarkerDetectionReviewDialog dialog(this);
            dialog.setController(_markerWorkspaceController);
            dialog.exec();
        });
    }
    if (_mainMenu && _mainMenu->printMarkersAction())
    {
        connect(_mainMenu->printMarkersAction(), &QAction::triggered, this, [this]()
        {
            xjw::gui::markers::PrintMarkersDialog dialog(this);
            if (_projectData && _projectData->hasProject())
            {
                dialog.setDefaultOutputDirectory(
                    QFileInfo(_projectData->currentProjectPath()).absolutePath());
            }
            else
            {
                dialog.setDefaultOutputDirectory(QDir::homePath());
            }
            dialog.exec();
        });
    }

    _menuWorkflowController = new MenuWorkflowController(this, this);
    _menuWorkflowController->setProjectManager(_projectManager);
 
    _reconController = new ReconstructionWorkflowController(this, this);
    _reconController->setProjectManager(_projectManager);
    // 连接特征显示选项更新到CanvasWidget
    connect(_menuWorkflowController, &MenuWorkflowController::requestApplyFeatureDisplayOptions,
            this, [this](const LayerRenderer::FeatureDisplayOptions &opts)
            {
                if (_canvas)
                {
                    _canvas->applyFeatureDisplayOptions(opts);
                }
            });

    if (_photoStrip)
    {
        connect(_photoStrip, &PhotoStripWidget::photoSelected, this, [this](const QString &path)
        {
            selectPhoto(path, false);
        });
        connect(_photoStrip, &PhotoStripWidget::photoActivated, this, [this](const QString &path)
        {
            selectPhoto(path, true);
        });
        connect(_photoStrip,
                &PhotoStripWidget::generateMaskRequested,
                this,
                [this](const QStringList &imagePaths)
                {
                    if (_projectManager)
                    {
                        _projectManager->openGenerateMaskDialogForImages(imagePaths);
                    }
                });
    }
    // 画布切换影像时，持久化活跃影像路径。
    if (_canvas)
    {
        connect(_canvas, &CanvasWidget::activeImageChanged, this, [this](const QString &path)
        {
            const QString stateKey = projectImageStateKey(path);
            int rotation =
                xjw::gui::config::imageViewRotationForPath(
                    _imageViewRotations, stateKey);
            if (rotation == 0 && stateKey != path)
            {
                rotation =
                    xjw::gui::config::imageViewRotationForPath(
                        _imageViewRotations, path);
            }
            _canvas->setViewRotationDegrees(
                rotation);
            saveUiSetting(QJsonObject{
                {QStringLiteral("active_image_id"), stateKey},
                {QStringLiteral("active_image_path"), QString()}
            });
            if (_projectManager)
            {
                _projectManager->setActiveImagePath(path);
            }
        });
        connect(_canvas, &CanvasWidget::viewRotationChanged, this,
                [this](const QString &path, int degrees)
        {
            const QString stateKey = projectImageStateKey(path);
            if (stateKey != path)
            {
                _imageViewRotations.remove(
                    xjw::gui::config::imageViewRotationPathKey(path));
            }
            _imageViewRotations = xjw::gui::config::withImageViewRotation(
                _imageViewRotations, stateKey, degrees);
            saveUiSetting(QJsonObject{
                {QStringLiteral("image_view_rotations"), _imageViewRotations}
            });
        });
    }

    if (_log)
    {
        connect(_log, &LogPanel::displayLevelChanged, this, &MainWindow::onLogDisplayLevelChanged);
    }

    if (_config)
    {
        _projectManager->setFileDialogStateManager(_config->fileDialogs());
    }

    if (_mainMenu)
    {
        if (_mainMenu->newAction())
        {
            connect(_mainMenu->newAction(), &QAction::triggered, this, [this]()
            {
                persistCurrentUiSettings();
                _projectManager->createNewProject();
            });
        }
        if (_mainMenu->openAction())
        {
            connect(_mainMenu->openAction(), &QAction::triggered, this, [this]()
            {
                persistCurrentUiSettings();
                _projectManager->openProject();
            });
        }
        if (_mainMenu->addPhotoAction())
        {
            connect(_mainMenu->addPhotoAction(), &QAction::triggered, _projectManager, &ProjectManager::addPhoto);
        }
        if (_mainMenu->addFolderAction())
        {
            connect(_mainMenu->addFolderAction(), &QAction::triggered, _projectManager, &ProjectManager::addFolder);
        }
        if (_mainMenu->saveAction())
        {
            _mainMenu->saveAction()->setEnabled(false);
            connect(_mainMenu->saveAction(), &QAction::triggered, _projectManager, &ProjectManager::saveProject);
        }
        if (_mainMenu->exportMatchedPairsAction())
        {
            connect(_mainMenu->exportMatchedPairsAction(), &QAction::triggered,
                    this, &MainWindow::onExportMatchedPairs);
        }

        if (_menuWorkflowController)
        {
            _menuWorkflowController->bindActions(_mainMenu);
        }

        // 工作流程菜单中的模型处理入口。
        if (_reconController)
        {
            auto connectRecon = [&](QAction *act, void (ReconstructionWorkflowController::*slot)())
            {
                if (act)
                {
                    connect(act, &QAction::triggered, _reconController, slot);
                }
            };
            connectRecon(_mainMenu->generateModelAction(),     &ReconstructionWorkflowController::openGenerateModelDialog);
            connectRecon(_mainMenu->generateTextureAction(),
                         &ReconstructionWorkflowController::openTextureMappingDialog);
        }

        auto openMatchViewer = [this](bool modal)
        {
            showMatchViewer(QString(), modal);
        };

        auto startMatchPhotosTask =
            [this](xjw::matchphotos::MatchPhotosOptions options,
                   const QStringList &manualPairKeys,
                   const QString &taskTitle)
        {
            if (!_tiePointWorkflowController)
            {
                LOG_ERROR(QStringLiteral("无法运行连接点匹配：工作流控制器未初始化"));
                QMessageBox::warning(this,
                                     tr("连接点匹配"),
                                     tr("连接点工作流尚未初始化。"));
                return;
            }

            _tiePointWorkflowController->start(std::move(options), manualPairKeys, taskTitle);
        };
        if (_mainMenu->createTiePointsAction())
        {
            connect(_mainMenu->createTiePointsAction(), &QAction::triggered, this, [this, startMatchPhotosTask]()
            {
                CreateTiePointsDialog dlg(this);
                bool hasAllReferenceCameras = false;
                const QStringList images = _projectManager ? _projectManager->getAllImages() : QStringList();
                const int cameraCount = _projectManager
                    ? _projectManager->getCamerasForImages(images, &hasAllReferenceCameras).size()
                    : 0;
                dlg.setReferencePreselectionAvailable(
                    hasAllReferenceCameras && cameraCount == images.size() && images.size() >= 2,
                    cameraCount,
                    images.size());
                if (dlg.exec() == QDialog::Accepted)
                {
                    xjw::matchphotos::MatchPhotosOptions options;
                    options.profile = tiePointProfileFromAccuracy(dlg.accuracy());
                    options.device = xjw::matchphotos::ComputeDevice::Cuda;
                    options.maxImageDim = maxImageDimFromAccuracy(dlg.accuracy());
                    options.useExplicitKeypointLimit = true;
                    options.maxKeypoints = dlg.useGuidedMatching() ? 0 : dlg.keypointLimit();
                    options.keypointLimitPerMegapixel = dlg.useGuidedMatching()
                        ? dlg.keypointLimitPerMegapixel()
                        : 0;
                    options.maxTiePointsPerImage = dlg.tiePointLimit();
                    options.excludeStationaryTiePoints = dlg.excludePinnedTiePoints();
                    options.matchThreshold = 0.15f;
                    options.enableGuidedMatching = dlg.useGuidedMatching();
                    options.useGenericPreselection = dlg.useGenericPreselection();
                    options.useReferencePreselection = dlg.useReferencePreselection();
                    options.maskApplyMode = dlg.maskApplyMode();
                    options.reuseExistingFeatures = true;
                    if (options.maskApplyMode == QStringLiteral("keypoints"))
                    {
                        // 关键点蒙版改变的是特征文件本身，不能复用旧的未过滤特征。
                        options.reuseExistingFeatures = false;
                    }
                    options.pairPolicy = xjw::matchphotos::makePairSelectionPolicy(
                        pairPresetFromAccuracy(dlg.accuracy()));
                    options.pairPolicy.includeVocabularyOverlap = options.useGenericPreselection;
                    options.pairPolicy.includeCameraOverlap = options.useReferencePreselection;
                    if (!dlg.useGenericPreselection() && !dlg.useReferencePreselection())
                    {
                        options.pairPolicy.mode = xjw::matchphotos::PairSelectionMode::Exhaustive;
                    }

                    startMatchPhotosTask(options, QStringList(), tr("创建连接点"));
                }
            });
        }

        if (_mainMenu->thinTiePointsAction())
        {
            connect(_mainMenu->thinTiePointsAction(), &QAction::triggered, this, [this]()
            {
                ThinTiePointsDialog dlg(this);
                if (dlg.exec() == QDialog::Accepted)
                {
                    statusBar()->showMessage(
                        tr("连接点稀释参数已确认：连接点限制 %1").arg(dlg.tiePointLimit()),
                        3000);
                }
            });
        }

        if (_mainMenu->cleanTiePointsAction())
        {
            connect(_mainMenu->cleanTiePointsAction(), &QAction::triggered, this, [this]()
            {
                CleanTiePointsDialog dlg(this);
                if (dlg.exec() == QDialog::Accepted)
                {
                    const QString operation = dlg.deleteRequested() ? tr("删除") : tr("筛选");
                    statusBar()->showMessage(
                        tr("连接点清理参数已确认：%1，%2，级别 %3")
                            .arg(dlg.criterionText(), operation, QString::number(dlg.level(), 'g', 3)),
                        3000);
                }
            });
        }

        if (_mainMenu->viewTiePointMatchesAction())
        {
            connect(_mainMenu->viewTiePointMatchesAction(), &QAction::triggered, this, [openMatchViewer]()
            {
                openMatchViewer(true);
            });
        }

        if (_mainMenu->intersectionCheckAction())
        {
            connect(_mainMenu->intersectionCheckAction(), &QAction::triggered, this, [this]()
            {
                if (!_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开前方交汇检测：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new ForwardIntersectionCheckDialog(_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
            });
        }

        if (_mainMenu->intersectionViewResultsAction())
        {
            connect(_mainMenu->intersectionViewResultsAction(), &QAction::triggered, this, [this]()
            {
                if (!_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开前方交汇结果：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new ForwardIntersectionResultsDialog(_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
            });
        }

        connect(_projectManager, &ProjectManager::projectOpened,  this, &MainWindow::onProjectOpened);
        connect(_projectManager, &ProjectManager::projectClosed, this, &MainWindow::onProjectClosed);

        // 当特征提取完成后，切换 Canvas 到对应后缀并刷新显示
        connect(_projectManager, &ProjectManager::ipfindResultAppended, this,
            [this](const QString &imagePath, const QString &suffix)
        {
            if (_canvas)
            {
                const bool isCurrentImage =
                    QDir::cleanPath(imagePath) == QDir::cleanPath(_canvas->currentImagePath());
                if (!suffix.isEmpty())
                    _canvas->setActiveFeatureSuffix(suffix);
                if (isCurrentImage)
                {
                    _canvas->reloadInterestPoints(imagePath);
                }
            }
        });

        if (_config)
        {
            _mainMenu->setRecentProjects(_config->recentProjects()->recentProjects());
        }

        connect(_mainMenu, &MainMenu::recentProjectSelected, this, [this](const QString &p)
        {
            openProjectFromPath(p);
        });
        connect(_mainMenu, &MainMenu::clearRecentRequested, this, &MainWindow::onClearRecentRequested);
    }

    
    connect(_projectManager, &ProjectManager::saveStarted,    this, &MainWindow::onSaveStarted);
    connect(_projectManager, &ProjectManager::saveFinished,   this, &MainWindow::onSaveFinished);
    connect(_projectManager, &ProjectManager::projectOpenStarted,
            this, &MainWindow::onProjectOpenStarted);
    connect(_projectManager, &ProjectManager::projectOpenProgressChanged,
            this, &MainWindow::onProjectOpenProgressChanged);
    connect(_projectManager, &ProjectManager::projectOpenFinished,
            this, &MainWindow::onProjectOpenFinished);
    connect(_projectManager, &ProjectManager::metadataDirtyChanged, this, &MainWindow::onMetadataDirtyChanged);
    connect(_projectManager, &ProjectManager::masksGenerated, this, [this](const QStringList &imagePaths)
    {
        if (!_canvas)
        {
            return;
        }

        const QString current = QDir::cleanPath(_canvas->currentImagePath());
        for (const QString &imagePath : imagePaths)
        {
            if (QDir::cleanPath(imagePath) == current)
            {
                _canvas->reloadMaskOverlay();
                return;
            }
        }
    });

    connect(_projectManager, &ProjectManager::projectMetadataUpdated, this, [this](const QString &)
    {
        if (_projectManager)
        {
            const QJsonObject meta = _projectManager->currentMeta();
            scheduleProjectMetadataRefresh(meta);
            if (_canvas)
            {
                _canvas->setProjectMetadata(meta);
            }
        }
    });
    connect(_projectManager, &ProjectManager::projectMetadataChanged, this, [this](const QJsonObject &meta)
    {
        scheduleProjectMetadataRefresh(meta);
        if (_canvas)
        {
            _canvas->setProjectMetadata(meta);
        }
    });
    connect(_projectManager,
            &ProjectManager::chunkListChanged,
            _dataTree,
            &DataTreeWidget::setChunkContext);

    connect(_dataTree, &DataTreeWidget::removeRequested,  _projectManager, &ProjectManager::removeResources);
    connect(_dataTree, &DataTreeWidget::deleteDataRequested, _projectManager, &ProjectManager::deleteGeneratedData);
    connect(_dataTree, &DataTreeWidget::viewMatchesRequested, this, [this](const QString &imagePath)
    {
        showMatchViewer(imagePath, true);
    });
    connect(_dataTree, &DataTreeWidget::sideOpenRequested, this, [this](const QString &section, const QString &path)
    {
        Q_UNUSED(section);
        if (!_workspaceCenter || path.trimmed().isEmpty())
        {
            return;
        }
        if (_lastSelectedImage.trimmed().isEmpty())
        {
            QMessageBox::information(this,
                                     QStringLiteral("侧边打开"),
                                     QStringLiteral("请先在中间打开一张二维影像，再选择另一张在侧边打开。"));
            return;
        }
        if (QDir::cleanPath(_lastSelectedImage) == QDir::cleanPath(path))
        {
            QMessageBox::information(this,
                                     QStringLiteral("侧边打开"),
                                     QStringLiteral("当前主影像与侧边影像相同，请选择另一张影像。"));
            return;
        }
        _workspaceCenter->showSideBySideImages(_lastSelectedImage, path);
    });
    connect(_dataTree, &DataTreeWidget::packRequested,    _projectManager, &ProjectManager::packResource);
    connect(_dataTree, &DataTreeWidget::createChunkRequested,
            _projectManager, &ProjectManager::createChunk);
    connect(_dataTree, &DataTreeWidget::renameChunkRequested,
            _projectManager, &ProjectManager::renameChunk);
    connect(_dataTree, &DataTreeWidget::removeChunkRequested,
            _projectManager, &ProjectManager::removeChunk);
    connect(_dataTree, &DataTreeWidget::switchChunkRequested,
            _projectManager, &ProjectManager::switchChunk);
    connect(_dataTree, &DataTreeWidget::openRequested, this, [](const QString &p)
    {
        QDesktopServices::openUrl(QUrl::fromLocalFile(p));
    });
    connect(_dataTree, &DataTreeWidget::revealRequested, this, [](const QString &p)
    {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(p).absolutePath()));
    });
    connect(_dataTree, &DataTreeWidget::resourceSelected, this, [this](const QString &section, const QString &path)
    {
        if (section == QStringLiteral("照片"))
        {
            selectPhoto(path, false);
            return;
        }
        selectResource(section, path);
    });
    connect(_dataTree, &DataTreeWidget::resourceActivated, this, [this](const QString &section, const QString &path)
    {
        if (!_workspaceCenter || !_projectManager)
        {
            return;
        }
        if (section == QStringLiteral("照片"))
        {
            selectPhoto(path, true);
            return;
        }
        selectResource(section, path);

        auto normalizedMeta = [this]()
        {
            return xjw::common::project::projectFilesRootObject(_projectManager->currentMeta());
        };

        if (section == QStringLiteral("深度图"))
        {
            // 深度图（16-bit PNG）通过 LayerRenderer 的 GDAL 归一化后显示
            _workspaceCenter->showImageView(path);
            _lastSelectedImage = path;
            return;
        }
        if (section == QStringLiteral("DEM") || section == QStringLiteral("正射影像"))
        {
            _workspaceCenter->showImageView(path);
            _lastSelectedImage = path;
            return;
        }
        if (section == QStringLiteral("3D模型"))
        {
            _workspaceCenter->showModelFile(path);
            return;
        }
        if (section == QStringLiteral("连接点"))
        {
            if (!path.isEmpty() && QFileInfo::exists(path))
            {
                const QString projectRoot =
                    QFileInfo(_projectManager->currentProjectPath()).absolutePath();
                auto resolveProjectPath = [&projectRoot](const QString &storedPath)
                {
                    const QString trimmedPath = storedPath.trimmed();
                    if (trimmedPath.isEmpty())
                    {
                        return QString();
                    }
                    if (QFileInfo(trimmedPath).isAbsolute())
                    {
                        return QDir::cleanPath(trimmedPath);
                    }
                    return QDir::cleanPath(QDir(projectRoot).filePath(trimmedPath));
                };

                QString sidecarPath;
                const QString selectedPathKey =
                    QDir::cleanPath(path).toCaseFolded();
                const QJsonArray results =
                    normalizedMeta().value(QStringLiteral("aerial_triangulation_results")).toArray();
                for (int index = results.size() - 1; index >= 0; --index)
                {
                    const QJsonObject files =
                        results.at(index).toObject().value(QStringLiteral("files")).toObject();
                    const QString cloudPath = resolveProjectPath(
                        files.value(QStringLiteral("sparse_cloud_xyz")).toString());
                    if (cloudPath.toCaseFolded() != selectedPathKey)
                    {
                        continue;
                    }
                    sidecarPath = resolveProjectPath(
                        files.value(QStringLiteral("sparse_cloud_points_json")).toString());
                    break;
                }
                _workspaceCenter->showTiePointCloudFile(path, sidecarPath);
            }
            return;
        }
        if (section == QStringLiteral("稠密点云"))
        {
            _workspaceCenter->showPointCloudFile(path);
            return;
        }
        if (section == QStringLiteral("匹配"))
        {
            if (!path.isEmpty() && QFileInfo::exists(path))
            {
                _workspaceCenter->showPointCloudFile(path);
                return;
            }
            _workspaceCenter->showModelView();
            return;
        }
        if (section == QStringLiteral("观测网络"))
        {
            if (!_projectManager)
            {
                return;
            }
            const QJsonObject meta = _projectManager->currentMeta();
            const QJsonArray results = meta.value(QStringLiteral("observation_network_results")).toArray();
            if (results.isEmpty())
            {
                return;
            }

            int idx = path.isEmpty() ? results.size() - 1 : path.toInt();
            if (idx < 0 || idx >= results.size())
            {
                idx = results.size() - 1;
            }

            const QJsonObject res = results[idx].toObject();

            // 从持久化文件或内联元数据恢复 ObservationNetwork
            xjw::ObservationNetwork net;

            // 优先从落盘 JSON 文件读取
            const QString netFile = res.value(QStringLiteral("network_file")).toString();
            bool loadedFromFile = false;
            if (!netFile.isEmpty() && QFileInfo::exists(netFile))
            {
                QFile nf(netFile);
                if (nf.open(QIODevice::ReadOnly))
                {
                    const QJsonDocument nd = QJsonDocument::fromJson(nf.readAll());
                    if (nd.isObject())
                    {
                        const QJsonObject no = nd.object();
                        for (const QJsonValue &v : no.value(QStringLiteral("node_names")).toArray())
                        {
                            net.nodeNames.push_back(v.toString().toStdString());
                        }
                        for (const QJsonValue &ev : no.value(QStringLiteral("edges")).toArray())
                        {
                            const QJsonObject eo = ev.toObject();
                            xjw::NetworkEdge e;
                            e.idx0 = eo.value(QStringLiteral("i")).toInt();
                            e.idx1 = eo.value(QStringLiteral("j")).toInt();
                            e.weight = eo.value(QStringLiteral("w")).toDouble(1.0);
                            e.numMatches = eo.value(QStringLiteral("n")).toInt(0);
                            net.edges.push_back(e);
                        }
                        loadedFromFile = !net.nodeNames.empty();
                    }
                }
            }

            // 回退：从元数据内联数据读取
            if (!loadedFromFile)
            {
                for (const QJsonValue &v : res.value(QStringLiteral("node_names")).toArray())
                {
                    net.nodeNames.push_back(v.toString().toStdString());
                }
                for (const QJsonValue &ev : res.value(QStringLiteral("edges")).toArray())
                {
                    const QJsonObject eo = ev.toObject();
                    xjw::NetworkEdge e;
                    e.idx0       = eo.value(QStringLiteral("i")).toInt();
                    e.idx1       = eo.value(QStringLiteral("j")).toInt();
                    e.weight     = eo.value(QStringLiteral("w")).toDouble(1.0);
                    e.numMatches = eo.value(QStringLiteral("n")).toInt(0);
                    net.edges.push_back(e);
                }
            }
            net.degrees.assign(net.nodeNames.size(), 0);
            for (const auto &e : net.edges)
            {
                if (e.idx0 >= 0 && e.idx0 < static_cast<int>(net.degrees.size()))
                {
                    net.degrees[e.idx0]++;
                }
                if (e.idx1 >= 0 && e.idx1 < static_cast<int>(net.degrees.size()))
                {
                    net.degrees[e.idx1]++;
                }
            }

            if (net.nodeNames.empty())
            {
                // 旧数据没有 node_names，显示提示并要求重新运行
                QMessageBox::information(this, QStringLiteral("观测网络"),
                    QStringLiteral("此结果不含详细边数据，请重新运行观测网络构建以生成可视化数据。"));
                return;
            }

            const QString algo = res.value(QStringLiteral("algorithm")).toString();
            const QString title = QStringLiteral("%1 [N:%2 E:%3]")
                .arg(algo).arg(net.numNodes()).arg(net.numEdges());
            _workspaceCenter->showObservationNetwork(net, title);
            return;
        }
    });

    connect(_referencePanel, &ReferencePanelWidget::exactImportRequested,
        _projectManager, &ProjectManager::importCameraForImage);
    connect(_referencePanel, &ReferencePanelWidget::batchImportRequested,
        _projectManager, &ProjectManager::importCamerasByFilenameBatch);
    connect(_referencePanel, &ReferencePanelWidget::clearCameraRequested,
        this, [this](const QStringList &paths)
        {
            int cleared = 0;
            QString err;
            if (_projectManager->clearImageCameras(paths, &cleared, &err))
            {
                LOG_INFO(QStringLiteral("已清除 %1 张影像的相机参数").arg(cleared));
            }
            else
            {
                LOG_WARN(QStringLiteral("清除相机参数失败: %1").arg(err));
            }
        });
    connect(_referencePanel, &ReferencePanelWidget::imageActivated,
        this, [this](const QString &p)
        {
            selectPhoto(p, true);
        });

    auto createTaskStatus = [this](int labelWidth, bool cancellable, const QString &cancellingText)
    {
        auto *widget = new TaskStatusWidget(this);
        widget->setLabelMinimumWidth(labelWidth);
        widget->setCancellable(cancellable);
        if (!cancellingText.isEmpty())
        {
            widget->setCancellingText(cancellingText);
        }
        connect(widget, &TaskStatusWidget::cancelRequested,
                this, &MainWindow::refreshDashboardTaskSnapshots);
        statusBar()->addPermanentWidget(widget);
        return widget;
    };

    // ── MVS 状态栏进度条 ──────────────────────────────────────────
    _mvsTaskStatus = createTaskStatus(220, true, tr("正在取消稠密重建..."));
    connect(_mvsTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        if (_projectManager)
        {
            _projectManager->cancelMvs();
        }
    });

    connect(_projectManager, &ProjectManager::mvsProgressChanged,
            this, &MainWindow::onMvsProgress);
    connect(_projectManager, &ProjectManager::mvsProgressFinished,
            this, &MainWindow::onMvsFinished);

    // ── 网格重建状态栏进度条 ──────────────────────────────────────
    _meshTaskStatus = createTaskStatus(220, true, tr("正在取消模型生成..."));
    connect(_meshTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        if (_projectManager)
        {
            _projectManager->cancelModelGeneration();
        }
    });

    connect(_projectManager, &ProjectManager::meshProgressChanged,
            this, &MainWindow::onMeshProgress);
    connect(_projectManager, &ProjectManager::meshProgressFinished,
            this, &MainWindow::onMeshFinished);

    // ── 空三（AT）/光束法平差状态栏进度条 ───────────────────────
    _atTaskStatus = createTaskStatus(220, true, tr("正在取消空三/光束法平差..."));
    connect(_atTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        if (_projectManager)
        {
            _projectManager->cancelAt();
        }
    });

    connect(_projectManager, &ProjectManager::atProgressChanged,
            this, &MainWindow::onAtProgress);
    connect(_projectManager, &ProjectManager::atProgressFinished,
            this, &MainWindow::onAtFinished);

    // ── 特征匹配状态栏进度条 ────────────────────────────
    _sgTaskStatus = createTaskStatus(180, true, tr("正在取消特征匹配..."));
    connect(_sgTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        emit sgCancelRequested();
    });

    // ── 照片蒙版生成状态栏进度条 ─────────────────────────────────────
    _maskTaskStatus = createTaskStatus(180, true, tr("正在取消生成蒙版..."));
    connect(_maskTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        if (_projectManager)
        {
            _projectManager->cancelMaskGeneration();
        }
    });

    connect(_projectManager, &ProjectManager::maskGenerationProgressChanged,
            this, &MainWindow::onMaskGenerationProgress);
    connect(_projectManager, &ProjectManager::maskGenerationFinished,
            this, &MainWindow::onMaskGenerationFinished);

    refreshDashboardTaskSnapshots();
}

void MainWindow::showMatchViewer(const QString &initialImagePath, bool modal)
{
    if (!_projectManager)
    {
        LOG_ERROR(QStringLiteral("无法打开匹配查看：ProjectManager 未初始化"));
        return;
    }

    auto *dialog = new MatchPairSelectorDialog(_projectManager, this);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    if (!initialImagePath.trimmed().isEmpty())
    {
        dialog->setInitialImagePath(initialImagePath);
    }

    if (modal)
    {
        dialog->exec();
    }
    else
    {
        dialog->show();
    }
}

void MainWindow::refreshDashboardTaskSnapshots()
{
    if (!_dashboard)
    {
        return;
    }

    QJsonArray tasks;
    auto appendTask = [&tasks](const QString &name, const TaskStatusWidget *widget)
    {
        if (!widget || (!widget->isActive() && !widget->isCancelling()))
        {
            return;
        }

        QJsonObject record;
        record[QStringLiteral("name")] = name;
        record[QStringLiteral("status_text")] = widget->statusText();
        record[QStringLiteral("active")] = widget->isActive();
        record[QStringLiteral("cancelling")] = widget->isCancelling();
        record[QStringLiteral("progress_value")] = widget->progressValue();
        record[QStringLiteral("progress_maximum")] = widget->progressMaximum();
        tasks.append(record);
    };

    appendTask(tr("MVS/稠密重建"), _mvsTaskStatus);
    appendTask(tr("网格重建"), _meshTaskStatus);
    appendTask(tr("空三/光束法平差"), _atTaskStatus);
    appendTask(tr("特征匹配"), _sgTaskStatus);
    appendTask(tr("生成蒙版"), _maskTaskStatus);

    _dashboard->setTaskSnapshots(tasks);
}

bool MainWindow::exportMatchedPairsToLis(QString *outputPath, QString *errorMessage) const
{
    if (outputPath)
    {
        outputPath->clear();
    }
    if (errorMessage)
    {
        errorMessage->clear();
    }

    if (!_projectManager)
    {
        if (errorMessage)
        {
            *errorMessage = tr("项目管理器未初始化");
        }
        return false;
    }

    const QString plascanPath = _projectManager->currentProjectPath();
    if (plascanPath.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("请先打开项目");
        }
        return false;
    }

    const QVector<QPair<QString, QString>> matchedPairs =
        xjw::common::project::collectMatchedImageNamePairs(plascanPath, _projectManager->currentMeta());
    if (matchedPairs.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前项目中没有可导出的匹配对");
        }
        return false;
    }

    const QString projectRoot = xjw::common::project::ProjectIO::projectRootFromPlascan(plascanPath);
    const QString exportDirPath = QDir(projectRoot).filePath(QStringLiteral("export"));
    QDir exportDir;
    if (!exportDir.mkpath(exportDirPath))
    {
        if (errorMessage)
        {
            *errorMessage = tr("无法创建导出目录: %1").arg(exportDirPath);
        }
        return false;
    }

    const QString lisPath = QDir(exportDirPath).filePath(QStringLiteral("matched_pairs.lis"));
    QSaveFile outputFile(lisPath);
    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        if (errorMessage)
        {
            *errorMessage = tr("无法写入导出文件: %1").arg(lisPath);
        }
        return false;
    }

    QTextStream out(&outputFile);
    for (const QPair<QString, QString> &matchedPair : matchedPairs)
    {
        out << matchedPair.first << ' ' << matchedPair.second << '\n';
    }

    if (!outputFile.commit())
    {
        if (errorMessage)
        {
            *errorMessage = tr("保存导出文件失败: %1").arg(lisPath);
        }
        return false;
    }

    if (outputPath)
    {
        *outputPath = lisPath;
    }
    return true;
}

void MainWindow::onExportMatchedPairs()
{
    QString outputPath;
    QString errorMessage;
    if (!exportMatchedPairsToLis(&outputPath, &errorMessage))
    {
        QMessageBox::warning(this, tr("导出匹配对"), errorMessage);
        return;
    }

    statusBar()->showMessage(tr("匹配对已导出到: %1").arg(outputPath), 5000);
    QMessageBox::information(this,
                             tr("导出匹配对"),
                             tr("导出成功，文件已保存到：\n%1").arg(outputPath));
}

void MainWindow::onManualPointCloudPrune()
{
    if (!_workspaceCenter || !_workspaceCenter->modelView())
    {
        return;
    }

    auto *modelView = _workspaceCenter->modelView();
    const bool enable = !modelView->isManualPruneModeEnabled();
    QString errorMessage;
    if (!modelView->setManualPruneModeEnabled(enable, &errorMessage))
    {
        QMessageBox::warning(this, tr("手动点云剔除"), errorMessage);
        return;
    }

    if (enable)
    {
        _workspaceCenter->showModelView();
        QMessageBox::information(this,
                                 tr("手动点云剔除"),
                                 tr("已进入手动剔除模式。\n"
                                                "请在点云视图中按住鼠标右键拖拽框选，高亮目标点。\n"
                                                "按鼠标前进侧键执行删除并覆盖保存。\n"
                                                "误删可按 Ctrl+Z 撤销。\n"
                                    "再次点击 工具-手动点云剔除 可退出。"));
    }
    else
    {
        statusBar()->showMessage(tr("已退出手动点云剔除模式"), 3500);
    }
}


// ============================================================
//  MVS 进度状态栏槽
// ============================================================

void MainWindow::onMvsProgress(const QString &stage, int percent)
{
    if (!_mvsTaskStatus)
    {
        return;
    }
    if (!_mvsTaskStatus->isActive())
    {
        _mvsTaskStatus->begin(stage, 0, 100);
    }
    _mvsTaskStatus->updateProgress(stage, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());   // 清空普通消息，让 permanent widget 露出
}

void MainWindow::onMvsFinished(bool success)
{
    if (!_mvsTaskStatus)
    {
        return;
    }
    _mvsTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("稠密重建完成") : tr("稠密重建已取消或失败"), 4000);
}

// ============================================================
//  网格重建进度状态栏槽
// ============================================================

void MainWindow::onMeshProgress(const QString &stage, int percent)
{
    if (!_meshTaskStatus)
    {
        return;
    }
    if (!_meshTaskStatus->isActive())
    {
        _meshTaskStatus->begin(stage, 0, 100);
    }
    _meshTaskStatus->updateProgress(stage, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onMeshFinished(bool success)
{
    if (!_meshTaskStatus)
    {
        return;
    }
    _meshTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("网格重建完成") : tr("网格重建失败"), 4000);
}

// ============================================================
//  AT（空三）/光束法平差进度状态栏槽
// ============================================================

void MainWindow::onAtProgress(const QString &stage, int percent)
{
    if (!_atTaskStatus)
    {
        return;
    }
    // 显示阶段信息和百分比，例如 "正在匹配特征点... 50%"
    QString statusText = stage;
    if (percent > 0 && percent < 100)
    {
        statusText = QStringLiteral("%1 %2%").arg(stage).arg(percent);
    }
    if (!_atTaskStatus->isActive())
    {
        _atTaskStatus->begin(statusText, 0, 100);
    }
    _atTaskStatus->updateProgress(statusText, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());   // 清空普通消息，让 permanent widget 露出
}

void MainWindow::onAtFinished(bool success)
{
    if (!_atTaskStatus)
    {
        return;
    }
    _atTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("空三/光束法平差完成") : tr("空三/光束法平差已取消或失败"), 4000);
}

void MainWindow::onCancelAt()
{
    if (_projectManager)
    {
        _projectManager->cancelAt();
    }
}

// ============================================================
//  照片蒙版生成进度状态栏槽
// ============================================================

void MainWindow::onMaskGenerationProgress(const QString &stage, int done, int total)
{
    if (!_maskTaskStatus)
    {
        return;
    }

    const int safeTotal = std::max(1, total);
    const int clampedDone = std::clamp(done, 0, safeTotal);
    const QString defaultText = tr("生成蒙版 %1/%2").arg(clampedDone).arg(safeTotal);
    const QString stageText = stage.trimmed();
    const QString statusText = stageText.isEmpty() || stageText == tr("生成蒙版")
        ? defaultText
        : tr("%1 %2/%3").arg(stageText).arg(clampedDone).arg(safeTotal);

    if (!_maskTaskStatus->isActive())
    {
        _maskTaskStatus->begin(statusText, 0, safeTotal);
    }
    _maskTaskStatus->updateProgress(statusText, clampedDone);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onMaskGenerationFinished(bool success)
{
    if (!_maskTaskStatus)
    {
        return;
    }
    _maskTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(success ? tr("蒙版生成完成") : tr("蒙版生成已取消或失败"), 4000);
}

void MainWindow::showSgProgress(int total)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    _sgTaskStatus->begin(tr("特征匹配 0/%1").arg(total), 0, total);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::updateSgProgress(int done)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    const int total = _sgTaskStatus->progressMaximum();
    const int clampedDone = std::clamp(done, 0, total);
    _sgTaskStatus->updateProgress(
        tr("特征匹配 %1/%2").arg(clampedDone).arg(total), clampedDone);
    refreshDashboardTaskSnapshots();
}

void MainWindow::hideSgProgress(bool ok)
{
    if (!_sgTaskStatus)
    {
        return;
    }
    _sgTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(ok ? tr("匹配完成") : tr("匹配已取消"), 4000);
}

// ============================================================
//  辅助方法
// ============================================================

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

void MainWindow::onProjectOpenStarted(const QString &plascanPath)
{
    if (!_openProgressDialog)
    {
        _openProgressDialog = new QProgressDialog(tr("正在打开项目..."), QString(), 0, 100, this);
        _openProgressDialog->setWindowModality(Qt::ApplicationModal);
        _openProgressDialog->setCancelButton(nullptr);
        _openProgressDialog->setMinimumDuration(0);
        _openProgressDialog->setAutoClose(false);
        _openProgressDialog->setAutoReset(false);
    }
    _openProgressDialog->setLabelText(tr("正在打开项目：%1").arg(QFileInfo(plascanPath).fileName()));
    _openProgressDialog->setValue(0);
    _openProgressDialog->show();
}

void MainWindow::onProjectOpenProgressChanged(const QString &message, int percent)
{
    if (!_openProgressDialog)
    {
        return;
    }

    _openProgressDialog->setLabelText(message.isEmpty() ? tr("正在打开项目...") : message);
    _openProgressDialog->setValue(std::clamp(percent, 0, 100));
}

void MainWindow::onProjectOpenFinished(bool success, const QString &message)
{
    if (_openProgressDialog)
    {
        _openProgressDialog->hide();
        _openProgressDialog->deleteLater();
        _openProgressDialog = nullptr;
    }

    statusBar()->showMessage(success ? message : tr("打开项目失败"), success ? 3000 : 5000);
}

void MainWindow::onSaveStarted()
{
    if (!_saveProgressDialog)
    {
        _saveProgressDialog = new QProgressDialog(tr("正在保存项目..."), QString(), 0, 0, this);
        _saveProgressDialog->setWindowModality(Qt::ApplicationModal);
        _saveProgressDialog->setCancelButton(nullptr);
        _saveProgressDialog->setMinimumDuration(0);
    }
    _saveProgressDialog->show();
}

void MainWindow::onSaveFinished(bool ok)
{
    if (_saveProgressDialog)
    {
        _saveProgressDialog->hide();
        _saveProgressDialog->deleteLater();
        _saveProgressDialog = nullptr;
    }
    statusBar()->showMessage(ok ? tr("保存完成") : tr("保存失败"), ok ? 3000 : 5000);
}

void MainWindow::onMetadataDirtyChanged(bool dirty)
{
    const QString projPath = _projectManager
        ? _projectManager->currentProjectPath()
        : QString();
    if (projPath.trimmed().isEmpty())
    {
        setWindowTitle(QStringLiteral("PlaScan"));
        return;
    }

    QString name = QFileInfo(projPath).baseName();
    if (name.isEmpty())
    {
        name = QFileInfo(projPath).fileName();
    }
    setWindowTitle(dirty ? QStringLiteral("PlaScan - %1*").arg(name)
                         : QStringLiteral("PlaScan - %1").arg(name));
}

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

    // 特征后缀恢复不能依赖主窗口 UI 设置存在；旧项目可能只有 assets/ip/*.dsk。
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
            const QString projectPath = _projectManager ? _projectManager->currentProjectPath() : QString();
            QTimer::singleShot(100, this, [this, imagePath, projectPath]()
            {
                if (!_projectManager || _projectManager->currentProjectPath() != projectPath)
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
            // 同步保存，不使用事件循环
            _projectManager->saveProject();
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
