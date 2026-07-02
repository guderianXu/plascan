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
#include <QToolButton>
#include <QButtonGroup>
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
#include <QEventLoop>
#include <QtConcurrent>
#include <QFutureWatcher>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QPointer>
#include <QSignalBlocker>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

#include <algorithm>

#include "AlgorithmCompat.h"
#include "CanvasWidget.h"
#include "ProjectSupportUtils.h"
#include "LogPanel.h"
#include "MainMenu.h"
#include "MenuWorkflowController.h"
#include "ReconstructionWorkflowController.h"
#include "MatchPairSelectorDialog.h"
#include "FeatureMatchingDialog.h"
#include "FeatureMatchRunner.h"
#include "ForwardIntersectionCheckDialog.h"
#include "ForwardIntersectionResultsDialog.h"
#include "ProjectManager.h"
#include "ProjectIO.h"
#include "ProjectData.h"
#include "ProjectDashboardWidget.h"
#include "PhotoStripWidget.h"
#include "AppConfigManager.h"
#include "DialogSettingStore.h"
#include "DialogSettingKeys.h"
#include "DataTreeWidget.h"
#include "ReferencePanelWidget.h"
#include "SelectionPropertiesWidget.h"
#include "TaskStatusWidget.h"
#include "ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"
#include "WorkspaceCenterWidget.h"
#include "CameraModel3DDialog.h"
#include "LayerRenderer.h"
#include "Logger.h"
#include "ModelDropSupport.h"

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
    _config->windowState()->load(this);

    if (windowState().testFlag(Qt::WindowFullScreen))
    {
        setWindowState((windowState() & ~Qt::WindowFullScreen) | Qt::WindowMaximized);
    }

    setupBottomPanel();
    setupMenuConnections();
    setupProjectManager();

    setAcceptDrops(true);
    statusBar()->showMessage(tr("就绪"));
}

MainWindow::~MainWindow()
{
    delete _ui;
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
    _workspaceCenter = _ui->workspaceCenter;
    _canvas       = _workspaceCenter->canvas();
    _mainSplitter->setStretchFactor(1, 1);

    _log = _ui->logPanel;
    _logDock = _ui->logDock;
    _logDock->setAllowedAreas(Qt::BottomDockWidgetArea);
    _logDock->setVisible(false);

    LOG_INFO("%s", qUtf8Printable(tr("日志面板已就绪")));

}

void MainWindow::setupSelectionPanels()
{
    if (!_mainSplitter || !_leftTabs || !_workspaceCenter || _leftPanelSplitter)
    {
        return;
    }

    const int leftIndex = _mainSplitter->indexOf(_leftTabs);
    const int workspaceIndex = _mainSplitter->indexOf(_workspaceCenter);
    if (leftIndex < 0 || workspaceIndex < 0)
    {
        return;
    }

    _selectionProperties = new SelectionPropertiesWidget(this);
    _selectionProperties->setObjectName(QStringLiteral("selectionProperties"));
    _selectionProperties->setMinimumHeight(145);

    _leftPanelSplitter = new QSplitter(Qt::Vertical);
    _leftPanelSplitter->setObjectName(QStringLiteral("leftPanelSplitter"));
    _leftPanelSplitter->setChildrenCollapsible(false);

    _rightPanelSplitter = new QSplitter(Qt::Vertical);
    _rightPanelSplitter->setObjectName(QStringLiteral("rightPanelSplitter"));
    _rightPanelSplitter->setChildrenCollapsible(false);

    auto *photosFrame = new QFrame(_rightPanelSplitter);
    photosFrame->setObjectName(QStringLiteral("photosPanel"));
    photosFrame->setFrameShape(QFrame::StyledPanel);
    photosFrame->setMinimumHeight(170);
    photosFrame->setMaximumHeight(320);

    auto *photosLayout = new QVBoxLayout(photosFrame);
    photosLayout->setContentsMargins(0, 0, 0, 0);
    photosLayout->setSpacing(0);

    auto *photosTitleBar = new QWidget(photosFrame);
    photosTitleBar->setObjectName(QStringLiteral("photosTitleBar"));
    auto *photosTitleLayout = new QHBoxLayout(photosTitleBar);
    photosTitleLayout->setContentsMargins(6, 2, 4, 2);
    photosTitleLayout->setSpacing(4);
    auto *photosTitle = new QLabel(tr("照片"), photosTitleBar);
    auto *photosClose = new QToolButton(photosTitleBar);
    photosClose->setObjectName(QStringLiteral("photosCloseButton"));
    photosClose->setText(QStringLiteral("x"));
    photosClose->setAutoRaise(true);
    photosClose->setToolTip(tr("隐藏照片面板"));
    photosTitleLayout->addWidget(photosTitle);
    photosTitleLayout->addStretch(1);
    photosTitleLayout->addWidget(photosClose);
    photosLayout->addWidget(photosTitleBar);

    _photoStrip = new PhotoStripWidget(photosFrame);
    photosLayout->addWidget(_photoStrip, 1);
    _photosPanel = photosFrame;

    connect(photosClose, &QToolButton::clicked, this, [this]()
    {
        if (_mainMenu && _mainMenu->togglePhotosAction())
        {
            _mainMenu->togglePhotosAction()->setChecked(false);
            return;
        }
        if (_photosPanel)
        {
            _photosPanel->setVisible(false);
        }
        saveUiSetting(QJsonObject{{QStringLiteral("photos_visible"), false}});
    });

    QWidget *oldLeftTabs = _mainSplitter->replaceWidget(leftIndex, _leftPanelSplitter);
    if (!oldLeftTabs)
    {
        oldLeftTabs = _leftTabs;
    }
    QWidget *oldWorkspace = _mainSplitter->replaceWidget(workspaceIndex, _rightPanelSplitter);
    if (!oldWorkspace)
    {
        oldWorkspace = _workspaceCenter;
    }

    _leftPanelSplitter->addWidget(oldLeftTabs);
    _leftPanelSplitter->addWidget(_selectionProperties);
    _leftPanelSplitter->setStretchFactor(0, 3);
    _leftPanelSplitter->setStretchFactor(1, 1);
    _leftPanelSplitter->setSizes({560, 190});

    _rightPanelSplitter->addWidget(oldWorkspace);
    _rightPanelSplitter->addWidget(_photosPanel);
    _rightPanelSplitter->setStretchFactor(0, 5);
    _rightPanelSplitter->setStretchFactor(1, 1);
    _rightPanelSplitter->setSizes({620, 210});

    _mainSplitter->insertWidget(0, _leftPanelSplitter);
    _mainSplitter->insertWidget(1, _rightPanelSplitter);
    _mainSplitter->setCollapsible(0, false);
    _mainSplitter->setCollapsible(1, false);
    _mainSplitter->setStretchFactor(0, 0);
    _mainSplitter->setStretchFactor(1, 1);
    _mainSplitter->setSizes({320, 960});
}

// ============================================================
//  setupBottomPanel — 底部面板标题栏切换按钮
// ============================================================

void MainWindow::setupBottomPanel()
{
    QWidget* titleBar = new QWidget();

    _logBtn = new QToolButton(titleBar);
    _logBtn->setText(tr("日志"));
    _logBtn->setCheckable(true);
    _logBtn->setChecked(false);
    _logBtn->setVisible(false);

    auto *grp = new QButtonGroup(titleBar);
    grp->setExclusive(true);
    grp->addButton(_logBtn, 0);

    _logDock->setTitleBarWidget(titleBar);

    connect(_logBtn, &QToolButton::clicked, this, &MainWindow::onLogBtnClicked);

    Q_UNUSED(titleBar);
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

    if (_mainMenu->zoomInAction())
    {
        connect(_mainMenu->zoomInAction(), &QAction::triggered, _canvas, &CanvasWidget::zoomIn);
    }
    if (_mainMenu->zoomOutAction())
    {
        connect(_mainMenu->zoomOutAction(), &QAction::triggered, _canvas, &CanvasWidget::zoomOut);
    }
    if (_mainMenu->resetViewAction())
    {
        connect(_mainMenu->resetViewAction(), &QAction::triggered, _canvas, &CanvasWidget::resetView);
    }

    if (_mainMenu->toggleLogAction())
    {
        connect(_mainMenu->toggleLogAction(), &QAction::toggled, this, &MainWindow::onToggleLogAction);
    }

    connectDockAction(_mainMenu->toggleWorkspaceAction(),
                      _leftTabs,
                      QStringLiteral("workspace_visible"));
    connectDockAction(_mainMenu->togglePropertiesAction(),
                      _selectionProperties,
                      QStringLiteral("properties_visible"));
    if (_mainMenu->togglePhotosAction() && _photosPanel)
    {
        auto *photosAction = _mainMenu->togglePhotosAction();
        photosAction->setCheckable(true);
        photosAction->setChecked(!_photosPanel->isHidden());
        connect(photosAction, &QAction::toggled, this, [this](bool on)
        {
            if (_photosPanel)
            {
                _photosPanel->setVisible(on);
            }
            if (on)
            {
                if (_logDock)
                {
                    _logDock->setVisible(false);
                }
                if (_logBtn)
                {
                    _logBtn->setChecked(false);
                    _logBtn->setVisible(false);
                }
                if (_mainMenu && _mainMenu->toggleLogAction())
                {
                    const QSignalBlocker blocker(_mainMenu->toggleLogAction());
                    _mainMenu->toggleLogAction()->setChecked(false);
                }
                saveUiSetting(QJsonObject{
                    {QStringLiteral("photos_visible"), true},
                    {QStringLiteral("log_visible"), false}
                });
                return;
            }
            saveUiSetting(QJsonObject{{QStringLiteral("photos_visible"), false}});
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
    if (_mainMenu->toggleCamerasAction() && _workspaceCenter && _workspaceCenter->modelView())
    {
        connect(_mainMenu->toggleCamerasAction(), &QAction::toggled,
                _workspaceCenter->modelView(), &CameraSceneWidget::setShowCameras);
    }
    if (_mainMenu->toggleWorldOriginAction() && _workspaceCenter && _workspaceCenter->modelView())
    {
        connect(_mainMenu->toggleWorldOriginAction(), &QAction::toggled,
                _workspaceCenter->modelView(), &CameraSceneWidget::setShowWorldOrigin);
        connect(_mainMenu->toggleWorldOriginAction(), &QAction::toggled, this, [this](bool on)
        {
            saveUiSetting(QJsonObject{{QStringLiteral("world_origin_visible"), on}});
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
        connect(_photoStrip, &PhotoStripWidget::photoActivated, this, [this](const QString &path)
        {
            selectPhoto(path, true);
        });
    }
    // 画布切换影像时，通过 DialogSettingStore 持久化活跃影像路径
    if (_canvas)
    {
        connect(_canvas, &CanvasWidget::activeImageChanged, this, [this](const QString &path)
        {
            saveUiSetting(QJsonObject{{QStringLiteral("active_image_path"), path}});
        });
    }

    if (_mainMenu)
    {
        if (_mainMenu->toggleLogAction())
        {
            connect(_mainMenu->toggleLogAction(), &QAction::toggled, this, &MainWindow::onLogVisiblePersist);
        }
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
            connect(_mainMenu->newAction(), &QAction::triggered, _projectManager, &ProjectManager::createNewProject);
        }
        if (_mainMenu->openAction())
        {
            connect(_mainMenu->openAction(), &QAction::triggered, _projectManager, &ProjectManager::openProject);
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

        // ── 重建菜单 → ReconstructionWorkflowController ──
        if (_reconController)
        {
            auto connectRecon = [&](QAction *act, void (ReconstructionWorkflowController::*slot)())
            {
                if (act)
                {
                    connect(act, &QAction::triggered, _reconController, slot);
                }
            };
            // 稀疏重建
            connectRecon(_mainMenu->buildObsNetworkAction(),   &ReconstructionWorkflowController::openObservationNetworkDialog);
            connectRecon(_mainMenu->initCameraPoseAction(),    &ReconstructionWorkflowController::openInitCameraPoseDialog);
            connectRecon(_mainMenu->triangulateAction(),       &ReconstructionWorkflowController::openTriangulationDialog);
            connectRecon(_mainMenu->reconBundleAdjustAction(), &ReconstructionWorkflowController::openReconBundleAdjustDialog);
            connectRecon(_mainMenu->sparseCloudPostProcessAction(), &ReconstructionWorkflowController::openSparseCloudPostProcessDialog);
            // 密集重建
            connectRecon(_mainMenu->denseMatchAction(),       &ReconstructionWorkflowController::openDenseMatchDialog);
            connectRecon(_mainMenu->depthMapEstimateAction(),  &ReconstructionWorkflowController::openDepthMapEstimateDialog);
            connectRecon(_mainMenu->fuseDepthMapsAction(),     &ReconstructionWorkflowController::openDepthFusionDialog);
            connectRecon(_mainMenu->refineDenseCloudAction(),  &ReconstructionWorkflowController::openDenseCloudRefineDialog);
            // 模型生成
            connectRecon(_mainMenu->meshReconstructAction(),   &ReconstructionWorkflowController::openMeshReconstructionDialog);
            connectRecon(_mainMenu->textureMappingAction(),    &ReconstructionWorkflowController::openTextureMappingDialog);
            connectRecon(_mainMenu->exportModelAction(),       &ReconstructionWorkflowController::openModelExportDialog);
        }

        // 连接匹配相关菜单动作到对话框（如果 ProjectManager 可用则传入以便填充数据）
        if (_mainMenu->matchFeaturesAction())
        {
            connect(_mainMenu->matchFeaturesAction(), &QAction::triggered, this, [this]()
            {
                if (!_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开匹配配置：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new FeatureMatchingDialog(this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->setProjectImages(_projectManager->getAllImages());

                // 从项目整体收集可用的特征后缀并设置到下拉框
                const QStringList projectSuffixes = xjw::gui::project::projectFeatureSuffixes(
                    _projectManager->currentProjectPath(), _projectManager->currentMeta());
                if (!projectSuffixes.isEmpty())
                {
                    dlg->setAvailableFeatureSuffixes(projectSuffixes);
                }

                // 懒初始化特征匹配记忆化设置管理器
                if (!_featureMatchingSetting)
                {
                    _featureMatchingSetting = new DialogSettingStore(DialogSettingKeys::FeatureMatching, this);
                }
                _featureMatchingSetting->setProjectPath(_projectManager->currentProjectPath());
                const QJsonObject saved = _featureMatchingSetting->load();
                if (!saved.isEmpty())
                {
                    dlg->applySettings(saved);
                }

                // 默认输出目录（仅当用户未保存过 output_dir 时才设置）
                const QString assetsDir = ProjectIO::projectAssetsDir(_projectManager->currentProjectPath());
                if (!assetsDir.isEmpty() && saved.value("output_dir").toString().isEmpty())
                {
                    QJsonObject defaultOutput = saved;  // 基于已恢复的设置，只覆盖 output_dir
                    defaultOutput.insert("output_dir", QDir(assetsDir).filePath(QStringLiteral("matches")));
                    dlg->applySettings(defaultOutput);
                }

                // 实时保存参数到项目配置（通过 DialogSettingStore）
                connect(dlg, &FeatureMatchingDialog::settingsChanged, this, [this](const QJsonObject &s)
                {
                    if (_featureMatchingSetting)
                    {
                        _featureMatchingSetting->save(s);
                    }
                });

                // 连接运行请求信号
                connect(dlg, &FeatureMatchingDialog::runRequested, this,
                    [this](const QJsonObject &config, const QStringList &imagePairs)
                {
                    QPointer<ProjectManager> pmGuard(_projectManager);
                    if (!pmGuard)
                    {
                        LOG_ERROR(QStringLiteral("无法运行特征匹配：项目管理器未初始化"));
                        return;
                    }

                    if (imagePairs.isEmpty())
                    {
                        LOG_WARN(QStringLiteral("未生成匹配对，请先点击【生成匹配对】按钮"));
                        return;
                    }

                    const QString algorithm = config.value("match_algorithm").toString("superglue");
                    // 在后台线程执行匹配，避免界面卡死
                    LOG_INFO("%s", qUtf8Printable(QString("开始在后台线程执行 %1 匹配...").arg(algorithm)));

                    // 在状态栏展示特征匹配进度
                    QPointer<MainWindow> self(this);
                    auto cancelFlag    = std::make_shared<std::atomic<bool>>(false);
                    auto progressCount = std::make_shared<std::atomic<int>>(0);
                    int total = imagePairs.size();
                    const QString featureSuffix = config.value(QStringLiteral("feature_suffix")).toString();
                    if (featureSuffix == QStringLiteral("__all__"))
                    {
                        const QStringList compatibleSuffixes =
                            xjw::feature_match::compatibleFeatureSuffixes(algorithm);
                        const QStringList availableSuffixes = xjw::gui::project::projectFeatureSuffixes(
                            pmGuard->currentProjectPath(), pmGuard->currentMeta());
                        int suffixCount = 0;
                        if (availableSuffixes.isEmpty())
                        {
                            suffixCount = compatibleSuffixes.size();
                        }
                        else
                        {
                            for (const QString &suffix : compatibleSuffixes)
                            {
                                if (availableSuffixes.contains(suffix))
                                {
                                    ++suffixCount;
                                }
                            }
                        }
                        if (suffixCount > 1)
                        {
                            total *= suffixCount;
                        }
                    }

                    showSgProgress(total);

                    auto cancelConn = connect(this, &MainWindow::sgCancelRequested,
                                              this, [cancelFlag]()
                    {
                        cancelFlag->store(true);
                    });

                    // 定时刷新进度（100ms 轮询）
                    auto *timer = new QTimer(this);
                    timer->setInterval(100);
                    connect(timer, &QTimer::timeout, timer,
                            [self, progressCount, total]()
                    {
                        Q_UNUSED(total);
                        if (!self)
                        {
                            return;
                        }
                        self->updateSgProgress(progressCount->load());
                    });
                    timer->start();

                    auto *watcher = new QFutureWatcher<void>(this);
                    connect(watcher, &QFutureWatcher<void>::finished, watcher,
                            [self, cancelFlag, timer, watcher, cancelConn]()
                        {
                        timer->stop();
                        timer->deleteLater();
                        QObject::disconnect(cancelConn);
                        if (self)
                        {
                            self->hideSgProgress(!cancelFlag->load());
                        }
                        watcher->deleteLater();
                    });

                    watcher->setFuture(QtConcurrent::run(
                        [config, imagePairs, pmGuard, cancelFlag, progressCount]()
                        {
                            FeatureMatchRunner::run(config, imagePairs, pmGuard, *cancelFlag, *progressCount);
                        }));
                });

                // 连接"查看匹配"信号：在对话框中直接打开匹配查看器
                connect(dlg, &FeatureMatchingDialog::viewMatchesRequested, this, [this]()
                {
                    if (!_projectManager)
                    {
                        return;
                    }

                    auto *matchDlg = new MatchPairSelectorDialog(_projectManager, this);
                    matchDlg->setAttribute(Qt::WA_DeleteOnClose);
                    matchDlg->show();
                });

                dlg->exec();
            });
        }

        if (_mainMenu->viewMatchesAction())
        {
            connect(_mainMenu->viewMatchesAction(), &QAction::triggered, this, [this]()
            {
                if (!_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开匹配查看：ProjectManager 未初始化"));
                    return;
                }
                // 打开匹配对选择对话框以便用户选择要查看的匹配对
                auto *dlg = new MatchPairSelectorDialog(_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
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

        connect(_projectManager, &ProjectManager::projectCreated, this, &MainWindow::onProjectOpened);
        connect(_projectManager, &ProjectManager::projectOpened,  this, &MainWindow::onProjectOpened);

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
            if (!p.isEmpty() && _projectManager)
            {
                _projectManager->openProjectFromPath(p);
            }
        });
        connect(_mainMenu, &MainMenu::clearRecentRequested, this, &MainWindow::onClearRecentRequested);
    }

    
    connect(_projectManager, &ProjectManager::saveStarted,    this, &MainWindow::onSaveStarted);
    connect(_projectManager, &ProjectManager::saveFinished,   this, &MainWindow::onSaveFinished);
    connect(_projectManager, &ProjectManager::metadataDirtyChanged, this, &MainWindow::onMetadataDirtyChanged);

    connect(_projectManager, &ProjectManager::projectOpened, this, [this](const QString &)
    {
        if (_dataTree)
        {
            _dataTree->loadFromJson(_projectManager->currentMeta());
        }
        if (_dashboard)
        {
            _dashboard->loadFromJson(_projectManager->currentMeta());
        }
        if (_referencePanel)
        {
            _referencePanel->loadFromJson(_projectManager->currentMeta());
        }
    });
    connect(_projectManager, &ProjectManager::projectMetadataUpdated, this, [this](const QString &)
    {
        if (_dataTree)
        {
            _dataTree->loadFromJson(_projectManager->currentMeta());
        }
        if (_dashboard)
        {
            _dashboard->loadFromJson(_projectManager->currentMeta());
        }
        if (_referencePanel)
        {
            _referencePanel->loadFromJson(_projectManager->currentMeta());
        }
    });
    connect(_projectManager, &ProjectManager::projectMetadataChanged, _dashboard, &ProjectDashboardWidget::loadFromJson);
    connect(_projectManager, &ProjectManager::projectMetadataChanged, _dataTree, &DataTreeWidget::loadFromJson);
    connect(_projectManager, &ProjectManager::projectMetadataChanged, _referencePanel, &ReferencePanelWidget::loadFromJson);
    connect(_projectManager, &ProjectManager::projectMetadataChanged, this, [this](const QJsonObject &meta)
    {
        if (_photoStrip)
        {
            if (_projectManager)
            {
                _photoStrip->setProjectPath(_projectManager->currentProjectPath());
            }
            _photoStrip->loadFromJson(meta);
        }
    });

    connect(_dataTree, &DataTreeWidget::removeRequested,  _projectManager, &ProjectManager::removeResources);
    connect(_dataTree, &DataTreeWidget::deleteDataRequested, _projectManager, &ProjectManager::deleteGeneratedData);
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
            return xjw::gui::project::projectFilesRootObject(_projectManager->currentMeta());
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
            // 直接加载稀疏点云 XYZ 文件到 3D 视图
            if (!path.isEmpty() && QFileInfo::exists(path))
            {
                _workspaceCenter->showPointCloudFile(path);
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

    connect(_projectManager, &ProjectManager::projectMetadataChanged, this, [this](const QJsonObject &meta)
    {
        if (_workspaceCenter)
        {
            _workspaceCenter->setProjectMeta(meta);
        }
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
    _meshTaskStatus = createTaskStatus(220, false, QString());

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

    // ── 特征点提取状态栏进度条 ──────────────────────────
    _spTaskStatus = createTaskStatus(180, true, tr("正在取消特征提取..."));
    connect(_spTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        emit spCancelRequested();
    });

    // ── 密集匹配进度条 ─────────────────────────────────────────
    _dmTaskStatus = createTaskStatus(180, true, tr("正在取消密集匹配..."));
    connect(_dmTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        emit dmCancelRequested();
    });

    // ── 重叠对获取进度条 ─────────────────────────────────────────
    _overlapTaskStatus = createTaskStatus(200, true, tr("正在取消重叠对获取..."));
    connect(_overlapTaskStatus, &TaskStatusWidget::cancelRequested, this, [this]()
    {
        emit overlapCancelRequested();
    });

    // ── 观测网络构建状态栏进度条 ─────────────────────────────────────
    _obsNetTaskStatus = createTaskStatus(200, false, QString());

    connect(_projectManager, &ProjectManager::obsNetProgressChanged,
            this, &MainWindow::onObsNetProgress);
    connect(_projectManager, &ProjectManager::obsNetProgressFinished,
            this, &MainWindow::onObsNetFinished);

    refreshDashboardTaskSnapshots();
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
    appendTask(tr("特征提取"), _spTaskStatus);
    appendTask(tr("密集匹配"), _dmTaskStatus);
    appendTask(tr("重叠对获取"), _overlapTaskStatus);
    appendTask(tr("观测网络"), _obsNetTaskStatus);

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
        xjw::gui::project::collectMatchedImageNamePairs(plascanPath, _projectManager->currentMeta());
    if (matchedPairs.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("当前项目中没有可导出的匹配对");
        }
        return false;
    }

    const QString projectRoot = ProjectIO::projectRootFromPlascan(plascanPath);
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
//  观测网络构建进度状态栏槽
// ============================================================

void MainWindow::onObsNetProgress(const QString &stage, int percent)
{
    if (!_obsNetTaskStatus)
    {
        return;
    }
    const QString statusText = QStringLiteral("观测网络: %1 %2%").arg(stage).arg(percent);
    if (!_obsNetTaskStatus->isActive())
    {
        _obsNetTaskStatus->begin(statusText, 0, 100);
    }
    _obsNetTaskStatus->updateProgress(statusText, percent);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onObsNetFinished(bool success)
{
    if (!_obsNetTaskStatus)
    {
        return;
    }
    _obsNetTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(
        success ? tr("观测网络构建完成") : tr("观测网络构建失败"), 4000);
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

void MainWindow::showDmProgress(int total)
{
    if (!_dmTaskStatus)
    {
        return;
    }
    _dmTaskStatus->begin(tr("密集匹配 0/%1").arg(total), 0, total);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::updateDmProgress(int done)
{
    if (!_dmTaskStatus)
    {
        return;
    }
    const int total = _dmTaskStatus->progressMaximum();
    const int clampedDone = std::clamp(done, 0, total);
    _dmTaskStatus->updateProgress(
        tr("密集匹配 %1/%2").arg(clampedDone).arg(total), clampedDone);
    refreshDashboardTaskSnapshots();
}

void MainWindow::hideDmProgress(bool ok)
{
    if (!_dmTaskStatus)
    {
        return;
    }
    _dmTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(ok ? tr("密集匹配完成") : tr("密集匹配已取消"), 4000);
}

void MainWindow::onOverlapProgress(const QString &stage, int percent)
{
    if (!_overlapTaskStatus)
    {
        return;
    }
    const QString statusText = percent > 0 && percent < 100
        ? tr("重叠对: %1 %2%").arg(stage).arg(percent)
        : tr("重叠对: %1").arg(stage);
    if (!_overlapTaskStatus->isActive())
    {
        _overlapTaskStatus->begin(statusText, 0, 100);
    }
    _overlapTaskStatus->updateProgress(statusText, std::clamp(percent, 0, 100));
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::onOverlapFinished(bool success)
{
    if (!_overlapTaskStatus)
    {
        return;
    }
    _overlapTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(success ? tr("重叠对获取完成") : tr("重叠对获取已取消或失败"), 4000);
}

// ============================================================
//  特征提取状态栏进度条 slots
// ============================================================

void MainWindow::showSpProgress(int total)
{
    if (!_spTaskStatus)
    {
        return;
    }
    _spTaskStatus->begin(tr("特征提取 0/%1").arg(total), 0, total);
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(QString());
}

void MainWindow::updateSpProgress(int done)
{
    if (!_spTaskStatus)
    {
        return;
    }
    const int total = _spTaskStatus->progressMaximum();
    const int clampedDone = std::clamp(done, 0, total);
    _spTaskStatus->updateProgress(
        tr("特征提取 %1/%2").arg(clampedDone).arg(total), clampedDone);
    refreshDashboardTaskSnapshots();
}

void MainWindow::hideSpProgress(bool ok)
{
    if (!_spTaskStatus)
    {
        return;
    }
    _spTaskStatus->finish();
    refreshDashboardTaskSnapshots();
    statusBar()->showMessage(ok ? tr("特征提取完成") : tr("特征提取已取消"), 4000);
}

// ============================================================
//  辅助方法
// ============================================================

void MainWindow::connectDockAction(QAction *action, QWidget *panel, const QString &settingKey)
{
    if (!action || !panel)
    {
        return;
    }

    action->setCheckable(true);
    action->setChecked(!panel->isHidden());

    connect(action, &QAction::toggled, panel, [this, panel, settingKey](bool on)
    {
        panel->setVisible(on);
        saveUiSetting(QJsonObject{{settingKey, on}});
    });

    if (auto *dock = qobject_cast<QDockWidget *>(panel))
    {
        connect(dock, &QDockWidget::visibilityChanged, action, [this, action, settingKey](bool on)
        {
            const QSignalBlocker blocker(action);
            action->setChecked(on);
            saveUiSetting(QJsonObject{{settingKey, on}});
        });
    }
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
    const QJsonArray images = xjw::gui::project::projectImageEntries(currentProjectMeta());

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
    saveUiSetting(QJsonObject{{QStringLiteral("active_image_path"), imagePath}});
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

void MainWindow::saveUiSetting(const QJsonObject &partial)
{
    if (_uiSetting)
    {
        _uiSetting->merge(partial);
    }
}

QString MainWindow::currentBottomPanelKey() const
{
    if (_photosPanel && _photosPanel->isVisible())
    {
        return QStringLiteral("photos");
    }
    return QStringLiteral("log");
}

void MainWindow::switchToLogPanel()
{
    if (_photosPanel)
    {
        _photosPanel->setVisible(false);
    }
    if (_mainMenu && _mainMenu->togglePhotosAction())
    {
        const QSignalBlocker blocker(_mainMenu->togglePhotosAction());
        _mainMenu->togglePhotosAction()->setChecked(false);
    }
    if (_logDock)
    {
        _logDock->setWidget(_log);
        _logDock->setVisible(true);
    }
    if (_log)
    {
        _log->loadFromLogFile();
    }
    if (_logBtn)
    {
        _logBtn->setChecked(true);
    }
}


// ============================================================
//  底部面板按钮槽
// ============================================================

void MainWindow::onLogBtnClicked()
{
    switchToLogPanel();
    QJsonObject s;
    s[QStringLiteral("bottom_panel")] = QStringLiteral("log");
    if (_mainMenu && _mainMenu->toggleLogAction())
    {
        s[QStringLiteral("log_visible")] = _mainMenu->toggleLogAction()->isChecked();
    }
    saveUiSetting(s);
}

// Interest-point panel removed: onIpBtnClicked is a no-op now.

// ============================================================
//  菜单动作响应
// ============================================================

void MainWindow::onToggleLogAction(bool on)
{
    if (_logBtn)
    {
        _logBtn->setVisible(on);
    }
    if (on)
    {
        switchToLogPanel();
        if (_log)
        {
            _log->loadFromLogFile();
        }
        if (_logBtn)
        {
            _logBtn->setChecked(true);
        }
    }
    else
    {
        if (_logDock)
        {
            _logDock->setVisible(false);
        }
    }
}

// Interest-point info UI removed: related slots are no-ops / deleted.

// ============================================================
//  菜单联动持久化槽
// ============================================================

void MainWindow::onLogVisiblePersist(bool on)
{
    QJsonObject s;
    s[QStringLiteral("log_visible")] = on;
    s[QStringLiteral("bottom_panel")] = currentBottomPanelKey();
    saveUiSetting(s);
}

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
    QString projPath = _projectManager->currentProjectPath();
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

void MainWindow::onProjectCreated(const QString &plascanPath)
{
    onProjectOpened(plascanPath);
}

void MainWindow::onProjectOpened(const QString &plascanPath)
{
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
    if (_workspaceCenter)
    {
        _workspaceCenter->showModelView();
    }
    if (_dataTree)
    {
        _dataTree->setProjectPath(plascanPath);
    }
    if (_photoStrip)
    {
        _photoStrip->setProjectPath(plascanPath);
    }
    if (_projectManager && _dataTree)
        _dataTree->loadFromJson(_projectManager->currentMeta());
    if (_projectManager && _dashboard)
        _dashboard->loadFromJson(_projectManager->currentMeta());
    if (_projectManager && _workspaceCenter)
        _workspaceCenter->setProjectMeta(_projectManager->currentMeta());
    if (_projectManager && _photoStrip)
        _photoStrip->loadFromJson(_projectManager->currentMeta());

    if (_config && _mainMenu)
    {
        _config->recentProjects()->addRecentProject(plascanPath);
        _mainMenu->setRecentProjects(_config->recentProjects()->recentProjects());
    }

    if (!_projectManager)
    {
        return;
    }

    // 初始化/更新项目级 UI 设置路径
    if (!_uiSetting)
    {
        _uiSetting = new DialogSettingStore(DialogSettingKeys::MainWindowUi, this);
    }
    _uiSetting->setProjectPath(plascanPath);
    if (!_featureMatchingSetting)
    {
        _featureMatchingSetting = new DialogSettingStore(DialogSettingKeys::FeatureMatching, this);
    }
    _featureMatchingSetting->setProjectPath(plascanPath);

    QJsonObject ui = _uiSetting->load();
    // 向后兼容：若新文件中无数据，尝试从旧 project_config.json 读取
    if (ui.isEmpty())
    {
        ui = _projectManager->loadUiSettings();
    }
    applyUiSettings(ui);

    // feature-info panel removed: do not reload interest points into a removed panel here
}

// ============================================================
//  applyUiSettings — 恢复项目范围内的 UI 设置
// ============================================================

void MainWindow::applyUiSettings(const QJsonObject &ui)
{
    // 特征后缀恢复不能依赖主窗口 UI 设置存在；旧项目可能只有 assets/ip/*.dsk。
    if (_menuWorkflowController)
    {
        _menuWorkflowController->applySavedFeatureDisplayOptions(ui);
    }

    if (ui.isEmpty())
    {
        return;
    }

    // feature-info panel removed: no ip button handling
    if (_logBtn && ui.contains(QStringLiteral("log_visible")))
    {
        bool lvis = ui.value(QStringLiteral("log_visible")).toBool();
        _logBtn->setChecked(lvis);
        _logBtn->setVisible(lvis);
    }

    if (ui.contains(QStringLiteral("log_display_level")) && _log)
    {
        int lvl = ui.value(QStringLiteral("log_display_level")).toInt(static_cast<int>(Logger::Info));
        _log->setDisplayLevel(static_cast<Logger::Level>(lvl));
    }

    bool panelHandledByLog = false;
    if (ui.contains(QStringLiteral("log_visible")) && _mainMenu && _mainMenu->toggleLogAction())
    {
        bool on = ui.value(QStringLiteral("log_visible")).toBool();
        _mainMenu->toggleLogAction()->blockSignals(true);
        _mainMenu->toggleLogAction()->setChecked(on);
        _mainMenu->toggleLogAction()->blockSignals(false);
        if (_logBtn)
        {
            _logBtn->setVisible(on);
        }
        if (on)
        {
            switchToLogPanel();
            panelHandledByLog = true;
        }
    }

    // feature-info visibility handling removed
    auto applyVisibility = [](QAction *action, QWidget *panel, const QJsonObject &settings, const QString &key)
    {
        if (!action || !panel || !settings.contains(key))
        {
            return;
        }

        const bool on = settings.value(key).toBool();
        {
            const QSignalBlocker actionBlocker(action);
            action->setChecked(on);
        }
        {
            const QSignalBlocker panelBlocker(panel);
            panel->setVisible(on);
        }
    };

    if (_mainMenu)
    {
        applyVisibility(_mainMenu->toggleWorkspaceAction(),
                        _leftTabs,
                        ui,
                        QStringLiteral("workspace_visible"));
        applyVisibility(_mainMenu->togglePropertiesAction(),
                        _selectionProperties,
                        ui,
                        QStringLiteral("properties_visible"));
        applyVisibility(_mainMenu->togglePhotosAction(),
                        _photosPanel,
                        ui,
                        QStringLiteral("photos_visible"));

        const bool photosVisible = _photosPanel && _photosPanel->isVisible();
        if (photosVisible)
        {
            if (_logDock)
            {
                _logDock->setVisible(false);
            }
            if (_logBtn)
            {
                _logBtn->setChecked(false);
                _logBtn->setVisible(false);
            }
            if (_mainMenu->toggleLogAction())
            {
                const QSignalBlocker blocker(_mainMenu->toggleLogAction());
                _mainMenu->toggleLogAction()->setChecked(false);
            }
        }

        if (ui.contains(QStringLiteral("world_origin_visible")) && _mainMenu->toggleWorldOriginAction())
        {
            const bool on = ui.value(QStringLiteral("world_origin_visible")).toBool();
            {
                const QSignalBlocker blocker(_mainMenu->toggleWorldOriginAction());
                _mainMenu->toggleWorldOriginAction()->setChecked(on);
            }
            if (_workspaceCenter && _workspaceCenter->modelView())
            {
                _workspaceCenter->modelView()->setShowWorldOrigin(on);
            }
        }
    }

    if (ui.contains(QStringLiteral("active_image_path")) && _canvas)
    {
        QString imagePath = ui.value(QStringLiteral("active_image_path")).toString();
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
    if (_config)
    {
        _config->windowState()->save(this);
    }

    if (_projectManager && _projectManager->isDirty())
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

            // 保存完成后接受事件，自动退出
            event->accept();
            return;
        }
        else if (btn == QMessageBox::Discard)
        {
            _projectManager->discardTemporaryMeta();
        }
    }

    event->accept();
    QMainWindow::closeEvent(event);
}
