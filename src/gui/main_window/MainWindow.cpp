#include "MainWindow.h"

#include <QApplication>
#include <QSplitter>
#include <QDockWidget>
#include <QToolBar>
#include <QMenuBar>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QFileInfo>
#include <QDesktopServices>
#include <QUrl>
#include <QListWidget>
#include <QToolButton>
#include <QButtonGroup>
#include <QJsonObject>
#include <QMessageBox>
#include <QProgressDialog>
#include <QProgressBar>
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
#include "AppConfigManager.h"
#include "DialogSettingStore.h"
#include "DialogSettingKeys.h"
#include "DataTreeWidget.h"
#include "ReferencePanelWidget.h"
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
{
    Qt::WindowFlags flags = windowFlags();
    flags |= Qt::Window;
    flags |= Qt::WindowMinimizeButtonHint;
    flags |= Qt::WindowCloseButtonHint;
    flags |= Qt::WindowSystemMenuHint;
    flags &= ~Qt::FramelessWindowHint;
    setWindowFlags(flags);
    setWindowTitle(QStringLiteral("PlaScan"));
    m_mainMenu = new MainMenu(this);
    m_config   = new AppConfigManager(this);
    m_config->windowState()->load(this);

    if (windowState().testFlag(Qt::WindowFullScreen))
    {
        setWindowState((windowState() & ~Qt::WindowFullScreen) | Qt::WindowMaximized);
    }

    setupUi();
    setupBottomPanel();
    setupMenuConnections();
    setupProjectManager();

    setAcceptDrops(true);
    statusBar()->showMessage(tr("就绪"));
}

MainWindow::~MainWindow() = default;

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

    if (m_workspaceCenter)
    {
        m_workspaceCenter->showModelFile(modelPath);
        statusBar()->showMessage(tr("已加载三维模型：%1").arg(QFileInfo(modelPath).fileName()), 5000);
    }
    event->acceptProposedAction();
}

// ============================================================
//  setupUi — 创建核心布局组件
// ============================================================

void MainWindow::setupUi()
{
    m_mainSplitter = new QSplitter(this);
    m_leftTabs     = new QTabWidget();
    m_dataTree     = new DataTreeWidget();
    m_referencePanel = new ReferencePanelWidget();
    m_workspaceCenter = new WorkspaceCenterWidget();
    m_canvas       = m_workspaceCenter->canvas();

    m_leftTabs->addTab(m_dataTree, tr("工作区"));
    m_leftTabs->addTab(m_referencePanel, tr("参考"));

    QWidget* centerWidget = new QWidget();
    QVBoxLayout* centerLayout = new QVBoxLayout(centerWidget);
    centerLayout->setContentsMargins(0, 0, 0, 0);
    centerLayout->addWidget(m_workspaceCenter);

    m_mainSplitter->addWidget(m_leftTabs);
    m_mainSplitter->addWidget(centerWidget);
    m_mainSplitter->setStretchFactor(1, 1);
    setCentralWidget(m_mainSplitter);

    m_log = new LogPanel();
    m_logDock = new QDockWidget(tr("日志"), this);
    m_logDock->setAllowedAreas(Qt::BottomDockWidgetArea);
    m_logDock->setWidget(m_log);
    addDockWidget(Qt::BottomDockWidgetArea, m_logDock);

    LOG_INFO("%s", qUtf8Printable(tr("日志面板已就绪")));

}

// ============================================================
//  setupBottomPanel — 底部面板标题栏切换按钮
// ============================================================

void MainWindow::setupBottomPanel()
{
    QWidget* titleBar = new QWidget();

    m_logBtn = new QToolButton(titleBar);
    m_logBtn->setText(tr("日志"));
    m_logBtn->setCheckable(true);
    m_logBtn->setChecked(true);

    auto *grp = new QButtonGroup(titleBar);
    grp->setExclusive(true);
    grp->addButton(m_logBtn, 0);

    m_logDock->setTitleBarWidget(titleBar);

    connect(m_logBtn, &QToolButton::clicked, this, &MainWindow::onLogBtnClicked);

    Q_UNUSED(titleBar);
}

// ============================================================
//  setupMenuConnections — 菜单/工具栏信号连接
// ============================================================

void MainWindow::setupMenuConnections()
{
    if (!m_mainMenu)
    {
        return;
    }

    if (m_mainMenu->zoomInAction())
    {
        connect(m_mainMenu->zoomInAction(), &QAction::triggered, m_canvas, &CanvasWidget::zoomIn);
    }
    if (m_mainMenu->zoomOutAction())
    {
        connect(m_mainMenu->zoomOutAction(), &QAction::triggered, m_canvas, &CanvasWidget::zoomOut);
    }
    if (m_mainMenu->resetViewAction())
    {
        connect(m_mainMenu->resetViewAction(), &QAction::triggered, m_canvas, &CanvasWidget::resetView);
    }

    if (m_mainMenu->toggleLogAction())
    {
        connect(m_mainMenu->toggleLogAction(), &QAction::toggled, this, &MainWindow::onToggleLogAction);
    }

    if (m_mainMenu->minimizeAction())
    {
        connect(m_mainMenu->minimizeAction(), &QAction::triggered, this, &QWidget::showMinimized);
    }

    if (m_mainMenu->toggleGizmoAction() && m_workspaceCenter && m_workspaceCenter->modelView())
    {
        connect(m_mainMenu->toggleGizmoAction(), &QAction::toggled,
                m_workspaceCenter->modelView(), &CameraSceneWidget::setShowGizmo);
    }

    if (m_mainMenu->manualPointCloudPruneAction())
    {
        connect(m_mainMenu->manualPointCloudPruneAction(), &QAction::triggered,
                this, &MainWindow::onManualPointCloudPrune);
    }

    if (m_workspaceCenter && m_workspaceCenter->modelView())
    {
        auto *modelView = m_workspaceCenter->modelView();
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
    m_projectData = new ProjectData(this);
    m_projectManager = new ProjectManager(m_projectData, this);
    m_projectManager->setObjectName(QStringLiteral("ProjectManager"));

    m_menuWorkflowController = new MenuWorkflowController(this, this);
    m_menuWorkflowController->setProjectManager(m_projectManager);
 
    m_reconController = new ReconstructionWorkflowController(this, this);
    m_reconController->setProjectManager(m_projectManager);
    // 连接特征显示选项更新到CanvasWidget
    connect(m_menuWorkflowController, &MenuWorkflowController::requestApplyFeatureDisplayOptions,
            this, [this](const LayerRenderer::FeatureDisplayOptions &opts)
            {
                if (m_canvas)
                {
                    m_canvas->applyFeatureDisplayOptions(opts);
                }
            });

    if (m_dataTree)
    {
        connect(m_dataTree, &DataTreeWidget::imageActivated,
                this, [this](const QString &p)
                {
                    m_lastSelectedImage = p;
                });
    }
    if (m_referencePanel)
    {
        connect(m_referencePanel, &ReferencePanelWidget::imageActivated,
                this, [this](const QString &p)
                {
                    m_lastSelectedImage = p;
                });
    }
    // 画布切换影像时，通过 DialogSettingStore 持久化活跃影像路径
    if (m_canvas)
    {
        connect(m_canvas, &CanvasWidget::activeImageChanged, this, [this](const QString &path)
        {
            saveUiSetting(QJsonObject{{QStringLiteral("active_image_path"), path}});
        });
    }

    if (m_mainMenu)
    {
        if (m_mainMenu->toggleLogAction())
        {
            connect(m_mainMenu->toggleLogAction(), &QAction::toggled, this, &MainWindow::onLogVisiblePersist);
        }
    }
    if (m_log)
    {
        connect(m_log, &LogPanel::displayLevelChanged, this, &MainWindow::onLogDisplayLevelChanged);
    }

    if (m_config)
    {
        m_projectManager->setFileDialogStateManager(m_config->fileDialogs());
    }

    if (m_mainMenu)
    {
        if (m_mainMenu->newAction())
        {
            connect(m_mainMenu->newAction(), &QAction::triggered, m_projectManager, &ProjectManager::createNewProject);
        }
        if (m_mainMenu->openAction())
        {
            connect(m_mainMenu->openAction(), &QAction::triggered, m_projectManager, &ProjectManager::openProject);
        }
        if (m_mainMenu->addPhotoAction())
        {
            connect(m_mainMenu->addPhotoAction(), &QAction::triggered, m_projectManager, &ProjectManager::addPhoto);
        }
        if (m_mainMenu->addFolderAction())
        {
            connect(m_mainMenu->addFolderAction(), &QAction::triggered, m_projectManager, &ProjectManager::addFolder);
        }
        if (m_mainMenu->saveAction())
        {
            connect(m_mainMenu->saveAction(), &QAction::triggered, m_projectManager, &ProjectManager::saveProject);
        }
        if (m_mainMenu->exportMatchedPairsAction())
        {
            connect(m_mainMenu->exportMatchedPairsAction(), &QAction::triggered,
                    this, &MainWindow::onExportMatchedPairs);
        }

        if (m_menuWorkflowController)
        {
            if (m_mainMenu->detectFeaturesAction())
            {
                connect(m_mainMenu->detectFeaturesAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openFeatureExtractionDialog);
            }
            if (m_mainMenu->featureVisualizationAction())
            {
                connect(m_mainMenu->featureVisualizationAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openSuperPointVisualizationDialog);
            }
            if (m_mainMenu->aerialTriangulationAction())
            {
                connect(m_mainMenu->aerialTriangulationAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openAerialTriangulationDialog);
            }
            if (m_mainMenu->createPointCloudAction())
            {
                connect(m_mainMenu->createPointCloudAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openCreatePointCloudDialog);
            }
            if (m_mainMenu->generateModelAction())
            {
                connect(m_mainMenu->generateModelAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::startGenerateModelWorkflow);
            }
            if (m_mainMenu->overlapAnalysisAction())
            {
                connect(m_mainMenu->overlapAnalysisAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openOverlapAnalysisDialog);
            }
            if (m_mainMenu->createDEMAction())
            {
                connect(m_mainMenu->createDEMAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openCreateDemDialog);
            }
            if (m_mainMenu->generateOrthoAction())
            {
                connect(m_mainMenu->generateOrthoAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openMapProjectDialog);
            }
            if (m_mainMenu->viewWorkflowReportAction())
            {
                connect(m_mainMenu->viewWorkflowReportAction(), &QAction::triggered,
                        m_menuWorkflowController, &MenuWorkflowController::openWorkflowReportDialog);
            }
        }

        // ── 重建菜单 → ReconstructionWorkflowController ──
        if (m_reconController)
        {
            auto connectRecon = [&](QAction *act, void (ReconstructionWorkflowController::*slot)())
            {
                if (act)
                {
                    connect(act, &QAction::triggered, m_reconController, slot);
                }
            };
            // 稀疏重建
            connectRecon(m_mainMenu->buildObsNetworkAction(),   &ReconstructionWorkflowController::openObservationNetworkDialog);
            connectRecon(m_mainMenu->initCameraPoseAction(),    &ReconstructionWorkflowController::openInitCameraPoseDialog);
            connectRecon(m_mainMenu->triangulateAction(),       &ReconstructionWorkflowController::openTriangulationDialog);
            connectRecon(m_mainMenu->reconBundleAdjustAction(), &ReconstructionWorkflowController::openReconBundleAdjustDialog);
            connectRecon(m_mainMenu->sparseCloudPostProcessAction(), &ReconstructionWorkflowController::openSparseCloudPostProcessDialog);
            // 密集重建
            connectRecon(m_mainMenu->denseMatchAction(),       &ReconstructionWorkflowController::openDenseMatchDialog);
            connectRecon(m_mainMenu->depthMapEstimateAction(),  &ReconstructionWorkflowController::openDepthMapEstimateDialog);
            connectRecon(m_mainMenu->fuseDepthMapsAction(),     &ReconstructionWorkflowController::openDepthFusionDialog);
            connectRecon(m_mainMenu->refineDenseCloudAction(),  &ReconstructionWorkflowController::openDenseCloudRefineDialog);
            // 模型生成
            connectRecon(m_mainMenu->meshReconstructAction(),   &ReconstructionWorkflowController::openMeshReconstructionDialog);
            connectRecon(m_mainMenu->textureMappingAction(),    &ReconstructionWorkflowController::openTextureMappingDialog);
            connectRecon(m_mainMenu->exportModelAction(),       &ReconstructionWorkflowController::openModelExportDialog);
        }

        // 连接匹配相关菜单动作到对话框（如果 ProjectManager 可用则传入以便填充数据）
        if (m_mainMenu->matchFeaturesAction())
        {
            connect(m_mainMenu->matchFeaturesAction(), &QAction::triggered, this, [this]()
            {
                if (!m_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开匹配配置：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new FeatureMatchingDialog(this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->setProjectImages(m_projectManager->getAllImages());

                // 从当前项目影像收集可用的特征后缀并设置到下拉框
                if (m_canvas)
                {
                    const QStringList projectSuffixes = m_canvas->availableFeatureSuffixes();
                    if (!projectSuffixes.isEmpty())
                    {
                        dlg->setAvailableFeatureSuffixes(projectSuffixes);
                    }
                }

                // 懒初始化 SuperGlue 记忆化设置管理器
                if (!m_sgSetting)
                {
                    m_sgSetting = new DialogSettingStore(DialogSettingKeys::SuperGlue, this);
                }
                m_sgSetting->setProjectPath(m_projectManager->currentProjectPath());
                const QJsonObject saved = m_sgSetting->load();
                if (!saved.isEmpty())
                {
                    dlg->applySettings(saved);
                }

                // 默认输出目录（仅当用户未保存过 output_dir 时才设置）
                const QString assetsDir = ProjectIO::projectAssetsDir(m_projectManager->currentProjectPath());
                if (!assetsDir.isEmpty() && saved.value("output_dir").toString().isEmpty())
                {
                    QJsonObject defaultOutput = saved;  // 基于已恢复的设置，只覆盖 output_dir
                    defaultOutput.insert("output_dir", QDir(assetsDir).filePath(QStringLiteral("matches")));
                    dlg->applySettings(defaultOutput);
                }

                // 实时保存参数到项目配置（通过 DialogSettingStore）
                connect(dlg, &FeatureMatchingDialog::settingsChanged, this, [this](const QJsonObject &s)
                {
                    if (m_sgSetting)
                    {
                        m_sgSetting->save(s);
                    }
                });

                // 连接运行请求信号
                connect(dlg, &FeatureMatchingDialog::runRequested, this,
                    [this](const QJsonObject &config, const QStringList &imagePairs)
                {
                    if (!m_projectManager)
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

                    // 在状态栏展示 SuperGlue 匹配进度
                    auto cancelFlag    = std::make_shared<std::atomic<bool>>(false);
                    auto progressCount = std::make_shared<std::atomic<int>>(0);
                    const int total    = imagePairs.size();

                    showSgProgress(total);

                    // 取消按钒临时连接
                    auto cancelConn = connect(m_sgCancelBtn, &QToolButton::clicked,
                                              this, [cancelFlag]()
                                              {
                                                  cancelFlag->store(true);
                                              });

                    // 定时刷新进度（100ms 轮询）
                    auto *timer = new QTimer(this);
                    timer->setInterval(100);
                    connect(timer, &QTimer::timeout, this,
                            [this, progressCount, total]()
                    {
                        Q_UNUSED(total);
                        updateSgProgress(progressCount->load());
                    });
                    timer->start();

                    auto *watcher = new QFutureWatcher<void>(this);
                    connect(watcher, &QFutureWatcher<void>::finished, this,
                            [this, cancelFlag, timer, watcher, cancelConn]()
                        {
                        timer->stop();
                        timer->deleteLater();
                        QObject::disconnect(cancelConn);
                        hideSgProgress(!cancelFlag->load());
                        watcher->deleteLater();
                    });

                    watcher->setFuture(QtConcurrent::run(
                        [config, imagePairs, pm = m_projectManager, cancelFlag, progressCount]()
                        {
                            FeatureMatchRunner::run(config, imagePairs, pm, *cancelFlag, *progressCount);
                        }));
                });

                // 连接"查看匹配"信号：在对话框中直接打开匹配查看器
                connect(dlg, &FeatureMatchingDialog::viewMatchesRequested, this, [this]()
                {
                    if (!m_projectManager)
                    {
                        return;
                    }

                    auto *matchDlg = new MatchPairSelectorDialog(m_projectManager, this);
                    matchDlg->setAttribute(Qt::WA_DeleteOnClose);
                    matchDlg->show();
                });

                dlg->exec();
            });
        }

        if (m_mainMenu->viewMatchesAction())
        {
            connect(m_mainMenu->viewMatchesAction(), &QAction::triggered, this, [this]()
            {
                if (!m_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开匹配查看：ProjectManager 未初始化"));
                    return;
                }
                // 打开匹配对选择对话框以便用户选择要查看的匹配对
                auto *dlg = new MatchPairSelectorDialog(m_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
            });
        }

        if (m_mainMenu->intersectionCheckAction())
        {
            connect(m_mainMenu->intersectionCheckAction(), &QAction::triggered, this, [this]()
            {
                if (!m_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开前方交汇检测：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new ForwardIntersectionCheckDialog(m_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
            });
        }

        if (m_mainMenu->intersectionViewResultsAction())
        {
            connect(m_mainMenu->intersectionViewResultsAction(), &QAction::triggered, this, [this]()
            {
                if (!m_projectManager)
                {
                    LOG_ERROR(QStringLiteral("无法打开前方交汇结果：ProjectManager 未初始化"));
                    return;
                }
                auto *dlg = new ForwardIntersectionResultsDialog(m_projectManager, this);
                dlg->setAttribute(Qt::WA_DeleteOnClose);
                dlg->exec();
            });
        }

        connect(m_projectManager, &ProjectManager::projectCreated, this, &MainWindow::onProjectOpened);
        connect(m_projectManager, &ProjectManager::projectOpened,  this, &MainWindow::onProjectOpened);

        // 当特征提取完成后，切换 Canvas 到对应后缀并刷新显示
        connect(m_projectManager, &ProjectManager::ipfindResultAppended, this,
            [this](const QString &imagePath, const QString &suffix)
        {
            if (m_canvas)
            {
                if (!suffix.isEmpty())
                    m_canvas->setActiveFeatureSuffix(suffix);
                m_canvas->immediateReloadInterestPoints(imagePath);
            }
        });

        if (m_config)
        {
            m_mainMenu->setRecentProjects(m_config->recentProjects()->recentProjects());
        }

        connect(m_mainMenu, &MainMenu::recentProjectSelected, this, [this](const QString &p)
        {
            if (!p.isEmpty() && m_projectManager)
            {
                m_projectManager->openProjectFromPath(p);
            }
        });
        connect(m_mainMenu, &MainMenu::clearRecentRequested, this, &MainWindow::onClearRecentRequested);
    }

    
    connect(m_projectManager, &ProjectManager::saveStarted,    this, &MainWindow::onSaveStarted);
    connect(m_projectManager, &ProjectManager::saveFinished,   this, &MainWindow::onSaveFinished);
    connect(m_projectManager, &ProjectManager::metadataDirtyChanged, this, &MainWindow::onMetadataDirtyChanged);

    connect(m_projectManager, &ProjectManager::projectOpened, this, [this](const QString &)
    {
        if (m_dataTree)
        {
            m_dataTree->loadFromJson(m_projectManager->currentMeta());
        }
        if (m_referencePanel)
        {
            m_referencePanel->loadFromJson(m_projectManager->currentMeta());
        }
    });
    connect(m_projectManager, &ProjectManager::projectMetadataUpdated, this, [this](const QString &)
    {
        if (m_dataTree)
        {
            m_dataTree->loadFromJson(m_projectManager->currentMeta());
        }
        if (m_referencePanel)
        {
            m_referencePanel->loadFromJson(m_projectManager->currentMeta());
        }
    });
    connect(m_projectManager, &ProjectManager::projectMetadataChanged, m_dataTree, &DataTreeWidget::loadFromJson);
    connect(m_projectManager, &ProjectManager::projectMetadataChanged, m_referencePanel, &ReferencePanelWidget::loadFromJson);

    connect(m_dataTree, &DataTreeWidget::removeRequested,  m_projectManager, &ProjectManager::removeResources);
    connect(m_dataTree, &DataTreeWidget::deleteDataRequested, m_projectManager, &ProjectManager::deleteGeneratedData);
    connect(m_dataTree, &DataTreeWidget::sideOpenRequested, this, [this](const QString &section, const QString &path)
    {
        Q_UNUSED(section);
        if (!m_workspaceCenter || path.trimmed().isEmpty())
        {
            return;
        }
        if (m_lastSelectedImage.trimmed().isEmpty())
        {
            QMessageBox::information(this,
                                     QStringLiteral("侧边打开"),
                                     QStringLiteral("请先在中间打开一张二维影像，再选择另一张在侧边打开。"));
            return;
        }
        if (QDir::cleanPath(m_lastSelectedImage) == QDir::cleanPath(path))
        {
            QMessageBox::information(this,
                                     QStringLiteral("侧边打开"),
                                     QStringLiteral("当前主影像与侧边影像相同，请选择另一张影像。"));
            return;
        }
        m_workspaceCenter->showSideBySideImages(m_lastSelectedImage, path);
    });
    connect(m_dataTree, &DataTreeWidget::packRequested,    m_projectManager, &ProjectManager::packResource);
    connect(m_dataTree, &DataTreeWidget::openRequested, this, [](const QString &p)
    {
        QDesktopServices::openUrl(QUrl::fromLocalFile(p));
    });
    connect(m_dataTree, &DataTreeWidget::revealRequested, this, [](const QString &p)
    {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(p).absolutePath()));
    });
    connect(m_dataTree, &DataTreeWidget::resourceActivated, this, [this](const QString &section, const QString &path)
    {
        if (!m_workspaceCenter || !m_projectManager)
        {
            return;
        }
        if (section == QStringLiteral("照片"))
        {
            return;
        }

        auto normalizedMeta = [this]()
        {
            return xjw::gui::project::projectFilesRootObject(m_projectManager->currentMeta());
        };

        if (section == QStringLiteral("深度图"))
        {
            // 深度图（16-bit PNG）通过 LayerRenderer 的 GDAL 归一化后显示
            m_workspaceCenter->showImageView(path);
            m_lastSelectedImage = path;
            return;
        }
        if (section == QStringLiteral("DEM") || section == QStringLiteral("正射影像"))
        {
            m_workspaceCenter->showImageView(path);
            m_lastSelectedImage = path;
            return;
        }
        if (section == QStringLiteral("3D模型"))
        {
            m_workspaceCenter->showModelFile(path);
            return;
        }
        if (section == QStringLiteral("连接点"))
        {
            // 直接加载稀疏点云 XYZ 文件到 3D 视图
            if (!path.isEmpty() && QFileInfo::exists(path))
            {
                m_workspaceCenter->showPointCloudFile(path);
            }
            return;
        }
        if (section == QStringLiteral("稠密点云"))
        {
            m_workspaceCenter->showPointCloudFile(path);
            return;
        }
        if (section == QStringLiteral("匹配"))
        {
            if (!path.isEmpty() && QFileInfo::exists(path))
            {
                m_workspaceCenter->showPointCloudFile(path);
                return;
            }
            m_workspaceCenter->showModelView();
            return;
        }
        if (section == QStringLiteral("观测网络"))
        {
            if (!m_projectManager)
            {
                return;
            }
            const QJsonObject meta = m_projectManager->currentMeta();
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
            m_workspaceCenter->showObservationNetwork(net, title);
            return;
        }
    });

    connect(m_referencePanel, &ReferencePanelWidget::exactImportRequested,
        m_projectManager, &ProjectManager::importCameraForImage);
    connect(m_referencePanel, &ReferencePanelWidget::batchImportRequested,
        m_projectManager, &ProjectManager::importCamerasByFilenameBatch);
    connect(m_referencePanel, &ReferencePanelWidget::clearCameraRequested,
        this, [this](const QStringList &paths)
        {
            int cleared = 0;
            QString err;
            if (m_projectManager->clearImageCameras(paths, &cleared, &err))
            {
                LOG_INFO(QStringLiteral("已清除 %1 张影像的相机参数").arg(cleared));
            }
            else
            {
                LOG_WARN(QStringLiteral("清除相机参数失败: %1").arg(err));
            }
        });
    connect(m_referencePanel, &ReferencePanelWidget::imageActivated,
        this, [this](const QString &p)
        {
            if (m_workspaceCenter)
            {
                m_workspaceCenter->showImageView(p);
                m_lastSelectedImage = p;
            }
        });

    connect(m_dataTree, &DataTreeWidget::imageActivated, this, [this](const QString &p)
    {
        if (m_workspaceCenter)
        {
            m_workspaceCenter->showImageView(p);
            m_lastSelectedImage = p;
        }
    });

    connect(m_projectManager, &ProjectManager::projectMetadataChanged, this, [this](const QJsonObject &meta)
    {
        if (m_workspaceCenter)
        {
            m_workspaceCenter->setProjectMeta(meta);
        }
    });

    // ── MVS 状态栏进度条 ──────────────────────────────────────────
    m_mvsStatusLabel = new QLabel(this);
    m_mvsStatusLabel->setMinimumWidth(220);
    m_mvsStatusLabel->hide();

    m_mvsProgressBar = new QProgressBar(this);
    m_mvsProgressBar->setRange(0, 100);
    m_mvsProgressBar->setFixedWidth(200);
    m_mvsProgressBar->setFixedHeight(16);
    m_mvsProgressBar->setTextVisible(false);
    m_mvsProgressBar->hide();

    m_mvsCancelBtn = new QToolButton(this);
    m_mvsCancelBtn->setText(tr("取消"));
    m_mvsCancelBtn->setFixedHeight(18);
    m_mvsCancelBtn->hide();
    connect(m_mvsCancelBtn, &QToolButton::clicked,
            m_projectManager, &ProjectManager::cancelMvs);

    statusBar()->addPermanentWidget(m_mvsStatusLabel);
    statusBar()->addPermanentWidget(m_mvsProgressBar);
    statusBar()->addPermanentWidget(m_mvsCancelBtn);

    connect(m_projectManager, &ProjectManager::mvsProgressChanged,
            this, &MainWindow::onMvsProgress);
    connect(m_projectManager, &ProjectManager::mvsProgressFinished,
            this, &MainWindow::onMvsFinished);

    // ── 网格重建状态栏进度条 ──────────────────────────────────────
    m_meshStatusLabel = new QLabel(this);
    m_meshStatusLabel->setMinimumWidth(220);
    m_meshStatusLabel->hide();

    m_meshProgressBar = new QProgressBar(this);
    m_meshProgressBar->setRange(0, 100);
    m_meshProgressBar->setFixedWidth(200);
    m_meshProgressBar->setFixedHeight(16);
    m_meshProgressBar->setTextVisible(false);
    m_meshProgressBar->hide();

    statusBar()->addPermanentWidget(m_meshStatusLabel);
    statusBar()->addPermanentWidget(m_meshProgressBar);

    connect(m_projectManager, &ProjectManager::meshProgressChanged,
            this, &MainWindow::onMeshProgress);
    connect(m_projectManager, &ProjectManager::meshProgressFinished,
            this, &MainWindow::onMeshFinished);

    // ── 空三（AT）状态栏进度条 ──────────────────────────────────
    m_atStatusLabel = new QLabel(this);
    m_atStatusLabel->setMinimumWidth(220);
    m_atStatusLabel->hide();

    m_atProgressBar = new QProgressBar(this);
    m_atProgressBar->setRange(0, 100);
    m_atProgressBar->setFixedWidth(200);
    m_atProgressBar->setFixedHeight(16);
    m_atProgressBar->setTextVisible(false);
    m_atProgressBar->hide();

    m_atCancelBtn = new QToolButton(this);
    m_atCancelBtn->setText(tr("取消"));
    m_atCancelBtn->setFixedHeight(18);
    m_atCancelBtn->hide();
    connect(m_atCancelBtn, &QToolButton::clicked,
            m_projectManager, &ProjectManager::cancelAt);

    statusBar()->addPermanentWidget(m_atStatusLabel);
    statusBar()->addPermanentWidget(m_atProgressBar);
    statusBar()->addPermanentWidget(m_atCancelBtn);

    connect(m_projectManager, &ProjectManager::atProgressChanged,
            this, &MainWindow::onAtProgress);
    connect(m_projectManager, &ProjectManager::atProgressFinished,
            this, &MainWindow::onAtFinished);

    // ── SuperGlue 连接点匹配状态栏进度条 ────────────────────────────
    m_sgStatusLabel = new QLabel(this);
    m_sgStatusLabel->setMinimumWidth(180);
    m_sgStatusLabel->hide();

    m_sgProgressBar = new QProgressBar(this);
    m_sgProgressBar->setRange(0, 100);
    m_sgProgressBar->setFixedWidth(200);
    m_sgProgressBar->setFixedHeight(16);
    m_sgProgressBar->setTextVisible(false);
    m_sgProgressBar->hide();

    m_sgCancelBtn = new QToolButton(this);
    m_sgCancelBtn->setText(tr("取消"));
    m_sgCancelBtn->setFixedHeight(18);
    m_sgCancelBtn->hide();

    statusBar()->addPermanentWidget(m_sgStatusLabel);
    statusBar()->addPermanentWidget(m_sgProgressBar);
    statusBar()->addPermanentWidget(m_sgCancelBtn);

    // ── SuperPoint 特征点提取状态栏进度条 ──────────────────────────
    m_spStatusLabel = new QLabel(this);
    m_spStatusLabel->setMinimumWidth(180);
    m_spStatusLabel->hide();

    m_spProgressBar = new QProgressBar(this);
    m_spProgressBar->setRange(0, 100);
    m_spProgressBar->setFixedWidth(200);
    m_spProgressBar->setFixedHeight(16);
    m_spProgressBar->setTextVisible(false);
    m_spProgressBar->hide();

    m_spCancelBtn = new QToolButton(this);
    m_spCancelBtn->setText(tr("取消"));
    m_spCancelBtn->setFixedHeight(18);
    m_spCancelBtn->hide();
    connect(m_spCancelBtn, &QToolButton::clicked, this, &MainWindow::spCancelRequested);

    statusBar()->addPermanentWidget(m_spStatusLabel);
    statusBar()->addPermanentWidget(m_spProgressBar);
    statusBar()->addPermanentWidget(m_spCancelBtn);

    // ── 密集匹配进度条 ─────────────────────────────────────────
    m_dmProgressBar = new QProgressBar(this);
    m_dmProgressBar->setRange(0, 100);
    m_dmProgressBar->setFixedWidth(200);
    m_dmProgressBar->setFixedHeight(16);
    m_dmProgressBar->setTextVisible(false);
    m_dmProgressBar->hide();
    statusBar()->addPermanentWidget(m_dmProgressBar);

    // ── 观测网络构建状态栏进度条 ─────────────────────────────────────
    m_obsNetStatusLabel = new QLabel(this);
    m_obsNetStatusLabel->setMinimumWidth(200);
    m_obsNetStatusLabel->hide();

    m_obsNetProgressBar = new QProgressBar(this);
    m_obsNetProgressBar->setRange(0, 100);
    m_obsNetProgressBar->setFixedWidth(200);
    m_obsNetProgressBar->setFixedHeight(16);
    m_obsNetProgressBar->setTextVisible(false);
    m_obsNetProgressBar->hide();

    statusBar()->addPermanentWidget(m_obsNetStatusLabel);
    statusBar()->addPermanentWidget(m_obsNetProgressBar);

    connect(m_projectManager, &ProjectManager::obsNetProgressChanged,
            this, &MainWindow::onObsNetProgress);
    connect(m_projectManager, &ProjectManager::obsNetProgressFinished,
            this, &MainWindow::onObsNetFinished);
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

    if (!m_projectManager)
    {
        if (errorMessage)
        {
            *errorMessage = tr("项目管理器未初始化");
        }
        return false;
    }

    const QString plascanPath = m_projectManager->currentProjectPath();
    if (plascanPath.isEmpty())
    {
        if (errorMessage)
        {
            *errorMessage = tr("请先打开项目");
        }
        return false;
    }

    const QVector<QPair<QString, QString>> matchedPairs =
        xjw::gui::project::collectMatchedImageNamePairs(plascanPath, m_projectManager->currentMeta());
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
    if (!m_workspaceCenter || !m_workspaceCenter->modelView())
    {
        return;
    }

    auto *modelView = m_workspaceCenter->modelView();
    const bool enable = !modelView->isManualPruneModeEnabled();
    QString errorMessage;
    if (!modelView->setManualPruneModeEnabled(enable, &errorMessage))
    {
        QMessageBox::warning(this, tr("手动点云剔除"), errorMessage);
        return;
    }

    if (enable)
    {
        m_workspaceCenter->showModelView();
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
    if (!m_mvsProgressBar)
    {
        return;
    }
    m_mvsStatusLabel->setText(stage);
    m_mvsProgressBar->setValue(percent);
    m_mvsStatusLabel->show();
    m_mvsProgressBar->show();
    m_mvsCancelBtn->show();
    statusBar()->showMessage(QString());   // 清空普通消息，让 permanent widget 露出
}

void MainWindow::onMvsFinished(bool success)
{
    if (!m_mvsProgressBar)
    {
        return;
    }
    m_mvsStatusLabel->hide();
    m_mvsProgressBar->hide();
    m_mvsCancelBtn->hide();
    statusBar()->showMessage(
        success ? tr("稠密重建完成") : tr("稠密重建已取消或失败"), 4000);
}

// ============================================================
//  网格重建进度状态栏槽
// ============================================================

void MainWindow::onMeshProgress(const QString &stage, int percent)
{
    if (!m_meshProgressBar)
    {
        return;
    }
    m_meshStatusLabel->setText(stage);
    m_meshProgressBar->setValue(percent);
    m_meshStatusLabel->show();
    m_meshProgressBar->show();
    statusBar()->showMessage(QString());
}

void MainWindow::onMeshFinished(bool success)
{
    if (!m_meshProgressBar)
    {
        return;
    }
    m_meshStatusLabel->hide();
    m_meshProgressBar->hide();
    statusBar()->showMessage(
        success ? tr("网格重建完成") : tr("网格重建失败"), 4000);
}

// ============================================================
//  AT（空三）进度状态栏槽
// ============================================================

void MainWindow::onAtProgress(const QString &stage, int percent)
{
    if (!m_atProgressBar)
    {
        return;
    }
    // 显示阶段信息和百分比，例如 "正在匹配特征点... 50%"
    if (percent > 0 && percent < 100)
    {
        m_atStatusLabel->setText(QStringLiteral("%1 %2%").arg(stage).arg(percent));
    }
    else
    {
        m_atStatusLabel->setText(stage);
    }
    m_atProgressBar->setValue(percent);
    m_atStatusLabel->show();
    m_atProgressBar->show();
    if (m_atCancelBtn) m_atCancelBtn->show();
    statusBar()->showMessage(QString());   // 清空普通消息，让 permanent widget 露出
}

void MainWindow::onAtFinished(bool success)
{
    if (!m_atProgressBar)
    {
        return;
    }
    m_atStatusLabel->hide();
    m_atProgressBar->hide();
    if (m_atCancelBtn) m_atCancelBtn->hide();
    statusBar()->showMessage(
        success ? tr("空三（AT）完成") : tr("空三（AT）已取消或失败"), 4000);
}

void MainWindow::onCancelAt()
{
    if (m_projectManager)
    {
        m_projectManager->cancelAt();
    }
}

// ============================================================
//  观测网络构建进度状态栏槽
// ============================================================

void MainWindow::onObsNetProgress(const QString &stage, int percent)
{
    if (!m_obsNetProgressBar)
    {
        return;
    }
    m_obsNetStatusLabel->setText(QStringLiteral("观测网络: %1 %2%").arg(stage).arg(percent));
    m_obsNetProgressBar->setValue(percent);
    m_obsNetStatusLabel->show();
    m_obsNetProgressBar->show();
    statusBar()->showMessage(QString());
}

void MainWindow::onObsNetFinished(bool success)
{
    if (!m_obsNetProgressBar)
    {
        return;
    }
    m_obsNetStatusLabel->hide();
    m_obsNetProgressBar->hide();
    statusBar()->showMessage(
        success ? tr("观测网络构建完成") : tr("观测网络构建失败"), 4000);
}



void MainWindow::showSgProgress(int total)
{
    if (!m_sgProgressBar)
    {
        return;
    }
    m_sgProgressBar->setRange(0, total);
    m_sgProgressBar->setValue(0);
    m_sgStatusLabel->setText(tr("特征匹配 0/%1").arg(total));
    m_sgStatusLabel->show();
    m_sgProgressBar->show();
    m_sgCancelBtn->show();
    statusBar()->showMessage(QString());
}

void MainWindow::updateSgProgress(int done)
{
    if (!m_sgProgressBar)
    {
        return;
    }
    m_sgProgressBar->setValue(done);
    m_sgStatusLabel->setText(tr("特征匹配 %1/%2").arg(done).arg(m_sgProgressBar->maximum()));
}

void MainWindow::hideSgProgress(bool ok)
{
    if (!m_sgProgressBar)
    {
        return;
    }
    m_sgStatusLabel->hide();
    m_sgProgressBar->hide();
    m_sgCancelBtn->hide();
    statusBar()->showMessage(ok ? tr("匹配完成") : tr("匹配已取消"), 4000);
}

void MainWindow::showDmProgress(int total)
{
    if (!m_dmProgressBar) return;
    m_dmProgressBar->setRange(0, total);
    m_dmProgressBar->setValue(0);
    m_dmProgressBar->show();
}

void MainWindow::updateDmProgress(int done)
{
    if (!m_dmProgressBar) return;
    m_dmProgressBar->setValue(done);
}

void MainWindow::hideDmProgress(bool ok)
{
    if (!m_dmProgressBar) return;
    m_dmProgressBar->hide();
    statusBar()->showMessage(ok ? tr("密集匹配完成") : tr("密集匹配已取消"), 4000);
}

// ============================================================
//  SuperPoint 状态栏进度条 slots
// ============================================================

void MainWindow::showSpProgress(int total)
{
    if (!m_spProgressBar)
    {
        return;
    }
    m_spProgressBar->setRange(0, total);
    m_spProgressBar->setValue(0);
    m_spStatusLabel->setText(tr("特征提取 0/%1").arg(total));
    m_spStatusLabel->show();
    m_spProgressBar->show();
    m_spCancelBtn->show();
    statusBar()->showMessage(QString());
}

void MainWindow::updateSpProgress(int done)
{
    if (!m_spProgressBar)
    {
        return;
    }
    m_spProgressBar->setValue(done);
    m_spStatusLabel->setText(tr("特征提取 %1/%2").arg(done).arg(m_spProgressBar->maximum()));
}

void MainWindow::hideSpProgress(bool ok)
{
    if (!m_spProgressBar)
    {
        return;
    }
    m_spStatusLabel->hide();
    m_spProgressBar->hide();
    m_spCancelBtn->hide();
    statusBar()->showMessage(ok ? tr("特征提取完成") : tr("特征提取已取消"), 4000);
}

// ============================================================
//  辅助方法
// ============================================================

void MainWindow::saveUiSetting(const QJsonObject &partial)
{
    if (m_uiSetting)
    {
        m_uiSetting->merge(partial);
    }
}

QString MainWindow::currentBottomPanelKey() const
{
    return QStringLiteral("log");
}

void MainWindow::switchToLogPanel()
{
    if (m_logDock)
    {
        m_logDock->setWidget(m_log);
    }
    if (m_log)
    {
        m_log->loadFromLogFile();
    }
    if (m_logBtn)
    {
        m_logBtn->setChecked(true);
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
    if (m_mainMenu && m_mainMenu->toggleLogAction())
    {
        s[QStringLiteral("log_visible")] = m_mainMenu->toggleLogAction()->isChecked();
    }
    saveUiSetting(s);
}

// Interest-point panel removed: onIpBtnClicked is a no-op now.

// ============================================================
//  菜单动作响应
// ============================================================

void MainWindow::onToggleLogAction(bool on)
{
    if (m_logBtn)
    {
        m_logBtn->setVisible(on);
    }
    if (on)
    {
        m_logDock->setWidget(m_log);
        m_logDock->setVisible(true);
        if (m_log)
        {
            m_log->loadFromLogFile();
        }
        if (m_logBtn)
        {
            m_logBtn->setChecked(true);
        }
    }
    else
    {
        // feature panel removed; nothing to switch to
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
    if (!m_saveProgressDialog)
    {
        m_saveProgressDialog = new QProgressDialog(tr("正在保存项目..."), QString(), 0, 0, this);
        m_saveProgressDialog->setWindowModality(Qt::ApplicationModal);
        m_saveProgressDialog->setCancelButton(nullptr);
        m_saveProgressDialog->setMinimumDuration(0);
    }
    m_saveProgressDialog->show();
}

void MainWindow::onSaveFinished(bool ok)
{
    if (m_saveProgressDialog)
    {
        m_saveProgressDialog->hide();
        m_saveProgressDialog->deleteLater();
        m_saveProgressDialog = nullptr;
    }
    statusBar()->showMessage(ok ? tr("保存完成") : tr("保存失败"), ok ? 3000 : 5000);
}

void MainWindow::onMetadataDirtyChanged(bool dirty)
{
    QString projPath = m_projectManager->currentProjectPath();
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
    if (!m_config)
    {
        return;
    }
    auto btn = QMessageBox::question(this, tr("清空最近打开"),
        tr("确定要清空最近打开的项目列表吗？"),
        QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    if (btn == QMessageBox::Yes)
    {
        m_config->recentProjects()->clearRecentProjects();
        if (m_mainMenu)
        {
            m_mainMenu->setRecentProjects(m_config->recentProjects()->recentProjects());
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
    QFileInfo fi(plascanPath);
    QString name = fi.baseName();
    if (name.isEmpty())
    {
        name = fi.fileName();
    }
    setWindowTitle(QStringLiteral("PlaScan - %1").arg(name));
    statusBar()->showMessage(tr("已打开项目：%1").arg(plascanPath), 4000);

    if (m_canvas)
    {
        m_canvas->setProperty("currentProjectPath", plascanPath);
    }
    if (m_workspaceCenter)
    {
        m_workspaceCenter->showModelView();
    }
    if (m_dataTree)
    {
        m_dataTree->loadFromArchive(plascanPath);
    }
    if (m_projectManager && m_dataTree)
        m_dataTree->loadFromJson(m_projectManager->currentMeta());
    if (m_projectManager && m_workspaceCenter)
        m_workspaceCenter->setProjectMeta(m_projectManager->currentMeta());

    if (m_config && m_mainMenu)
    {
        m_config->recentProjects()->addRecentProject(plascanPath);
        m_mainMenu->setRecentProjects(m_config->recentProjects()->recentProjects());
    }

    if (!m_projectManager)
    {
        return;
    }

    // 初始化/更新项目级 UI 设置路径
    if (!m_uiSetting)
    {
        m_uiSetting = new DialogSettingStore(DialogSettingKeys::MainWindowUi, this);
    }
    m_uiSetting->setProjectPath(plascanPath);
    if (!m_sgSetting)
    {
        m_sgSetting = new DialogSettingStore(DialogSettingKeys::SuperGlue, this);
    }
    m_sgSetting->setProjectPath(plascanPath);

    QJsonObject ui = m_uiSetting->load();
    // 向后兼容：若新文件中无数据，尝试从旧 project_config.json 读取
    if (ui.isEmpty())
    {
        ui = m_projectManager->loadUiSettings();
    }
    applyUiSettings(ui);

    // feature-info panel removed: do not reload interest points into a removed panel here
}

// ============================================================
//  applyUiSettings — 恢复项目范围内的 UI 设置
// ============================================================

void MainWindow::applyUiSettings(const QJsonObject &ui)
{
    if (ui.isEmpty())
    {
        return;
    }

    // feature-info panel removed: no ip button handling
    if (m_logBtn && ui.contains(QStringLiteral("log_visible")))
    {
        bool lvis = ui.value(QStringLiteral("log_visible")).toBool();
        m_logBtn->setChecked(lvis);
        m_logBtn->setVisible(lvis);
    }

    if (ui.contains(QStringLiteral("log_display_level")) && m_log)
    {
        int lvl = ui.value(QStringLiteral("log_display_level")).toInt(static_cast<int>(Logger::Info));
        m_log->setDisplayLevel(static_cast<Logger::Level>(lvl));
    }

    bool panelHandledByLog = false;
    if (ui.contains(QStringLiteral("log_visible")) && m_mainMenu && m_mainMenu->toggleLogAction())
    {
        bool on = ui.value(QStringLiteral("log_visible")).toBool();
        m_mainMenu->toggleLogAction()->blockSignals(true);
        m_mainMenu->toggleLogAction()->setChecked(on);
        m_mainMenu->toggleLogAction()->blockSignals(false);
        if (m_logBtn)
        {
            m_logBtn->setVisible(on);
        }
        if (on)
        {
            switchToLogPanel();
            panelHandledByLog = true;
        }
    }

    // feature-info visibility handling removed

    if (ui.contains(QStringLiteral("active_image_path")) && m_canvas)
    {
        QString imagePath = ui.value(QStringLiteral("active_image_path")).toString();
        if (!imagePath.isEmpty() && QFileInfo::exists(imagePath))
        {
            QTimer::singleShot(100, this, [this, imagePath]()
            {
                if (m_workspaceCenter)
                {
                    m_workspaceCenter->showImageView(imagePath);
                }
            });
        }
    }
    // Apply saved feature display options (visualization) if present
    if (m_menuWorkflowController)
    {
        m_menuWorkflowController->applySavedFeatureDisplayOptions(ui);
    }
}

// ============================================================
//  closeEvent — 退出时保存/提示
// ============================================================

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_config)
    {
        m_config->windowState()->save(this);
    }

    if (m_projectManager && m_projectManager->isDirty())
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
            m_projectManager->saveProject();

            // 保存完成后接受事件，自动退出
            event->accept();
            return;
        }
        else if (btn == QMessageBox::Discard)
        {
            m_projectManager->discardTemporaryMeta();
        }
    }

    event->accept();
    QMainWindow::closeEvent(event);
}
