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
#include "ProjectTaskStatusController.h"
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
void MainWindow::setupProjectManager()
{
    _projectData = new ProjectData(this);
    _projectManager = new ProjectManager(_projectData, this);
    _projectLifecyclePresenter = new ProjectLifecyclePresenter(
        _projectManager, this, statusBar(), this);
    connect(_projectLifecyclePresenter,
            &ProjectLifecyclePresenter::closeAfterSaveRequested,
            this,
            &QWidget::close,
            Qt::QueuedConnection);
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
            connectRecon(_mainMenu->createPointCloudAction(),
                         &ReconstructionWorkflowController::openCreatePointCloudDialog);
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
                    options.reuseExistingMatches = true;
                    if (options.maskApplyMode == QStringLiteral("keypoints"))
                    {
                        // 关键点蒙版改变的是特征文件本身，不能复用旧的未过滤特征。
                        options.reuseExistingMatches = false;
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

        // 匹配分片提交后刷新当前影像的匹配观测。SIFT 描述子不再落盘，
        // 因此这里不再切换旧的特征文件后缀。
        connect(_projectManager, &ProjectManager::imageMatchResultAppended, this,
            [this](const QString &imagePath)
        {
            if (_canvas)
            {
                const bool isCurrentImage =
                    QDir::cleanPath(imagePath) == QDir::cleanPath(_canvas->currentImagePath());
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

    _taskStatusController = new ProjectTaskStatusController(
        _projectManager, _dashboard, statusBar(), this, this);
    connect(_taskStatusController,
            &ProjectTaskStatusController::tiePointCancelRequested,
            this,
            &MainWindow::sgCancelRequested);
}
