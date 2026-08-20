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
#include <QDialog>
#include <QDialogButtonBox>
#include <QUrl>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
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
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

#include "CanvasWidget.h"
#include "ImageViewRotationSettings.h"
#include "ProjectCameraIO.h"
#include "project/ProjectMatchCatalog.h"
#include "project/ProjectMetadata.h"
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
#include "ProjectTiePointResultService.h"
#include "project/ProjectIO.h"
#include "project/SparseResultQuality.h"
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
#include "CameraSceneWidget.h"
#include "WorkspacePanelController.h"
#include "ProjectUiHydrator.h"
#include "TiePointWorkflowController.h"
#include "ProjectTaskStatusController.h"
#include "ProjectLifecyclePresenter.h"
#include "tie_points/ThinTiePointsDialog.h"
#include "LayerRenderer.h"
#include "ModelDropSupport.h"
#include "MarkerWorkspaceController.h"
#include "MarkerReferencePanel.h"
#include "reference/CameraReferenceController.h"
#include "reference/ProjectCameraReferenceRepository.h"
#include "MarkerFocusMeasurementDialog.h"
#include "DetectMarkersDialog.h"
#include "MarkerDetectionReviewDialog.h"
#include "PrintMarkersDialog.h"

namespace
{
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
    _taskStatusController = new ProjectTaskStatusController(
        _projectManager, _dashboard, statusBar(), this, this);
    connect(_photoStrip,
            &PhotoStripWidget::imageLoadingProgressChanged,
            _taskStatusController,
            &ProjectTaskStatusController::updateImageLoading);
    connect(_photoStrip,
            &PhotoStripWidget::imageLoadingFinished,
            _taskStatusController,
            &ProjectTaskStatusController::finishImageLoading);

    _tiePointWorkflowController = new TiePointWorkflowController(_projectManager, this);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressStarted,
            _taskStatusController,
            &ProjectTaskStatusController::showTiePointProgress);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressUpdated,
            _taskStatusController,
            &ProjectTaskStatusController::updateTiePointProgress);
    connect(_tiePointWorkflowController,
            &TiePointWorkflowController::progressFinished,
            _taskStatusController,
            &ProjectTaskStatusController::finishTiePointProgress);
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
    connect(_taskStatusController,
            &ProjectTaskStatusController::tiePointCancelRequested,
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
    _cameraReferenceRepository =
        new xjw::gui::reference::ProjectCameraReferenceRepository(_projectData, this);
    _cameraReferenceController = new xjw::gui::reference::CameraReferenceController(
        _projectData, _cameraReferenceRepository, this, this);
    _referencePanel->setMarkerController(_markerWorkspaceController);
    _referencePanel->setCameraReferenceRepository(_cameraReferenceRepository);
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
        if (_mainMenu->importPointCloudAction())
        {
            connect(_mainMenu->importPointCloudAction(),
                    &QAction::triggered,
                    _projectManager,
                    &ProjectManager::importPointCloud);
        }
        if (_mainMenu->importModelAction())
        {
            connect(_mainMenu->importModelAction(),
                    &QAction::triggered,
                    _projectManager,
                    &ProjectManager::importModel);
        }
        if (_mainMenu->importReferenceAction())
        {
            connect(_mainMenu->importReferenceAction(), &QAction::triggered, this, [this]()
            {
                if (!_projectData || !_projectData->hasProject())
                {
                    QMessageBox::information(this,
                                             QStringLiteral("导入参考"),
                                             QStringLiteral("请先创建或打开项目"));
                    return;
                }

                QMessageBox reference_type_dialog(
                    QMessageBox::Question,
                    QStringLiteral("导入参考"),
                    QStringLiteral("请选择要导入的参考数据类型。\n"
                                   "相机参考用于位置/姿态，并自动读取同目录的 GNSS_offset.txt；\n"
                                   "标记参考用于控制点和检查点。"),
                    QMessageBox::Cancel,
                    this);
                QPushButton *camera_reference_button = reference_type_dialog.addButton(
                    QStringLiteral("相机参考（Cameras_WGS84.txt）..."), QMessageBox::ActionRole);
                QPushButton *marker_reference_button = reference_type_dialog.addButton(
                    QStringLiteral("标记参考（GCPs_WGS84.txt / CSV）..."), QMessageBox::ActionRole);
                reference_type_dialog.setDefaultButton(camera_reference_button);
                reference_type_dialog.exec();

                if (reference_type_dialog.clickedButton() == camera_reference_button &&
                    _cameraReferenceController)
                {
                    _cameraReferenceController->importMetashapeReference();
                }
                else if (reference_type_dialog.clickedButton() == marker_reference_button &&
                         _projectManager)
                {
                    _projectManager->openSurveyControlDialog();
                }
            });
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
                    options.device = xjw::matchphotos::ComputeDevice::Auto;
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
                if (!_projectManager || !_workspaceCenter || !_workspaceCenter->modelView())
                {
                    QMessageBox::warning(this,
                                         tr("清理连接点"),
                                         tr("三维视图或项目管理器尚未就绪。"));
                    return;
                }

                const QString projectPath = _projectManager->currentProjectPath();
                const auto selection =
                    xjw::gui::project::ProjectTiePointResultService::selectCurrent(
                        _projectManager->currentMeta(), projectPath);
                if (!selection.isValid())
                {
                    QMessageBox::information(this,
                                             tr("清理连接点"),
                                             tr("当前项目中没有可清理的连接点成果。"));
                    return;
                }
                if (!xjw::gui::project::isProductionSparseResult(selection.record))
                {
                    QString reason = xjw::gui::project::sparseResultBlockingReason(selection.record);
                    if (reason.trimmed().isEmpty())
                    {
                        reason = tr("当前成果不是正式 SfM/BA 连接点，不能执行清理。");
                    }
                    QMessageBox::warning(this, tr("清理连接点"), reason);
                    return;
                }

                const auto resolveProjectPath = [&projectPath](const QString &storedPath)
                {
                    const QString path = storedPath.trimmed();
                    if (path.isEmpty())
                    {
                        return QString();
                    }
                    return QFileInfo(path).isAbsolute()
                        ? QDir::cleanPath(path)
                        : xjw::common::project::ProjectIO::resolveProjectResourcePath(
                              projectPath, path);
                };

                const QJsonObject resultFiles =
                    selection.record.value(QStringLiteral("files")).toObject();
                QString sidecarPath = resolveProjectPath(
                    resultFiles.value(QStringLiteral("sparse_cloud_points_json")).toString());
                if (sidecarPath.isEmpty())
                {
                    const QString outputDir = resolveProjectPath(
                        selection.record.value(QStringLiteral("output_dir")).toString());
                    if (!outputDir.isEmpty())
                    {
                        sidecarPath = QDir(outputDir).filePath(
                            QStringLiteral("sparse_cloud_points.json"));
                    }
                }
                if (!QFileInfo(sidecarPath).isFile())
                {
                    QMessageBox::warning(
                        this,
                        tr("清理连接点"),
                        tr("当前连接点缺少逐点质量数据，无法进行阈值预览。\n%1")
                            .arg(sidecarPath));
                    return;
                }

                auto *modelView = _workspaceCenter->modelView();
                modelView->clearTiePointPrunePreview();
                _workspaceCenter->showTiePointCloudFile(selection.sparseCloudPath, sidecarPath);

                CleanTiePointsDialog dlg(this);
                using DialogCriterion = CleanTiePointsDialog::Criterion;
                using QualityCriterion = CameraSceneWidget::TiePointQualityCriterion;

                const auto toQualityCriterion = [](DialogCriterion criterion,
                                                   QualityCriterion *qualityCriterion)
                {
                    if (!qualityCriterion)
                    {
                        return false;
                    }
                    switch (criterion)
                    {
                    case DialogCriterion::ReprojectionError:
                        *qualityCriterion = QualityCriterion::ReprojectionError;
                        return true;
                    case DialogCriterion::ImageCount:
                        *qualityCriterion = QualityCriterion::ImageCount;
                        return true;
                    case DialogCriterion::MinimumTriangulationAngle:
                        *qualityCriterion = QualityCriterion::MinimumTriangulationAngle;
                        return true;
                    case DialogCriterion::None:
                    case DialogCriterion::ReconstructionUncertainty:
                    case DialogCriterion::ProjectionAccuracy:
                        return false;
                    }
                    return false;
                };

                const auto refreshCriterionConfiguration = [&dlg, modelView]()
                {
                    const auto configure = [&dlg, modelView](
                        DialogCriterion dialogCriterion,
                        QualityCriterion qualityCriterion,
                        double defaultLevel,
                        double singleStep,
                        int decimals)
                    {
                        auto configuration = dlg.criterionConfiguration(dialogCriterion);
                        configuration.available =
                            modelView->hasTiePointQualityMetadata(qualityCriterion);
                        configuration.defaultLevel = defaultLevel;
                        configuration.singleStep = singleStep;
                        configuration.decimals = decimals;
                        if (configuration.available)
                        {
                            const auto range = modelView->tiePointQualityRange(qualityCriterion);
                            if (qualityCriterion == QualityCriterion::ReprojectionError)
                            {
                                configuration.minimum = 0.0;
                                configuration.maximum = std::max(
                                    range.maximum, defaultLevel);
                            }
                            else if (qualityCriterion == QualityCriterion::ImageCount)
                            {
                                configuration.minimum = std::min(
                                    range.minimum, defaultLevel);
                                configuration.maximum = std::max(
                                    range.maximum, defaultLevel);
                            }
                            else
                            {
                                configuration.minimum = 0.0;
                                configuration.maximum = std::max(
                                    range.maximum, defaultLevel);
                            }
                            configuration.unavailableReason.clear();
                        }
                        else
                        {
                            configuration.unavailableReason =
                                QObject::tr("正在加载该指标，或当前成果未保存该逐点数据。");
                        }
                        dlg.setCriterionConfiguration(dialogCriterion, configuration);
                    };

                    configure(DialogCriterion::ReprojectionError,
                              QualityCriterion::ReprojectionError,
                              1.0,
                              0.01,
                              3);
                    configure(DialogCriterion::ImageCount,
                              QualityCriterion::ImageCount,
                              3.0,
                              1.0,
                              0);
                    configure(DialogCriterion::MinimumTriangulationAngle,
                              QualityCriterion::MinimumTriangulationAngle,
                              2.0,
                              0.1,
                              2);
                };

                connect(&dlg,
                        &CleanTiePointsDialog::previewRequested,
                        &dlg,
                        [this, modelView, &dlg, toQualityCriterion](
                            DialogCriterion criterion, double level)
                        {
                            QualityCriterion qualityCriterion;
                            if (!toQualityCriterion(criterion, &qualityCriterion))
                            {
                                modelView->clearTiePointPrunePreview();
                                return;
                            }
                            QString errorMessage;
                            if (!modelView->requestTiePointPrunePreview(
                                    CameraSceneWidget::TiePointPrunePreviewQuery{
                                        qualityCriterion, level},
                                    &errorMessage))
                            {
                                dlg.setCandidateCount(-1, -1);
                                if (!errorMessage.trimmed().isEmpty())
                                {
                                    statusBar()->showMessage(errorMessage, 3000);
                                }
                            }
                        });
                connect(&dlg,
                        &CleanTiePointsDialog::previewCleared,
                        &dlg,
                        [modelView]()
                        {
                            modelView->clearTiePointPrunePreview();
                        });
                connect(modelView,
                        &CameraSceneWidget::tiePointPrunePreviewChanged,
                        &dlg,
                        [&dlg, modelView](int candidateCount)
                        {
                            dlg.setCandidateCount(candidateCount,
                                                  modelView->tiePointQualityPointCount());
                        });
                connect(modelView,
                        &CameraSceneWidget::tiePointQualityMetadataReady,
                        &dlg,
                        [&dlg, refreshCriterionConfiguration](bool ready)
                        {
                            refreshCriterionConfiguration();
                            if (!ready)
                            {
                                dlg.setCandidateCount(-1, -1);
                                return;
                            }
                            if (dlg.criterion() == DialogCriterion::None)
                            {
                                if (dlg.criterionConfiguration(
                                        DialogCriterion::ReprojectionError).available)
                                {
                                    dlg.setCriterion(DialogCriterion::ReprojectionError);
                                }
                                else if (dlg.criterionConfiguration(
                                             DialogCriterion::ImageCount).available)
                                {
                                    dlg.setCriterion(DialogCriterion::ImageCount);
                                }
                                else if (dlg.criterionConfiguration(
                                             DialogCriterion::MinimumTriangulationAngle).available)
                                {
                                    dlg.setCriterion(
                                        DialogCriterion::MinimumTriangulationAngle);
                                }
                            }
                        });

                refreshCriterionConfiguration();
                if (modelView->tiePointQualityPointCount() > 0)
                {
                    if (dlg.criterionConfiguration(
                            DialogCriterion::ReprojectionError).available)
                    {
                        dlg.setCriterion(DialogCriterion::ReprojectionError);
                    }
                    else if (dlg.criterionConfiguration(
                                 DialogCriterion::ImageCount).available)
                    {
                        dlg.setCriterion(DialogCriterion::ImageCount);
                    }
                    else if (dlg.criterionConfiguration(
                                 DialogCriterion::MinimumTriangulationAngle).available)
                    {
                        dlg.setCriterion(DialogCriterion::MinimumTriangulationAngle);
                    }
                }

                if (dlg.exec() != QDialog::Accepted)
                {
                    modelView->clearTiePointPrunePreview();
                    return;
                }

                const int candidateCount = dlg.candidateCount();
                const int totalPointCount = dlg.totalPointCount();
                if (!dlg.deleteRequested())
                {
                    statusBar()->showMessage(
                        tr("已在三维视图高亮 %1 / %2 个候选剔除点（%3，阈值 %4）。")
                            .arg(candidateCount)
                            .arg(totalPointCount)
                            .arg(dlg.criterionText())
                            .arg(QString::number(dlg.level(), 'g', 5)),
                        5000);
                    return;
                }

                if (candidateCount <= 0 || totalPointCount <= 0
                    || candidateCount >= totalPointCount)
                {
                    modelView->clearTiePointPrunePreview();
                    QMessageBox::warning(
                        this,
                        tr("清理连接点"),
                        tr("候选点数无效，已取消删除。请调整阈值，保留至少一个连接点。"));
                    return;
                }

                const auto answer = QMessageBox::question(
                    this,
                    tr("删除候选连接点"),
                    tr("将删除 %1 / %2 个连接点，并生成新的稀疏点云成果。\n"
                       "现有深度图、稠密点云、模型、DEM 和 DOM 将失效，需重新生成。\n\n"
                       "是否继续？")
                        .arg(candidateCount)
                        .arg(totalPointCount),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No);
                if (answer != QMessageBox::Yes)
                {
                    statusBar()->showMessage(tr("已取消删除，候选点仍保持高亮。"), 3000);
                    return;
                }

                QJsonObject settings;
                settings[QStringLiteral("sourceKind")] = QStringLiteral("project_result");
                settings[QStringLiteral("sourceAtIndex")] = selection.sourceIndex;
                settings[QStringLiteral("sourceSparseCloudPath")] = selection.sparseCloudPath;
                settings[QStringLiteral("sourceSidecarPath")] = sidecarPath;
                settings[QStringLiteral("filterByReprojError")] = false;
                settings[QStringLiteral("filterByTrackLen")] = false;
                settings[QStringLiteral("filterByTriAngle")] = false;
                settings[QStringLiteral("filterByStatistical")] = false;
                settings[QStringLiteral("filterByDensity")] = false;

                switch (dlg.criterion())
                {
                case DialogCriterion::ReprojectionError:
                    settings[QStringLiteral("filterByReprojError")] = true;
                    settings[QStringLiteral("maxReprojError")] = dlg.level();
                    break;
                case DialogCriterion::ImageCount:
                    settings[QStringLiteral("filterByTrackLen")] = true;
                    settings[QStringLiteral("minTrackLen")] =
                        static_cast<int>(std::lround(dlg.level()));
                    break;
                case DialogCriterion::MinimumTriangulationAngle:
                    settings[QStringLiteral("filterByTriAngle")] = true;
                    settings[QStringLiteral("minTriAngleDeg")] = dlg.level();
                    break;
                case DialogCriterion::None:
                case DialogCriterion::ReconstructionUncertainty:
                case DialogCriterion::ProjectionAccuracy:
                    modelView->clearTiePointPrunePreview();
                    QMessageBox::warning(this,
                                         tr("清理连接点"),
                                         tr("所选标准尚无可执行的剔除算法。"));
                    return;
                }

                modelView->clearTiePointPrunePreview();
                statusBar()->showMessage(
                    tr("正在后台删除 %1 个候选连接点…").arg(candidateCount),
                    5000);
                _projectManager->startSparseCloudOutlierRemovalAsync(settings);
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
            &ProjectManager::tiePointResultReady,
            this,
            [this](const QString &sparseCloudPath, const QString &sidecarPath)
            {
                if (_workspaceCenter && QFileInfo(sparseCloudPath).isFile())
                {
                    _workspaceCenter->showTiePointCloudFile(sparseCloudPath, sidecarPath);
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

    connect(_referencePanel, &ReferencePanelWidget::imageActivated,
        this, [this](const QString &p)
        {
            selectPhoto(p, true);
        });
    connect(_referencePanel, &ReferencePanelWidget::markerActivated,
            this, [this](const QString &markerId)
    {
        openMarkerFocusMeasurement(markerId, _canvas ? _canvas->currentImagePath() : QString());
    });
    connect(_referencePanel, &ReferencePanelWidget::markerPropertiesRequested,
            this, [this](const QString &markerId)
    {
        auto *dialog = new QDialog(this);
        dialog->setAttribute(Qt::WA_DeleteOnClose);
        dialog->setWindowTitle(QStringLiteral("标记属性"));
        dialog->resize(760, 640);
        auto *layout = new QVBoxLayout(dialog);
        auto *panel = new xjw::gui::markers::MarkerReferencePanel(dialog);
        panel->setController(_markerWorkspaceController);
        panel->selectMarker(markerId);
        layout->addWidget(panel, 1);
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, dialog);
        connect(buttons, &QDialogButtonBox::rejected, dialog, &QDialog::reject);
        layout->addWidget(buttons);
        connect(panel,
                &xjw::gui::markers::MarkerReferencePanel::focusMeasurementRequested,
                this,
                [this](const QString &id)
        {
            openMarkerFocusMeasurement(id, _canvas ? _canvas->currentImagePath() : QString());
        });
        dialog->show();
    });
    connect(_projectManager, &ProjectManager::surveyControlChanged, this, [this]()
    {
        QString error;
        if (!_markerWorkspaceController->openProject(&error))
        {
            QMessageBox::warning(this, QStringLiteral("标记参考"), error);
        }
    });
    connect(_referencePanel, &ReferencePanelWidget::importMarkerReferencesRequested,
            this, [this]()
    {
        _projectManager->openSurveyControlDialog();
    });
    connect(_referencePanel,
            &ReferencePanelWidget::importCameraReferencesRequested,
            _cameraReferenceController,
            &xjw::gui::reference::CameraReferenceController::importMetashapeReference);
    connect(_referencePanel,
            &ReferencePanelWidget::exportCameraReferencesRequested,
            _cameraReferenceController,
            &xjw::gui::reference::CameraReferenceController::exportReferences);
    connect(_referencePanel,
            &ReferencePanelWidget::cameraReferenceSettingsRequested,
            _cameraReferenceController,
            &xjw::gui::reference::CameraReferenceController::showSettingsSummary);

}
