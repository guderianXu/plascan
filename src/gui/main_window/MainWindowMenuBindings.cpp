#include "MainWindow.h"

#include <QAction>
#include <QStatusBar>
#include <QJsonObject>
#include <QMessageBox>
#include <QSignalBlocker>

#include "CanvasWidget.h"
#include "LogPanel.h"
#include "MainMenu.h"
#include "WorkspaceCenterWidget.h"
#include "WorkspacePanelController.h"
#include "CameraSceneWidget.h"
#include "Logger.h"

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
    _workspacePanels->registerDock(WorkspacePanelId::Work,
                                   _mainMenu->toggleWorkAction(),
                                   _workDock);
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
        connect(modelView,
                &CameraSceneWidget::modelColorModeChanged,
                this,
                [this](CameraSceneWidget::ModelColorMode mode)
        {
            QAction *modeAction = nullptr;
            switch (mode)
            {
            case CameraSceneWidget::ModelColorMode::Texture:
                modeAction = _mainMenu->modelTextureModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::Shaded:
                modeAction = _mainMenu->modelShadedModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::Solid:
                modeAction = _mainMenu->modelSolidModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::Wireframe:
                modeAction = _mainMenu->modelWireframeModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::Elevation:
                modeAction = _mainMenu->modelElevationModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::Confidence:
                modeAction = _mainMenu->modelConfidenceModeAction();
                break;
            case CameraSceneWidget::ModelColorMode::AssignedImage:
                modeAction = _mainMenu->modelAssignedImageModeAction();
                break;
            }
            if (modeAction && !modeAction->isChecked())
            {
                modeAction->setChecked(true);
            }
        });
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
