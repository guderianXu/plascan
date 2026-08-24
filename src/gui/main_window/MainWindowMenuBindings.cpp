#include "MainWindow.h"

#include <QAction>
#include <QStatusBar>
#include <QToolBar>
#include <QToolButton>
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
            _propertiesDockSuppressed = false;
            updatePropertiesDockForCurrentTab();
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
    if (_mainMenu->rectangleMaskAction())
    {
        connect(_mainMenu->rectangleMaskAction(), &QAction::triggered,
                _canvas, &CanvasWidget::useRectangleMaskTool);
    }
    if (_mainMenu->scissorsMaskAction())
    {
        connect(_mainMenu->scissorsMaskAction(), &QAction::triggered,
                _canvas, &CanvasWidget::useScissorsMaskTool);
    }
    if (_mainMenu->smartPaintMaskAction())
    {
        connect(_mainMenu->smartPaintMaskAction(), &QAction::triggered,
                _canvas, &CanvasWidget::useSmartPaintMaskTool);
    }
    if (_mainMenu->magicWandMaskAction())
    {
        connect(_mainMenu->magicWandMaskAction(), &QAction::triggered,
                _canvas, &CanvasWidget::useMagicWandMaskTool);
    }
    if (_mainMenu->maskEditorSettingsAction())
    {
        connect(_mainMenu->maskEditorSettingsAction(), &QAction::triggered,
                _canvas, &CanvasWidget::showMaskEditorSettings);
    }
    if (_mainMenu->resetMaskSelectionAction())
    {
        connect(_mainMenu->resetMaskSelectionAction(), &QAction::triggered,
                _canvas, &CanvasWidget::resetMaskSelection);
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
        connect(_canvas,
                &CanvasWidget::maskSelectionActiveChanged,
                _mainMenu,
                &MainMenu::setMaskSelectionActive);
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
            QAction *lock_action = _mainMenu->lockCameraImageAction();
            QAction *show_image_action = _mainMenu->toggleCameraImagesAction();
            lock_action->setEnabled(
                show_image_action && show_image_action->isChecked());
            if (show_image_action)
            {
                connect(show_image_action, &QAction::toggled,
                        lock_action, &QAction::setEnabled);
            }
            connect(lock_action, &QAction::toggled,
                    modelView, &CameraSceneWidget::setCameraImageLocked);
            connect(modelView, &CameraSceneWidget::cameraImageLockedChanged,
                    lock_action, [lock_action](bool locked)
            {
                const QSignalBlocker blocker(lock_action);
                lock_action->setChecked(locked);
            });
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
        auto sync_point_tool_actions = [this](CameraSceneWidget::PointInteractionMode mode)
        {
            const QSignalBlocker navigation_blocker(_mainMenu->navigationAction());
            const QSignalBlocker rectangle_blocker(_mainMenu->rectangleSelectionAction());
            const QSignalBlocker circle_blocker(_mainMenu->circleSelectionAction());
            const QSignalBlocker freehand_blocker(_mainMenu->freehandSelectionAction());
            _mainMenu->navigationAction()->setChecked(
                mode == CameraSceneWidget::PointInteractionMode::Navigation);
            _mainMenu->rectangleSelectionAction()->setChecked(
                mode == CameraSceneWidget::PointInteractionMode::RectangleSelection);
            _mainMenu->circleSelectionAction()->setChecked(
                mode == CameraSceneWidget::PointInteractionMode::CircleSelection);
            _mainMenu->freehandSelectionAction()->setChecked(
                mode == CameraSceneWidget::PointInteractionMode::FreehandSelection);
            auto *selection_button = _mainMenu->toolBar()->findChild<QToolButton *>(
                QStringLiteral("toolButtonPointSelection"));
            if (selection_button
                && mode != CameraSceneWidget::PointInteractionMode::Navigation)
            {
                QAction *selection_action = _mainMenu->rectangleSelectionAction();
                if (mode == CameraSceneWidget::PointInteractionMode::CircleSelection)
                {
                    selection_action = _mainMenu->circleSelectionAction();
                }
                else if (mode == CameraSceneWidget::PointInteractionMode::FreehandSelection)
                {
                    selection_action = _mainMenu->freehandSelectionAction();
                }
                selection_button->setDefaultAction(selection_action);
            }
        };
        auto activate_point_tool = [this, modelView, sync_point_tool_actions](
                                       CameraSceneWidget::PointInteractionMode mode)
        {
            QString error_message;
            if (!modelView->setPointInteractionMode(mode, &error_message))
            {
                sync_point_tool_actions(modelView->pointInteractionMode());
                QMessageBox::warning(this, tr("点云选择"), error_message);
                return;
            }
            _workspaceCenter->showModelView();
            statusBar()->showMessage(
                mode == CameraSceneWidget::PointInteractionMode::Navigation
                    ? tr("已切换到导航工具")
                    : tr("选择点云后按 Delete 删除，按 Ctrl+Z 撤销"),
                3500);
        };
        connect(_mainMenu->navigationAction(), &QAction::triggered, this,
                [activate_point_tool]()
        {
            activate_point_tool(CameraSceneWidget::PointInteractionMode::Navigation);
        });
        connect(_mainMenu->rectangleSelectionAction(), &QAction::triggered, this,
                [activate_point_tool]()
        {
            activate_point_tool(CameraSceneWidget::PointInteractionMode::RectangleSelection);
        });
        connect(_mainMenu->circleSelectionAction(), &QAction::triggered, this,
                [activate_point_tool]()
        {
            activate_point_tool(CameraSceneWidget::PointInteractionMode::CircleSelection);
        });
        connect(_mainMenu->freehandSelectionAction(), &QAction::triggered, this,
                [activate_point_tool]()
        {
            activate_point_tool(CameraSceneWidget::PointInteractionMode::FreehandSelection);
        });
        connect(modelView, &CameraSceneWidget::pointInteractionModeChanged,
                this, sync_point_tool_actions);
        sync_point_tool_actions(modelView->pointInteractionMode());

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
