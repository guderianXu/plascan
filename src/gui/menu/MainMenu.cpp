/**
 * @file MainMenu.cpp
 * @brief MainMenu 的实现文件。
 *
 * 包含菜单栏构建、最近项目子菜单的动态重建，以及全部访问器方法的实现。
 * 本文件不包含任何业务逻辑，所有 QAction 的 triggered 信号由主窗口负责连接。
 */
#include "MainMenu.h"
#include "AboutDialog.h"
#include "ToolbarButton.h"

#include <QDir>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QToolBar>
#include <QApplication>
#include <QWidgetAction>
#include <QFileInfo>
#include <QIcon>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>
#include <QPolygonF>
#include <QSize>
#include <QStyle>
#include <QToolButton>
#include <Qt>

namespace {

template <typename T>
T *findNamedChild(QObject *root, const char *name)
{
    return root ? root->findChild<T *>(QString::fromLatin1(name)) : nullptr;
}

QAction *ensureCheckableAction(QObject *root,
                               QObject *actionParent,
                               QMenu *menu,
                               const QString &objectName,
                               const QString &text,
                               bool checked,
                               QAction *before = nullptr)
{
    auto *action = root ? root->findChild<QAction *>(objectName) : nullptr;
    if (!action)
    {
        action = new QAction(text, actionParent);
        action->setObjectName(objectName);
    }

    action->setText(text);
    action->setCheckable(true);
    action->setChecked(checked);

    if (menu && !menu->actions().contains(action))
    {
        if (before)
        {
            menu->insertAction(before, action);
        }
        else
        {
            menu->addAction(action);
        }
    }

    return action;
}

QAction *ensurePlainAction(QObject *root,
                           QObject *actionParent,
                           QMenu *menu,
                           const QString &objectName,
                           const QString &text,
                           QAction *before = nullptr)
{
    auto *action = root ? root->findChild<QAction *>(objectName) : nullptr;
    if (!action)
    {
        action = new QAction(text, actionParent);
        action->setObjectName(objectName);
    }

    action->setText(text);

    if (menu && !menu->actions().contains(action))
    {
        if (before && menu->actions().contains(before))
        {
            menu->insertAction(before, action);
        }
        else
        {
            menu->addAction(action);
        }
    }

    return action;
}

QMenu *ensureSubMenu(QObject *root,
                     QMenu *parent,
                     const QString &objectName,
                     const QString &title,
                     QAction *before = nullptr)
{
    auto *menu = root ? root->findChild<QMenu *>(objectName) : nullptr;
    if (!menu)
    {
        menu = new QMenu(title, parent);
        menu->setObjectName(objectName);
    }

    menu->setTitle(title);

    if (parent && !parent->actions().contains(menu->menuAction()))
    {
        if (before && parent->actions().contains(before))
        {
            parent->insertMenu(before, menu);
        }
        else
        {
            parent->addMenu(menu);
        }
    }

    return menu;
}

QIcon makeCameraToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor bodyColor(142, 145, 148);
    const QColor lensColor(118, 121, 125);
    const QColor highlightColor(226, 228, 230);
    painter.setPen(Qt::NoPen);

    painter.setBrush(bodyColor);
    const QRectF cameraBodyRect(2.0, 17.0, 52.0, 35.0);
    painter.drawRoundedRect(cameraBodyRect, 4.0, 4.0);

    const QRectF cameraTopRect(10.0, 7.0, 22.0, 13.0);
    painter.drawRoundedRect(cameraTopRect, 3.0, 3.0);

    const QRectF cameraShutterRect(40.0, 10.0, 9.0, 6.0);
    painter.drawRoundedRect(cameraShutterRect, 1.8, 1.8);

    painter.setBrush(lensColor);
    const QRectF cameraLensRect(19.0, 24.0, 18.0, 18.0);
    painter.drawEllipse(cameraLensRect);

    painter.setBrush(highlightColor);
    painter.drawEllipse(QRectF(24.0, 29.0, 7.0, 7.0));
    painter.drawEllipse(QRectF(43.0, 20.5, 3.2, 3.2));

    return QIcon(pixmap);
}

QIcon makeCameraImageToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor frameColor(128, 132, 136);
    const QColor imageColor(182, 185, 188);
    const QColor detailColor(96, 100, 105);
    painter.setPen(Qt::NoPen);

    painter.setBrush(frameColor);
    const QRectF imageFrameRect(3.0, 3.0, 50.0, 50.0);
    painter.drawRoundedRect(imageFrameRect, 3.0, 3.0);

    painter.setBrush(imageColor);
    painter.drawRect(QRectF(10.0, 10.0, 36.0, 36.0));

    painter.setBrush(detailColor);
    QPolygonF imageMountain;
    imageMountain << QPointF(10.0, 46.0)
                  << QPointF(21.0, 28.0)
                  << QPointF(29.0, 37.0)
                  << QPointF(34.0, 33.0)
                  << QPointF(46.0, 46.0);
    painter.drawPolygon(imageMountain);

    const QRectF imageSunRect(36.0, 15.0, 7.0, 7.0);
    painter.drawEllipse(imageSunRect);

    return QIcon(pixmap);
}

QIcon makeImageRotationToolbarIcon(bool rotateLeft)
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor iconColor(126, 131, 136);
    painter.setPen(QPen(iconColor, 3.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(Qt::NoBrush);

    QPainterPath arrowPath;
    if (rotateLeft)
    {
        arrowPath.moveTo(49.0, 18.0);
        arrowPath.cubicTo(44.0, 1.0, 12.0, 1.0, 7.0, 18.0);
    }
    else
    {
        arrowPath.moveTo(7.0, 18.0);
        arrowPath.cubicTo(12.0, 1.0, 44.0, 1.0, 49.0, 18.0);
    }
    painter.drawPath(arrowPath);

    painter.setPen(Qt::NoPen);
    painter.setBrush(iconColor);
    QPolygonF arrowHead;
    if (rotateLeft)
    {
        arrowHead << QPointF(7.0, 20.0) << QPointF(1.0, 9.0) << QPointF(19.0, 13.0);
    }
    else
    {
        arrowHead << QPointF(49.0, 20.0) << QPointF(37.0, 13.0) << QPointF(55.0, 9.0);
    }
    painter.drawPolygon(arrowHead);

    painter.drawRoundedRect(QRectF(4.0, 22.0, 48.0, 33.0), 2.5, 2.5);
    painter.setBrush(QColor(218, 221, 224));
    painter.drawRect(QRectF(9.0, 27.0, 38.0, 23.0));
    painter.setBrush(iconColor);
    QPolygonF mountain;
    mountain << QPointF(9.0, 50.0) << QPointF(20.0, 34.0)
             << QPointF(28.0, 43.0) << QPointF(35.0, 37.0) << QPointF(47.0, 50.0);
    painter.drawPolygon(mountain);
    painter.drawEllipse(QRectF(38.0, 29.0, 6.0, 6.0));

    return QIcon(pixmap);
}

QIcon makeZoomToolbarIcon(bool zoomIn)
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor iconColor(142, 145, 148);
    painter.setPen(QPen(iconColor, 4.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(QRectF(4.0, 4.0, 36.0, 36.0));
    painter.drawLine(QPointF(34.0, 34.0), QPointF(53.0, 53.0));
    painter.drawLine(QPointF(13.0, 22.0), QPointF(31.0, 22.0));
    if (zoomIn)
    {
        painter.drawLine(QPointF(22.0, 13.0), QPointF(22.0, 31.0));
    }

    return QIcon(pixmap);
}

QIcon makeFeaturePointsToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(128, 132, 136));
    for (const QPointF &point : {QPointF(13, 13), QPointF(28, 11), QPointF(43, 15),
                                 QPointF(10, 29), QPointF(27, 28), QPointF(45, 31),
                                 QPointF(15, 44), QPointF(31, 45), QPointF(46, 43)})
    {
        painter.drawEllipse(point, 4.2, 4.2);
    }
    return QIcon(pixmap);
}

QIcon makeMaskToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor color(132, 136, 140);
    painter.setPen(QPen(color, 4.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRect(QRectF(6.0, 6.0, 44.0, 44.0));
    painter.setBrush(color);
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(QRectF(16.0, 16.0, 24.0, 24.0));
    return QIcon(pixmap);
}

QIcon makeDepthOverlayToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);

    const QRectF frame(5.0, 7.0, 46.0, 42.0);
    painter.setPen(QPen(QColor(128, 132, 136), 2.5));
    painter.setBrush(Qt::NoBrush);
    painter.drawRoundedRect(frame, 2.0, 2.0);

    painter.setPen(Qt::NoPen);
    const QList<QColor> colors = {
        QColor(49, 54, 149), QColor(69, 117, 180), QColor(116, 173, 209),
        QColor(171, 217, 233), QColor(224, 243, 248), QColor(254, 224, 144),
        QColor(253, 174, 97), QColor(244, 109, 67), QColor(165, 0, 38)};
    const qreal band_width = 42.0 / static_cast<qreal>(colors.size());
    for (int index = 0; index < colors.size(); ++index)
    {
        painter.setBrush(colors[index]);
        painter.drawRect(QRectF(7.0 + index * band_width, 9.0, band_width + 0.5, 38.0));
    }
    return QIcon(pixmap);
}

QIcon makeResetViewToolbarIcon()
{
    QPixmap pixmap(56, 56);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor color(132, 136, 140);
    painter.setPen(QPen(color, 3.5, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(QPointF(28, 20), QPointF(28, 5));
    painter.drawLine(QPointF(28, 36), QPointF(28, 51));
    painter.drawLine(QPointF(20, 28), QPointF(5, 28));
    painter.drawLine(QPointF(36, 28), QPointF(51, 28));
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawPolygon(QPolygonF{QPointF(28, 3), QPointF(21, 12), QPointF(35, 12)});
    painter.drawPolygon(QPolygonF{QPointF(28, 53), QPointF(21, 44), QPointF(35, 44)});
    painter.drawPolygon(QPolygonF{QPointF(3, 28), QPointF(12, 21), QPointF(12, 35)});
    painter.drawPolygon(QPolygonF{QPointF(53, 28), QPointF(44, 21), QPointF(44, 35)});
    painter.drawRect(QRectF(22, 22, 12, 12));
    return QIcon(pixmap);
}

QIcon makeSaveToolbarIcon()
{
    QPixmap pixmap(32, 32);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor iconColor(135, 140, 145);
    painter.setPen(Qt::NoPen);
    painter.setBrush(iconColor);
    painter.drawRoundedRect(QRectF(4.0, 3.0, 24.0, 26.0), 1.5, 1.5);
    painter.setBrush(QColor(230, 232, 234));
    painter.drawRect(QRectF(8.0, 4.0, 14.0, 9.0));
    painter.drawRoundedRect(QRectF(8.0, 18.0, 16.0, 10.0), 1.0, 1.0);
    painter.setBrush(iconColor);
    painter.drawRect(QRectF(18.0, 5.0, 3.0, 7.0));

    return QIcon(pixmap);
}

QIcon makePointCloudPruneToolbarIcon()
{
    QPixmap pixmap(32, 32);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    const QColor iconColor(135, 140, 145);
    painter.setPen(Qt::NoPen);
    painter.setBrush(iconColor);
    for (const QPointF &point : {QPointF(6.0, 7.0), QPointF(15.0, 5.0), QPointF(24.0, 8.0),
                                 QPointF(8.0, 16.0), QPointF(17.0, 14.0), QPointF(25.0, 18.0),
                                 QPointF(6.0, 25.0), QPointF(15.0, 23.0)})
    {
        painter.drawEllipse(point, 2.2, 2.2);
    }
    painter.setPen(QPen(iconColor, 3.0, Qt::SolidLine, Qt::RoundCap));
    painter.drawLine(QPointF(20.0, 22.0), QPointF(29.0, 31.0));
    painter.drawLine(QPointF(29.0, 22.0), QPointF(20.0, 31.0));

    return QIcon(pixmap);
}

} // namespace

// ============================================================
//  构造函数 — 创建完整的菜单结构
// ============================================================

/**
 * @brief 构造函数：在 mainWindow 上创建所有菜单项和工具栏。
 *
 * 菜单创建顺序（与菜单栏从左到右排列一致）：
 *   1. 项目菜单（新建、打开、最近打开子菜单、保存、退出）
 *   2. 视图菜单（缩放、可视化设置、窗口面板子菜单）
 *   3. 工作流程菜单（空三、模型、DEM、正射影像）
 *   4. 工具菜单（连接点、重叠度、交汇、报告）
 *   5. 帮助菜单（关于）
 *   6. 主工具栏
 *
 * @param mainWindow 目标主窗口，不可为 nullptr。
 */
MainMenu::MainMenu(QMainWindow *mainWindow)
    : QObject(mainWindow), _mainWindow(mainWindow)
{
    if (!_mainWindow) return;

    auto installDepthOverlayActions = [this](QMenu *view_menu, QObject *action_parent)
    {
        _showDepthOverlayAct = ensureCheckableAction(
            _mainWindow,
            action_parent,
            view_menu,
            QStringLiteral("actionShowDepthOverlay"),
            tr("显示深度图"),
            false,
            _featureVisualizationAct);
        _showDepthOverlayAct->setIcon(makeDepthOverlayToolbarIcon());
        _showDepthOverlayAct->setToolTip(tr("在当前照片上叠加显示深度信息"));

        _depthOverlayAllLevelsAct = ensureCheckableAction(
            _mainWindow, action_parent, nullptr,
            QStringLiteral("actionDepthOverlayAllLevels"), tr("所有级别"), true);
        _depthOverlayLevel1Act = ensureCheckableAction(
            _mainWindow, action_parent, nullptr,
            QStringLiteral("actionDepthOverlayLevel1"), tr("级别 1"), false);
        _depthOverlayLevel2Act = ensureCheckableAction(
            _mainWindow, action_parent, nullptr,
            QStringLiteral("actionDepthOverlayLevel2"), tr("级别 2"), false);
        _depthOverlayLevel3Act = ensureCheckableAction(
            _mainWindow, action_parent, nullptr,
            QStringLiteral("actionDepthOverlayLevel3"), tr("级别 3"), false);
        _showDepthIntensityAct = ensureCheckableAction(
            _mainWindow, action_parent, nullptr,
            QStringLiteral("actionShowDepthIntensity"), tr("显示强度"), false);

        _depthOverlayLevelGroup = new QActionGroup(this);
        _depthOverlayLevelGroup->setObjectName(QStringLiteral("depthOverlayLevelActionGroup"));
        _depthOverlayLevelGroup->setExclusive(true);
        _depthOverlayLevelGroup->addAction(_depthOverlayAllLevelsAct);
        _depthOverlayLevelGroup->addAction(_depthOverlayLevel1Act);
        _depthOverlayLevelGroup->addAction(_depthOverlayLevel2Act);
        _depthOverlayLevelGroup->addAction(_depthOverlayLevel3Act);
    };

    auto installTiePointsMenu = [this](QMenu *toolsMenu, QAction *before = nullptr)
    {
        QMenu *tiePointsMenu = ensureSubMenu(_mainWindow,
                                             toolsMenu,
                                             QStringLiteral("menuTiePoints"),
                                             tr("连接点"),
                                             before);
        if (!tiePointsMenu)
        {
            return;
        }

        _createTiePointsAct = ensurePlainAction(_mainWindow,
                                                tiePointsMenu,
                                                tiePointsMenu,
                                                QStringLiteral("actionCreateTiePoints"),
                                                tr("创建连接点..."));
        _createTiePointsAct->setToolTip(tr("打开连接点创建参数对话框"));

        _thinTiePointsAct = ensurePlainAction(_mainWindow,
                                              tiePointsMenu,
                                              tiePointsMenu,
                                              QStringLiteral("actionThinTiePoints"),
                                              tr("稀释连接点..."));
        _thinTiePointsAct->setToolTip(tr("按连接点数量限制稀释当前连接点"));

        _cleanTiePointsAct = ensurePlainAction(_mainWindow,
                                               tiePointsMenu,
                                               tiePointsMenu,
                                               QStringLiteral("actionCleanTiePoints"),
                                               QStringLiteral("Clean Tie Points..."));
        _cleanTiePointsAct->setToolTip(tr("按误差或观测指标筛选连接点"));

        auto *separator = _mainWindow->findChild<QAction *>(QStringLiteral("actionTiePointsViewSeparator"));
        if (!separator)
        {
            separator = new QAction(tiePointsMenu);
            separator->setObjectName(QStringLiteral("actionTiePointsViewSeparator"));
            separator->setSeparator(true);
        }
        if (!tiePointsMenu->actions().contains(separator))
        {
            tiePointsMenu->addAction(separator);
        }

        _viewTiePointMatchesAct = ensurePlainAction(_mainWindow,
                                                    tiePointsMenu,
                                                    tiePointsMenu,
                                                    QStringLiteral("actionViewTiePointMatches"),
                                                    tr("查看匹配..."));
        _viewTiePointMatchesAct->setToolTip(tr("打开当前项目的匹配查看器"));
    };

    auto installModelDisplayMenu = [this](QMenu *modelMenu)
    {
        if (!modelMenu)
        {
            return;
        }
        _modelMenu = modelMenu;

        QMenu *displayMenu = ensureSubMenu(_mainWindow,
                                           modelMenu,
                                           QStringLiteral("menuModelDisplayHideItems"),
                                           tr("显示/隐藏项目"));
        if (!displayMenu)
        {
            return;
        }
        _modelDisplayHideMenu = displayMenu;

        if (_toggleGizmoAct)
        {
            _toggleGizmoAct->setText(tr("显示轨迹球"));
            _toggleGizmoAct->setToolTip(tr("显示或隐藏 3D 视图中的旋转轨迹球"));
            if (!displayMenu->actions().contains(_toggleGizmoAct))
            {
                displayMenu->addAction(_toggleGizmoAct);
            }
        }

        if (_toggleCamerasAct && !displayMenu->actions().contains(_toggleCamerasAct))
        {
            displayMenu->addAction(_toggleCamerasAct);
        }

        _toggleDependentCamerasAct = ensurePlainAction(_mainWindow,
                                                       displayMenu,
                                                       displayMenu,
                                                       QStringLiteral("actionToggleDependentCameras"),
                                                       tr("显示从属相机"));
        _toggleDependentCamerasAct->setEnabled(false);
        _toggleDependentCamerasAct->setToolTip(tr("暂未建立模型与从属相机关系，当前版本无法单独显示从属相机"));

        _toggleCameraThumbnailsAct = ensureCheckableAction(_mainWindow,
                                                           displayMenu,
                                                           displayMenu,
                                                           QStringLiteral("actionToggleCameraThumbnails"),
                                                           tr("显示缩略图"),
                                                           true);
        _toggleCameraThumbnailsAct->setToolTip(tr("在相机平面上显示项目缩略图"));

        QMenu *imageMenu = ensureSubMenu(_mainWindow,
                                         displayMenu,
                                         QStringLiteral("menuModelDisplayImages"),
                                         tr("显示图像"));
        if (!imageMenu)
        {
            return;
        }

        _toggleCameraImagesAct = ensureCheckableAction(_mainWindow,
                                                       imageMenu,
                                                       imageMenu,
                                                       QStringLiteral("actionToggleCameraImages"),
                                                       tr("显示图像"),
                                                       false);
        _toggleCameraImagesAct->setToolTip(tr("显示或隐藏与当前模型观察方向最接近的相机图像"));

        imageMenu->addSeparator();

        _showCameraImagesInForegroundAct = ensureCheckableAction(_mainWindow,
                                                                 imageMenu,
                                                                 imageMenu,
                                                                 QStringLiteral("actionShowCameraImagesInForeground"),
                                                                 tr("在前景中显示"),
                                                                 true);
        _showCameraImagesInForegroundAct->setToolTip(tr("将当前相机图像绘制在三维模型前方，图像覆盖模型"));

        _showCameraImagesInBackgroundAct = ensureCheckableAction(_mainWindow,
                                                                 imageMenu,
                                                                 imageMenu,
                                                                 QStringLiteral("actionShowCameraImagesInBackground"),
                                                                 tr("在后景中显示"),
                                                                 false);
        _showCameraImagesInBackgroundAct->setToolTip(tr("将当前相机图像绘制在三维模型后方，模型覆盖图像"));

        auto *displayLayerGroup =
            imageMenu->findChild<QActionGroup *>(QStringLiteral("actionGroupCameraImageDisplayLayer"));
        if (!displayLayerGroup)
        {
            displayLayerGroup = new QActionGroup(imageMenu);
            displayLayerGroup->setObjectName(QStringLiteral("actionGroupCameraImageDisplayLayer"));
        }
        displayLayerGroup->setExclusive(true);
        if (_showCameraImagesInForegroundAct->actionGroup() != displayLayerGroup)
        {
            displayLayerGroup->addAction(_showCameraImagesInForegroundAct);
        }
        if (_showCameraImagesInBackgroundAct->actionGroup() != displayLayerGroup)
        {
            displayLayerGroup->addAction(_showCameraImagesInBackgroundAct);
        }

        imageMenu->addSeparator();

        _lockCameraImageAct = ensureCheckableAction(_mainWindow,
                                                    imageMenu,
                                                    imageMenu,
                                                    QStringLiteral("actionLockCameraImage"),
                                                    tr("锁定图像"),
                                                    false);
        _lockCameraImageAct->setToolTip(tr("固定当前相机图像；模型视角仍可自由旋转"));
    };

    auto installViewMenuLayout = [this](QMenu *viewMenu, QMenu *windowMenu)
    {
        if (!viewMenu || !windowMenu)
        {
            return;
        }

        QMenu *imageMenu = ensureSubMenu(_mainWindow,
                                         viewMenu,
                                         QStringLiteral("menuViewImageDisplay"),
                                         tr("影像显示"));
        _imageDisplayMenu = imageMenu;
        auto removeAllActions = [](QMenu *menu)
        {
            if (!menu)
            {
                return;
            }
            const QList<QAction *> actions = menu->actions();
            for (QAction *action : actions)
            {
                menu->removeAction(action);
            }
        };

        removeAllActions(imageMenu);
        removeAllActions(windowMenu);
        removeAllActions(viewMenu);

        for (QAction *action : {_zoomInAct, _zoomOutAct, _resetViewAct, _toggleFullScreenAct})
        {
            if (action)
            {
                viewMenu->addAction(action);
            }
        }
        viewMenu->addSeparator();

        for (QAction *action : {_rotateImageLeftAct, _rotateImageRightAct})
        {
            if (action)
            {
                imageMenu->addAction(action);
            }
        }
        imageMenu->addSeparator();
        for (QAction *action : {_showFeaturePointsAct,
                                _showFeatureResidualsAct,
                                _showMaskOverlayAct,
                                _showDepthOverlayAct})
        {
            if (action)
            {
                imageMenu->addAction(action);
            }
        }
        imageMenu->addSeparator();
        if (_featureVisualizationAct)
        {
            imageMenu->addAction(_featureVisualizationAct);
        }
        viewMenu->addMenu(imageMenu);
        viewMenu->addSeparator();

        for (QAction *action : {_toggleWorkspaceAct,
                                _togglePropertiesAct,
                                _togglePhotosAct,
                                _toggleLogAct})
        {
            if (action)
            {
                windowMenu->addAction(action);
            }
        }
        windowMenu->addSeparator();
        if (_toggleMainToolbarAct)
        {
            windowMenu->addAction(_toggleMainToolbarAct);
        }
        if (_toggleHenanUniversityBrandAct)
        {
            windowMenu->addAction(_toggleHenanUniversityBrandAct);
        }
        viewMenu->addMenu(windowMenu);

        QStyle *style = _mainWindow->style();
        if (style)
        {
            _toggleWorkspaceAct->setIcon(style->standardIcon(QStyle::SP_DirHomeIcon));
            _togglePropertiesAct->setIcon(style->standardIcon(QStyle::SP_FileDialogDetailedView));
            _togglePhotosAct->setIcon(style->standardIcon(QStyle::SP_FileDialogContentsView));
            _toggleLogAct->setIcon(style->standardIcon(QStyle::SP_FileDialogInfoView));
            _toggleMainToolbarAct->setIcon(
                style->standardIcon(QStyle::SP_ToolBarHorizontalExtensionButton));
        }
        updateImageActionAvailability();
    };

    auto installCameraToolbarButton = [this]()
    {
        if (!_toolBar || !_toggleCamerasAct || !_toggleCameraThumbnailsAct || !_toggleDependentCamerasAct)
        {
            return;
        }
        if (_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraVisibility")))
        {
            return;
        }

        _toggleCamerasAct->setIcon(makeCameraToolbarIcon());
        _toggleCamerasAct->setText(tr("显示相机"));
        _toggleCamerasAct->setToolTip(tr("显示相机"));

        const bool showLocalAxes = !_toggleGizmoAct || _toggleGizmoAct->isChecked();
        _toggleLocalAxesAct = ensureCheckableAction(_mainWindow,
                                                    _mainWindow,
                                                    nullptr,
                                                    QStringLiteral("actionToggleLocalAxes"),
                                                    tr("显示本地轴"),
                                                    showLocalAxes);
        _toggleLocalAxesAct->setToolTip(tr("显示或隐藏模型视图中的本地轴"));

        auto *cameraMenu = new QMenu(_toolBar);
        cameraMenu->setObjectName(QStringLiteral("menuToolbarCameraVisibility"));
        cameraMenu->addAction(_toggleCameraThumbnailsAct);
        cameraMenu->addAction(_toggleDependentCamerasAct);
        cameraMenu->addAction(_toggleLocalAxesAct);
        _cameraToolbarWidgetAct = xjw::gui::toolbar::createToolbarSplitButton(
            _toolBar,
            _toggleCamerasAct,
            cameraMenu,
            QStringLiteral("toolButtonModelCameraVisibility"),
            _toolbarEditingSeparatorAct);
    };

    auto installCameraImageToolbarButton = [this]()
    {
        if (!_toolBar || !_toggleCameraImagesAct ||
            !_showCameraImagesInForegroundAct || !_showCameraImagesInBackgroundAct || !_lockCameraImageAct)
        {
            return;
        }
        if (_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonModelCameraImageVisibility")))
        {
            return;
        }

        _toggleCameraImagesAct->setIcon(makeCameraImageToolbarIcon());
        _toggleCameraImagesAct->setText(tr("显示图像"));
        _toggleCameraImagesAct->setToolTip(tr("显示图像"));

        auto *imageMenu = new QMenu(_toolBar);
        imageMenu->setObjectName(QStringLiteral("menuToolbarCameraImageVisibility"));
        imageMenu->addAction(_showCameraImagesInForegroundAct);
        imageMenu->addAction(_showCameraImagesInBackgroundAct);
        imageMenu->addSeparator();
        imageMenu->addAction(_lockCameraImageAct);
        _cameraImageToolbarWidgetAct = xjw::gui::toolbar::createToolbarSplitButton(
            _toolBar,
            _toggleCameraImagesAct,
            imageMenu,
            QStringLiteral("toolButtonModelCameraImageVisibility"),
            _toolbarEditingSeparatorAct);
    };

    auto installImageRotationToolbarButtons = [this]()
    {
        if (!_toolBar || !_rotateImageLeftAct || !_rotateImageRightAct)
        {
            return;
        }

        auto installButton = [this](QAction *action,
                                    const QString &objectName,
                                    QAction *&toolbarWidgetAction)
        {
            if (_toolBar->findChild<QToolButton *>(objectName))
            {
                return;
            }

            toolbarWidgetAction = xjw::gui::toolbar::createToolbarButton(
                _toolBar, action, objectName, _toolbarEditingSeparatorAct);
        };

        installButton(_rotateImageLeftAct,
                      QStringLiteral("toolButtonRotateImageLeft"),
                      _rotateImageLeftToolbarWidgetAct);
        installButton(_rotateImageRightAct,
                      QStringLiteral("toolButtonRotateImageRight"),
                      _rotateImageRightToolbarWidgetAct);
    };

    auto installImageOverlayToolbarButtons = [this]()
    {
        if (!_toolBar || !_showFeaturePointsAct || !_showFeatureResidualsAct
            || !_showMaskOverlayAct || !_showDepthOverlayAct || !_depthOverlayAllLevelsAct
            || !_depthOverlayLevel1Act || !_depthOverlayLevel2Act || !_depthOverlayLevel3Act
            || !_showDepthIntensityAct || !_resetViewAct || !_featureVisualizationAct)
        {
            return;
        }

        if (!_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonResetImageView")))
        {
            _resetImageViewToolbarWidgetAct = xjw::gui::toolbar::createToolbarButton(
                _toolBar, _resetViewAct, QStringLiteral("toolButtonResetImageView"),
                _toolbarEditingSeparatorAct);
        }
        if (!_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowFeaturePoints")))
        {
            auto *pointsMenu = new QMenu(_toolBar);
            pointsMenu->setObjectName(QStringLiteral("menuToolbarFeaturePoints"));
            pointsMenu->addAction(_showFeatureResidualsAct);
            pointsMenu->addSeparator();
            pointsMenu->addAction(_featureVisualizationAct);
            _showFeaturePointsToolbarWidgetAct = xjw::gui::toolbar::createToolbarSplitButton(
                _toolBar, _showFeaturePointsAct, pointsMenu,
                QStringLiteral("toolButtonShowFeaturePoints"), _toolbarEditingSeparatorAct);
        }
        if (!_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowMaskOverlay")))
        {
            _showMaskOverlayToolbarWidgetAct = xjw::gui::toolbar::createToolbarButton(
                _toolBar, _showMaskOverlayAct, QStringLiteral("toolButtonShowMaskOverlay"),
                _toolbarEditingSeparatorAct);
        }
        if (!_toolBar->findChild<QToolButton *>(QStringLiteral("toolButtonShowDepthOverlay")))
        {
            auto *depth_menu = new QMenu(_toolBar);
            depth_menu->setObjectName(QStringLiteral("menuToolbarDepthOverlay"));
            depth_menu->addAction(_depthOverlayAllLevelsAct);
            depth_menu->addAction(_depthOverlayLevel1Act);
            depth_menu->addAction(_depthOverlayLevel2Act);
            depth_menu->addAction(_depthOverlayLevel3Act);
            depth_menu->addSeparator();
            depth_menu->addAction(_showDepthIntensityAct);
            _showDepthOverlayToolbarWidgetAct = xjw::gui::toolbar::createToolbarSplitButton(
                _toolBar,
                _showDepthOverlayAct,
                depth_menu,
                QStringLiteral("toolButtonShowDepthOverlay"),
                _toolbarEditingSeparatorAct);
        }
    };

    auto installZoomToolbarButtons = [this]()
    {
        if (!_toolBar || !_zoomInAct || !_zoomOutAct)
        {
            return;
        }

        auto installButton = [this](QAction *action,
                                    const QString &objectName,
                                    QAction *&toolbarWidgetAction)
        {
            if (_toolBar->findChild<QToolButton *>(objectName))
            {
                return;
            }

            toolbarWidgetAction = xjw::gui::toolbar::createToolbarButton(
                _toolBar, action, objectName, _toolbarEditingSeparatorAct);
        };

        installButton(_zoomInAct, QStringLiteral("toolButtonZoomIn"), _zoomInToolbarWidgetAct);
        installButton(_zoomOutAct, QStringLiteral("toolButtonZoomOut"), _zoomOutToolbarWidgetAct);
    };

    auto initializeToolbar = [this, &installZoomToolbarButtons]()
    {
        if (!_toolBar)
        {
            return;
        }

        _toolBar->clear();
        xjw::gui::toolbar::configureToolbar(_toolBar);
        if (_saveAct)
        {
            _saveAct->setIcon(makeSaveToolbarIcon());
            _saveAct->setToolTip(tr("保存项目"));
            _saveToolbarWidgetAct = xjw::gui::toolbar::createToolbarButton(
                _toolBar, _saveAct, QStringLiteral("toolButtonSaveProject"));
        }
        _toolBar->addSeparator();
        installZoomToolbarButtons();
        _toolbarEditingSeparatorAct = _toolBar->addSeparator();
        if (_manualPointCloudPruneAct)
        {
            _manualPointCloudPruneAct->setIcon(makePointCloudPruneToolbarIcon());
            _manualPointCloudPruneAct->setToolTip(tr("手动点云剔除"));
            _manualPointCloudPruneToolbarWidgetAct = xjw::gui::toolbar::createToolbarButton(
                _toolBar,
                _manualPointCloudPruneAct,
                QStringLiteral("toolButtonManualPointCloudPrune"));
        }
    };

    if (findNamedChild<QAction>(_mainWindow, "actionNewProject"))
    {
        _fileMenu = findNamedChild<QMenu>(_mainWindow, "menuProject");
        _recentMenu = findNamedChild<QMenu>(_mainWindow, "menuRecentProjects");
        auto *viewMenu = findNamedChild<QMenu>(_mainWindow, "menuView");
        auto *windowMenu = findNamedChild<QMenu>(_mainWindow, "menuWindow");
        auto *workflowMenu = findNamedChild<QMenu>(_mainWindow, "menuWorkflow");
        auto *reconstructionMenu = findNamedChild<QMenu>(_mainWindow, "menuReconstruction");
        auto *toolsMenu = findNamedChild<QMenu>(_mainWindow, "menuTools");
        auto *modelMenu = findNamedChild<QMenu>(_mainWindow, "menuModel");

        _newAct = findNamedChild<QAction>(_mainWindow, "actionNewProject");
        _openAct = findNamedChild<QAction>(_mainWindow, "actionOpenProject");
        _saveAct = findNamedChild<QAction>(_mainWindow, "actionSaveProject");
        _exportMatchedPairsAct = findNamedChild<QAction>(_mainWindow, "actionExportMatchedPairs");
        _minimizeAct = findNamedChild<QAction>(_mainWindow, "actionMinimize");
        _exitAct = findNamedChild<QAction>(_mainWindow, "actionExit");

        _zoomInAct = findNamedChild<QAction>(_mainWindow, "actionZoomIn");
        _zoomOutAct = findNamedChild<QAction>(_mainWindow, "actionZoomOut");
        _resetViewAct = findNamedChild<QAction>(_mainWindow, "actionResetView");
        _toggleFullScreenAct = ensurePlainAction(_mainWindow,
                                                 viewMenu,
                                                 nullptr,
                                                 QStringLiteral("actionToggleFullScreen"),
                                                 tr("全屏"));
        _toggleFullScreenAct->setShortcut(QKeySequence(Qt::Key_F11));
        _toggleFullScreenAct->setToolTip(tr("切换全屏显示"));
        if (_zoomInAct)
        {
            _zoomInAct->setIcon(makeZoomToolbarIcon(true));
            _zoomInAct->setToolTip(tr("放大"));
            _zoomInAct->setShortcuts({QKeySequence::ZoomIn});
        }
        if (_zoomOutAct)
        {
            _zoomOutAct->setIcon(makeZoomToolbarIcon(false));
            _zoomOutAct->setToolTip(tr("缩小"));
            _zoomOutAct->setShortcuts({QKeySequence::ZoomOut});
        }
        QObject *rotationActionParent = viewMenu
            ? static_cast<QObject *>(viewMenu)
            : static_cast<QObject *>(_mainWindow);
        _rotateImageLeftAct = ensurePlainAction(_mainWindow,
                                                rotationActionParent,
                                                viewMenu,
                                                QStringLiteral("actionRotateImageLeft"),
                                                tr("向左旋转"),
                                                _zoomInAct);
        _rotateImageRightAct = ensurePlainAction(_mainWindow,
                                                 rotationActionParent,
                                                 viewMenu,
                                                 QStringLiteral("actionRotateImageRight"),
                                                 tr("向右旋转"),
                                                 _zoomInAct);
        _rotateImageLeftAct->setToolTip(tr("向左旋转"));
        _rotateImageRightAct->setToolTip(tr("向右旋转"));
        _rotateImageLeftAct->setIcon(makeImageRotationToolbarIcon(true));
        _rotateImageRightAct->setIcon(makeImageRotationToolbarIcon(false));
        _rotateImageLeftAct->setEnabled(false);
        _rotateImageRightAct->setEnabled(false);
        _toggleGizmoAct = findNamedChild<QAction>(_mainWindow, "actionToggleGizmo");
        _toggleCamerasAct = findNamedChild<QAction>(_mainWindow, "actionToggleCameras");
        _toggleHenanUniversityBrandAct =
            findNamedChild<QAction>(_mainWindow, "actionToggleHenanUniversityBrand");
        _featureVisualizationAct = findNamedChild<QAction>(_mainWindow, "actionFeatureVisualization");
        _showFeaturePointsAct = ensureCheckableAction(_mainWindow, _mainWindow, viewMenu,
                                                      QStringLiteral("actionShowFeaturePoints"),
                                                      tr("显示点"), true,
                                                      _featureVisualizationAct);
        _showFeatureResidualsAct = ensureCheckableAction(_mainWindow, _mainWindow, viewMenu,
                                                         QStringLiteral("actionShowFeatureResiduals"),
                                                         tr("显示点残差"), false,
                                                         _featureVisualizationAct);
        _showMaskOverlayAct = ensureCheckableAction(_mainWindow, _mainWindow, viewMenu,
                                                    QStringLiteral("actionShowMaskOverlay"),
                                                    tr("显示蒙版"), true,
                                                    _featureVisualizationAct);
        installDepthOverlayActions(viewMenu, _mainWindow);
        _showFeaturePointsAct->setIcon(makeFeaturePointsToolbarIcon());
        _showMaskOverlayAct->setIcon(makeMaskToolbarIcon());
        if (_resetViewAct)
        {
            _resetViewAct->setIcon(makeResetViewToolbarIcon());
            _resetViewAct->setToolTip(tr("重置视图"));
        }
        _toggleLogAct = findNamedChild<QAction>(_mainWindow, "actionToggleLog");
        QObject *windowActionParent = windowMenu
            ? static_cast<QObject *>(windowMenu)
            : static_cast<QObject *>(_mainWindow);
        _toggleWorkspaceAct = ensureCheckableAction(_mainWindow,
                                                    windowActionParent,
                                                    nullptr,
                                                    QStringLiteral("actionToggleWorkspace"),
                                                    tr("工作区"),
                                                    true);
        _toggleWorkspaceAct->setToolTip(tr("显示或隐藏工作区面板"));
        _togglePropertiesAct = ensureCheckableAction(_mainWindow,
                                                     windowActionParent,
                                                     nullptr,
                                                     QStringLiteral("actionToggleProperties"),
                                                     tr("属性"),
                                                     true);
        _togglePropertiesAct->setToolTip(tr("显示或隐藏选择对象属性面板"));
        _togglePhotosAct = ensureCheckableAction(_mainWindow,
                                                 windowActionParent,
                                                 nullptr,
                                                 QStringLiteral("actionTogglePhotos"),
                                                 tr("照片"),
                                                 true);
        _togglePhotosAct->setToolTip(tr("显示或隐藏照片面板"));
        if (!_toggleLogAct)
        {
            _toggleLogAct = ensureCheckableAction(_mainWindow,
                                                  windowActionParent,
                                                  nullptr,
                                                  QStringLiteral("actionToggleLog"),
                                                  tr("日志"),
                                                  false);
        }
        _toggleLogAct->setText(tr("日志"));
        _toggleLogAct->setCheckable(true);
        _toggleLogAct->setToolTip(tr("显示或隐藏日志面板"));
        _toggleMainToolbarAct = ensureCheckableAction(_mainWindow,
                                                      windowActionParent,
                                                      nullptr,
                                                      QStringLiteral("actionToggleMainToolbar"),
                                                      tr("主工具栏"),
                                                      true);
        _toggleMainToolbarAct->setToolTip(tr("显示或隐藏主工具栏"));
        QObject *viewActionParent = viewMenu
            ? static_cast<QObject *>(viewMenu)
            : static_cast<QObject *>(_mainWindow);
        _toggleGizmoAct = ensureCheckableAction(_mainWindow,
                                                viewActionParent,
                                                viewMenu,
                                                QStringLiteral("actionToggleGizmo"),
                                                tr("显示轨迹球"),
                                                true,
                                                _toggleCamerasAct);
        _toggleGizmoAct->setToolTip(tr("显示或隐藏 3D 视图中的旋转轨迹球"));
        _toggleHenanUniversityBrandAct = ensureCheckableAction(
            _mainWindow,
            viewActionParent,
            viewMenu,
            QStringLiteral("actionToggleHenanUniversityBrand"),
            tr("河南大学校徽"),
            true,
            _featureVisualizationAct);
        _toggleHenanUniversityBrandAct->setToolTip(tr("显示或隐藏主工具栏中的河南大学校徽"));

        _addPhotoAct = findNamedChild<QAction>(_mainWindow, "actionAddPhoto");
        _addFolderAct = findNamedChild<QAction>(_mainWindow, "actionAddFolder");
        _workflowAerialTriangulationAct =
            findNamedChild<QAction>(_mainWindow, "actionWorkflowAerialTriangulation");
        _threeDReconstructionAct = findNamedChild<QAction>(_mainWindow, "actionThreeDReconstruction");
        _generateModelAct = findNamedChild<QAction>(_mainWindow, "actionGenerateModel");
        _createDEMAct = findNamedChild<QAction>(_mainWindow, "actionCreateDEM");
        _generateOrthoAct = findNamedChild<QAction>(_mainWindow, "actionGenerateOrtho");

        _detectFeaturesAct = findNamedChild<QAction>(_mainWindow, "actionDetectFeatures");
        _vocabularyOverlapAct = findNamedChild<QAction>(_mainWindow, "actionVocabularyOverlap");
        _matchFeaturesAct = findNamedChild<QAction>(_mainWindow, "actionMatchFeatures");
        _aerialTriangulationAct = findNamedChild<QAction>(_mainWindow, "actionAerialTriangulation");
        _buildObsNetworkAct = findNamedChild<QAction>(_mainWindow, "actionBuildObsNetwork");
        _initCameraPoseAct = findNamedChild<QAction>(_mainWindow, "actionInitCameraPose");
        _triangulateAct = findNamedChild<QAction>(_mainWindow, "actionTriangulate");
        _reconBundleAdjustAct = findNamedChild<QAction>(_mainWindow, "actionReconBundleAdjust");
        _sparseCloudPostProcessAct = findNamedChild<QAction>(_mainWindow, "actionSparseCloudPostProcess");
        _denseMatchAct = findNamedChild<QAction>(_mainWindow, "actionDenseMatch");
        _depthMapEstimateAct = findNamedChild<QAction>(_mainWindow, "actionDepthMapEstimate");
        _fuseDepthMapsAct = findNamedChild<QAction>(_mainWindow, "actionFuseDepthMaps");
        _refineDenseCloudAct = findNamedChild<QAction>(_mainWindow, "actionRefineDenseCloud");
        _overlapAnalysisAct = findNamedChild<QAction>(_mainWindow, "actionOverlapAnalysis");
        _intersectionCheckAct = findNamedChild<QAction>(_mainWindow, "actionIntersectionCheck");
        _intersectionViewResultsAct = findNamedChild<QAction>(_mainWindow, "actionIntersectionViewResults");
        _manualPointCloudPruneAct = findNamedChild<QAction>(_mainWindow, "actionManualPointCloudPrune");
        _generateMaskAct = findNamedChild<QAction>(_mainWindow, "actionGenerateMask");
        _viewMatchesAct = findNamedChild<QAction>(_mainWindow, "actionViewMatches");
        _viewWorkflowReportAct = findNamedChild<QAction>(_mainWindow, "actionViewWorkflowReport");
        _cameraConvertAct = findNamedChild<QAction>(_mainWindow, "actionCameraConvert");
        _surveyControlAct = findNamedChild<QAction>(_mainWindow, "actionSurveyControl");
        _detectMarkersAct = findNamedChild<QAction>(_mainWindow, "actionDetectMarkers");
        _reviewMarkerDetectionsAct = findNamedChild<QAction>(
            _mainWindow, "actionReviewMarkerDetections");
        _printMarkersAct = findNamedChild<QAction>(_mainWindow, "actionPrintMarkers");
        _markersMenu = findNamedChild<QMenu>(_mainWindow, "menuMarkers");
        _importReferenceDatasetAct = findNamedChild<QAction>(_mainWindow, "actionImportReferenceDataset");
        _referenceQualityCheckAct = findNamedChild<QAction>(_mainWindow, "actionReferenceQualityCheck");
        _referenceTerrainBundleAdjustAct = findNamedChild<QAction>(_mainWindow, "actionReferenceTerrainBundleAdjust");
        if (toolsMenu)
        {
            QAction *firstToolAction = toolsMenu->actions().isEmpty() ? nullptr : toolsMenu->actions().first();
            installTiePointsMenu(toolsMenu, firstToolAction);
        }
        if (!_workflowAerialTriangulationAct)
        {
            QObject *actionParent = workflowMenu
                ? static_cast<QObject *>(workflowMenu)
                : static_cast<QObject *>(_mainWindow);
            _workflowAerialTriangulationAct = new QAction(tr("空中三角测量..."), actionParent);
            _workflowAerialTriangulationAct->setObjectName(
                QStringLiteral("actionWorkflowAerialTriangulation"));
            _workflowAerialTriangulationAct->setToolTip(tr("打开对齐照片参数对话框"));
            if (workflowMenu)
            {
                if (_threeDReconstructionAct)
                {
                    workflowMenu->insertAction(_threeDReconstructionAct, _workflowAerialTriangulationAct);
                }
                else
                {
                    workflowMenu->addAction(_workflowAerialTriangulationAct);
                }
            }
        }
        if (!_generateModelAct)
        {
            QObject *actionParent = workflowMenu
                ? static_cast<QObject *>(workflowMenu)
                : static_cast<QObject *>(_mainWindow);
            _generateModelAct = new QAction(tr("生成模型..."), actionParent);
            _generateModelAct->setObjectName(QStringLiteral("actionGenerateModel"));
            _generateModelAct->setToolTip(tr("选择连接点、点云或已有模型作为源数据生成三维模型"));
            if (workflowMenu)
            {
                if (_createDEMAct)
                {
                    workflowMenu->insertAction(_createDEMAct, _generateModelAct);
                }
                else
                {
                    workflowMenu->addAction(_generateModelAct);
                }
            }
        }
        if (!_toggleCamerasAct)
        {
            QObject *actionParent = viewMenu
                ? static_cast<QObject *>(viewMenu)
                : static_cast<QObject *>(_mainWindow);
            _toggleCamerasAct = new QAction(tr("显示相机"), actionParent);
            _toggleCamerasAct->setObjectName(QStringLiteral("actionToggleCameras"));
            _toggleCamerasAct->setCheckable(true);
            _toggleCamerasAct->setChecked(true);
            _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
            if (viewMenu)
            {
                QAction *before = _featureVisualizationAct ? _featureVisualizationAct : nullptr;
                if (before)
                {
                    viewMenu->insertAction(before, _toggleCamerasAct);
                }
                else
                {
                    viewMenu->addAction(_toggleCamerasAct);
                }
            }
        }
        else
        {
            _toggleCamerasAct->setCheckable(true);
            _toggleCamerasAct->setChecked(true);
            _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
        }
        if (!modelMenu)
        {
            modelMenu = new QMenu(tr("模型"), _mainWindow);
            modelMenu->setObjectName(QStringLiteral("menuModel"));
            if (_mainWindow->menuBar())
            {
                QAction *before = reconstructionMenu
                    ? reconstructionMenu->menuAction()
                    : (toolsMenu ? toolsMenu->menuAction() : nullptr);
                if (before)
                {
                    _mainWindow->menuBar()->insertMenu(before, modelMenu);
                }
                else
                {
                    _mainWindow->menuBar()->addMenu(modelMenu);
                }
            }
        }
        installModelDisplayMenu(modelMenu);
        if (!_cameraConvertAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _cameraConvertAct = new QAction(tr("相机格式转换..."), actionParent);
            _cameraConvertAct->setObjectName(QStringLiteral("actionCameraConvert"));
            _cameraConvertAct->setToolTip(tr("将外部相机文件转换为 PlaScan tsai 和 image_camera.lis"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _cameraConvertAct);
                    toolsMenu->insertSeparator(_viewWorkflowReportAct);
                }
                else
                {
                    toolsMenu->addAction(_cameraConvertAct);
                }
            }
        }
        if (!_generateMaskAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _generateMaskAct = new QAction(tr("生成蒙版..."), actionParent);
            _generateMaskAct->setObjectName(QStringLiteral("actionGenerateMask"));
            _generateMaskAct->setToolTip(tr("根据照片背景或阈值生成蒙版，并在照片视图中显示轮廓"));
            if (toolsMenu)
            {
                if (_cameraConvertAct)
                {
                    toolsMenu->insertAction(_cameraConvertAct, _generateMaskAct);
                }
                else if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _generateMaskAct);
                }
                else
                {
                    toolsMenu->addAction(_generateMaskAct);
                }
            }
        }
        if (!_importReferenceDatasetAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _importReferenceDatasetAct = new QAction(tr("导入参考 DEM/LiDAR..."), actionParent);
            _importReferenceDatasetAct->setObjectName(QStringLiteral("actionImportReferenceDataset"));
            _importReferenceDatasetAct->setToolTip(tr("以外部引用方式导入 DEM、LAS/LAZ/COPC 或点云文件，用于精度检查和后续软约束"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _importReferenceDatasetAct);
                }
                else
                {
                    toolsMenu->addAction(_importReferenceDatasetAct);
                }
            }
        }
        if (!_surveyControlAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _surveyControlAct = new QAction(tr("测绘控制..."), actionParent);
            _surveyControlAct->setObjectName(QStringLiteral("actionSurveyControl"));
            _surveyControlAct->setToolTip(tr("导入并查看控制点、检查点和比例尺约束"));
            if (toolsMenu)
            {
                if (_importReferenceDatasetAct)
                {
                    toolsMenu->insertAction(_importReferenceDatasetAct, _surveyControlAct);
                }
                else if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _surveyControlAct);
                }
                else
                {
                    toolsMenu->addAction(_surveyControlAct);
                }
            }
        }
        if (!_markersMenu && toolsMenu)
        {
            _markersMenu = new QMenu(tr("标记"), toolsMenu);
            _markersMenu->setObjectName(QStringLiteral("menuMarkers"));
            QAction *before = _surveyControlAct
                ? _surveyControlAct
                : (_importReferenceDatasetAct ? _importReferenceDatasetAct : _viewWorkflowReportAct);
            if (before)
            {
                toolsMenu->insertMenu(before, _markersMenu);
            }
            else
            {
                toolsMenu->addMenu(_markersMenu);
            }
        }
        if (!_detectMarkersAct)
        {
            QObject *action_parent = _markersMenu
                ? static_cast<QObject *>(_markersMenu)
                : static_cast<QObject *>(_mainWindow);
            _detectMarkersAct = new QAction(tr("检测标靶..."), action_parent);
            _detectMarkersAct->setObjectName(QStringLiteral("actionDetectMarkers"));
            _detectMarkersAct->setToolTip(tr("在项目照片中后台检测编码或非编码标靶"));
        }
        if (_markersMenu && !_markersMenu->actions().contains(_detectMarkersAct))
        {
            _markersMenu->addAction(_detectMarkersAct);
        }
        if (!_reviewMarkerDetectionsAct)
        {
            QObject *action_parent = _markersMenu
                ? static_cast<QObject *>(_markersMenu)
                : static_cast<QObject *>(_mainWindow);
            _reviewMarkerDetectionsAct = new QAction(tr("复核检测候选..."), action_parent);
            _reviewMarkerDetectionsAct->setObjectName(
                QStringLiteral("actionReviewMarkerDetections"));
            _reviewMarkerDetectionsAct->setToolTip(tr("检查未归并候选和自动检测冲突"));
        }
        if (_markersMenu && !_markersMenu->actions().contains(_reviewMarkerDetectionsAct))
        {
            _markersMenu->addAction(_reviewMarkerDetectionsAct);
        }
        if (!_printMarkersAct)
        {
            QObject *action_parent = _markersMenu
                ? static_cast<QObject *>(_markersMenu)
                : static_cast<QObject *>(_mainWindow);
            _printMarkersAct = new QAction(tr("打印标靶..."), action_parent);
            _printMarkersAct->setObjectName(QStringLiteral("actionPrintMarkers"));
            _printMarkersAct->setToolTip(tr("生成具有明确物理尺寸的编码或非编码标靶 PDF"));
        }
        if (_markersMenu && !_markersMenu->actions().contains(_printMarkersAct))
        {
            _markersMenu->addAction(_printMarkersAct);
        }
        if (!_referenceQualityCheckAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _referenceQualityCheckAct = new QAction(tr("点云/DEM 精度检查..."), actionParent);
            _referenceQualityCheckAct->setObjectName(QStringLiteral("actionReferenceQualityCheck"));
            _referenceQualityCheckAct->setToolTip(tr("根据已导入参考 DEM/LiDAR 和当前 DEM/点云成果生成精度检查报告"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _referenceQualityCheckAct);
                }
                else
                {
                    toolsMenu->addAction(_referenceQualityCheckAct);
                }
            }
        }
        if (!_referenceTerrainBundleAdjustAct)
        {
            QObject *actionParent = toolsMenu
                ? static_cast<QObject *>(toolsMenu)
                : static_cast<QObject *>(_mainWindow);
            _referenceTerrainBundleAdjustAct = new QAction(tr("参考地形约束重新平差..."), actionParent);
            _referenceTerrainBundleAdjustAct->setObjectName(QStringLiteral("actionReferenceTerrainBundleAdjust"));
            _referenceTerrainBundleAdjustAct->setToolTip(tr("检查参考 DEM/LiDAR 是否可作为 BA 软约束，并生成前置检查报告"));
            if (toolsMenu)
            {
                if (_viewWorkflowReportAct)
                {
                    toolsMenu->insertAction(_viewWorkflowReportAct, _referenceTerrainBundleAdjustAct);
                }
                else
                {
                    toolsMenu->addAction(_referenceTerrainBundleAdjustAct);
                }
            }
        }

        if (_exitAct)
        {
            connect(_exitAct, &QAction::triggered, qApp, &QCoreApplication::quit);
        }
        if (auto *aboutAct = findNamedChild<QAction>(_mainWindow, "actionAbout"))
        {
            connect(aboutAct, &QAction::triggered, _mainWindow, [mw = _mainWindow]() {
                AboutDialog dialog(mw);
                dialog.exec();
            });
        }

        _toolBar = findNamedChild<QToolBar>(_mainWindow, "mainToolBar");
        if (_addPhotoAct)
        {
            _addPhotoAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_FileIcon));
        }
        if (_addFolderAct)
        {
            _addFolderAct->setIcon(_mainWindow->style()->standardIcon(QStyle::SP_DirOpenIcon));
        }
        const QList<QAction *> workflowActions{
            _workflowAerialTriangulationAct,
            _threeDReconstructionAct,
            _generateModelAct,
            _createDEMAct,
            _generateOrthoAct
        };
        for (QAction *action : workflowActions)
        {
            if (action)
            {
                action->setIcon(QIcon());
            }
        }
        initializeToolbar();
        installCameraToolbarButton();
        installCameraImageToolbarButton();
        installImageRotationToolbarButtons();
        installImageOverlayToolbarButtons();
        setContextualToolbarVisibility(true, false);
        installViewMenuLayout(viewMenu, windowMenu);

        return;
    }

    // ---- 项目菜单 ----
    // 创建顶级"项目"菜单并依次添加固定动作
    _fileMenu = _mainWindow->menuBar()->addMenu(tr("项目"));
    _newAct   = _fileMenu->addAction(tr("新建项目"));
    _openAct  = _fileMenu->addAction(tr("打开项目"));
    _saveAct  = _fileMenu->addAction(tr("保存项目"));

    // "最近打开"子菜单插入到"保存项目"之前，保持菜单项顺序符合直觉
    _recentMenu = new QMenu(tr("最近打开"), _fileMenu);
    _fileMenu->insertMenu(_saveAct, _recentMenu);

    auto *exportMenu = _fileMenu->addMenu(tr("导出"));
    _exportMatchedPairsAct = exportMenu->addAction(tr("导出匹配对(.lis)"));

    _fileMenu->addSeparator();
    _minimizeAct = _fileMenu->addAction(tr("最小化"));
    // 退出动作直接连接到 QCoreApplication::quit，无需额外连接
    _exitAct = _fileMenu->addAction(tr("退出"), qApp, &QCoreApplication::quit);

    // ---- 视图菜单 ----
    auto *viewMenu = _mainWindow->menuBar()->addMenu(tr("视图"));
    _rotateImageLeftAct = viewMenu->addAction(makeImageRotationToolbarIcon(true), tr("向左旋转"));
    _rotateImageRightAct = viewMenu->addAction(makeImageRotationToolbarIcon(false), tr("向右旋转"));
    _rotateImageLeftAct->setObjectName(QStringLiteral("actionRotateImageLeft"));
    _rotateImageRightAct->setObjectName(QStringLiteral("actionRotateImageRight"));
    _rotateImageLeftAct->setToolTip(tr("向左旋转"));
    _rotateImageRightAct->setToolTip(tr("向右旋转"));
    _rotateImageLeftAct->setEnabled(false);
    _rotateImageRightAct->setEnabled(false);
    _zoomInAct    = viewMenu->addAction(tr("放大"));
    _zoomOutAct   = viewMenu->addAction(tr("缩小"));
    _resetViewAct = viewMenu->addAction(tr("重置视图"));
    _toggleFullScreenAct = new QAction(tr("全屏"), viewMenu);
    _toggleFullScreenAct->setObjectName(QStringLiteral("actionToggleFullScreen"));
    _toggleFullScreenAct->setShortcut(QKeySequence(Qt::Key_F11));
    _toggleFullScreenAct->setToolTip(tr("切换全屏显示"));
    _zoomInAct->setIcon(makeZoomToolbarIcon(true));
    _zoomOutAct->setIcon(makeZoomToolbarIcon(false));
    _zoomInAct->setToolTip(tr("放大"));
    _zoomOutAct->setToolTip(tr("缩小"));
    _zoomInAct->setShortcuts({QKeySequence::ZoomIn});
    _zoomOutAct->setShortcuts({QKeySequence::ZoomOut});
    viewMenu->addSeparator();
    // 轨迹球显示/隐藏切换
    _toggleGizmoAct = new QAction(tr("显示轨迹球"), viewMenu);
    _toggleGizmoAct->setObjectName(QStringLiteral("actionToggleGizmo"));
    _toggleGizmoAct->setCheckable(true);
    _toggleGizmoAct->setChecked(true);  // 默认显示
    _toggleGizmoAct->setToolTip(tr("显示或隐藏 3D 视图中的旋转轨迹球"));
    viewMenu->addAction(_toggleGizmoAct);
    _toggleCamerasAct = new QAction(tr("显示相机"), viewMenu);
    _toggleCamerasAct->setObjectName(QStringLiteral("actionToggleCameras"));
    _toggleCamerasAct->setCheckable(true);
    _toggleCamerasAct->setChecked(true);
    _toggleCamerasAct->setToolTip(tr("显示或隐藏 3D 视图中的相机光心、视锥体和文件名标签"));
    viewMenu->addAction(_toggleCamerasAct);
    _toggleHenanUniversityBrandAct = ensureCheckableAction(
        _mainWindow,
        viewMenu,
        viewMenu,
        QStringLiteral("actionToggleHenanUniversityBrand"),
        tr("河南大学校徽"),
        true);
    _toggleHenanUniversityBrandAct->setToolTip(tr("显示或隐藏主工具栏中的河南大学校徽"));
    viewMenu->addSeparator();
    // 特征点可视化设置对话框入口
    _featureVisualizationAct = viewMenu->addAction(tr("特征点 可视化设置..."));
    _showFeaturePointsAct = ensureCheckableAction(_mainWindow, viewMenu, viewMenu,
                                                  QStringLiteral("actionShowFeaturePoints"),
                                                  tr("显示点"), true,
                                                  _featureVisualizationAct);
    _showFeatureResidualsAct = ensureCheckableAction(_mainWindow, viewMenu, viewMenu,
                                                     QStringLiteral("actionShowFeatureResiduals"),
                                                     tr("显示点残差"), false,
                                                     _featureVisualizationAct);
    _showMaskOverlayAct = ensureCheckableAction(_mainWindow, viewMenu, viewMenu,
                                                QStringLiteral("actionShowMaskOverlay"),
                                                tr("显示蒙版"), true,
                                                _featureVisualizationAct);
    installDepthOverlayActions(viewMenu, viewMenu);
    _showFeaturePointsAct->setIcon(makeFeaturePointsToolbarIcon());
    _showMaskOverlayAct->setIcon(makeMaskToolbarIcon());
    _resetViewAct->setIcon(makeResetViewToolbarIcon());
    _resetViewAct->setToolTip(tr("重置视图"));
    viewMenu->addSeparator();

    // 窗口面板子菜单使用原生 QAction，保持平台菜单的紧凑布局和勾选交互。
    auto *windowMenu = viewMenu->addMenu(tr("窗口"));
    _toggleWorkspaceAct = ensureCheckableAction(_mainWindow,
                                                windowMenu,
                                                windowMenu,
                                                QStringLiteral("actionToggleWorkspace"),
                                                tr("工作区"),
                                                true);
    _toggleWorkspaceAct->setToolTip(tr("显示或隐藏工作区面板"));
    _togglePropertiesAct = ensureCheckableAction(_mainWindow,
                                                 windowMenu,
                                                 windowMenu,
                                                 QStringLiteral("actionToggleProperties"),
                                                 tr("属性"),
                                                 true);
    _togglePropertiesAct->setToolTip(tr("显示或隐藏选择对象属性面板"));
    _togglePhotosAct = ensureCheckableAction(_mainWindow,
                                             windowMenu,
                                             windowMenu,
                                             QStringLiteral("actionTogglePhotos"),
                                             tr("照片"),
                                             true);
    _togglePhotosAct->setToolTip(tr("显示或隐藏照片面板"));
    _toggleLogAct = new QAction(tr("日志"), windowMenu);
    _toggleLogAct->setCheckable(true);  // 可切换：勾选时面板可见
    _toggleLogAct->setChecked(false);
    _toggleLogAct->setObjectName(QStringLiteral("actionToggleLog"));
    _toggleLogAct->setToolTip(tr("显示或隐藏日志面板"));
    _toggleMainToolbarAct = new QAction(tr("主工具栏"), windowMenu);
    _toggleMainToolbarAct->setObjectName(QStringLiteral("actionToggleMainToolbar"));
    _toggleMainToolbarAct->setCheckable(true);
    _toggleMainToolbarAct->setChecked(true);
    _toggleMainToolbarAct->setToolTip(tr("显示或隐藏主工具栏"));

    // ---- 工作流程菜单 ----
    // 提供高层一键式处理流程入口，适合不需要分步调试的普通用户
    auto *workflowMenu = _mainWindow->menuBar()->addMenu(tr("工作流程"));
    _addPhotoAct       = workflowMenu->addAction(tr("添加 照片"));
    _addFolderAct      = workflowMenu->addAction(tr("添加 文件夹"));
    workflowMenu->addSeparator();
    _workflowAerialTriangulationAct = workflowMenu->addAction(tr("空中三角测量...")); // 对齐照片参数对话框
    _threeDReconstructionAct = workflowMenu->addAction(tr("三维重建"));     // 一键完整建模流程
    _generateModelAct = workflowMenu->addAction(tr("生成模型..."));        // Metashape 风格源数据选择
    _createDEMAct      = workflowMenu->addAction(tr("创建 DEM"));          // DEM 完整流程
    _generateOrthoAct  = workflowMenu->addAction(tr("生成 正射影像"));     // 正射影像完整流程

    _modelMenu = _mainWindow->menuBar()->addMenu(tr("模型"));
    _modelMenu->setObjectName(QStringLiteral("menuModel"));
    installModelDisplayMenu(_modelMenu);

    // ---- 重建菜单 ----
    // 重建菜单只保留稀疏、密集阶段；模型统一从“工作流程 → 生成模型”进入。
    auto *reconMenu = _mainWindow->menuBar()->addMenu(tr("重建"));

    // ── 稀疏重建 ──
    auto *sparseReconMenu = reconMenu->addMenu(tr("稀疏重建"));
    _detectFeaturesAct = sparseReconMenu->addAction(tr("特征点提取"));
    _vocabularyOverlapAct = sparseReconMenu->addAction(tr("重叠对规划..."));
    _matchFeaturesAct  = sparseReconMenu->addAction(tr("连接点匹配"));
    _aerialTriangulationAct = sparseReconMenu->addAction(tr("空中三角测量..."));
    _sparseCloudPostProcessAct = sparseReconMenu->addAction(tr("稀疏点云后处理..."));

    sparseReconMenu->addSeparator();
    auto *advancedSparseMenu = sparseReconMenu->addMenu(tr("高级工具"));
    _viewMatchesAct = advancedSparseMenu->addAction(tr("查看匹配"));
    _buildObsNetworkAct = advancedSparseMenu->addAction(tr("构建观测网络..."));
    _initCameraPoseAct = advancedSparseMenu->addAction(tr("初始化相机位姿..."));
    _triangulateAct = advancedSparseMenu->addAction(tr("生成两视预览云..."));
    _reconBundleAdjustAct = advancedSparseMenu->addAction(tr("单独光束法平差..."));

    // ── 密集重建 ──
    auto *denseReconMenu = reconMenu->addMenu(tr("密集重建"));
    _denseMatchAct       = denseReconMenu->addAction(tr("密集匹配..."));
    _depthMapEstimateAct = denseReconMenu->addAction(tr("深度图估计..."));
    _fuseDepthMapsAct    = denseReconMenu->addAction(tr("深度图融合生成密集点云..."));
    _refineDenseCloudAct = denseReconMenu->addAction(tr("密集点云后处理..."));

    // ---- 工具菜单 ----
    // 提供细粒度的单步工具入口，供高级用户和调试场景使用
    auto *toolsMenu = _mainWindow->menuBar()->addMenu(tr("工具"));

    installTiePointsMenu(toolsMenu);
    toolsMenu->addSeparator();

    _markersMenu = toolsMenu->addMenu(tr("标记"));
    _markersMenu->setObjectName(QStringLiteral("menuMarkers"));
    _detectMarkersAct = _markersMenu->addAction(tr("检测标靶..."));
    _detectMarkersAct->setObjectName(QStringLiteral("actionDetectMarkers"));
    _detectMarkersAct->setToolTip(tr("在项目照片中后台检测编码或非编码标靶"));
    _reviewMarkerDetectionsAct = _markersMenu->addAction(tr("复核检测候选..."));
    _reviewMarkerDetectionsAct->setObjectName(
        QStringLiteral("actionReviewMarkerDetections"));
    _reviewMarkerDetectionsAct->setToolTip(tr("检查未归并候选和自动检测冲突"));
    _printMarkersAct = _markersMenu->addAction(tr("打印标靶..."));
    _printMarkersAct->setObjectName(QStringLiteral("actionPrintMarkers"));
    _printMarkersAct->setToolTip(tr("生成具有明确物理尺寸的编码或非编码标靶 PDF"));

    // 质量检查工具
    _overlapAnalysisAct = toolsMenu->addAction(tr("重叠度获取"));
    QMenu *intersectionMenu = toolsMenu->addMenu(tr("前方交汇精度检验"));
    _intersectionCheckAct = intersectionMenu->addAction(tr("检测交汇"));
    _intersectionViewResultsAct = intersectionMenu->addAction(tr("查看结果"));
    toolsMenu->addSeparator();
    _manualPointCloudPruneAct = toolsMenu->addAction(tr("手动点云剔除"));
    _generateMaskAct = toolsMenu->addAction(tr("生成蒙版..."));
    _generateMaskAct->setObjectName(QStringLiteral("actionGenerateMask"));
    _generateMaskAct->setToolTip(tr("根据照片背景或阈值生成蒙版，并在照片视图中显示轮廓"));

    // 相机格式转换：外部 benchmark/摄影测量相机 -> tsai + image_camera.lis
    toolsMenu->addSeparator();
    _cameraConvertAct = toolsMenu->addAction(tr("相机格式转换..."));

    // 参考数据：外部 DEM/LiDAR 只登记引用，用于精度检查和后续 BA 软约束
    _surveyControlAct = toolsMenu->addAction(tr("测绘控制..."));
    _surveyControlAct->setObjectName(QStringLiteral("actionSurveyControl"));
    _surveyControlAct->setToolTip(tr("导入并查看控制点、检查点和比例尺约束"));
    _importReferenceDatasetAct = toolsMenu->addAction(tr("导入参考 DEM/LiDAR..."));
    _importReferenceDatasetAct->setObjectName(QStringLiteral("actionImportReferenceDataset"));
    _importReferenceDatasetAct->setToolTip(tr("以外部引用方式导入 DEM、LAS/LAZ/COPC 或点云文件，用于精度检查和后续软约束"));
    _referenceQualityCheckAct = toolsMenu->addAction(tr("点云/DEM 精度检查..."));
    _referenceQualityCheckAct->setObjectName(QStringLiteral("actionReferenceQualityCheck"));
    _referenceQualityCheckAct->setToolTip(tr("根据已导入参考 DEM/LiDAR 和当前 DEM/点云成果生成精度检查报告"));
    _referenceTerrainBundleAdjustAct = toolsMenu->addAction(tr("参考地形约束重新平差..."));
    _referenceTerrainBundleAdjustAct->setObjectName(QStringLiteral("actionReferenceTerrainBundleAdjust"));
    _referenceTerrainBundleAdjustAct->setToolTip(tr("检查参考 DEM/LiDAR 是否可作为 BA 软约束，并生成前置检查报告"));

    // 报告：查看各工作流程的历史统计报告
    toolsMenu->addSeparator();
    _viewWorkflowReportAct = toolsMenu->addAction(tr("查看工作流程报告..."));

    // ---- 帮助菜单 ----
    auto *helpMenu = _mainWindow->menuBar()->addMenu(tr("帮助"));
    helpMenu->addAction(tr("关于"), _mainWindow, [mw = _mainWindow]() {
        AboutDialog dialog(mw);
        dialog.exec();
    });

    // ---- 主工具栏 ----
    _toolBar = _mainWindow->addToolBar(tr("工具"));
    if (_toolBar)
    {
        initializeToolbar();
        installCameraToolbarButton();
        installCameraImageToolbarButton();
        installImageRotationToolbarButtons();
        installImageOverlayToolbarButtons();
        setContextualToolbarVisibility(true, false);
    }
    installViewMenuLayout(viewMenu, windowMenu);
}

/** @brief 析构函数（默认实现，所有成员由 Qt 对象树释放）。 */
MainMenu::~MainMenu() = default;

void MainMenu::setContextualToolbarVisibility(bool showModelTools, bool showImageTools)
{
    _imageToolsVisible = showImageTools;
    updateImageActionAvailability();

    if (!_toolBar)
    {
        return;
    }

    auto setButtonVisible = [this](QAction *toolbarAction, bool visible)
    {
        if (!toolbarAction)
        {
            return;
        }
        const bool isInToolbar = _toolBar->actions().contains(toolbarAction);
        if (!visible)
        {
            if (isInToolbar)
            {
                _toolBar->removeAction(toolbarAction);
            }
            return;
        }
        if (isInToolbar)
        {
            toolbarAction->setVisible(true);
            if (QWidget *widget = _toolBar->widgetForAction(toolbarAction))
            {
                widget->setVisible(true);
            }
            return;
        }
        if (_toolbarEditingSeparatorAct && _toolBar->actions().contains(_toolbarEditingSeparatorAct))
        {
            _toolBar->insertAction(_toolbarEditingSeparatorAct, toolbarAction);
        }
        else
        {
            _toolBar->addAction(toolbarAction);
        }
        toolbarAction->setVisible(true);
        if (QWidget *widget = _toolBar->widgetForAction(toolbarAction))
        {
            widget->setVisible(true);
        }
    };

    setButtonVisible(_cameraToolbarWidgetAct, showModelTools);
    setButtonVisible(_cameraImageToolbarWidgetAct, showModelTools);
    setButtonVisible(_rotateImageLeftToolbarWidgetAct, showImageTools);
    setButtonVisible(_rotateImageRightToolbarWidgetAct, showImageTools);
    setButtonVisible(_resetImageViewToolbarWidgetAct, showImageTools);
    setButtonVisible(_showFeaturePointsToolbarWidgetAct, showImageTools);
    setButtonVisible(_showMaskOverlayToolbarWidgetAct, showImageTools);
    setButtonVisible(_showDepthOverlayToolbarWidgetAct, showImageTools);
    const bool zoomEnabled = showModelTools || showImageTools;
    _zoomInAct->setEnabled(zoomEnabled);
    _zoomOutAct->setEnabled(zoomEnabled);
}

void MainMenu::setImageDisplayReady(bool ready)
{
    _imageDisplayReady = ready;
    updateImageActionAvailability();
}

void MainMenu::setDepthOverlayAvailable(bool available)
{
    _depthOverlayAvailable = available;
    if (!available)
    {
        _depthOverlayFinalAvailable = false;
        _depthOverlayLevel1Available = false;
        _depthOverlayLevel2Available = false;
        _depthOverlayLevel3Available = false;
    }
    updateImageActionAvailability();
}

void MainMenu::setDepthOverlayLevelsAvailable(bool finalAvailable,
                                              bool level1Available,
                                              bool level2Available,
                                              bool level3Available,
                                              const QString &finalReason,
                                              const QString &level1Reason,
                                              const QString &level2Reason,
                                              const QString &level3Reason)
{
    _depthOverlayFinalAvailable = finalAvailable;
    _depthOverlayLevel1Available = level1Available;
    _depthOverlayLevel2Available = level2Available;
    _depthOverlayLevel3Available = level3Available;
    const auto updateUnavailableHint = [](QAction *action,
                                          bool available,
                                          const QString &levelLabel,
                                          const QString &reason)
    {
        if (!action)
        {
            return;
        }
        const QString hint = available
            ? QString()
            : (reason.trimmed().isEmpty()
                ? QStringLiteral("%1栅格未保存；请重新生成深度图并保存该级别的可视化栅格。")
                      .arg(levelLabel)
                : reason);
        action->setToolTip(hint);
        action->setStatusTip(hint);
    };
    updateUnavailableHint(_depthOverlayAllLevelsAct,
                          finalAvailable,
                          QStringLiteral("最终层"),
                          finalReason);
    updateUnavailableHint(_depthOverlayLevel1Act,
                          level1Available,
                          QStringLiteral("Level 1"),
                          level1Reason);
    updateUnavailableHint(_depthOverlayLevel2Act,
                          level2Available,
                          QStringLiteral("Level 2"),
                          level2Reason);
    updateUnavailableHint(_depthOverlayLevel3Act,
                          level3Available,
                          QStringLiteral("Level 3"),
                          level3Reason);
    updateImageActionAvailability();
}

void MainMenu::updateImageActionAvailability()
{
    if (_imageDisplayMenu)
    {
        _imageDisplayMenu->setEnabled(_imageToolsVisible);
    }

    const bool imageReady = _imageToolsVisible && _imageDisplayReady;
    for (QAction *action : {_rotateImageLeftAct,
                            _rotateImageRightAct,
                            _showFeaturePointsAct,
                            _showFeatureResidualsAct,
                            _showMaskOverlayAct})
    {
        if (action)
        {
            action->setEnabled(imageReady);
        }
    }

    if (_featureVisualizationAct)
    {
        _featureVisualizationAct->setEnabled(_imageToolsVisible);
    }

    const bool depthReady = imageReady && _depthOverlayAvailable;
    for (QAction *action : {_showDepthOverlayAct, _showDepthIntensityAct})
    {
        if (action)
        {
            action->setEnabled(depthReady);
        }
    }
    if (_depthOverlayAllLevelsAct)
    {
        _depthOverlayAllLevelsAct->setEnabled(depthReady && _depthOverlayFinalAvailable);
    }
    if (_depthOverlayLevel1Act)
    {
        _depthOverlayLevel1Act->setEnabled(depthReady && _depthOverlayLevel1Available);
    }
    if (_depthOverlayLevel2Act)
    {
        _depthOverlayLevel2Act->setEnabled(depthReady && _depthOverlayLevel2Available);
    }
    if (_depthOverlayLevel3Act)
    {
        _depthOverlayLevel3Act->setEnabled(depthReady && _depthOverlayLevel3Available);
    }
}

// ============================================================
//  最近打开子菜单的动态重建
// ============================================================

/**
 * @brief 根据路径列表动态重建"最近打开"子菜单。
 *
 * 每次调用时先清空旧菜单项，然后：
 * 1. 为每条路径创建编号动作，点击时 emit recentProjectSelected；
 * 2. 若列表为空则添加灰色"（无）"占位项；
 * 3. 末尾添加"清空最近打开"分隔项，可用时（列表非空）才启用。
 *
 * @param paths 最近打开的项目路径列表。
 */
void MainMenu::setRecentProjects(const QStringList &paths)
{
    if (!_recentMenu) return;
    _recentMenu->clear(); // 清除旧的菜单项

    int idx = 0;
    for (const QString &p : paths) {
        ++idx;
        // 转换为平台原生路径分隔符，并取绝对路径，使显示更友好
        QString abs = QDir::toNativeSeparators(QFileInfo(p).absoluteFilePath());
        // 菜单项格式：序号. 路径（例如 "1. /home/user/project.plascan"）
        QAction *a = _recentMenu->addAction(QStringLiteral("%1. %2").arg(idx).arg(abs));
        a->setToolTip(abs); // 鼠标悬停时显示完整路径
        // 点击时以 lambda 捕获路径，emit 信号通知主窗口打开项目
        connect(a, &QAction::triggered, this, [this, abs]() { emit recentProjectSelected(abs); });
    }

    if (paths.isEmpty()) {
        // 无记录时显示灰色占位项，防止子菜单空白
        auto *na = _recentMenu->addAction(tr("（无）"));
        na->setEnabled(false);
    }

    _recentMenu->addSeparator();
    auto *clearAct = _recentMenu->addAction(tr("清空最近打开"));
    // 只有列表非空时才允许清空操作，防止用户对空列表执行无意义操作
    clearAct->setEnabled(!paths.isEmpty());
    connect(clearAct, &QAction::triggered, this, [this]() { emit clearRecentRequested(); });
}

// ============================================================
//  访问器方法实现（均为简单的指针返回，无逻辑）
// ============================================================

QAction *MainMenu::toggleLogAction() const   { return _toggleLogAct; }
QAction *MainMenu::toggleMainToolbarAction() const { return _toggleMainToolbarAct; }
QToolBar *MainMenu::toolBar() const          { return _toolBar; }

QAction *MainMenu::newAction() const  { return _newAct; }
QAction *MainMenu::openAction() const { return _openAct; }
QAction *MainMenu::saveAction() const { return _saveAct; }
QAction *MainMenu::minimizeAction() const { return _minimizeAct; }
QAction *MainMenu::exitAction() const { return _exitAct; }

QAction *MainMenu::zoomInAction() const    { return _zoomInAct; }
QAction *MainMenu::zoomOutAction() const   { return _zoomOutAct; }
QAction *MainMenu::resetViewAction() const { return _resetViewAct; }
QAction *MainMenu::toggleFullScreenAction() const { return _toggleFullScreenAct; }
QAction *MainMenu::rotateImageLeftAction() const { return _rotateImageLeftAct; }
QAction *MainMenu::rotateImageRightAction() const { return _rotateImageRightAct; }
QAction *MainMenu::showFeaturePointsAction() const { return _showFeaturePointsAct; }
QAction *MainMenu::showFeatureResidualsAction() const { return _showFeatureResidualsAct; }
QAction *MainMenu::showMaskOverlayAction() const { return _showMaskOverlayAct; }
QAction *MainMenu::showDepthOverlayAction() const { return _showDepthOverlayAct; }
QAction *MainMenu::depthOverlayAllLevelsAction() const { return _depthOverlayAllLevelsAct; }
QAction *MainMenu::depthOverlayLevel1Action() const { return _depthOverlayLevel1Act; }
QAction *MainMenu::depthOverlayLevel2Action() const { return _depthOverlayLevel2Act; }
QAction *MainMenu::depthOverlayLevel3Action() const { return _depthOverlayLevel3Act; }
QAction *MainMenu::showDepthIntensityAction() const { return _showDepthIntensityAct; }
QAction *MainMenu::toggleGizmoAction() const { return _toggleGizmoAct; }
QAction *MainMenu::toggleCamerasAction() const { return _toggleCamerasAct; }
QAction *MainMenu::toggleDependentCamerasAction() const { return _toggleDependentCamerasAct; }
QAction *MainMenu::toggleCameraThumbnailsAction() const { return _toggleCameraThumbnailsAct; }
QAction *MainMenu::toggleLocalAxesAction() const
{
    return _toggleLocalAxesAct;
}
QAction *MainMenu::toggleCameraImagesAction() const { return _toggleCameraImagesAct; }
QAction *MainMenu::showCameraImagesInForegroundAction() const { return _showCameraImagesInForegroundAct; }
QAction *MainMenu::showCameraImagesInBackgroundAction() const { return _showCameraImagesInBackgroundAct; }
QAction *MainMenu::lockCameraImageAction() const { return _lockCameraImageAct; }
QAction *MainMenu::toggleHenanUniversityBrandAction() const { return _toggleHenanUniversityBrandAct; }

QAction *MainMenu::addPhotoAction() const       { return _addPhotoAct; }
QAction *MainMenu::addFolderAction() const      { return _addFolderAct; }
QAction *MainMenu::detectFeaturesAction() const { return _detectFeaturesAct; }
QAction *MainMenu::vocabularyOverlapAction() const { return _vocabularyOverlapAct; }
QAction *MainMenu::featureVisualizationAction() const { return _featureVisualizationAct; }
QAction *MainMenu::matchFeaturesAction() const  { return _matchFeaturesAct; }
QAction *MainMenu::viewMatchesAction() const    { return _viewMatchesAct; }
QAction *MainMenu::workflowAerialTriangulationAction() const { return _workflowAerialTriangulationAct; }
QAction *MainMenu::threeDReconstructionAction() const { return _threeDReconstructionAct; }
QAction *MainMenu::overlapAnalysisAction() const { return _overlapAnalysisAct; }
QAction *MainMenu::intersectionCheckAction() const { return _intersectionCheckAct; }
QAction *MainMenu::intersectionViewResultsAction() const { return _intersectionViewResultsAct; }
QAction *MainMenu::createDEMAction() const      { return _createDEMAct; }
QAction *MainMenu::generateOrthoAction() const  { return _generateOrthoAct; }
QAction *MainMenu::generateModelAction() const  { return _generateModelAct; }

QAction *MainMenu::viewWorkflowReportAction() const         { return _viewWorkflowReportAct; }
QAction *MainMenu::createTiePointsAction() const           { return _createTiePointsAct; }
QAction *MainMenu::thinTiePointsAction() const             { return _thinTiePointsAct; }
QAction *MainMenu::cleanTiePointsAction() const            { return _cleanTiePointsAct; }
QAction *MainMenu::viewTiePointMatchesAction() const       { return _viewTiePointMatchesAct; }
QAction *MainMenu::manualPointCloudPruneAction() const      { return _manualPointCloudPruneAct; }
QAction *MainMenu::cameraConvertAction() const              { return _cameraConvertAct; }
QAction *MainMenu::generateMaskAction() const               { return _generateMaskAct; }
QAction *MainMenu::surveyControlAction() const              { return _surveyControlAct; }
QAction *MainMenu::detectMarkersAction() const               { return _detectMarkersAct; }
QAction *MainMenu::reviewMarkerDetectionsAction() const      { return _reviewMarkerDetectionsAct; }
QAction *MainMenu::printMarkersAction() const                { return _printMarkersAct; }
QAction *MainMenu::importReferenceDatasetAction() const     { return _importReferenceDatasetAct; }
QAction *MainMenu::referenceQualityCheckAction() const      { return _referenceQualityCheckAct; }
QAction *MainMenu::referenceTerrainBundleAdjustAction() const { return _referenceTerrainBundleAdjustAct; }

QAction *MainMenu::buildObsNetworkAction() const     { return _buildObsNetworkAct; }
QAction *MainMenu::initCameraPoseAction() const      { return _initCameraPoseAct; }
QAction *MainMenu::aerialTriangulationAction() const { return _aerialTriangulationAct; }
QAction *MainMenu::triangulateAction() const         { return _triangulateAct; }
QAction *MainMenu::reconBundleAdjustAction() const   { return _reconBundleAdjustAct; }
QAction *MainMenu::sparseCloudPostProcessAction() const { return _sparseCloudPostProcessAct; }
QAction *MainMenu::depthMapEstimateAction() const    { return _depthMapEstimateAct; }
QAction *MainMenu::fuseDepthMapsAction() const       { return _fuseDepthMapsAct; }
QAction *MainMenu::refineDenseCloudAction() const    { return _refineDenseCloudAct; }
QAction *MainMenu::exportMatchedPairsAction() const  { return _exportMatchedPairsAct; }

QAction *MainMenu::denseMatchAction() const { return _denseMatchAct; }
QAction *MainMenu::toggleWorkspaceAction() const { return _toggleWorkspaceAct; }
QAction *MainMenu::togglePropertiesAction() const { return _togglePropertiesAct; }
QAction *MainMenu::togglePhotosAction() const { return _togglePhotosAct; }
