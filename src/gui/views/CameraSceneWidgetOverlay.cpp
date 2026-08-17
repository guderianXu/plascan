// =============================================================================
// 文件: CameraSceneWidgetOverlay.cpp
// 功能: CameraSceneWidget 的 QWidget/QPainter 覆盖层绘制
// =============================================================================
#include "CameraSceneWidget.h"

#include <QFileInfo>
#include <QPainter>
#include <QPainterPath>
#include <QRadialGradient>
#include <QtMath>

#include <cmath>

void CameraSceneWidget::drawRotationGizmo(QPainter &painter) const
{
    if (!_showGizmo)
    {
        return;
    }

    const QPointF center2d = manipCenterScreen();
    const qreal radiusPx = manipRadiusPx();
    QRadialGradient gradient(
        center2d - QPointF(radiusPx * 0.18, radiusPx * 0.18),
        radiusPx * 1.25);
    gradient.setColorAt(0.0, QColor(245, 245, 248, 40));
    gradient.setColorAt(1.0, QColor(175, 178, 186, 28));
    painter.setPen(QPen(QColor(210, 210, 216, 44), 1.0));
    painter.setBrush(gradient);
    painter.drawEllipse(center2d, radiusPx, radiusPx);

    const auto axisPen = [this](HoverAxis axis, const QColor &base)
    {
        const bool highlighted =
            (_hoverAxis == axis) || (_dragAxis == axis && _leftDragging);
        QColor color = base;
        if (highlighted)
        {
            color = color.lighter(150);
        }
        return QPen(color, highlighted ? 4.0 : 2.0);
    };
    const auto drawGreatCircle = [&](HoverAxis axis, const QColor &color)
    {
        painter.setPen(axisPen(axis, color));
        QPointF previous;
        QPointF first;
        bool hasPrevious = false;
        bool previousVisible = false;
        bool firstVisible = false;
        for (int index = 0; index <= 128; ++index)
        {
            const qreal angle = (2.0 * M_PI * index) / 128.0;
            QVector3D localPoint;
            if (axis == HoverAxis::X)
            {
                localPoint = QVector3D(0.0f, float(std::cos(angle)), float(std::sin(angle)));
            }
            else if (axis == HoverAxis::Y)
            {
                localPoint = QVector3D(float(std::cos(angle)), 0.0f, float(std::sin(angle)));
            }
            else
            {
                localPoint = QVector3D(float(std::cos(angle)), float(std::sin(angle)), 0.0f);
            }
            const QVector3D viewPoint = applyViewRotation(localPoint);
            const bool visible = viewPoint.z() > 0.0f;
            const QPointF current = center2d + QPointF(
                viewPoint.x() * radiusPx,
                -viewPoint.y() * radiusPx);
            if (!hasPrevious)
            {
                first = current;
                firstVisible = visible;
            }
            else if (previousVisible && visible)
            {
                painter.drawLine(previous, current);
            }
            previous = current;
            previousVisible = visible;
            hasPrevious = true;
        }
        if (hasPrevious && firstVisible && previousVisible)
        {
            painter.drawLine(previous, first);
        }
    };
    drawGreatCircle(HoverAxis::X, QColor(255, 110, 110, 150));
    drawGreatCircle(HoverAxis::Y, QColor(110, 255, 150, 150));
    drawGreatCircle(HoverAxis::Z, QColor(110, 170, 255, 150));
}

void CameraSceneWidget::paintOverlay(QPainter &painter)
{
    if (!painter.isActive())
    {
        return;
    }

    if (!_renderError.isEmpty())
    {
        painter.fillRect(rect(), Qt::white);
        painter.setPen(QColor(180, 42, 42));
        painter.drawText(rect().adjusted(12, 12, -12, -12),
                         Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap,
                         _renderError);
        return;
    }

    if (cameraImageRegistrationActive())
    {
        const int pose_index = displayedCameraImagePoseIndex();
        const CameraPose &pose = _poses.at(pose_index);
        const QString composition =
            _cameraImageDisplayLayer == CameraImageDisplayLayer::Foreground
            ? tr("原图/网格 50% 表面投影")
            : tr("原图网格表面投影");
        const QString lock_state = _cameraImageLocked
            ? tr("已锁定")
            : tr("自动匹配");
        const QString label = tr("模型视角影像配准 · %1 · %2 · %3")
            .arg(QFileInfo(pose.imagePath).fileName(), lock_state, composition);
        QRectF label_rect(painter.fontMetrics().boundingRect(label));
        label_rect = label_rect.adjusted(-8.0, -5.0, 8.0, 5.0);
        label_rect.translate(10.0, 10.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(25, 31, 42, 184));
        painter.drawRoundedRect(label_rect, 4.0, 4.0);
        painter.setPen(Qt::white);
        painter.drawText(label_rect, Qt::AlignCenter, label);
    }
    else if (_showCameraImage && !_poses.isEmpty())
    {
        QString label;
        if (!_meshHasFaces || _meshIsPointPreview)
        {
            label = tr("影像配准不可用：请加载带三角面的网格模型");
        }
        else if (displayedCameraImagePoseIndex() < 0)
        {
            label = tr("影像配准：当前视角没有方向相近的 SfM 照片，请旋转模型（阈值 30°）");
        }
        else
        {
            label = tr("影像配准：正在载入匹配照片...");
        }
        QRectF label_rect(painter.fontMetrics().boundingRect(label));
        label_rect = label_rect.adjusted(-8.0, -5.0, 8.0, 5.0);
        label_rect.translate(10.0, 10.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(116, 70, 12, 196));
        painter.drawRoundedRect(label_rect, 4.0, 4.0);
        painter.setPen(Qt::white);
        painter.drawText(label_rect, Qt::AlignCenter, label);
    }

    painter.save();
    const bool interactive_camera_motion = isNavigationDragging();

    // 点云完全由 RHI 点图元管线绘制；覆盖层只保留交互控件和文字，
    // 避免在每帧中通过 QPainter 再次遍历全部点。

    if (_showCameras && !_cameraResourceFailed)
    {
        const int labelBudget = maxVisibleCameraLabels();
        const int cameraCount = static_cast<int>(_poses.size());
        const bool drawAllCameraLabels = _poses.size() <= maxVisibleCameraLabels();
        const int cameraLabelStride = drawAllCameraLabels
            ? 1
            : qMax(1, static_cast<int>(std::ceil(double(cameraCount) / double(qMax(1, labelBudget)))));

        if (_poses.isEmpty())
        {
            painter.setPen(QColor(120, 120, 120));
            painter.drawText(rect(), Qt::AlignCenter, tr("暂无相机参数，显示默认模型球"));
        }

        // 相机卡片和方位线已在 RHI 三维场景中批量绘制。覆盖层只保留
        // 少量文件名；拖动时跳过全部文字布局，避免影响轨迹球帧率。
        if (!interactive_camera_motion)
        {
            const QMatrix4x4 camera_model_view = sceneMatrices().modelView;
            for (qsizetype poseIndex = 0; poseIndex < _poses.size(); ++poseIndex)
            {
                const CameraPose &pose = _poses.at(poseIndex);
                const bool highlighted = isCameraHighlighted(pose);
                const bool drawCameraLabel = highlighted
                    || drawAllCameraLabels
                    || poseIndex == 0
                    || poseIndex == _poses.size() - 1
                    || poseIndex % cameraLabelStride == 0;
                if (!drawCameraLabel)
                {
                    continue;
                }

                bool centerOk = false;
                const QPointF center = projectToScreen(pose.center, &centerOk);
                if (!centerOk)
                {
                    continue;
                }
                QVector3D leaderStart;
                QVector3D leaderEnd;
                const float halfExtent = cameraImagePlaneHalfExtent(
                    pose, camera_model_view);
                const bool hasLeader = cameraDirectionLeaderSegment(
                    pose, halfExtent, &leaderStart, &leaderEnd);
                bool leaderEndOk = false;
                const QPointF leaderEndScreen = hasLeader
                    ? projectToScreen(leaderEnd, &leaderEndOk)
                    : QPointF();
                const QPointF labelAnchor = leaderEndOk ? leaderEndScreen : center;
                const bool placeLabelLeft = leaderEndOk && leaderEndScreen.x() < center.x();
                const QString labelSource = pose.imagePath.isEmpty() ? pose.name : pose.imagePath;
                const QString label = QFileInfo(labelSource).fileName().isEmpty()
                    ? pose.name
                    : QFileInfo(labelSource).fileName();
                const qreal labelWidth = painter.fontMetrics().horizontalAdvance(label);
                const QPointF textOffset = placeLabelLeft
                    ? QPointF(-labelWidth - 5.0, -2.0)
                    : QPointF(5.0, -2.0);
                painter.setPen(highlighted
                    ? QColor(210, 45, 65, 230)
                    : (drawAllCameraLabels ? QColor(60, 60, 60) : QColor(45, 45, 45, 170)));
                painter.drawText(labelAnchor + textOffset, label);
            }
        }

    }

    // 操控球是交互前景层，必须在点云和相机标注之后绘制，避免缩小时
    // 被高密度点云遮挡而无法识别或拖动。
    drawRotationGizmo(painter);
    drawFloorPivotCross(painter);
    painter.restore();

    if (!interactive_camera_motion)
    {
        drawTiePointLegend(painter);
        drawModelLegend(painter);
    }

    const QPoint origin(width() - 64, height() - 64);
    const QVector3D ex = applyViewRotation(QVector3D(1, 0, 0)).normalized();
    const QVector3D ey = applyViewRotation(QVector3D(0, 1, 0)).normalized();
    const QVector3D ez = applyViewRotation(QVector3D(0, 0, 1)).normalized();
    auto drawMiniAxis = [&](const QVector3D &dir, const QColor &color, const QString &label) {
        const QPoint end(origin.x() + int(dir.x() * 28.0f), origin.y() - int(dir.y() * 28.0f));
        painter.setPen(QPen(color, 2));
        painter.drawLine(origin, end);
        painter.setPen(color);
        painter.drawText(end + QPoint(4, -2), label);
    };
    painter.setPen(QPen(QColor(80, 80, 80), 1));
    painter.setBrush(QColor(80, 80, 80));
    painter.drawEllipse(QPointF(origin), 2.5, 2.5);
    drawMiniAxis(ex, QColor(210, 50, 50), QStringLiteral("X"));
    drawMiniAxis(ey, QColor(30, 160, 60), QStringLiteral("Y"));
    drawMiniAxis(ez, QColor(40, 100, 220), QStringLiteral("Z"));
    painter.setPen(QColor(100, 100, 110));
    const QVector3D euler = eulerAnglesDeg();
    painter.drawText(origin + QPoint(-84, 26),
                     QStringLiteral("Yaw %1°  Pitch %2°  Roll %3°")
                         .arg(QString::number(euler.y(), 'f', 1))
                         .arg(QString::number(euler.x(), 'f', 1))
                         .arg(QString::number(euler.z(), 'f', 1)));

    if (_manualPruneMode)
    {
        painter.setPen(QPen(QColor(255, 90, 90, 220), 1.5, Qt::DashLine));
        painter.setBrush(QColor(255, 90, 90, 40));
        if (!_manualSelectRect.isNull())
        {
            painter.drawRect(_manualSelectRect.normalized());
        }

        painter.setPen(QColor(235, 80, 80));
        QString manual_status = tr(
            "手动剔除：左键框选，右键/中键平移，中央轨迹球旋转，前进侧键删除，Ctrl+Z 撤销（已选 %1）")
            .arg(static_cast<int>(_manualPreviewIndices.size()));
        if (_manualSelectionRunning)
        {
            manual_status = tr("正在后台计算框选点...");
        }
        else if (_manualEditRunning)
        {
            manual_status = tr("正在后台更新并保存点云...");
        }
        painter.drawText(QPointF(14.0, 24.0), manual_status);
    }

    if (!_renderWarning.isEmpty())
    {
        const QRect warningRect = QRect(14, height() - 112, qMax(120, width() - 150), 44)
            .intersected(rect().adjusted(8, 8, -8, -8));
        painter.setPen(QPen(QColor(166, 102, 20), 1.0));
        painter.setBrush(QColor(255, 244, 214, 232));
        painter.drawRoundedRect(warningRect, 5.0, 5.0);
        painter.setPen(QColor(116, 70, 12));
        painter.drawText(warningRect.adjusted(9, 5, -9, -5),
                         Qt::AlignLeft | Qt::AlignVCenter | Qt::TextWordWrap,
                         _renderWarning);
    }

    drawPlyLoadProgressOverlay(painter);
    drawCameraThumbnailProgressOverlay(painter);
}
