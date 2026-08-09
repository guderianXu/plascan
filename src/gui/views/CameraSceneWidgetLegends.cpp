// =============================================================================
// 文件: CameraSceneWidgetLegends.cpp
// 功能: CameraSceneWidget 的图例和后台加载进度绘制
// =============================================================================
#include "CameraSceneWidget.h"

#include <QLinearGradient>
#include <QPainter>
#include <QtMath>

void CameraSceneWidget::drawTiePointLegend(QPainter &painter) const
{
    if (!_isTiePointCloud || _tiePointColorMode == TiePointColorMode::Color ||
        _cloud.size() == 0)
    {
        return;
    }

    const bool elevationMode = _tiePointColorMode == TiePointColorMode::Elevation;
    const bool imageCountReady =
        _tiePointImageCounts.size() == static_cast<qsizetype>(_cloud.size());
    if (!elevationMode && !imageCountReady)
    {
        const QString status = _tiePointMetadataLoading
            ? tr("影像数：正在读取观测数据...")
            : tr("影像数：%1").arg(
                  _tiePointMetadataError.isEmpty() ? tr("无观测数据")
                                                   : _tiePointMetadataError);
        const QRectF statusRect(24.0, height() - 58.0, 310.0, 30.0);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 255, 255, 220));
        painter.drawRoundedRect(statusRect, 4.0, 4.0);
        painter.setPen(QColor(85, 85, 90));
        painter.drawText(statusRect.adjusted(10.0, 0.0, -8.0, 0.0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         status);
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode ? _tiePointElevationRange : _tiePointImageCountRange;
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const double rampValue = elevationMode ? 1.0 - position : position;
        gradient.setColorAt(position,
                            xjw::gui::tie_points::scalarRampColor(rampValue));
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("连接点 — 高程 (Z)")
                                   : tr("连接点 — 影像数"));

    auto formatValue = [elevationMode](double value)
    {
        if (!elevationMode)
        {
            return QString::number(qRound(value)) + QStringLiteral(" 张");
        }
        return QString::number(value, 'g', 7);
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}
void CameraSceneWidget::drawModelLegend(QPainter &painter) const
{
    const bool elevationMode = _modelColorMode == ModelColorMode::Elevation;
    const bool confidenceMode = _modelColorMode == ModelColorMode::Confidence;
    if (_isTiePointCloud || !_cloud.hasFaces() || (!elevationMode && !confidenceMode))
    {
        return;
    }

    const xjw::gui::tie_points::ScalarRange range =
        elevationMode
        ? _modelElevationRange
        : xjw::gui::tie_points::ScalarRange{1.0, 100.0};
    if (!range.isValid())
    {
        return;
    }

    const qreal legendHeight = qBound<qreal>(112.0, height() * 0.22, 188.0);
    const QRectF panel(22.0, height() - legendHeight - 70.0, 174.0, legendHeight + 48.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.top() + 28.0,
                     20.0,
                     legendHeight);

    painter.setPen(QPen(QColor(205, 205, 210, 190), 1.0));
    painter.setBrush(QColor(255, 255, 255, 220));
    painter.drawRoundedRect(panel, 5.0, 5.0);

    QLinearGradient gradient(bar.topLeft(), bar.bottomLeft());
    constexpr int colorStopCount = 5;
    for (int stopIndex = 0; stopIndex < colorStopCount; ++stopIndex)
    {
        const double position =
            static_cast<double>(stopIndex) / static_cast<double>(colorStopCount - 1);
        const QColor color = elevationMode
            ? xjw::gui::tie_points::scalarRampColor(1.0 - position)
            : xjw::gui::tie_points::imageCountColor(
                  qRound(100.0 - position * 99.0), range);
        gradient.setColorAt(position, color);
    }
    painter.fillRect(bar, gradient);
    painter.setPen(QColor(100, 100, 105));
    painter.drawRect(bar);

    painter.setPen(QColor(50, 50, 55));
    painter.drawText(QRectF(panel.left() + 12.0,
                            panel.top() + 4.0,
                            panel.width() - 24.0,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     elevationMode ? tr("模型 — 高程 (Z)")
                                   : tr("模型 — 可信度"));

    const auto formatValue = [elevationMode](double value)
    {
        return elevationMode
            ? QString::number(value, 'g', 7)
            : QString::number(qRound(value));
    };
    const double middle = (range.minimum + range.maximum) * 0.5;
    const qreal labelLeft = bar.right() + 10.0;
    const qreal labelWidth = panel.right() - labelLeft - 6.0;
    painter.drawText(QRectF(labelLeft, bar.top() - 9.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.maximum));
    painter.drawText(QRectF(labelLeft,
                            bar.center().y() - 10.0,
                            labelWidth,
                            20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(middle));
    painter.drawText(QRectF(labelLeft, bar.bottom() - 11.0, labelWidth, 20.0),
                     Qt::AlignLeft | Qt::AlignVCenter,
                     formatValue(range.minimum));
}

void CameraSceneWidget::drawPlyLoadProgressOverlay(QPainter &painter)
{
    if (!_loading || _plyLoadProgressPercent < 0)
    {
        return;
    }

    const int panelWidth = qMin(width() - 48, 520);
    if (panelWidth <= 160 || height() <= 100)
    {
        return;
    }

    const QRectF panel(24.0, height() - 72.0, panelWidth, 48.0);
    const QRectF bar(panel.left() + 16.0, panel.bottom() - 16.0, panel.width() - 32.0, 6.0);
    const qreal fillWidth = bar.width() * qBound(0, _plyLoadProgressPercent, 100) / 100.0;

    painter.save();
    painter.setPen(QPen(QColor(70, 82, 96, 160), 1.0));
    painter.setBrush(QColor(250, 252, 255, 235));
    painter.drawRoundedRect(panel, 6.0, 6.0);

    painter.setPen(QColor(34, 48, 68));
    const QString title = _plyLoadProgressText.isEmpty()
        ? tr("正在加载密集点云...")
        : _plyLoadProgressText;
    painter.drawText(QRectF(panel.left() + 16.0,
                            panel.top() + 8.0,
                            panel.width() - 96.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignLeft,
                     title);
    painter.drawText(QRectF(panel.right() - 70.0,
                            panel.top() + 8.0,
                            54.0,
                            20.0),
                     Qt::AlignVCenter | Qt::AlignRight,
                     QStringLiteral("%1%").arg(qBound(0, _plyLoadProgressPercent, 100)));

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(218, 226, 238));
    painter.drawRoundedRect(bar, 3.0, 3.0);
    painter.setBrush(QColor(36, 115, 218));
    painter.drawRoundedRect(QRectF(bar.left(), bar.top(), fillWidth, bar.height()), 3.0, 3.0);
    painter.restore();
}

void CameraSceneWidget::drawCameraThumbnailProgressOverlay(QPainter &painter) const
{
    if (!_showCameras || !_showCameraThumbnails || _cameraThumbnailLoadTotal <= 0
        || _cameraThumbnailLoadCompleted >= _cameraThumbnailLoadTotal)
    {
        return;
    }

    const int completed = qBound(
        0, _cameraThumbnailLoadCompleted, _cameraThumbnailLoadTotal);
    const int panel_width = qMin(width() - 48, 360);
    if (panel_width <= 160 || height() <= 100)
    {
        return;
    }

    const QRectF panel(width() - panel_width - 24.0, 24.0, panel_width, 46.0);
    const QRectF bar(panel.left() + 14.0,
                     panel.bottom() - 14.0,
                     panel.width() - 28.0,
                     6.0);
    const qreal fill_width = bar.width()
        * static_cast<qreal>(completed)
        / static_cast<qreal>(_cameraThumbnailLoadTotal);

    painter.save();
    painter.setPen(QPen(QColor(70, 82, 96, 150), 1.0));
    painter.setBrush(QColor(250, 252, 255, 230));
    painter.drawRoundedRect(panel, 6.0, 6.0);
    painter.setPen(QColor(34, 48, 68));
    painter.drawText(panel.adjusted(14.0, 4.0, -14.0, -16.0),
                     Qt::AlignVCenter | Qt::AlignLeft,
                     tr("正在加载相机影像 %1/%2")
                         .arg(completed)
                         .arg(_cameraThumbnailLoadTotal));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(218, 226, 238));
    painter.drawRoundedRect(bar, 3.0, 3.0);
    painter.setBrush(QColor(36, 115, 218));
    painter.drawRoundedRect(
        QRectF(bar.left(), bar.top(), fill_width, bar.height()), 3.0, 3.0);
    painter.restore();
}
