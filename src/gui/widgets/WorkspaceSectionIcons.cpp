#include <array>

#include <QBrush>
#include <QColor>
#include <QHash>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>
#include <QPolygonF>

#include "WorkspaceSectionIcons.h"

namespace xjw::gui::widgets
{
namespace
{

constexpr qreal DesignSize = 20.0;

QPen iconPen(const QColor &color, qreal width = 1.4)
{
    QPen pen(color, width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
    return pen;
}

void drawPhotos(QPainter &painter)
{
    const QColor color(QStringLiteral("#A4A9AE"));
    QPainterPath folder;
    folder.moveTo(1.5, 5.0);
    folder.lineTo(7.7, 5.0);
    folder.lineTo(9.7, 7.0);
    folder.lineTo(18.5, 7.0);
    folder.lineTo(17.3, 17.2);
    folder.lineTo(2.2, 17.2);
    folder.closeSubpath();
    painter.fillPath(folder, color);

    painter.setPen(iconPen(color.darker(112), 1.0));
    painter.drawLine(QPointF(2.0, 7.0), QPointF(17.8, 7.0));
}

void drawMasks(QPainter &painter)
{
    const QColor frameColor(QStringLiteral("#7F8994"));
    painter.setPen(iconPen(frameColor.darker(112), 1.0));
    painter.setBrush(frameColor);
    painter.drawRoundedRect(QRectF(2.0, 2.0, 16.0, 16.0), 2.0, 2.0);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(QStringLiteral("#F7F8FA")));
    painter.drawEllipse(QPointF(10.0, 10.0), 4.2, 4.2);
}

void drawObservationNetwork(QPainter &painter)
{
    const QColor color(QStringLiteral("#4B86C5"));
    const QPointF top(10.0, 2.8);
    const QPointF left(3.7, 15.5);
    const QPointF right(16.3, 15.5);

    painter.setPen(iconPen(color, 1.7));
    painter.drawLine(top, left);
    painter.drawLine(top, right);
    painter.drawLine(left, right);

    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    for (const QPointF &point : {top, left, right})
    {
        painter.drawEllipse(point, 2.35, 2.35);
    }
}

void drawTiePoints(QPainter &painter)
{
    const QColor color(QStringLiteral("#6482A4"));
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    for (const QPointF &point : {
             QPointF(5.0, 5.0), QPointF(15.0, 5.0),
             QPointF(5.0, 15.0), QPointF(15.0, 15.0)})
    {
        painter.drawEllipse(point, 2.8, 2.8);
    }
}

void drawDepthMaps(QPainter &painter)
{
    const QColor color(QStringLiteral("#58A6A0"));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 120));
    painter.drawRoundedRect(QRectF(2.0, 3.0, 12.5, 9.5), 1.6, 1.6);
    painter.setBrush(color);
    painter.drawRoundedRect(QRectF(5.5, 7.0, 12.5, 9.5), 1.6, 1.6);
    painter.setPen(iconPen(color.darker(118), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawRoundedRect(QRectF(5.5, 7.0, 12.5, 9.5), 1.6, 1.6);
}

void drawDenseCloud(QPainter &painter)
{
    const QColor color(QStringLiteral("#6482A4"));
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    const std::array<QPointF, 8> points = {
        QPointF(4.0, 7.0), QPointF(8.0, 3.5), QPointF(13.3, 4.8), QPointF(16.5, 9.0),
        QPointF(5.3, 12.3), QPointF(10.0, 9.6), QPointF(12.2, 15.3), QPointF(17.0, 15.7)
    };
    const std::array<qreal, 8> radii = {1.8, 1.5, 2.0, 1.4, 2.1, 1.6, 1.9, 1.4};
    for (std::size_t i = 0; i < points.size(); ++i)
    {
        painter.drawEllipse(points[i], radii[i], radii[i]);
    }
}

void drawModel3D(QPainter &painter)
{
    const QPointF top(10.0, 1.5);
    const QPointF upperRight(18.0, 6.0);
    const QPointF lowerRight(18.0, 14.0);
    const QPointF bottom(10.0, 18.5);
    const QPointF lowerLeft(2.0, 14.0);
    const QPointF upperLeft(2.0, 6.0);
    const QPointF center(10.0, 10.2);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(QStringLiteral("#9AA3AF")));
    painter.drawPolygon(QPolygonF{top, upperRight, center, upperLeft});
    painter.setBrush(QColor(QStringLiteral("#606A78")));
    painter.drawPolygon(QPolygonF{center, upperRight, lowerRight, bottom});
    painter.setBrush(QColor(QStringLiteral("#828C99")));
    painter.drawPolygon(QPolygonF{upperLeft, center, bottom, lowerLeft});

    painter.setPen(iconPen(QColor(QStringLiteral("#737D8B")), 1.0));
    painter.setBrush(Qt::NoBrush);
    painter.drawPolygon(QPolygonF{top, upperRight, lowerRight, bottom, lowerLeft, upperLeft});
}

void drawDem(QPainter &painter)
{
    const QColor color(QStringLiteral("#7A9563"));
    painter.setPen(iconPen(color, 1.2));
    painter.setBrush(QColor(QStringLiteral("#EDF2E8")));
    painter.drawRoundedRect(QRectF(1.5, 1.5, 17.0, 17.0), 2.0, 2.0);

    painter.setPen(iconPen(color, 1.45));
    painter.setBrush(Qt::NoBrush);
    QPainterPath upper;
    upper.moveTo(3.0, 8.0);
    upper.cubicTo(5.5, 3.0, 8.0, 10.5, 11.0, 5.0);
    upper.cubicTo(13.8, 0.8, 15.2, 7.5, 17.0, 4.0);
    painter.drawPath(upper);

    QPainterPath middle;
    middle.moveTo(3.0, 13.0);
    middle.cubicTo(6.0, 7.0, 8.0, 15.0, 11.5, 9.0);
    middle.cubicTo(14.0, 5.0, 15.8, 11.0, 17.0, 8.5);
    painter.drawPath(middle);

    QPainterPath lower;
    lower.moveTo(3.0, 16.5);
    lower.cubicTo(6.0, 12.0, 8.8, 18.0, 12.0, 13.5);
    lower.cubicTo(14.4, 10.0, 16.2, 14.0, 17.0, 12.0);
    painter.drawPath(lower);
}

void drawOrthomosaic(QPainter &painter)
{
    const QColor color(QStringLiteral("#4D8BAC"));
    painter.setPen(iconPen(color, 1.0));
    painter.setBrush(QColor(QStringLiteral("#E9F3F7")));
    painter.drawRoundedRect(QRectF(1.5, 1.5, 17.0, 17.0), 1.5, 1.5);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(QStringLiteral("#79A8BA")));
    painter.drawRect(QRectF(3.0, 3.0, 6.2, 6.2));
    painter.drawRect(QRectF(10.8, 10.8, 6.2, 6.2));
    painter.setBrush(QColor(QStringLiteral("#A5C8D6")));
    painter.drawRect(QRectF(10.8, 3.0, 6.2, 6.2));
    painter.drawRect(QRectF(3.0, 10.8, 6.2, 6.2));

    QPainterPath terrain;
    terrain.moveTo(3.0, 15.8);
    terrain.lineTo(7.0, 10.5);
    terrain.lineTo(10.0, 13.0);
    terrain.lineTo(13.0, 8.0);
    terrain.lineTo(17.0, 13.2);
    terrain.lineTo(17.0, 17.0);
    terrain.lineTo(3.0, 17.0);
    terrain.closeSubpath();
    painter.fillPath(terrain, QColor(color.red(), color.green(), color.blue(), 205));
}

void drawReferenceData(QPainter &painter)
{
    const QColor color(QStringLiteral("#807A9A"));
    painter.setPen(iconPen(color, 1.5));
    painter.setBrush(QColor(QStringLiteral("#F0EEF4")));
    painter.drawEllipse(QPointF(10.0, 10.0), 7.5, 7.5);
    painter.setBrush(Qt::NoBrush);
    painter.drawEllipse(QPointF(10.0, 10.0), 3.2, 3.2);
    painter.drawLine(QPointF(10.0, 1.0), QPointF(10.0, 6.0));
    painter.drawLine(QPointF(10.0, 14.0), QPointF(10.0, 19.0));
    painter.drawLine(QPointF(1.0, 10.0), QPointF(6.0, 10.0));
    painter.drawLine(QPointF(14.0, 10.0), QPointF(19.0, 10.0));
    painter.setPen(Qt::NoPen);
    painter.setBrush(color);
    painter.drawEllipse(QPointF(10.0, 10.0), 1.3, 1.3);
}

void drawReports(QPainter &painter)
{
    const QColor color(QStringLiteral("#9A7A55"));
    QPainterPath page;
    page.moveTo(4.0, 1.5);
    page.lineTo(12.5, 1.5);
    page.lineTo(17.0, 6.0);
    page.lineTo(17.0, 18.5);
    page.lineTo(4.0, 18.5);
    page.closeSubpath();
    painter.setPen(iconPen(color.darker(110), 1.1));
    painter.setBrush(QColor(QStringLiteral("#F2ECE5")));
    painter.drawPath(page);
    painter.drawLine(QPointF(12.5, 1.8), QPointF(12.5, 6.0));
    painter.drawLine(QPointF(12.5, 6.0), QPointF(16.7, 6.0));

    painter.setPen(iconPen(color, 1.25));
    painter.drawLine(QPointF(6.5, 9.0), QPointF(14.5, 9.0));
    painter.drawLine(QPointF(6.5, 12.0), QPointF(14.5, 12.0));
    painter.drawLine(QPointF(6.5, 15.0), QPointF(12.0, 15.0));
}

void drawUnknown(QPainter &painter)
{
    const QColor color(QStringLiteral("#87919D"));
    painter.setPen(iconPen(color, 1.5));
    painter.setBrush(QColor(QStringLiteral("#EEF0F2")));
    painter.drawRoundedRect(QRectF(2.0, 3.0, 16.0, 14.0), 2.0, 2.0);
    painter.setBrush(color);
    painter.drawEllipse(QPointF(10.0, 10.0), 2.2, 2.2);
}

void drawWorkspaceRoot(QPainter &painter)
{
    const QColor color(QStringLiteral("#7A828B"));
    painter.setPen(iconPen(color, 1.3));
    painter.setBrush(QColor(QStringLiteral("#E9ECEF")));
    painter.drawRect(QRectF(2.0, 2.0, 6.0, 5.0));
    painter.drawRect(QRectF(11.5, 11.0, 6.0, 5.0));
    painter.drawLine(QPointF(5.0, 7.0), QPointF(5.0, 13.5));
    painter.drawLine(QPointF(5.0, 13.5), QPointF(11.5, 13.5));
}

void drawWorkspaceChunk(QPainter &painter)
{
    const QColor dark(QStringLiteral("#8A9097"));
    const QColor light(QStringLiteral("#E8EAEC"));
    painter.setPen(iconPen(dark.darker(115), 0.9));
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            painter.setBrush((row + column) % 2 == 0 ? light : dark);
            painter.drawRect(QRectF(2.0 + column * 4.0,
                                    2.0 + row * 4.0,
                                    4.0,
                                    4.0));
        }
    }
}

void drawWorkspaceImage(QPainter &painter)
{
    const QColor frame(QStringLiteral("#7E848A"));
    painter.setPen(iconPen(frame.darker(112), 1.0));
    painter.setBrush(frame);
    painter.drawRect(QRectF(2.0, 3.0, 16.0, 14.0));
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor(QStringLiteral("#CDD1D5")));
    painter.drawEllipse(QPointF(6.2, 7.0), 1.6, 1.6);
    painter.setBrush(QColor(QStringLiteral("#AEB4BA")));
    painter.drawPolygon(QPolygonF{
        QPointF(3.5, 15.5),
        QPointF(8.0, 10.0),
        QPointF(11.0, 13.0),
        QPointF(14.0, 9.0),
        QPointF(17.0, 15.5)});
}

QPixmap drawSectionPixmap(WorkspaceSection section, int size)
{
    QPixmap pixmap(size, size);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.scale(size / DesignSize, size / DesignSize);

    switch (section)
    {
    case WorkspaceSection::Photos:
        drawPhotos(painter);
        break;
    case WorkspaceSection::Masks:
        drawMasks(painter);
        break;
    case WorkspaceSection::ObservationNetwork:
        drawObservationNetwork(painter);
        break;
    case WorkspaceSection::TiePoints:
        drawTiePoints(painter);
        break;
    case WorkspaceSection::DepthMaps:
        drawDepthMaps(painter);
        break;
    case WorkspaceSection::DenseCloud:
        drawDenseCloud(painter);
        break;
    case WorkspaceSection::Model3D:
        drawModel3D(painter);
        break;
    case WorkspaceSection::Dem:
        drawDem(painter);
        break;
    case WorkspaceSection::Orthomosaic:
        drawOrthomosaic(painter);
        break;
    case WorkspaceSection::ReferenceData:
        drawReferenceData(painter);
        break;
    case WorkspaceSection::Reports:
        drawReports(painter);
        break;
    case WorkspaceSection::Unknown:
        drawUnknown(painter);
        break;
    }

    return pixmap;
}

QIcon drawWorkspaceIcon(void (*draw)(QPainter &))
{
    QIcon icon;
    for (const int size : {16, 18, 20, 24, 32, 36, 48, 64})
    {
        QPixmap pixmap(size, size);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.scale(size / DesignSize, size / DesignSize);
        draw(painter);
        icon.addPixmap(pixmap);
    }
    return icon;
}

} // namespace

QIcon workspaceSectionIcon(WorkspaceSection section)
{
    static QHash<int, QIcon> cache;
    const int key = static_cast<int>(section);
    const auto found = cache.constFind(key);
    if (found != cache.constEnd())
    {
        return found.value();
    }

    QIcon icon;
    for (const int size : {16, 18, 20, 24, 32, 36, 48, 64})
    {
        icon.addPixmap(drawSectionPixmap(section, size));
    }
    cache.insert(key, icon);
    return icon;
}

QIcon workspaceRootIcon()
{
    static const QIcon icon = drawWorkspaceIcon(drawWorkspaceRoot);
    return icon;
}

QIcon workspaceChunkIcon()
{
    static const QIcon icon = drawWorkspaceIcon(drawWorkspaceChunk);
    return icon;
}

QIcon workspaceImageIcon()
{
    static const QIcon icon = drawWorkspaceIcon(drawWorkspaceImage);
    return icon;
}

} // namespace xjw::gui::widgets
