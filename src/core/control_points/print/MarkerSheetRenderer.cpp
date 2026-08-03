#include "MarkerSheetRenderer.h"

#include <apriltag/apriltag.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tagCircle21h7.h>
#include <apriltag/tagStandard41h12.h>
#include <apriltag/tagStandard52h13.h>

#include <QFont>
#include <QPainter>
#include <QPainterPath>
#include <QRect>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace xjw::control_points
{

namespace
{

using FamilyDestroy = void (*)(apriltag_family_t *);
using FamilyHandle = std::unique_ptr<apriltag_family_t, FamilyDestroy>;

int millimetresToPixels(double millimetres, int dpi)
{
    return qRound(millimetres * dpi / 25.4);
}

bool isAprilTag(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::AprilTag16h5
        && family <= MarkerTargetFamily::AprilTagStandard52h13;
}

bool isCircularCoded(MarkerTargetFamily family)
{
    return family >= MarkerTargetFamily::Circular12Bit
        && family <= MarkerTargetFamily::Circular20Bit;
}

FamilyHandle createAprilTagFamily(MarkerTargetFamily family)
{
    switch (family)
    {
    case MarkerTargetFamily::AprilTag16h5:
        return FamilyHandle(tag16h5_create(), tag16h5_destroy);
    case MarkerTargetFamily::AprilTag25h9:
        return FamilyHandle(tag25h9_create(), tag25h9_destroy);
    case MarkerTargetFamily::AprilTag36h10:
        return FamilyHandle(tag36h10_create(), tag36h10_destroy);
    case MarkerTargetFamily::AprilTag36h11:
        return FamilyHandle(tag36h11_create(), tag36h11_destroy);
    case MarkerTargetFamily::AprilTagCircle21h7:
        return FamilyHandle(tagCircle21h7_create(), tagCircle21h7_destroy);
    case MarkerTargetFamily::AprilTagStandard41h12:
        return FamilyHandle(tagStandard41h12_create(), tagStandard41h12_destroy);
    case MarkerTargetFamily::AprilTagStandard52h13:
        return FamilyHandle(tagStandard52h13_create(), tagStandard52h13_destroy);
    default:
        throw std::invalid_argument("Target family is not an AprilTag family");
    }
}

QImage aprilTagImage(MarkerTargetFamily family, int id, int targetPixels)
{
    FamilyHandle native_family = createAprilTagFamily(family);
    if (!native_family || id < 0 || static_cast<uint64_t>(id) >= native_family->ncodes)
    {
        throw std::invalid_argument("AprilTag ID is outside the selected family range");
    }
    const int total_width = native_family->total_width;
    QImage source(total_width, total_width, QImage::Format_Grayscale8);
    source.fill(Qt::black);
    const auto set_white = [&](int x, int y)
    {
        source.scanLine(y)[x] = 255;
    };

    const int white_border_width = native_family->width_at_border +
        (native_family->reversed_border ? 0 : 2);
    const int white_border_start = (total_width - white_border_width) / 2;
    for (int index = 0; index < white_border_width - 1; ++index)
    {
        set_white(white_border_start + index, white_border_start);
        set_white(total_width - 1 - white_border_start, white_border_start + index);
        set_white(white_border_start + index + 1, total_width - 1 - white_border_start);
        set_white(white_border_start, white_border_start + index + 1);
    }

    const int border_start = (total_width - native_family->width_at_border) / 2;
    const std::uint64_t code = native_family->codes[static_cast<std::size_t>(id)];
    for (std::uint32_t bit = 0; bit < native_family->nbits; ++bit)
    {
        if ((code & (std::uint64_t{1} << (native_family->nbits - bit - 1))) != 0)
        {
            set_white(native_family->bit_x[bit] + border_start,
                      native_family->bit_y[bit] + border_start);
        }
    }
    return source.scaled(targetPixels,
                         targetPixels,
                         Qt::IgnoreAspectRatio,
                         Qt::FastTransformation);
}

QImage nonCodedImage(MarkerTargetFamily family, int targetPixels)
{
    QImage image(targetPixels, targetPixels, QImage::Format_Grayscale8);
    image.fill(Qt::white);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(Qt::NoPen);
    const QRectF circle(1.0, 1.0, targetPixels - 2.0, targetPixels - 2.0);
    if (family == MarkerTargetFamily::NonCodedCircle)
    {
        painter.setBrush(Qt::black);
        painter.drawEllipse(circle);
    }
    else
    {
        QPainterPath clipping;
        clipping.addEllipse(circle);
        painter.setClipPath(clipping);
        painter.fillRect(image.rect(), Qt::white);
        painter.setBrush(Qt::black);
        const QPointF center = circle.center();
        painter.drawPie(circle, 0, 90 * 16);
        painter.drawPie(circle, 180 * 16, 90 * 16);
        painter.setClipping(false);
        painter.setPen(QPen(Qt::black, std::max(1.0, targetPixels / 100.0)));
        painter.setBrush(Qt::NoBrush);
        painter.drawEllipse(circle);
        painter.drawLine(QPointF(circle.left(), center.y()), QPointF(circle.right(), center.y()));
        painter.drawLine(QPointF(center.x(), circle.top()), QPointF(center.x(), circle.bottom()));
    }
    return image;
}

QImage markerImage(MarkerTargetFamily family, int id, int targetPixels)
{
    if (isAprilTag(family))
    {
        return aprilTagImage(family, id, targetPixels);
    }
    if (family == MarkerTargetFamily::NonCodedCircle
        || family == MarkerTargetFamily::NonCodedFourQuadrant)
    {
        return nonCodedImage(family, targetPixels);
    }
    throw std::invalid_argument("Unsupported printable marker family");
}

} // namespace

QImage MarkerSheetRenderer::renderMarkerImage(MarkerTargetFamily family,
                                               int id,
                                               int targetPixels)
{
    if (targetPixels <= 0)
    {
        throw std::invalid_argument("Marker image size must be positive");
    }
    return markerImage(family, id, targetPixels);
}

MarkerSheetRenderResult MarkerSheetRenderer::render(const MarkerPrintRequest &request, int dpi)
{
    MarkerSheetRenderResult result;
    result.dpi = dpi;
    if (isCircularCoded(request.family))
    {
        result.error = QStringLiteral("尚未安装经许可导出的 Metashape 圆形编码标靶语料");
        return result;
    }
    if (request.ids.isEmpty() || dpi <= 0 || request.targetDiameterMm <= 0.0
        || request.pageSizeMm.width() <= 0.0 || request.pageSizeMm.height() <= 0.0
        || request.marginMm < 0.0 || request.spacingMm < 0.0)
    {
        result.error = QStringLiteral("标靶打印参数无效");
        return result;
    }

    const int page_width = millimetresToPixels(request.pageSizeMm.width(), dpi);
    const int page_height = millimetresToPixels(request.pageSizeMm.height(), dpi);
    const int margin = millimetresToPixels(request.marginMm, dpi);
    const int target_size = millimetresToPixels(request.targetDiameterMm, dpi);
    const int spacing = millimetresToPixels(request.spacingMm, dpi);
    const int label_height = request.showLabels ? millimetresToPixels(6.0, dpi) : 0;
    const int footer_height = millimetresToPixels(7.0, dpi);
    const int usable_width = page_width - 2 * margin;
    const int usable_height = page_height - 2 * margin - footer_height;
    const int cell_width = target_size + spacing;
    const int cell_height = target_size + label_height + spacing;
    const int columns = cell_width > 0 ? (usable_width + spacing) / cell_width : 0;
    const int rows = cell_height > 0 ? (usable_height + spacing) / cell_height : 0;
    if (columns <= 0 || rows <= 0)
    {
        result.error = QStringLiteral("目标尺寸、页边距和间距无法在页面上排入一个标靶");
        return result;
    }

    try
    {
        const int per_page = columns * rows;
        for (int first = 0; first < request.ids.size(); first += per_page)
        {
            QImage page(page_width, page_height, QImage::Format_Grayscale8);
            page.fill(Qt::white);
            QPainter painter(&page);
            painter.setRenderHint(QPainter::Antialiasing, true);
            QFont label_font = painter.font();
            label_font.setPixelSize(std::max(8, millimetresToPixels(3.2, dpi)));
            painter.setFont(label_font);
            painter.setPen(Qt::black);

            const int count = std::min(per_page,
                                       static_cast<int>(request.ids.size()) - first);
            const int occupied_width = columns * target_size + (columns - 1) * spacing;
            const int x_origin = margin + std::max(0, (usable_width - occupied_width) / 2);
            for (int offset = 0; offset < count; ++offset)
            {
                const int row = offset / columns;
                const int column = offset % columns;
                const int x = x_origin + column * cell_width;
                const int y = margin + row * cell_height;
                const int id = request.ids[first + offset];
                painter.drawImage(QRect(x, y, target_size, target_size),
                                  renderMarkerImage(request.family, id, target_size));
                if (request.showLabels)
                {
                    painter.drawText(QRect(x, y + target_size, target_size, label_height),
                                     Qt::AlignCenter,
                                     QStringLiteral("%1 %2")
                                         .arg(markerTargetFamilyName(request.family))
                                         .arg(id));
                }
            }

            QFont footer_font = painter.font();
            footer_font.setPixelSize(std::max(7, millimetresToPixels(2.4, dpi)));
            painter.setFont(footer_font);
            painter.drawText(QRect(margin,
                                   page_height - margin - footer_height,
                                   usable_width,
                                   footer_height),
                             Qt::AlignCenter,
                             QStringLiteral("PlaScan | %1 | target %2 mm | %3 dpi")
                                 .arg(markerTargetFamilyName(request.family))
                                 .arg(request.targetDiameterMm, 0, 'f', 2)
                                 .arg(dpi));
            painter.end();
            result.pages.push_back(std::move(page));
        }
    }
    catch (const std::exception &exception)
    {
        result.pages.clear();
        result.error = QString::fromUtf8(exception.what());
        return result;
    }

    result.ok = !result.pages.isEmpty();
    return result;
}

} // namespace xjw::control_points
