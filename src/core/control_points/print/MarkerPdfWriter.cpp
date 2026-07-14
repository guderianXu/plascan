#include "MarkerPdfWriter.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMarginsF>
#include <QPageLayout>
#include <QPageSize>
#include <QPainter>
#include <QPdfWriter>
#include <QTemporaryFile>

namespace xjw::control_points
{

MarkerPdfWriteResult MarkerPdfWriter::write(const MarkerPrintRequest &request,
                                            const QString &outputPath,
                                            int dpi)
{
    MarkerPdfWriteResult result;
    result.outputPath = QDir::cleanPath(outputPath);
    const MarkerSheetRenderResult rendered = MarkerSheetRenderer::render(request, dpi);
    if (!rendered.ok)
    {
        result.error = rendered.error;
        return result;
    }

    const QFileInfo output_info(result.outputPath);
    if (!QDir().mkpath(output_info.absolutePath()))
    {
        result.error = QStringLiteral("无法创建 PDF 输出目录: %1").arg(output_info.absolutePath());
        return result;
    }
    QString temporary_path;
    {
        QTemporaryFile temporary(QDir(output_info.absolutePath())
                                    .filePath(QStringLiteral(".marker-sheet-XXXXXX.pdf")));
        temporary.setAutoRemove(false);
        if (!temporary.open())
        {
            result.error = QStringLiteral("无法创建 PDF 临时文件");
            return result;
        }
        temporary_path = temporary.fileName();
        temporary.close();
    }

    bool paint_ok = false;
    {
        QPdfWriter writer(temporary_path);
        writer.setResolution(dpi);
        writer.setPageSize(QPageSize(request.pageSizeMm,
                                     QPageSize::Millimeter,
                                     QStringLiteral("PlaScan marker sheet"),
                                     QPageSize::ExactMatch));
        writer.setPageMargins(QMarginsF(), QPageLayout::Millimeter);
        writer.setTitle(QStringLiteral("PlaScan %1 markers")
                            .arg(markerTargetFamilyName(request.family)));
        QPainter painter(&writer);
        paint_ok = painter.isActive();
        if (paint_ok)
        {
            for (int index = 0; index < rendered.pages.size(); ++index)
            {
                if (index > 0 && !writer.newPage())
                {
                    paint_ok = false;
                    break;
                }
                painter.drawImage(QRect(0, 0, writer.width(), writer.height()),
                                  rendered.pages[index]);
            }
            painter.end();
        }
    }
    if (!paint_ok)
    {
        QFile::remove(temporary_path);
        result.error = QStringLiteral("PDF 绘制失败");
        return result;
    }

    if (QFile::exists(result.outputPath) && !QFile::remove(result.outputPath))
    {
        QFile::remove(temporary_path);
        result.error = QStringLiteral("无法替换已有 PDF: %1").arg(result.outputPath);
        return result;
    }
    if (!QFile::rename(temporary_path, result.outputPath))
    {
        QFile::remove(temporary_path);
        result.error = QStringLiteral("无法提交 PDF 输出: %1").arg(result.outputPath);
        return result;
    }

    result.ok = true;
    result.pageCount = rendered.pages.size();
    return result;
}

} // namespace xjw::control_points
