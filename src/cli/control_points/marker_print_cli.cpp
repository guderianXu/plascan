#include "cli_common.h"

#include "detection/MarkerDetectorFactory.h"
#include "print/MarkerPdfWriter.h"

#include <QDir>
#include <QGuiApplication>
#include <QSizeF>

#include <cstdio>
#include <string>

namespace
{

QVector<int> parseIds(const std::string &text)
{
    const QStringList parts = QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()))
                                  .split(QLatin1Char(','), Qt::SkipEmptyParts);
    QVector<int> ids;
    ids.reserve(parts.size());
    for (const QString &part : parts)
    {
        bool ok = false;
        const int id = part.trimmed().toInt(&ok);
        if (!ok || id < 0)
        {
            cli::fatal("--ids 必须是逗号分隔的非负整数", cli::EXIT_ARG_ERR);
        }
        ids.push_back(id);
    }
    if (ids.isEmpty())
    {
        cli::fatal("--ids 不能为空", cli::EXIT_ARG_ERR);
    }
    return ids;
}

} // namespace

int main(int argc, char *argv[])
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
    {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }
    QGuiApplication application(argc, argv);
    CLI::App app{"PlaScan 标靶 PDF 生成工具"};
    cli::configureApp(app);

    std::string family_name;
    std::string ids_text;
    std::string output_path;
    std::string page_size = "a4";
    double diameter_mm = 30.0;
    double margin_mm = 12.0;
    double spacing_mm = 8.0;
    int dpi = 300;
    bool hide_labels = false;

    app.add_option("--family", family_name, "标靶族，例如 tag36h11")->required();
    app.add_option("--ids", ids_text, "逗号分隔的标靶编号")->required();
    app.add_option("--output", output_path, "输出 PDF 路径")->required();
    app.add_option("--diameter-mm", diameter_mm, "标靶外径，单位毫米");
    app.add_option("--page-size", page_size, "页面尺寸: a4 或 letter");
    app.add_option("--margin-mm", margin_mm, "页面边距，单位毫米");
    app.add_option("--spacing-mm", spacing_mm, "标靶间距，单位毫米");
    app.add_option("--dpi", dpi, "PDF 栅格绘制分辨率");
    app.add_flag("--hide-labels", hide_labels, "不打印标靶编号标签");
    CLI11_PARSE(app, argc, argv);

    const auto family = xjw::control_points::MarkerDetectorFactory::parseFamily(
        QString::fromUtf8(family_name.data(), static_cast<qsizetype>(family_name.size())));
    if (!family.has_value())
    {
        cli::fatal("不支持的标靶族: " + family_name, cli::EXIT_ARG_ERR);
    }
    if (diameter_mm <= 0.0 || margin_mm < 0.0 || spacing_mm < 0.0 || dpi < 72)
    {
        cli::fatal("物理尺寸必须非负且 DPI 不得低于 72", cli::EXIT_ARG_ERR);
    }

    xjw::control_points::MarkerPrintRequest request;
    request.family = *family;
    request.ids = parseIds(ids_text);
    request.targetDiameterMm = diameter_mm;
    request.marginMm = margin_mm;
    request.spacingMm = spacing_mm;
    request.showLabels = !hide_labels;
    const QString normalized_page = QString::fromUtf8(
        page_size.data(), static_cast<qsizetype>(page_size.size())).trimmed().toLower();
    if (normalized_page == QLatin1String("a4"))
    {
        request.pageSizeMm = QSizeF(210.0, 297.0);
    }
    else if (normalized_page == QLatin1String("letter"))
    {
        request.pageSizeMm = QSizeF(215.9, 279.4);
    }
    else
    {
        cli::fatal("--page-size 只支持 a4 或 letter", cli::EXIT_ARG_ERR);
    }

    const QString output = QDir::cleanPath(QString::fromUtf8(
        output_path.data(), static_cast<qsizetype>(output_path.size())));
    const auto result = xjw::control_points::MarkerPdfWriter::write(request, output, dpi);
    if (!result.ok)
    {
        cli::fatal(result.error.toStdString(), cli::EXIT_ALGO_ERR);
    }
    std::fprintf(stdout,
                 "标靶 PDF 已生成: %s, %d 页\n",
                 output_path.c_str(),
                 result.pageCount);
    return cli::EXIT_OK;
}
