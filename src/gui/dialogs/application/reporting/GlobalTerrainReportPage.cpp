#include "application/reporting/GlobalTerrainReportPage.h"

#include <QColor>
#include <QDir>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QJsonArray>
#include <QLabel>
#include <QPixmap>
#include <QScrollArea>
#include <QSizePolicy>
#include <QStringList>
#include <QVBoxLayout>

#include <cmath>

namespace
{

QString percentText(const QJsonObject &object, const QString &key)
{
    const QJsonValue value = object.value(key);
    if (!value.isDouble() || !std::isfinite(value.toDouble()))
    {
        return QStringLiteral("—");
    }
    return QStringLiteral("%1%").arg(value.toDouble() * 100.0, 0, 'f', 2);
}

QString numberText(const QJsonObject &object,
                   const QString &key,
                   int decimals,
                   const QString &suffix = QString())
{
    const QJsonValue value = object.value(key);
    if (!value.isDouble() || !std::isfinite(value.toDouble()))
    {
        return QStringLiteral("—");
    }
    return QStringLiteral("%1%2").arg(value.toDouble(), 0, 'f', decimals).arg(suffix);
}

QString centerText(const QJsonValue &value)
{
    const QJsonArray center = value.toArray();
    if (center.size() < 3)
    {
        return QStringLiteral("—");
    }
    return QStringLiteral("(%1, %2, %3) m")
        .arg(center.at(0).toDouble(), 0, 'f', 3)
        .arg(center.at(1).toDouble(), 0, 'f', 3)
        .arg(center.at(2).toDouble(), 0, 'f', 3);
}

QWidget *makeMetricCard(const QString &title,
                        const QString &value,
                        const QString &subtitle,
                        const QColor &accent)
{
    auto *card = new QWidget;
    card->setMinimumHeight(88);
    card->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    card->setStyleSheet(QStringLiteral(
        "QWidget{background:white;border-left:4px solid %1;border-radius:4px;}")
                            .arg(accent.name()));

    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(12, 8, 8, 8);
    layout->setSpacing(2);

    auto *title_label = new QLabel(title);
    title_label->setStyleSheet(QStringLiteral(
        "color:#777;background:transparent;border:none;"));
    auto *value_label = new QLabel(value);
    QFont value_font = value_label->font();
    value_font.setPointSize(17);
    value_font.setBold(true);
    value_label->setFont(value_font);
    value_label->setStyleSheet(QStringLiteral(
        "color:%1;background:transparent;border:none;").arg(accent.name()));

    layout->addWidget(title_label);
    layout->addWidget(value_label);
    if (!subtitle.isEmpty())
    {
        auto *subtitle_label = new QLabel(subtitle);
        subtitle_label->setStyleSheet(QStringLiteral(
            "color:#999;background:transparent;border:none;"));
        layout->addWidget(subtitle_label);
    }
    return card;
}

QLabel *makeValueLabel(const QString &value)
{
    auto *label = new QLabel(value.isEmpty() ? QStringLiteral("—") : value);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    label->setWordWrap(true);
    label->setStyleSheet(QStringLiteral("color:#444;"));
    return label;
}

void addDetailRow(QGridLayout *layout, int row, const QString &key, const QString &value)
{
    auto *key_label = new QLabel(key);
    QFont font = key_label->font();
    font.setBold(true);
    key_label->setFont(font);
    key_label->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    layout->addWidget(key_label, row, 0);
    layout->addWidget(makeValueLabel(value), row, 1);
}

QString normalizedConvention(QString value)
{
    return value.trimmed().toLower().replace(QLatin1Char('-'), QLatin1Char('_'))
        .replace(QLatin1Char(' '), QLatin1Char('_'));
}

QStringList coordinateWarnings(const QJsonObject &frame)
{
    QStringList warnings;
    const QString body_frame = frame.value(QStringLiteral("body_fixed_frame")).toString();
    const QString upper_frame = body_frame.trimmed().toUpper();
    if (body_frame.trimmed().isEmpty())
    {
        warnings.append(QStringLiteral("未记录体固连坐标系。"));
    }
    else if (upper_frame.contains(QStringLiteral("J2000"))
             || upper_frame.contains(QStringLiteral("ICRF"))
             || upper_frame.contains(QStringLiteral("INERTIAL")))
    {
        warnings.append(QStringLiteral("坐标系看起来是惯性系，不是体固连坐标系。"));
    }

    if (normalizedConvention(frame.value(QStringLiteral("latitude_type")).toString())
        != QStringLiteral("planetocentric"))
    {
        warnings.append(QStringLiteral("纬度类型不是 planetocentric（行星心纬度）。"));
    }
    if (normalizedConvention(frame.value(QStringLiteral("longitude_direction")).toString())
        != QStringLiteral("positive_east"))
    {
        warnings.append(QStringLiteral("经度方向不是 positive_east（东经为正）。"));
    }

    const QString domain = normalizedConvention(
        frame.value(QStringLiteral("longitude_domain")).toString());
    const bool recognized_domain = domain == QStringLiteral("0_360")
        || domain == QStringLiteral("0..360") || domain == QStringLiteral("[0,360)")
        || domain == QStringLiteral("0_to_360");
    if (!recognized_domain)
    {
        warnings.append(QStringLiteral("经度域不是预期的 0–360°。"));
    }

    const QJsonValue radius = frame.value(QStringLiteral("reference_radius_m"));
    if (!radius.isDouble() || !std::isfinite(radius.toDouble()) || radius.toDouble() <= 0.0)
    {
        warnings.append(QStringLiteral("参考半径缺失或无效。"));
    }
    if (frame.value(QStringLiteral("center_xyz_m")).toArray().size() < 3)
    {
        warnings.append(QStringLiteral("未记录完整的体心坐标。"));
    }
    return warnings;
}

QLabel *makeCoordinateNotice(const QJsonObject &report, const QJsonObject &frame)
{
    QStringList warnings = coordinateWarnings(frame);
    for (const QJsonValue &value : report.value(QStringLiteral("warnings")).toArray())
    {
        const QString warning = value.toString().trimmed();
        if (!warning.isEmpty() && !warnings.contains(warning))
        {
            warnings.append(warning);
        }
    }
    auto *notice = new QLabel;
    notice->setWordWrap(true);
    notice->setTextInteractionFlags(Qt::TextSelectableByMouse);
    if (warnings.isEmpty())
    {
        notice->setText(QStringLiteral(
            "<b>坐标约定检查通过：</b>体固连坐标、行星心纬度、东经为正、0–360° 经度域。"
            "本页只显示核心生成的地图，不执行坐标转换或重新投影。"));
        notice->setStyleSheet(QStringLiteral(
            "QLabel{background:#edf7f0;color:#28643b;border:1px solid #b9ddc4;"
            "border-radius:4px;padding:9px;}"));
        return notice;
    }

    QStringList escaped_warnings;
    for (const QString &warning : warnings)
    {
        escaped_warnings.append(QStringLiteral("• %1").arg(warning.toHtmlEscaped()));
    }
    notice->setText(QStringLiteral(
        "<b>坐标与产品警告：</b><br>%1<br>本页不会尝试校正坐标或重新计算地图。")
                        .arg(escaped_warnings.join(QStringLiteral("<br>"))));
    notice->setStyleSheet(QStringLiteral(
        "QLabel{background:#fff6e6;color:#8a5512;border:1px solid #efcf96;"
        "border-radius:4px;padding:9px;}"));
    return notice;
}

void addArtifactRow(QGridLayout *layout,
                    int row,
                    const QString &name,
                    const QString &path)
{
    auto *name_label = new QLabel(name);
    QFont font = name_label->font();
    font.setBold(true);
    name_label->setFont(font);
    name_label->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    auto *path_label = makeValueLabel(path.isEmpty() ? QStringLiteral("未记录") : path);
    path_label->setToolTip(path);

    auto *status_label = new QLabel;
    if (path.isEmpty())
    {
        status_label->setText(QStringLiteral("未记录"));
        status_label->setStyleSheet(QStringLiteral("color:#999;"));
    }
    else if (QFileInfo::exists(path))
    {
        status_label->setText(QStringLiteral("可用"));
        status_label->setStyleSheet(QStringLiteral("color:#2f7d46;"));
    }
    else
    {
        status_label->setText(QStringLiteral("未找到"));
        status_label->setStyleSheet(QStringLiteral("color:#b45a42;"));
    }
    status_label->setAlignment(Qt::AlignTop | Qt::AlignRight);

    layout->addWidget(name_label, row, 0);
    layout->addWidget(path_label, row, 1);
    layout->addWidget(status_label, row, 2);
}

QString resolveReportPath(const QJsonObject &report, const QString &path)
{
    if (path.trimmed().isEmpty() || QFileInfo(path).isAbsolute())
    {
        return QDir::cleanPath(path);
    }
    const QString report_directory =
        report.value(QStringLiteral("_report_directory")).toString();
    return report_directory.isEmpty()
        ? QDir::cleanPath(path)
        : QDir::cleanPath(QDir(report_directory).absoluteFilePath(path));
}

} // namespace

GlobalTerrainReportPage::GlobalTerrainReportPage(const QJsonObject &report, QWidget *parent)
    : QWidget(parent)
{
    buildUi(report);
}

void GlobalTerrainReportPage::buildUi(const QJsonObject &report)
{
    setStyleSheet(QStringLiteral("background:#f5f5f7;"));
    auto *outer_layout = new QVBoxLayout(this);
    outer_layout->setContentsMargins(0, 0, 0, 0);

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet(QStringLiteral("QScrollArea{background:#f5f5f7;}"));
    outer_layout->addWidget(scroll);

    auto *content = new QWidget;
    content->setStyleSheet(QStringLiteral("background:#f5f5f7;"));
    auto *layout = new QVBoxLayout(content);
    layout->setContentsMargins(16, 16, 16, 24);
    layout->setSpacing(12);
    scroll->setWidget(content);

    if (report.isEmpty())
    {
        auto *empty = new QLabel(QStringLiteral(
            "<br><br><center><b style='font-size:14px;color:#666;'>尚无全球 DEM/DOM 报告</b>"
            "<br><br><span style='color:#999;'>在当前 Chunk 生成全球地形产品后，"
            "对应成果报告会显示在此处。</span></center>"));
        empty->setAlignment(Qt::AlignCenter);
        empty->setWordWrap(true);
        layout->addWidget(empty);
        layout->addStretch();
        return;
    }

    const QJsonObject frame = report.value(QStringLiteral("frame")).toObject();
    const QJsonObject grid = report.value(QStringLiteral("grid")).toObject();
    const QJsonObject metrics = report.value(QStringLiteral("metrics")).toObject();
    const QJsonObject artifacts = report.value(QStringLiteral("artifacts")).toObject();
    const QString target_name = frame.value(QStringLiteral("target_name")).toString(
        QStringLiteral("未命名天体"));
    const QString created_at = report.value(QStringLiteral("created_at")).toString();
    const QString report_type = report.value(QStringLiteral("type")).toString();

    auto *header = new QLabel(QStringLiteral(
        "<b style='font-size:14px;'>%1 · 全球 DEM/DOM 报告</b>"
        "<span style='color:#888;'>　%2</span><br>"
        "<span style='color:#999;font-size:10px;'>报告类型：%3</span>")
                                  .arg(target_name.toHtmlEscaped(),
                                       created_at.isEmpty() ? QStringLiteral("未记录生成时间")
                                                            : created_at.toHtmlEscaped(),
                                       report_type.isEmpty() ? QStringLiteral("未记录")
                                                             : report_type.toHtmlEscaped()));
    header->setStyleSheet(QStringLiteral(
        "QLabel{background:white;border:1px solid #e0e0e0;border-radius:4px;padding:10px;}"));
    layout->addWidget(header);

    auto *metric_layout = new QGridLayout;
    metric_layout->setSpacing(8);
    metric_layout->addWidget(makeMetricCard(
        QStringLiteral("像素覆盖率"),
        percentText(metrics, QStringLiteral("coverage_ratio")),
        QStringLiteral("有效像素 / 全部像素"), QColor(70, 130, 210)), 0, 0);
    metric_layout->addWidget(makeMetricCard(
        QStringLiteral("固体角加权覆盖率"),
        percentText(
            metrics,
            metrics.contains(QStringLiteral("solid_angle_weighted_coverage_ratio"))
                ? QStringLiteral("solid_angle_weighted_coverage_ratio")
                : QStringLiteral("area_weighted_coverage_ratio")),
        QStringLiteral("按 cos(纬度) 加权"), QColor(60, 160, 100)), 0, 1);
    metric_layout->addWidget(makeMetricCard(
        QStringLiteral("歧义比例"),
        percentText(metrics, QStringLiteral("ambiguous_ratio")),
        QStringLiteral("多表面方向歧义"), QColor(200, 120, 70)), 0, 2);

    QString grid_size = QStringLiteral("—");
    if (grid.value(QStringLiteral("width")).isDouble()
        && grid.value(QStringLiteral("height")).isDouble())
    {
        grid_size = QStringLiteral("%1 × %2")
                        .arg(grid.value(QStringLiteral("width")).toInt())
                        .arg(grid.value(QStringLiteral("height")).toInt());
    }
    metric_layout->addWidget(makeMetricCard(
        QStringLiteral("全球栅格"), grid_size, QStringLiteral("宽 × 高"),
        QColor(140, 100, 200)), 1, 0);
    metric_layout->addWidget(makeMetricCard(
        QStringLiteral("角分辨率"),
        numberText(grid, QStringLiteral("angular_resolution_deg"), 6, QStringLiteral("°")),
        QStringLiteral("每像素"), QColor(90, 145, 160)), 1, 1);
    metric_layout->setColumnStretch(0, 1);
    metric_layout->setColumnStretch(1, 1);
    metric_layout->setColumnStretch(2, 1);
    layout->addLayout(metric_layout);

    auto *preview_box = new QGroupBox(QStringLiteral("核心生成的全球预览"));
    preview_box->setStyleSheet(QStringLiteral(
        "QGroupBox{background:white;border:1px solid #ddd;border-radius:4px;"
        "font-weight:bold;padding-top:10px;margin-top:4px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:10px;}"));
    auto *preview_layout = new QVBoxLayout(preview_box);
    const QString preview_path = resolveReportPath(
        report, artifacts.value(QStringLiteral("preview_png")).toString());
    auto *preview_label = new QLabel;
    preview_label->setAlignment(Qt::AlignCenter);
    preview_label->setMinimumHeight(260);
    preview_label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    preview_label->setStyleSheet(QStringLiteral(
        "QLabel{background:#23262b;color:#c8cdd4;border:1px solid #d8d8d8;"
        "border-radius:4px;padding:8px;}"));
    const QPixmap preview(preview_path);
    if (!preview.isNull())
    {
        const QPixmap scaled_preview = preview.scaled(
            QSize(760, 450), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        preview_label->setPixmap(scaled_preview);
        preview_label->setMinimumHeight(scaled_preview.height() + 16);
    }
    else if (preview_path.isEmpty())
    {
        preview_label->setText(QStringLiteral("报告未记录 preview_png，无法显示预览。"));
    }
    else
    {
        preview_label->setText(QStringLiteral("无法加载核心生成的预览图：\n%1").arg(preview_path));
        preview_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    }
    preview_layout->addWidget(preview_label);
    auto *preview_caption = makeValueLabel(
        preview_path.isEmpty() ? QStringLiteral("预览路径：未记录")
                               : QStringLiteral("预览路径：%1").arg(preview_path));
    preview_caption->setStyleSheet(QStringLiteral("color:#777;font-weight:normal;"));
    preview_layout->addWidget(preview_caption);
    layout->addWidget(preview_box);

    auto *frame_box = new QGroupBox(QStringLiteral("坐标框架"));
    frame_box->setStyleSheet(QStringLiteral(
        "QGroupBox{background:white;border:1px solid #ddd;border-radius:4px;"
        "font-weight:bold;padding-top:10px;margin-top:4px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:10px;}"));
    auto *frame_layout = new QGridLayout(frame_box);
    frame_layout->setColumnStretch(1, 1);
    addDetailRow(frame_layout, 0, QStringLiteral("目标天体"), target_name);
    addDetailRow(frame_layout, 1, QStringLiteral("体固连坐标系"),
                 frame.value(QStringLiteral("body_fixed_frame")).toString());
    addDetailRow(frame_layout, 2, QStringLiteral("框架验证状态"),
                 frame.value(QStringLiteral("frame_status")).toString());
    addDetailRow(frame_layout, 3, QStringLiteral("纬度类型"),
                 frame.value(QStringLiteral("latitude_type")).toString());
    addDetailRow(frame_layout, 4, QStringLiteral("经度方向"),
                 frame.value(QStringLiteral("longitude_direction")).toString());
    addDetailRow(frame_layout, 5, QStringLiteral("经度域"),
                 frame.value(QStringLiteral("longitude_domain")).toString());
    addDetailRow(frame_layout, 6, QStringLiteral("参考半径"),
                 numberText(frame, QStringLiteral("reference_radius_m"), 3,
                            QStringLiteral(" m")));
    addDetailRow(frame_layout, 7, QStringLiteral("体心坐标"),
                 centerText(frame.value(QStringLiteral("center_xyz_m"))));
    addDetailRow(frame_layout, 8, QStringLiteral("DOM 颜色来源"),
                 report.value(QStringLiteral("dom_color_source")).toString());
    addDetailRow(frame_layout, 9, QStringLiteral("输入网格单位"),
                 frame.value(QStringLiteral("source_surface_unit")).toString());
    addDetailRow(frame_layout, 10, QStringLiteral("中央经线"),
                 numberText(frame, QStringLiteral("central_meridian_deg"), 3,
                            QStringLiteral("°")));
    layout->addWidget(frame_box);
    layout->addWidget(makeCoordinateNotice(report, frame));

    auto *artifact_box = new QGroupBox(QStringLiteral("产物路径"));
    artifact_box->setStyleSheet(QStringLiteral(
        "QGroupBox{background:white;border:1px solid #ddd;border-radius:4px;"
        "font-weight:bold;padding-top:10px;margin-top:4px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:10px;}"));
    auto *artifact_layout = new QGridLayout(artifact_box);
    artifact_layout->setColumnStretch(1, 1);
    addArtifactRow(artifact_layout, 0, QStringLiteral("报告预览"), preview_path);
    addArtifactRow(artifact_layout, 1, QStringLiteral("径向 DEM"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("radial_dem")).toString()));
    addArtifactRow(artifact_layout, 2, QStringLiteral("高程 DEM"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("elevation_dem")).toString()));
    addArtifactRow(artifact_layout, 3, QStringLiteral("DOM"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("dom")).toString()));
    addArtifactRow(artifact_layout, 4, QStringLiteral("可靠性栅格"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("reliability")).toString()));
    addArtifactRow(artifact_layout, 5, QStringLiteral("覆盖栅格"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("coverage")).toString()));
    addArtifactRow(artifact_layout, 6, QStringLiteral("多表面歧义掩膜"),
                   resolveReportPath(
                       report, artifacts.value(QStringLiteral("ambiguity")).toString()));
    layout->addWidget(artifact_box);
    layout->addStretch();
}
