// =============================================================================
// 文件: WorkflowReportDialog.cpp
// =============================================================================

#include "application/WorkflowReportDialog.h"
#include "ui_WorkflowReportDialog.h"

#include <QApplication>
#include <QColor>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QLabel>
#include <QPainter>
#include <QPaintEvent>
#include <QPushButton>
#include <QComboBox>
#include <QScrollArea>
#include <QScrollBar>
#include <QSizePolicy>
#include <QSpacerItem>
#include <QTabWidget>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QVBoxLayout>

#include <cmath>
#include <algorithm>

// =============================================================================
// ReportChartWidget
// =============================================================================
ReportChartWidget::ReportChartWidget(ChartType type, QWidget *parent)
    : QWidget(parent), _type(type)
{
    setMinimumSize(200, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

QSize ReportChartWidget::sizeHint() const
{
    switch (_type) {
    case ChartType::ArcProgress:      return {220, 180};
    case ChartType::BarComparison:    return {400, 200};
    case ChartType::BarSeries:        return {400, 200};
    }
    return {400, 200};
}

void ReportChartWidget::setComparisonData(const QStringList &labels,
                                           const std::vector<double> &before,
                                           const std::vector<double> &after,
                                           const QString &unit)
{
    _labels = labels; _before = before; _after = after; _unit = unit;
    update();
}

void ReportChartWidget::setArcData(double value, double total, const QString &label)
{
    _arcValue = value; _arcTotal = total; _arcLabel = label;
    update();
}

void ReportChartWidget::setBarSeriesData(const QStringList &labels,
                                          const std::vector<double> &values,
                                          const QString &unit)
{
    _labels = labels; _values = values; _unit = unit;
    update();
}

void ReportChartWidget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    switch (_type) {
    case ChartType::BarComparison: drawBarComparison(p); break;
    case ChartType::ArcProgress:  drawArcProgress(p);   break;
    case ChartType::BarSeries:    drawBarSeries(p);     break;
    }
}

void ReportChartWidget::drawBarComparison(QPainter &p)
{
    if (_labels.isEmpty() || _before.empty()) return;

    const int n = qMin(_labels.size(), (int)_before.size());
    const QColor colBefore(220, 90, 80);
    const QColor colAfter (60, 160, 100);
    const QColor colGrid  (230, 230, 230);
    const QColor colText  (80, 80, 80);

    const int marginL = 60, marginR = 20, marginT = 30, marginB = 50;
    const int W = width() - marginL - marginR;
    const int H = height() - marginT - marginB;

    // 背景
    p.fillRect(rect(), QColor(250, 250, 252));

    // 最大值
    double maxVal = 0.01;
    for (int i = 0; i < n; ++i) {
        maxVal = std::max(maxVal, _before[i]);
        if (i < (int)_after.size()) maxVal = std::max(maxVal, _after[i]);
    }
    maxVal *= 1.15;

    // 网格线
    p.setPen(QPen(colGrid, 1));
    const int gridLines = 4;
    QFont smallFont = p.font();
    smallFont.setPointSize(8);
    p.setFont(smallFont);
    p.setPen(colText);
    for (int g = 0; g <= gridLines; ++g) {
        int y = marginT + H - (int)(H * g / gridLines);
        p.setPen(QPen(colGrid, 1));
        p.drawLine(marginL, y, marginL + W, y);
        double val = maxVal * g / gridLines;
        p.setPen(colText);
        p.drawText(0, y - 6, marginL - 4, 14, Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(val, 'f', 2));
    }

    // 单位标签
    p.setPen(colText);
    p.save();
    p.translate(10, marginT + H / 2);
    p.rotate(-90);
    p.drawText(-40, -6, 80, 14, Qt::AlignCenter, _unit);
    p.restore();

    // 左边框
    p.setPen(QPen(QColor(180,180,180), 1));
    p.drawLine(marginL, marginT, marginL, marginT + H);
    p.drawLine(marginL, marginT + H, marginL + W, marginT + H);

    const int barGroupW = W / n;
    const int barW = qMax(6, barGroupW / 3);
    const int gap  = 3;

    for (int i = 0; i < n; ++i) {
        int cx = marginL + i * barGroupW + barGroupW / 2;

        // 平差前（红）
        int h1 = (maxVal > 0) ? (int)(H * _before[i] / maxVal) : 0;
        p.fillRect(cx - barW - gap, marginT + H - h1, barW, h1, colBefore);
        // 数值
        p.setPen(colBefore);
        p.drawText(cx - barW - gap, marginT + H - h1 - 14, barW, 13,
                   Qt::AlignHCenter, QString::number(_before[i], 'f', 2));

        // 平差后（绿）
        if (i < (int)_after.size()) {
            int h2 = (maxVal > 0) ? (int)(H * _after[i] / maxVal) : 0;
            p.fillRect(cx + gap, marginT + H - h2, barW, h2, colAfter);
            p.setPen(colAfter);
            p.drawText(cx + gap, marginT + H - h2 - 14, barW, 13,
                       Qt::AlignHCenter, QString::number(_after[i], 'f', 2));
        }

        // X 轴标签
        p.setPen(colText);
        p.drawText(cx - barGroupW / 2, marginT + H + 4, barGroupW, 18,
                   Qt::AlignHCenter, _labels[i]);
    }

    // 图例
    const int legX = marginL + W - 140;
    const int legY = marginT + 8;
    p.fillRect(legX, legY, 14, 10, colBefore);
    p.setPen(colText);
    p.drawText(legX + 18, legY - 1, 60, 12, Qt::AlignLeft, "平差前");
    p.fillRect(legX + 80, legY, 14, 10, colAfter);
    p.drawText(legX + 98, legY - 1, 60, 12, Qt::AlignLeft, "平差后");
}

void ReportChartWidget::drawArcProgress(QPainter &p)
{
    p.fillRect(rect(), QColor(250, 250, 252));
    const QColor colArc  (60, 160, 100);
    const QColor colRemain(230, 230, 230);
    const QColor colText (60, 60, 60);

    const int cx = width() / 2;
    const int cy = height() / 2 + 5;
    const int r  = qMin(cx, cy) - 28;
    if (r < 20) return;

    const double ratio = (_arcTotal > 0) ? qBound(0.0, _arcValue / _arcTotal, 1.0) : 0.0;
    const int   spanned= (int)(ratio * 5760); // 0.01度单位，360°=5760

    // 灰色背景环
    p.setPen(QPen(colRemain, 18, Qt::SolidLine, Qt::RoundCap));
    p.drawArc(cx - r, cy - r, 2*r, 2*r, 90 * 16, -360 * 16);

    // 彩色前景弧
    if (spanned > 0) {
        p.setPen(QPen(colArc, 18, Qt::SolidLine, Qt::RoundCap));
        p.drawArc(cx - r, cy - r, 2*r, 2*r, 90 * 16, -spanned);
    }

    // 中心数值
    QFont bigFont = p.font();
    bigFont.setPointSize(18);
    bigFont.setBold(true);
    p.setFont(bigFont);
    p.setPen(colText);
    p.drawText(cx - r, cy - 22, 2*r, 28, Qt::AlignCenter,
               QString::number((int)(ratio * 100)) + "%");

    QFont smallFont = p.font();
    smallFont.setPointSize(9);
    smallFont.setBold(false);
    p.setFont(smallFont);
    p.setPen(QColor(120, 120, 120));
    p.drawText(cx - r, cy + 8, 2*r, 20, Qt::AlignCenter,
               QString("%1 / %2").arg((int)_arcValue).arg((int)_arcTotal));

    // 底部标签
    p.drawText(0, height() - 22, width(), 20, Qt::AlignCenter, _arcLabel);
}

void ReportChartWidget::drawBarSeries(QPainter &p)
{
    if (_labels.isEmpty() || _values.empty()) return;

    const int n = qMin(_labels.size(), (int)_values.size());
    const QColor colBar(70, 130, 210);
    const QColor colGrid(230, 230, 230);
    const QColor colText(80, 80, 80);

    const int marginL = 60, marginR = 20, marginT = 30, marginB = 50;
    const int W = width() - marginL - marginR;
    const int H = height() - marginT - marginB;

    p.fillRect(rect(), QColor(250, 250, 252));

    double maxVal = 0.01;
    for (int i = 0; i < n; ++i) maxVal = std::max(maxVal, _values[i]);
    maxVal *= 1.15;

    QFont smallFont = p.font(); smallFont.setPointSize(8); p.setFont(smallFont);

    for (int g = 0; g <= 4; ++g) {
        int y = marginT + H - (int)(H * g / 4);
        p.setPen(QPen(colGrid, 1));
        p.drawLine(marginL, y, marginL + W, y);
        p.setPen(colText);
        double val = maxVal * g / 4;
        p.drawText(0, y - 6, marginL - 4, 14, Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(val, 'f', 1));
    }

    p.setPen(QPen(QColor(180,180,180), 1));
    p.drawLine(marginL, marginT, marginL, marginT + H);
    p.drawLine(marginL, marginT + H, marginL + W, marginT + H);

    // Y 轴单位
    p.setPen(colText);
    p.save();
    p.translate(10, marginT + H / 2);
    p.rotate(-90);
    p.drawText(-40, -6, 80, 14, Qt::AlignCenter, _unit);
    p.restore();

    const int barW  = qMax(4, W / n - 8);
    const int cellW = W / n;

    for (int i = 0; i < n; ++i) {
        int x = marginL + i * cellW + (cellW - barW) / 2;
        int barH = (maxVal > 0) ? (int)(H * _values[i] / maxVal) : 0;
        int y = marginT + H - barH;

        p.fillRect(x, y, barW, barH, colBar);

        p.setPen(colBar.darker(130));
        p.drawText(x, y - 14, barW, 13, Qt::AlignHCenter,
                   QString::number(_values[i], 'f', 1));

        p.setPen(colText);
        const QString lbl = _labels[i];
        p.drawText(x - 4, marginT + H + 4, barW + 8, 18, Qt::AlignHCenter, lbl);
    }
}

// =============================================================================
// WorkflowReportDialog
// =============================================================================
static QString fmtNum(double v, int dec = 2) {
    return QString::number(v, 'f', dec);
}
static QString fmtInt(int v) {
    return QString::number(v);
}
static QString fmtPercent(int n, int d) {
    if (d <= 0) return "—";
    return QString("%1%").arg(QString::number(100.0 * n / d, 'f', 1));
}

WorkflowReportDialog::WorkflowReportDialog(const QString &projectAssetsDir, QWidget *parent)
    : QDialog(parent)
    , _assetsDir(projectAssetsDir)
{
    setWindowTitle(tr("工作流程历史报告"));
    setMinimumSize(750, 600);
    resize(860, 700);
    buildUi();
}

QJsonObject WorkflowReportDialog::loadReport(const QString &name) const
{
    const QString path = QDir(_assetsDir).filePath("reports/" + name);
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return {};
    return doc.object();
}

QJsonArray WorkflowReportDialog::loadReportHistory(const QString &name) const
{
    const QString path = QDir(_assetsDir).filePath("reports/" + name);
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};

    QJsonParseError err;
    auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError || !doc.isArray()) return {};
    return doc.array();
}

QWidget *WorkflowReportDialog::makeStatCard(const QString &title,
                                             const QString &value,
                                             const QString &subtitle,
                                             const QColor  &accent)
{
    auto *w = new QWidget;
    w->setFixedHeight(90);
    w->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    w->setStyleSheet(QString("QWidget { background: white; border-left: 4px solid %1; "
                             "border-radius: 4px; }")
                     .arg(accent.name()));

    auto *vl = new QVBoxLayout(w);
    vl->setContentsMargins(12, 8, 8, 8);
    vl->setSpacing(2);

    auto *lTitle = new QLabel(title);
    QFont f = lTitle->font();
    f.setPointSize(9);
    lTitle->setFont(f);
    lTitle->setStyleSheet("color:#888; background:transparent; border:none;");

    auto *lValue = new QLabel(value);
    QFont fv = lValue->font();
    fv.setPointSize(20);
    fv.setBold(true);
    lValue->setFont(fv);
    lValue->setStyleSheet(QString("color:%1; background:transparent; border:none;")
                          .arg(accent.name()));

    vl->addWidget(lTitle);
    vl->addWidget(lValue);

    if (!subtitle.isEmpty()) {
        auto *lSub = new QLabel(subtitle);
        QFont fs = lSub->font(); fs.setPointSize(8);
        lSub->setFont(fs);
        lSub->setStyleSheet("color:#aaa; background:transparent; border:none;");
        vl->addWidget(lSub);
    }

    return w;
}

QWidget *WorkflowReportDialog::makeSeparator()
{
    auto *f = new QFrame;
    f->setFrameShape(QFrame::HLine);
    f->setFrameShadow(QFrame::Sunken);
    f->setStyleSheet("QFrame { color: #e0e0e0; }");
    return f;
}

QLabel *WorkflowReportDialog::makeKVLabel(const QString &key, const QString &value)
{
    auto *lbl = new QLabel(QString("<b>%1</b>　%2").arg(key, value.toHtmlEscaped()));
    lbl->setStyleSheet("color:#444; padding:2px 0;");
    return lbl;
}

void WorkflowReportDialog::buildUi()
{
    Ui::WorkflowReportDialog form;
    form.setupUi(this);

    _refreshBtn = form.m_refreshBtn;
    _tabs = form.m_tabs;

    auto *titleLbl = form.reportTitleLabel;
    QFont tf = titleLbl->font();
    tf.setPointSize(13);
    tf.setBold(true);
    titleLbl->setFont(tf);

    connect(_refreshBtn, &QPushButton::clicked, this, &WorkflowReportDialog::refresh);

    _tabs->addTab(buildAtTab(),    tr("✈ 空中三角测量"));
    _tabs->addTab(buildDenseTab(), tr("☁ 稠密点云"));
    _tabs->addTab(buildMeshTab(),  tr("▲ 三维模型"));
}

void WorkflowReportDialog::refresh()
{
    _tabs->removeTab(2); _tabs->removeTab(1); _tabs->removeTab(0);
    _tabs->addTab(buildAtTab(),    tr("✈ 空中三角测量"));
    _tabs->addTab(buildDenseTab(), tr("☁ 稠密点云"));
    _tabs->addTab(buildMeshTab(),  tr("▲ 三维模型"));
}

// ─────────────────────────────────────────────────────────────────────────────
// AT 标签页
// ─────────────────────────────────────────────────────────────────────────────
QWidget *WorkflowReportDialog::buildAtTab()
{
    QJsonArray history = loadReportHistory("at_report_history.json");
    if (history.isEmpty())
    {
        const QJsonObject latest = loadReport("at_report.json");
        if (!latest.isEmpty())
        {
            history.append(latest);
        }
    }

    if (history.isEmpty())
    {
        return buildAtReportPage(QJsonObject());
    }

    auto *page = new QWidget;
    page->setStyleSheet("background:#f5f5f7;");
    auto *vl = new QVBoxLayout(page);
    vl->setContentsMargins(12, 12, 12, 12);
    vl->setSpacing(8);

    auto *toolbar = new QWidget(page);
    toolbar->setStyleSheet("background:white;border:1px solid #ddd;border-radius:4px;");
    auto *toolbarLayout = new QHBoxLayout(toolbar);
    toolbarLayout->setContentsMargins(10, 8, 10, 8);
    toolbarLayout->setSpacing(8);

    auto *label = new QLabel(tr("历史记录:"), toolbar);
    auto *combo = new QComboBox(toolbar);
    combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);

    for (int i = history.size() - 1; i >= 0; --i)
    {
        const QJsonObject report = history.at(i).toObject();
        const QString timestamp = report.value(QStringLiteral("timestamp")).toString();
        const QString mode = report.value(QStringLiteral("mode")).toString(QStringLiteral("sfm"));
        const QString source = report.value(QStringLiteral("source")).toString();
        QString sourceLabel;
        if (source == QStringLiteral("workflow_aerial_triangulation"))
            sourceLabel = tr("工作流程-空中三角测量");
        else if (source == QStringLiteral("reconstruction_bundle_adjust"))
            sourceLabel = tr("重建-稀疏重建-光束法平差优化");
        else
            sourceLabel = (mode == QStringLiteral("bundle_adjust"))
                ? tr("光束法平差")
                : tr("空中三角测量");

        const QString title = QStringLiteral("%1 | %2 | %3 张影像 | %4 点")
            .arg(timestamp)
            .arg(sourceLabel)
            .arg(report.value(QStringLiteral("num_images")).toInt())
            .arg(report.value(QStringLiteral("num_points_3d")).toInt());
        combo->addItem(title, report);
    }

    toolbarLayout->addWidget(label);
    toolbarLayout->addWidget(combo, 1);
    vl->addWidget(toolbar);

    auto *contentHolder = new QVBoxLayout;
    contentHolder->setContentsMargins(0, 0, 0, 0);
    vl->addLayout(contentHolder, 1);

    auto rebuild = [this, contentHolder](const QJsonObject &report)
    {
        while (contentHolder->count() > 0)
        {
            QLayoutItem *item = contentHolder->takeAt(0);
            if (item->widget())
            {
                item->widget()->deleteLater();
            }
            delete item;
        }
        contentHolder->addWidget(this->buildAtReportPage(report));
    };

    rebuild(combo->currentData().toJsonObject());
    QObject::connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), page,
        [combo, rebuild](int) {
            rebuild(combo->currentData().toJsonObject());
        });

    return page;
}

QWidget *WorkflowReportDialog::buildAtReportPage(const QJsonObject &r)
{

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea{background:#f5f5f7;}");

    auto *page = new QWidget;
    page->setStyleSheet("background:#f5f5f7;");
    auto *vl = new QVBoxLayout(page);
    vl->setContentsMargins(16, 16, 16, 24);
    vl->setSpacing(12);
    scroll->setWidget(page);

    if (r.isEmpty()) {
        auto *empty = new QLabel(tr(
            "<br><br>"
            "<center><span style='font-size:40px;'>✈</span><br><br>"
            "<b style='font-size:14px;color:#666;'>尚无空中三角测量报告</b><br><br>"
            "<span style='color:#999;'>执行空中三角测量后报告将自动显示在此处</span></center>"));
        empty->setAlignment(Qt::AlignCenter);
        vl->addWidget(empty);
        vl->addStretch();
        return scroll;
    }

    // ── 基本信息标题 ──────────────────────────────────────────────────────
    const QString ts         = r.value("timestamp").toString();
    const QString mode       = r.value("mode").toString("sfm");
    const QString source     = r.value("source").toString();
    const int    numImg      = r.value("num_images").toInt();
    const int    numReg      = r.value("num_registered").toInt();
    const int    numPoints   = r.value("num_points_3d").toInt();
    const double reproj      = r.value("mean_reproj_error_px").toDouble();
    const double rmsBefore   = r.value("ba_rms_before").toDouble();
    const double rmsAfter    = r.value("ba_rms_after").toDouble();
    const int    tracksTotal = r.value("ba_tracks_total").toInt();
    const int    tracksOpt   = r.value("ba_tracks_optimized").toInt();
    const int    tracksFilt  = r.value("ba_tracks_filtered").toInt();
    const double dur         = r.value("duration_s").toDouble(-1.0);
    const QString outDir     = r.value("output_dir").toString();
    const QString sparsePath = r.value("sparse_cloud_path").toString();

    QString sourceLabel;
    if (source == QStringLiteral("workflow_aerial_triangulation"))
        sourceLabel = tr("工作流程-空中三角测量");
    else if (source == QStringLiteral("reconstruction_bundle_adjust"))
        sourceLabel = tr("重建-稀疏重建-光束法平差优化");
    else
        sourceLabel = (mode == QStringLiteral("bundle_adjust"))
            ? tr("光束法平差")
            : tr("空中三角测量");

    // 标题行
    auto *headerLbl = new QLabel(QString("<b style='font-size:13px;'>空中三角测量报告</b>"
                                         "<span style='color:#999;font-size:10px;'>　%1　%2　来源：%3</span>")
                                 .arg(ts)
                                 .arg(mode == "sfm" ? "（SFM 模式）" : "（光束法平差模式）")
                                 .arg(sourceLabel));
    headerLbl->setStyleSheet("background:white;border-radius:4px;padding:10px;");
    vl->addWidget(headerLbl);

    // ── 统计卡片行 1：注册率 / 三维点 / 最终重投影误差 ──
    auto *cards1 = new QHBoxLayout;
    cards1->setSpacing(8);

    const QString regRatioStr = (numImg > 0)
        ? QString("%1 / %2").arg(numReg).arg(numImg)
        : "—";
    cards1->addWidget(makeStatCard(tr("已注册影像"), fmtInt(numReg),
                                   QString("共 %1 张，注册率 %2")
                                       .arg(numImg).arg(fmtPercent(numReg, numImg)),
                                   QColor(60, 160, 100)));
    cards1->addWidget(makeStatCard(tr("连接点（三维点）"), fmtInt(numPoints), "",
                                   QColor(70, 130, 210)));
    cards1->addWidget(makeStatCard(tr("最终重投影误差"), fmtNum(reproj) + " px",
                                   "（所有相机均值）",
                                   reproj < 1.5 ? QColor(60, 160, 100) :
                                   reproj < 3.0 ? QColor(200, 140, 60) :
                                                  QColor(200, 70, 60)));
    if (dur >= 0)
        cards1->addWidget(makeStatCard(tr("处理时长"),
                                       dur < 60 ? QString("%1 s").arg(fmtNum(dur, 1))
                                                : QString("%1 min").arg(fmtNum(dur/60, 1)),
                                       "",
                                       QColor(140, 100, 200)));
    vl->addLayout(cards1);

    const QJsonObject sfmDiag = r.value(QStringLiteral("sfm_diagnostics")).toObject();
    if (!sfmDiag.isEmpty()) {
        vl->addWidget(makeSeparator());

        auto *diagBox = new QGroupBox(tr("匹配与注册诊断"));
        diagBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *diagLayout = new QGridLayout(diagBox);
        diagLayout->setSpacing(8);

        const QJsonObject candidateGraph = sfmDiag.value(QStringLiteral("candidate_graph")).toObject();
        const QJsonObject actualGraph = sfmDiag.value(QStringLiteral("actual_match_graph")).toObject();
        const QJsonObject sparseQuality = sfmDiag.value(QStringLiteral("sparse_quality")).toObject();
        const QJsonObject pairPlan = sfmDiag.value(QStringLiteral("pair_plan")).toObject();
        const QJsonArray sourceTypes = pairPlan.value(QStringLiteral("source_types")).toArray();
        const QJsonObject sourceTypeCounts = pairPlan.value(QStringLiteral("source_type_counts")).toObject();
        QStringList sourceLabels;
        for (const QJsonValue &value : sourceTypes) {
            const QString sourceType = value.toString();
            if (sourceType.isEmpty()) continue;
            const int sourceCount = sourceTypeCounts.value(sourceType).toInt(-1);
            sourceLabels.append(sourceCount >= 0
                ? QStringLiteral("%1(%2)").arg(sourceType).arg(sourceCount)
                : sourceType);
        }

        auto addDiag = [&](int row, int col, const QString &key, const QString &value) {
            diagLayout->addWidget(makeKVLabel(key, value), row, col);
        };
        addDiag(0, 0, tr("候选对:"), fmtInt(sfmDiag.value(QStringLiteral("total_pairs")).toInt()));
        addDiag(0, 1, tr("有效匹配对:"), fmtInt(sfmDiag.value(QStringLiteral("actual_match_pairs")).toInt()));
        addDiag(1, 0, tr("待生成/缺失:"), fmtInt(sfmDiag.value(QStringLiteral("pending_pairs")).toInt()));
        addDiag(1, 1, tr("无匹配负缓存:"), fmtInt(sfmDiag.value(QStringLiteral("no_match_cache_skipped_pairs")).toInt()));
        addDiag(2, 0, tr("候选图分量:"), fmtInt(candidateGraph.value(QStringLiteral("component_count")).toInt()));
        addDiag(2, 1, tr("匹配图分量:"), fmtInt(actualGraph.value(QStringLiteral("component_count")).toInt()));
        addDiag(3, 0, tr("最大匹配分量:"), QStringLiteral("%1 / %2")
                .arg(actualGraph.value(QStringLiteral("largest_component_size")).toInt())
                .arg(actualGraph.value(QStringLiteral("node_count")).toInt()));
        addDiag(3, 1, tr("规划候选:"), QStringLiteral("%1 / %2")
                .arg(pairPlan.value(QStringLiteral("candidate_count")).toInt(
                    pairPlan.value(QStringLiteral("planned_pair_count")).toInt()))
                .arg(pairPlan.value(QStringLiteral("all_pair_count")).toInt()));
        addDiag(4, 0, tr("配对来源:"), sourceLabels.isEmpty() ? QStringLiteral("—") : sourceLabels.join(QStringLiteral(", ")));

        const QJsonObject triAngle = sparseQuality.value(QStringLiteral("triangulation_angle")).toObject();
        if (!triAngle.isEmpty()) {
            addDiag(5, 0, tr("平均三角角:"), QStringLiteral("%1°").arg(fmtNum(triAngle.value(QStringLiteral("mean")).toDouble(), 3)));
            addDiag(5, 1, tr("多视 track:"), fmtInt(sparseQuality.value(QStringLiteral("multi_view_track_count")).toInt()));
        }

        vl->addWidget(diagBox);
    }

    // ── 注册率弧形图 + BA 误差改善图 ─────────────────────────────────────
    auto *chartsRow = new QHBoxLayout;
    chartsRow->setSpacing(12);

    // 注册率弧形图
    auto *arcBox = new QGroupBox(tr("影像注册率"));
    arcBox->setStyleSheet("QGroupBox{background:white;border-radius:4px;"
                          "border:1px solid #ddd;font-weight:bold;padding-top:8px;}"
                          "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
    auto *arcLayout = new QVBoxLayout(arcBox);
    auto *arcChart = new ReportChartWidget(ReportChartWidget::ChartType::ArcProgress);
    arcChart->setFixedSize(200, 180);
    arcChart->setArcData(numReg, numImg, tr("影像"));
    arcLayout->addWidget(arcChart, 0, Qt::AlignCenter);
    chartsRow->addWidget(arcBox);

    // BA 重投影误差对比柱状图
    if (rmsBefore > 0.0 && rmsAfter > 0.0) {
        auto *baBox = new QGroupBox(tr("光束法平差 — 重投影误差改善（像素）"));
        baBox->setStyleSheet(arcBox->styleSheet());
        auto *baLayout = new QVBoxLayout(baBox);
        auto *baChart = new ReportChartWidget(ReportChartWidget::ChartType::BarComparison);
        baChart->setFixedHeight(200);
        baChart->setComparisonData(QStringList{tr("RMS 误差")},
                                   {rmsBefore}, {rmsAfter}, "px");
        baLayout->addWidget(baChart);

        // 信息文字
        double improve = (rmsBefore > 0) ? (rmsBefore - rmsAfter) / rmsBefore * 100.0 : 0.0;
        auto *infoLbl = new QLabel(
            QString("<span style='color:#888;font-size:9px;'>"
                    "平差前: <b>%1 px</b>　平差后: <b>%2 px</b>　"
                    "改善: <b style='color:%3;'>%4%</b>"
                    "</span>")
            .arg(fmtNum(rmsBefore), fmtNum(rmsAfter))
            .arg(improve > 10 ? "#3c9a50" : "#888")
            .arg(fmtNum(improve, 1)));
        infoLbl->setAlignment(Qt::AlignCenter);
        baLayout->addWidget(infoLbl);

        chartsRow->addWidget(baBox, 2);
    }

    vl->addLayout(chartsRow);

    // ── BA 统计详情 ───────────────────────────────────────────────────────
    if (tracksTotal > 0) {
        vl->addWidget(makeSeparator());

        auto *baSection = new QGroupBox(tr("光束法平差统计"));
        baSection->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *baSectionLayout = new QGridLayout(baSection);
        baSectionLayout->setSpacing(8);

        // 使轨迹柱状图
        auto *tracksChart = new ReportChartWidget(ReportChartWidget::ChartType::BarSeries);
        tracksChart->setFixedHeight(180);
        tracksChart->setBarSeriesData(
            QStringList{tr("总轨迹数"), tr("已优化"), tr("已过滤")},
            {(double)tracksTotal, (double)tracksOpt, (double)tracksFilt},
            "条");

        baSectionLayout->addWidget(tracksChart, 0, 0, 1, 2);

        auto addStat = [&](int row, int col, const QString &k, const QString &v) {
            baSectionLayout->addWidget(makeKVLabel(k, v), row, col);
        };
        addStat(1, 0, tr("总轨迹数:"), fmtInt(tracksTotal));
        addStat(1, 1, tr("已优化:"),  fmtInt(tracksOpt) + " (" + fmtPercent(tracksOpt, tracksTotal) + ")");
        addStat(2, 0, tr("过滤率:"),  fmtPercent(tracksFilt, tracksTotal));
        addStat(2, 1, tr("RMS (前→后):"),
                QString("%1 → %2 px").arg(fmtNum(rmsBefore), fmtNum(rmsAfter)));

        vl->addWidget(baSection);
    }

    // ── 逐相机残差表 ──────────────────────────────────────────────────────
    const QJsonArray camArr = r.value("per_camera").toArray();
    if (!camArr.isEmpty()) {
        vl->addWidget(makeSeparator());

        auto *camBox = new QGroupBox(tr("逐相机重投影残差"));
        camBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *camLayout = new QVBoxLayout(camBox);

        auto *table = new QTableWidget(camArr.size(), 4);
        table->setHorizontalHeaderLabels({tr("影像"), tr("已注册"), tr("残差 (px)"), tr("质量")});
        table->horizontalHeader()->setStretchLastSection(true);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setAlternatingRowColors(true);
        table->setSelectionBehavior(QAbstractItemView::SelectRows);
        table->setStyleSheet("QTableWidget{border:none;}"
                             "QHeaderView::section{background:#f0f0f2;padding:4px;"
                             "border:1px solid #e0e0e0;font-weight:bold;}");

        // 按残差排序数据
        QVector<QJsonObject> camList;
        for (const QJsonValue &v : camArr) camList.append(v.toObject());
        std::sort(camList.begin(), camList.end(), [](const QJsonObject &a, const QJsonObject &b){
            return a.value("residual_px").toDouble() < b.value("residual_px").toDouble();
        });

        for (int i = 0; i < camList.size(); ++i) {
            const QJsonObject &cam = camList[i];
            const QString path = cam.value("path").toString();
            const bool  reg    = cam.value("registered").toBool();
            const double res   = cam.value("residual_px").toDouble();

            table->setItem(i, 0, new QTableWidgetItem(QFileInfo(path).fileName()));
            auto *regItem = new QTableWidgetItem(reg ? tr("✓ 已注册") : tr("✗ 未注册"));
            regItem->setForeground(reg ? QColor(60,160,100) : QColor(200,70,60));
            table->setItem(i, 1, regItem);
            table->setItem(i, 2, new QTableWidgetItem(reg ? fmtNum(res) : "—"));

            QString quality;
            QColor  qualColor;
            if (!reg)          { quality = tr("未使用"); qualColor = QColor(150,150,150); }
            else if (res < 1.0){ quality = tr("优秀");   qualColor = QColor(40,160,80);  }
            else if (res < 2.0){ quality = tr("良好");   qualColor = QColor(70,130,210); }
            else if (res < 4.0){ quality = tr("一般");   qualColor = QColor(200,140,60); }
            else               { quality = tr("较差");   qualColor = QColor(200,70,60);  }
            auto *qItem = new QTableWidgetItem(quality);
            qItem->setForeground(qualColor);
            table->setItem(i, 3, qItem);
        }
        table->resizeColumnsToContents();
        table->setColumnWidth(0, 240);
        table->setMinimumHeight(qMin(camList.size() * 26 + 30, 300));
        camLayout->addWidget(table);
        vl->addWidget(camBox);
    }

    // ── 相机参数 BA 前后对比表 ────────────────────────────────────────────
    const QJsonArray camCompArr = r.value("camera_comparison").toArray();
    if (!camCompArr.isEmpty()) {
        vl->addWidget(makeSeparator());

        auto *ccBox = new QGroupBox(tr("相机参数变化（BA 前后对比）"));
        ccBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *ccLayout = new QVBoxLayout(ccBox);

        // 说明标签
        auto *hint = new QLabel(tr(
            "<span style='color:#666;font-size:11px;'>"
            "内方位：fu/fv 为等效焦距（像素），cu/cv 为像主点；"
            "外方位：位置偏移为三维空间距离，角度变化为欧拉角绝对差值。"
            "若平差前无外参记录，则位置/角度列显示 \xe2\x80\x94 。"
            "</span>"));
        hint->setWordWrap(true);
        ccLayout->addWidget(hint);

        // 8 列：影像|fu前|fu后|Δfu%|位置偏移(m)|Δ偏航°|Δ俯仰°|Δ横滚°
        const QStringList hdr = {
            tr("影像"),
            tr("fu 前(px)"), tr("fu 后(px)"), tr("Δfu%"),
            tr("位置偏移(m)"),
            tr("Δ偏航(°)"), tr("Δ俯仰(°)"), tr("Δ横滚(°)")
        };
        auto *ccTable = new QTableWidget(camCompArr.size(), hdr.size());
        ccTable->setHorizontalHeaderLabels(hdr);
        ccTable->horizontalHeader()->setStretchLastSection(true);
        ccTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ccTable->setAlternatingRowColors(true);
        ccTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        ccTable->setStyleSheet("QTableWidget{border:none;}"
                               "QHeaderView::section{background:#f0f0f2;padding:4px;"
                               "border:1px solid #e0e0e0;font-weight:bold;}");

        // 用于创建携带数值 UserRole 的 item（便于数字列排序）
        auto makeNumItem = [](double v, const QString &display) -> QTableWidgetItem* {
            auto *it = new QTableWidgetItem(display);
            it->setData(Qt::UserRole, v);
            return it;
        };

        for (int i = 0; i < camCompArr.size(); ++i) {
            const QJsonObject &cam = camCompArr[i].toObject();
            const bool   hadBefore = cam.value(QStringLiteral("had_before")).toBool();
            const QString name     = cam.value(QStringLiteral("name")).toString();
            const double fuB = cam.value(QStringLiteral("fu_before")).toDouble();
            const double fuA = cam.value(QStringLiteral("fu_after")).toDouble();
            const double dfu = hadBefore && fuB > 1e-6 ? (fuA - fuB) / fuB * 100.0 : 0.0;
            const double posDelta   = cam.value(QStringLiteral("pos_delta")).toDouble(-1.0);
            const double yawB       = cam.value(QStringLiteral("yaw_before")).toDouble();
            const double yawA       = cam.value(QStringLiteral("yaw_after")).toDouble();
            const double pitchB     = cam.value(QStringLiteral("pitch_before")).toDouble();
            const double pitchA     = cam.value(QStringLiteral("pitch_after")).toDouble();
            const double rollB      = cam.value(QStringLiteral("roll_before")).toDouble();
            const double rollA      = cam.value(QStringLiteral("roll_after")).toDouble();
            const double dYaw   = std::abs(yawA - yawB);
            const double dPitch = std::abs(pitchA - pitchB);
            const double dRoll  = std::abs(rollA - rollB);

            ccTable->setItem(i, 0, new QTableWidgetItem(name));
            ccTable->setItem(i, 1, makeNumItem(fuB, hadBefore ? fmtNum(fuB) : QStringLiteral("—")));
            ccTable->setItem(i, 2, makeNumItem(fuA, fmtNum(fuA)));
            // Δfu% 着色：> 1% 黄色，> 3% 红色
            auto *dfuItem = makeNumItem(std::abs(dfu), hadBefore ? QString::asprintf("%+.3f%%", dfu) : QStringLiteral("—"));
            if (hadBefore) {
                if (std::abs(dfu) > 3.0)      dfuItem->setForeground(QColor(200,70,60));
                else if (std::abs(dfu) > 1.0)  dfuItem->setForeground(QColor(200,140,60));
                else                           dfuItem->setForeground(QColor(40,160,80));
            }
            ccTable->setItem(i, 3, dfuItem);

            // 位置偏移
            auto *posItem = posDelta >= 0.0
                ? makeNumItem(posDelta, fmtNum(posDelta))
                : new QTableWidgetItem(QStringLiteral("—"));
            if (posDelta >= 0.0 && hadBefore) {
                if (posDelta > 10.0)     posItem->setForeground(QColor(200,70,60));
                else if (posDelta > 2.0) posItem->setForeground(QColor(200,140,60));
                else                     posItem->setForeground(QColor(40,160,80));
            }
            ccTable->setItem(i, 4, posItem);

            // 角度变化
            auto angItem = [&](int col, double delta) {
                auto *it = makeNumItem(delta, hadBefore ? QString::asprintf("%.4f°", delta) : QStringLiteral("—"));
                if (hadBefore) {
                    if (delta > 1.0)      it->setForeground(QColor(200,70,60));
                    else if (delta > 0.1) it->setForeground(QColor(200,140,60));
                    else                  it->setForeground(QColor(40,160,80));
                }
                ccTable->setItem(i, col, it);
            };
            angItem(5, dYaw);
            angItem(6, dPitch);
            angItem(7, dRoll);
        }

        ccTable->setSortingEnabled(true);
        ccTable->resizeColumnsToContents();
        ccTable->setColumnWidth(0, 200);
        ccTable->setMinimumHeight(qMin(camCompArr.size() * 26 + 30, 350));
        ccLayout->addWidget(ccTable);
        vl->addWidget(ccBox);
    }

    // ── BA 逐相机 RMS 前后对比 ────────────────────────────────────────────
    const QJsonArray camPrevArr = r.value("camera_preview").toArray();
    if (!camPrevArr.isEmpty()) {
        vl->addWidget(makeSeparator());

        auto *cpBox = new QGroupBox(tr("逐相机重投影误差（BA 前后对比）"));
        cpBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *cpLayout = new QVBoxLayout(cpBox);

        const QStringList cpHdr = {
            tr("影像"),
            tr("RMS 前(px)"), tr("RMS 后(px)"), tr("改善(px)"), tr("改善%"),
            tr("位移(m)"), tr("Δ偏航(°)"), tr("Δ俯仰(°)"), tr("Δ横滚(°)")
        };
        auto *cpTable = new QTableWidget(camPrevArr.size(), cpHdr.size());
        cpTable->setHorizontalHeaderLabels(cpHdr);
        cpTable->horizontalHeader()->setStretchLastSection(true);
        cpTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        cpTable->setAlternatingRowColors(true);
        cpTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        cpTable->setStyleSheet("QTableWidget{border:none;}"
                               "QHeaderView::section{background:#f0f0f2;padding:4px;"
                               "border:1px solid #e0e0e0;font-weight:bold;}");

        auto makeNI_cp = [](double v, const QString &txt) -> QTableWidgetItem* {
            auto *it = new QTableWidgetItem(txt);
            it->setData(Qt::UserRole, v);
            return it;
        };

        for (int i = 0; i < camPrevArr.size(); ++i) {
            const QJsonObject &cp = camPrevArr[i].toObject();
            const QString name  = cp.value("image_name").toString(
                                   QFileInfo(cp.value("image_path").toString()).fileName());
            const double rmsB   = cp.value("mean_rms_before").toDouble();
            const double rmsA   = cp.value("mean_rms_after").toDouble();
            const double dimpr  = rmsB - rmsA;
            const double pimpr  = (rmsB > 1e-9) ? dimpr / rmsB * 100.0 : 0.0;
            const double dC     = cp.value("delta_c_m").toDouble(-1.0);
            const double dYaw   = std::abs(cp.value("yaw_after").toDouble()   - cp.value("yaw_before").toDouble());
            const double dPitch = std::abs(cp.value("pitch_after").toDouble() - cp.value("pitch_before").toDouble());
            const double dRoll  = std::abs(cp.value("roll_after").toDouble()  - cp.value("roll_before").toDouble());

            cpTable->setItem(i, 0, new QTableWidgetItem(name));

            auto *rbItem = makeNI_cp(rmsB, fmtNum(rmsB));
            if (rmsB > 4.0)       rbItem->setForeground(QColor(200,70,60));
            else if (rmsB > 2.0)  rbItem->setForeground(QColor(200,140,60));
            cpTable->setItem(i, 1, rbItem);

            auto *raItem = makeNI_cp(rmsA, fmtNum(rmsA));
            if (rmsA > 4.0)       raItem->setForeground(QColor(200,70,60));
            else if (rmsA > 2.0)  raItem->setForeground(QColor(200,140,60));
            else                  raItem->setForeground(QColor(40,160,80));
            cpTable->setItem(i, 2, raItem);

            auto *diItem = makeNI_cp(dimpr, QString::asprintf("%+.4f", dimpr));
            diItem->setForeground(dimpr >= 0 ? QColor(40,160,80) : QColor(200,70,60));
            cpTable->setItem(i, 3, diItem);

            auto *piItem = makeNI_cp(pimpr, QString::asprintf("%+.1f%%", pimpr));
            piItem->setForeground(pimpr >= 0 ? QColor(40,160,80) : QColor(200,70,60));
            cpTable->setItem(i, 4, piItem);

            cpTable->setItem(i, 5, dC >= 0 ? makeNI_cp(dC, fmtNum(dC))
                                           : new QTableWidgetItem(QStringLiteral("\xe2\x80\x94")));

            auto angI_cp = [&](int col, double delta) {
                auto *it = makeNI_cp(delta, QString::asprintf("%.4f\xc2\xb0", delta));
                if (delta > 1.0)      it->setForeground(QColor(200,70,60));
                else if (delta > 0.1) it->setForeground(QColor(200,140,60));
                else                  it->setForeground(QColor(40,160,80));
                cpTable->setItem(i, col, it);
            };
            angI_cp(6, dYaw); angI_cp(7, dPitch); angI_cp(8, dRoll);
        }
        cpTable->setSortingEnabled(true);
        cpTable->resizeColumnsToContents();
        cpTable->setColumnWidth(0, 180);
        cpTable->setMinimumHeight(qMin(camPrevArr.size() * 26 + 30, 280));
        cpLayout->addWidget(cpTable);
        vl->addWidget(cpBox);
    }

    // ── 逐点平差误差变化 ──────────────────────────────────────────────────
    const QJsonArray ptResArr = r.value("point_residuals").toArray();
    if (!ptResArr.isEmpty()) {
        vl->addWidget(makeSeparator());

        // 统计有效点
        int validCnt = 0, convCnt = 0;
        double sumB = 0.0, sumA = 0.0, maxB = 0.0, maxA = 0.0;
        QVector<QJsonObject> ptList;
        for (const QJsonValue &pv : ptResArr) {
            const QJsonObject pj = pv.toObject();
            if (!pj.value("valid").toBool()) continue;
            ++validCnt;
            if (pj.value("converged").toBool()) ++convCnt;
            const double rb = pj.value("rms_before").toDouble();
            const double ra = pj.value("rms_after").toDouble();
            sumB += rb; sumA += ra;
            if (rb > maxB) maxB = rb;
            if (ra > maxA) maxA = ra;
            ptList.append(pj);
        }
        const double avgB = validCnt > 0 ? sumB / validCnt : 0.0;
        const double avgA = validCnt > 0 ? sumA / validCnt : 0.0;

        auto *ptBox = new QGroupBox(
            tr("连接点平差误差分布（%1 个有效点）").arg(validCnt));
        ptBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *ptLayout = new QVBoxLayout(ptBox);

        auto *ptSumLbl = new QLabel(
            tr("<span style='font-size:11px;color:#555;'>"
               "有效点：<b>%1</b>　已收敛：<b>%2</b>　"
               "平均误差 前\xe2\x86\x92后：<b>%3 px \xe2\x86\x92 %4 px</b>　"
               "最大误差 前\xe2\x86\x92后：<b>%5 px \xe2\x86\x92 %6 px</b>"
               "</span>")
                .arg(validCnt).arg(convCnt)
                .arg(fmtNum(avgB)).arg(fmtNum(avgA))
                .arg(fmtNum(maxB)).arg(fmtNum(maxA)));
        ptSumLbl->setWordWrap(true);
        ptLayout->addWidget(ptSumLbl);

        // 按 rms_before 降序（最差点排前）
        std::sort(ptList.begin(), ptList.end(), [](const QJsonObject &a, const QJsonObject &b){
            return a.value("rms_before").toDouble() > b.value("rms_before").toDouble();
        });

        const QStringList ptHdr = {
            tr("编号"), tr("已收敛"), tr("迭代次数"),
            tr("RMS 前(px)"), tr("RMS 后(px)"), tr("改善(px)"), tr("改善%")
        };
        const int showRows = qMin(ptList.size(), 500);
        auto *ptTable = new QTableWidget(showRows, ptHdr.size());
        ptTable->setHorizontalHeaderLabels(ptHdr);
        ptTable->horizontalHeader()->setStretchLastSection(true);
        ptTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ptTable->setAlternatingRowColors(true);
        ptTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        ptTable->setStyleSheet("QTableWidget{border:none;}"
                               "QHeaderView::section{background:#f0f0f2;padding:4px;"
                               "border:1px solid #e0e0e0;font-weight:bold;}");

        auto makeNI_pt = [](double v, const QString &txt) -> QTableWidgetItem* {
            auto *it = new QTableWidgetItem(txt);
            it->setData(Qt::UserRole, v);
            return it;
        };

        for (int i = 0; i < showRows; ++i) {
            const QJsonObject &pj = ptList[i];
            const int    idx   = pj.value("index").toInt();
            const bool   conv  = pj.value("converged").toBool();
            const int    iters = pj.value("iterations").toInt();
            const double rb    = pj.value("rms_before").toDouble();
            const double ra    = pj.value("rms_after").toDouble();
            const double impr  = rb - ra;
            const double ipct  = (rb > 1e-9) ? impr / rb * 100.0 : 0.0;

            ptTable->setItem(i, 0, makeNI_pt(idx, QString::number(idx)));

            auto *cItem = new QTableWidgetItem(conv ? tr("\xe2\x9c\x93") : tr("\xe2\x9c\x97"));
            cItem->setForeground(conv ? QColor(40,160,80) : QColor(200,70,60));
            ptTable->setItem(i, 1, cItem);

            ptTable->setItem(i, 2, makeNI_pt(iters, QString::number(iters)));

            auto *rbItem = makeNI_pt(rb, fmtNum(rb, 4));
            if (rb > avgB * 2.5)      rbItem->setForeground(QColor(200,70,60));
            else if (rb > avgB * 1.5) rbItem->setForeground(QColor(200,140,60));
            ptTable->setItem(i, 3, rbItem);

            auto *raItem = makeNI_pt(ra, fmtNum(ra, 4));
            if (ra > avgA * 2.5)      raItem->setForeground(QColor(200,70,60));
            else if (ra > avgA * 1.5) raItem->setForeground(QColor(200,140,60));
            else                      raItem->setForeground(QColor(40,160,80));
            ptTable->setItem(i, 4, raItem);

            auto *dItem = makeNI_pt(impr, QString::asprintf("%+.4f", impr));
            dItem->setForeground(impr >= 0 ? QColor(40,160,80) : QColor(200,70,60));
            ptTable->setItem(i, 5, dItem);

            auto *pItem = makeNI_pt(ipct, QString::asprintf("%+.1f%%", ipct));
            pItem->setForeground(ipct >= 0 ? QColor(40,160,80) : QColor(200,70,60));
            ptTable->setItem(i, 6, pItem);
        }
        ptTable->setSortingEnabled(true);
        ptTable->resizeColumnsToContents();
        ptTable->setMinimumHeight(qMin(showRows * 24 + 30, 420));
        if (ptList.size() > showRows)
            ptLayout->addWidget(new QLabel(
                tr("<span style='color:#999;font-size:10px;'>"
                   "（仅显示前 500 个最差点，共 %1 个有效点）</span>").arg(ptList.size())));
        ptLayout->addWidget(ptTable);
        vl->addWidget(ptBox);
    }

    // ── 文件路径 ──────────────────────────────────────────────────────────
    if (!outDir.isEmpty() || !sparsePath.isEmpty()) {
        vl->addWidget(makeSeparator());
        auto *pathBox = new QGroupBox(tr("输出文件"));
        pathBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;margin-top:4px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *pathLayout = new QVBoxLayout(pathBox);
        if (!outDir.isEmpty())     pathLayout->addWidget(makeKVLabel(tr("输出目录:"), outDir));
        if (!sparsePath.isEmpty()) pathLayout->addWidget(makeKVLabel(tr("稀疏点云:"), sparsePath));
        vl->addWidget(pathBox);
    }

    vl->addStretch();
    return scroll;
}

// ─────────────────────────────────────────────────────────────────────────────
// 稠密点云标签页
// ─────────────────────────────────────────────────────────────────────────────
QWidget *WorkflowReportDialog::buildDenseTab()
{
    const QJsonObject r = loadReport("dense_report.json");

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea{background:#f5f5f7;}");

    auto *page = new QWidget;
    page->setStyleSheet("background:#f5f5f7;");
    auto *vl = new QVBoxLayout(page);
    vl->setContentsMargins(16, 16, 16, 24);
    vl->setSpacing(12);
    scroll->setWidget(page);

    if (r.isEmpty()) {
        auto *empty = new QLabel(tr(
            "<br><br>"
            "<center><span style='font-size:40px;'>☁</span><br><br>"
            "<b style='font-size:14px;color:#666;'>尚无稠密点云报告</b><br><br>"
            "<span style='color:#999;'>执行稠密点云生成后报告将自动显示在此处</span></center>"));
        empty->setAlignment(Qt::AlignCenter);
        vl->addWidget(empty);
        vl->addStretch();
        return scroll;
    }

    const QString ts       = r.value("timestamp").toString();
    const int    numPts    = r.value("num_points").toInt();
    const double dur       = r.value("duration_s").toDouble(-1.0);
    const QString outPath  = r.value("output_path").toString();
    const int    depthMaps = r.value("num_depth_maps").toInt();
    const int    imagesUsed= r.value("num_images_used").toInt();

    auto *headerLbl = new QLabel(
        QString("<b style='font-size:13px;'>稠密点云报告</b>"
                "<span style='color:#999;font-size:10px;'>　%1</span>").arg(ts));
    headerLbl->setStyleSheet("background:white;border-radius:4px;padding:10px;");
    vl->addWidget(headerLbl);

    auto *cards = new QHBoxLayout;
    cards->setSpacing(8);
    cards->addWidget(makeStatCard(tr("点云点数"), fmtInt(numPts), "", QColor(70,130,210)));
    if (imagesUsed > 0)
        cards->addWidget(makeStatCard(tr("参与影像"), fmtInt(imagesUsed), "", QColor(60,160,100)));
    if (depthMaps > 0)
        cards->addWidget(makeStatCard(tr("深度图数"), fmtInt(depthMaps), "", QColor(140,100,200)));
    if (dur >= 0)
        cards->addWidget(makeStatCard(tr("处理时长"),
                          dur < 60 ? QString("%1 s").arg(fmtNum(dur,1))
                                   : QString("%1 min").arg(fmtNum(dur/60,1)),
                          "", QColor(200,140,60)));
    vl->addLayout(cards);

    if (!outPath.isEmpty()) {
        auto *pathBox = new QGroupBox(tr("输出文件"));
        pathBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *pl = new QVBoxLayout(pathBox);
        pl->addWidget(makeKVLabel(tr("点云文件:"), outPath));
        vl->addWidget(pathBox);
    }

    vl->addStretch();
    return scroll;
}

// ─────────────────────────────────────────────────────────────────────────────
// 三维模型标签页
// ─────────────────────────────────────────────────────────────────────────────
QWidget *WorkflowReportDialog::buildMeshTab()
{
    const QJsonObject r = loadReport("mesh_report.json");

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setStyleSheet("QScrollArea{background:#f5f5f7;}");

    auto *page = new QWidget;
    page->setStyleSheet("background:#f5f5f7;");
    auto *vl = new QVBoxLayout(page);
    vl->setContentsMargins(16, 16, 16, 24);
    vl->setSpacing(12);
    scroll->setWidget(page);

    if (r.isEmpty()) {
        auto *empty = new QLabel(tr(
            "<br><br>"
            "<center><span style='font-size:40px;'>▲</span><br><br>"
            "<b style='font-size:14px;color:#666;'>尚无三维模型报告</b><br><br>"
            "<span style='color:#999;'>执行三维模型生成后报告将自动显示在此处</span></center>"));
        empty->setAlignment(Qt::AlignCenter);
        vl->addWidget(empty);
        vl->addStretch();
        return scroll;
    }

    const QString ts       = r.value("timestamp").toString();
    const int    numVerts  = r.value("num_vertices").toInt();
    const int    numFaces  = r.value("num_faces").toInt();
    const double dur       = r.value("duration_s").toDouble(-1.0);
    const QString outPath  = r.value("output_path").toString();
    const QString pclPath  = r.value("input_cloud_path").toString();

    auto *headerLbl = new QLabel(
        QString("<b style='font-size:13px;'>三维模型报告</b>"
                "<span style='color:#999;font-size:10px;'>　%1</span>").arg(ts));
    headerLbl->setStyleSheet("background:white;border-radius:4px;padding:10px;");
    vl->addWidget(headerLbl);

    auto *cards = new QHBoxLayout;
    cards->setSpacing(8);
    cards->addWidget(makeStatCard(tr("顶点数"), fmtInt(numVerts), "", QColor(70,130,210)));
    cards->addWidget(makeStatCard(tr("面片数"),  fmtInt(numFaces),  "", QColor(60,160,100)));
    if (dur >= 0)
        cards->addWidget(makeStatCard(tr("处理时长"),
                          dur < 60 ? QString("%1 s").arg(fmtNum(dur,1))
                                   : QString("%1 min").arg(fmtNum(dur/60,1)),
                          "", QColor(200,140,60)));
    // 面顶比
    if (numVerts > 0)
        cards->addWidget(makeStatCard(tr("面顶比"),
                          fmtNum(numFaces > 0 ? (double)numFaces / numVerts : 0.0, 2),
                          tr("理想值：约 2.0"), QColor(140,100,200)));
    vl->addLayout(cards);

    if (!outPath.isEmpty() || !pclPath.isEmpty()) {
        auto *pathBox = new QGroupBox(tr("文件路径"));
        pathBox->setStyleSheet(
            "QGroupBox{background:white;border-radius:4px;border:1px solid #ddd;"
            "font-weight:bold;padding-top:8px;}"
            "QGroupBox::title{subcontrol-origin:margin;left:10px;}");
        auto *pl = new QVBoxLayout(pathBox);
        if (!pclPath.isEmpty()) pl->addWidget(makeKVLabel(tr("输入点云:"), pclPath));
        if (!outPath.isEmpty()) pl->addWidget(makeKVLabel(tr("模型文件:"), outPath));
        vl->addWidget(pathBox);
    }

    vl->addStretch();
    return scroll;
}
