// =============================================================================
// 文件: ObservationNetworkDialog.cpp
// =============================================================================
#include "ObservationNetworkDialog.h"
#include "../widgets/ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QTabWidget>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QMessageBox>
#include <QScrollArea>

// ── 预设表 ─────────────────────────────────────────────────────────────────
namespace {
struct Preset {
    int    k;
    int    minMatch;
    double minOverlap;
    double pruneThresh;
    double verifyThresh;
    bool   pruneWeak;
};
static const Preset kPresets[] = {
    /* 低  */ {10, 10, 0.05, 0.05, 8.0, false},
    /* 中  */ {20, 30, 0.10, 0.15, 3.0, true },
    /* 高  */ {40, 60, 0.20, 0.25, 1.5, true },
    /* 自定义 */ {0, 0, 0.0, 0.0, 0.0, false},   // sentinel — no-op
};
}

// ── 构造 ──────────────────────────────────────────────────────────────────
ObservationNetworkDialog::ObservationNetworkDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("构建观测网络模型"));
    setMinimumSize(620, 560);

    auto *root = new QVBoxLayout(this);

    m_tabs = new QTabWidget(this);
    root->addWidget(m_tabs);

    // ====================================================================
    // 标签页 1 — 参数
    // ====================================================================
    {
        auto *paramScrollContent = new QWidget;
        auto *pv = new QVBoxLayout(paramScrollContent);
        pv->setSpacing(8);

        // ── 预设 ──
        {
            auto *box = new QGroupBox(tr("快速预设"));
            auto *fl = new QFormLayout(box);

            m_presetCombo = new QComboBox;
            m_presetCombo->addItems({tr("低 (快速/稀疏)"),
                                     tr("中 (均衡)"),
                                     tr("高 (精细/密集)"),
                                     tr("自定义")});
            m_presetCombo->setCurrentIndex(1);   // 默认：中
            m_presetCombo->setToolTip(tr("选择预设档位可一键设置所有关键参数。"
                                         "选 [自定义]后可手动调整每个参数。"));
            fl->addRow(tr("预设档位:"), m_presetCombo);
            pv->addWidget(box);
        }

        // ── 连接策略 ──
        {
            auto *box = new QGroupBox(tr("连接策略"));
            auto *fl = new QFormLayout(box);

            m_graphAlgoCombo = new QComboBox;
            m_graphAlgoCombo->addItems({
                tr("完全图 (Complete)"),
                tr("K 近邻 (KNN)"),
                tr("最小生成树 (MST)"),
                tr("空间邻近 (Spatial)"),
                tr("空间 KDTree")
            });
            m_graphAlgoCombo->setCurrentIndex(1);   // 默认 KNN
            m_graphAlgoCombo->setToolTip(
                tr("图构建算法。\n"
                   "• Complete: 所有有效匹配对均建立连接，适合小数据集\n"
                   "• KNN: 每张图保留 K 个最佳邻居，推荐首选\n"
                   "• MST: 最大生成树，使网络连通且冗余最小\n"
                   "• Spatial: 仅连接序号相邻的图像，适合序列影像\n"
                   "• KDTree: 基于地理/序列坐标的空间近邻搜索"));
            fl->addRow(tr("图算法:"), m_graphAlgoCombo);

            m_maxNeighborsSpin = new QSpinBox;
            m_maxNeighborsSpin->setRange(2, 200);
            m_maxNeighborsSpin->setValue(20);
            m_maxNeighborsSpin->setToolTip(
                tr("KNN / KDTree 算法中每张图像保留的最大邻居数。★ 推荐 20"));
            fl->addRow(tr("最大邻居数 K:"), m_maxNeighborsSpin);

            m_minMatchCountSpin = new QSpinBox;
            m_minMatchCountSpin->setRange(5, 5000);
            m_minMatchCountSpin->setValue(30);
            m_minMatchCountSpin->setToolTip(
                tr("建立连接所需的最少匹配点数。★ 推荐 30"));
            fl->addRow(tr("最少匹配数:"), m_minMatchCountSpin);

            m_minOverlapSpin = new QDoubleSpinBox;
            m_minOverlapSpin->setRange(0.0, 1.0);
            m_minOverlapSpin->setDecimals(2);
            m_minOverlapSpin->setSingleStep(0.05);
            m_minOverlapSpin->setValue(0.0);  // 默认不过滤（无重叠率数据时 0.1 会过滤掉所有边）
            m_minOverlapSpin->setToolTip(
                tr("建立连接所需的最小重叠率 (0~1)。★ 推荐 0.10"));
            fl->addRow(tr("最小重叠率:"), m_minOverlapSpin);

            pv->addWidget(box);
        }

        // ── 几何验证 ──
        {
            auto *box = new QGroupBox(tr("几何验证"));
            auto *fl = new QFormLayout(box);

            m_verifyMethodCombo = new QComboBox;
            m_verifyMethodCombo->addItems({
                tr("基础矩阵 (F)"),
                tr("本质矩阵 (E)"),
                tr("单应矩阵 (H)")
            });
            m_verifyMethodCombo->setCurrentIndex(0);
            m_verifyMethodCombo->setToolTip(tr("用于验证匹配一致性的几何模型"));
            fl->addRow(tr("验证方法:"), m_verifyMethodCombo);

            m_verifyThreshSpin = new QDoubleSpinBox;
            m_verifyThreshSpin->setRange(0.1, 20.0);
            m_verifyThreshSpin->setDecimals(1);
            m_verifyThreshSpin->setSingleStep(0.5);
            m_verifyThreshSpin->setValue(3.0);
            m_verifyThreshSpin->setSuffix(tr(" px"));
            m_verifyThreshSpin->setToolTip(tr("RANSAC 内点阈值。★ 推荐 3.0 px"));
            fl->addRow(tr("验证阈值:"), m_verifyThreshSpin);

            pv->addWidget(box);
        }

        // ── 剪枝 ──
        {
            auto *box = new QGroupBox(tr("弱连接剪枝"));
            auto *fl = new QFormLayout(box);

            m_pruneWeakCheck = new QCheckBox(tr("启用弱连接剪枝"));
            m_pruneWeakCheck->setChecked(true);
            m_pruneWeakCheck->setToolTip(tr("移除内点率过低的边，减少噪声"));
            fl->addRow(m_pruneWeakCheck);

            m_pruneThreshSpin = new QDoubleSpinBox;
            m_pruneThreshSpin->setRange(0.0, 1.0);
            m_pruneThreshSpin->setDecimals(2);
            m_pruneThreshSpin->setSingleStep(0.05);
            m_pruneThreshSpin->setValue(0.15);
            m_pruneThreshSpin->setToolTip(tr("内点率低于此值的边将被剪掉。★ 推荐 0.15"));
            fl->addRow(tr("剪枝阈值 (内点率):"), m_pruneThreshSpin);

            pv->addWidget(box);
        }

        // ── 系统 ──
        {
            auto *box = new QGroupBox(tr("系统"));
            auto *fl = new QFormLayout(box);
            m_threadsSpin = new QSpinBox;
            m_threadsSpin->setRange(1, 128);
            m_threadsSpin->setValue(8);
            m_threadsSpin->setToolTip(tr("并行线程数"));
            fl->addRow(tr("线程数:"), m_threadsSpin);
            pv->addWidget(box);
        }

        // ── 构建预览按钮 ──
        {
            auto *hb = new QHBoxLayout;
            hb->addStretch();
            auto *previewBtn = new QPushButton(tr("构建预览"));
            previewBtn->setToolTip(tr("根据当前参数立即构建观测网络并切换到网络图标签页预览"));
            hb->addWidget(previewBtn);
            pv->addLayout(hb);
            connect(previewBtn, &QPushButton::clicked, this, &ObservationNetworkDialog::onPreview);
        }

        pv->addStretch();

        auto *paramScroll = new QScrollArea;
        paramScroll->setWidget(paramScrollContent);
        paramScroll->setWidgetResizable(true);
        paramScroll->setFrameShape(QFrame::NoFrame);
        m_tabs->addTab(paramScroll, tr("参数"));
    }

    // ====================================================================
    // 标签页 2 — 网络图
    // ====================================================================
    {
        auto *vw = new QVBoxLayout;
        vw->setSpacing(6);

        m_netView = new ObservationNetworkView(this);
        m_netView->setMinimumSize(500, 380);
        vw->addWidget(m_netView, 1);

        auto *hb = new QHBoxLayout;
        m_statsLabel = new QLabel(tr("尚未构建 — 请在 [参数] 标签页点击 [构建预览]"));
        m_statsLabel->setWordWrap(true);
        hb->addWidget(m_statsLabel, 1);

        auto *forceBtn = new QPushButton(tr("力导向布局"));
        forceBtn->setToolTip(tr("重新运行 Fruchterman-Reingold 力导向布局算法"));
        hb->addWidget(forceBtn);
        vw->addLayout(hb);

        connect(forceBtn, &QPushButton::clicked,
                m_netView, &ObservationNetworkView::startForceLayout);

        auto *netWidget = new QWidget;
        netWidget->setLayout(vw);
        m_tabs->addTab(netWidget, tr("网络图"));
    }

    // ====================================================================
    // 底部按钮行
    // ====================================================================
    auto *btnBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("执行构建"));
    root->addWidget(btnBox);

    // ── 信号连接 ──
    auto changed = [this]() { emitSettingsNow(); };
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ObservationNetworkDialog::onPresetChanged);
    connect(m_graphAlgoCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_minMatchCountSpin, QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);
    connect(m_maxNeighborsSpin,  QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);
    connect(m_minOverlapSpin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_verifyMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_verifyThreshSpin,  QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_pruneWeakCheck,    &QCheckBox::toggled, this, changed);
    connect(m_pruneThreshSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_threadsSpin,       QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &ObservationNetworkDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // 应用中预设（触发 onPresetChanged 设参数）
    onPresetChanged(m_presetCombo->currentIndex());
}

// ── 注入数据 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::setMatchEdges(
    const QVector<xjw::MatchEdge>  &edges,
    const QStringList              &imageNames,
    const QVector<xjw::GpsCoord>   &gps)
{
    m_matchEdges  = edges;
    m_imageNames  = imageNames;
    m_gpsCoords   = gps;

    const QString info = tr("已加载 %1 张图像，%2 个匹配对")
                         .arg(imageNames.size()).arg(edges.size());
    m_statsLabel->setText(info);
}

// ── 预设切换 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::onPresetChanged(int idx)
{
    if (idx < 0 || idx >= 3) return;   // 自定义 → no-op

    const Preset &p = kPresets[idx];

    // 暂断 settingsChanged，避免多次触发
    m_maxNeighborsSpin->blockSignals(true);
    m_minMatchCountSpin->blockSignals(true);
    m_minOverlapSpin->blockSignals(true);
    m_pruneWeakCheck->blockSignals(true);
    m_pruneThreshSpin->blockSignals(true);
    m_verifyThreshSpin->blockSignals(true);

    m_maxNeighborsSpin->setValue(p.k);
    m_minMatchCountSpin->setValue(p.minMatch);
    m_minOverlapSpin->setValue(p.minOverlap);
    m_pruneWeakCheck->setChecked(p.pruneWeak);
    m_pruneThreshSpin->setValue(p.pruneThresh);
    m_verifyThreshSpin->setValue(p.verifyThresh);

    m_maxNeighborsSpin->blockSignals(false);
    m_minMatchCountSpin->blockSignals(false);
    m_minOverlapSpin->blockSignals(false);
    m_pruneWeakCheck->blockSignals(false);
    m_pruneThreshSpin->blockSignals(false);
    m_verifyThreshSpin->blockSignals(false);

    emitSettingsNow();
}

// ── 构建预览 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::onPreview()
{
    if (m_matchEdges.isEmpty())
    {
        QMessageBox::information(this, tr("无数据"),
            tr("尚未加载匹配数据，请先完成特征点匹配步骤再预览观测网络。"));
        return;
    }

    xjw::ObservationNetworkBuilder builder;
    // 将 Qt 容器转换为 std::vector
    std::vector<std::string> nodeNames;
    nodeNames.reserve(static_cast<size_t>(m_imageNames.size()));
    for (const QString &name : m_imageNames)
    {
        nodeNames.push_back(name.toStdString());
    }

    std::vector<xjw::MatchEdge> edges(m_matchEdges.begin(), m_matchEdges.end());
    std::vector<xjw::GpsCoord>  gps(m_gpsCoords.begin(), m_gpsCoords.end());

    xjw::ObservationNetwork net =
        builder.build(nodeNames, edges, gps, buildConfig());

    int nodeCount = net.nodeNames.size();
    int edgeCount = net.edges.size();
    double avgDeg = nodeCount > 0
        ? static_cast<double>(edgeCount * 2) / nodeCount
        : 0.0;

    QString statsText = tr("节点: %1  |  边: %2  |  平均度: %3")
        .arg(nodeCount)
        .arg(edgeCount)
        .arg(avgDeg, 0, 'f', 2);

    if (nodeCount >= 180 || edgeCount >= 1600)
    {
        statsText += tr("\n已自动启用大规模显示模式：默认抽样显示强连接；点击节点可聚焦局部关系。"
                        "力导向按钮此时将改用快速分层布局。");
    }

    m_statsLabel->setText(statsText);

    m_netView->setNetwork(net);
    m_netView->startForceLayout();
    m_tabs->setCurrentIndex(1);   // 切换到网络图标签
}

// ── 执行构建 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}

// ── buildConfig ───────────────────────────────────────────────────────────
xjw::ObservationNetworkConfig ObservationNetworkDialog::buildConfig() const
{
    using Algo = xjw::ObservationNetworkConfig::Algorithm;
    xjw::ObservationNetworkConfig cfg;

    static const Algo algoMap[] = {
        Algo::Complete, Algo::KNN, Algo::MST, Algo::Spatial, Algo::KDTree
    };
    int ai = m_graphAlgoCombo->currentIndex();
    cfg.algorithm   = (ai >= 0 && ai < 5) ? algoMap[ai] : Algo::KNN;

    cfg.k           = m_maxNeighborsSpin->value();
    cfg.minMatches  = m_minMatchCountSpin->value();
    cfg.minOverlap  = m_minOverlapSpin->value();
    cfg.pruneWeak   = m_pruneWeakCheck->isChecked();
    cfg.pruneThresh = m_pruneThreshSpin->value();
    return cfg;
}

// ── collectSettings ──────────────────────────────────────────────────────
QJsonObject ObservationNetworkDialog::collectSettings() const
{
    QJsonObject o;
    o["graphAlgorithm"]  = m_graphAlgoCombo->currentText();
    o["graphAlgoIndex"]  = m_graphAlgoCombo->currentIndex();
    o["maxNeighbors"]    = m_maxNeighborsSpin->value();
    o["minMatchCount"]   = m_minMatchCountSpin->value();
    o["minOverlap"]      = m_minOverlapSpin->value();
    o["verifyMethod"]    = m_verifyMethodCombo->currentText();
    o["verifyThreshold"] = m_verifyThreshSpin->value();
    o["pruneWeak"]       = m_pruneWeakCheck->isChecked();
    o["pruneThreshold"]  = m_pruneThreshSpin->value();
    o["threads"]         = m_threadsSpin->value();
    o["preset"]          = m_presetCombo->currentText();
    return o;
}

// ── applySettings ────────────────────────────────────────────────────────
void ObservationNetworkDialog::applySettings(const QJsonObject &s)
{
    // 先切换自定义（避免 preset 覆盖即将设置的值）
    m_presetCombo->setCurrentIndex(3);

    if (s.contains("graphAlgoIndex")) {
        m_graphAlgoCombo->setCurrentIndex(s["graphAlgoIndex"].toInt());
    } else if (s.contains("graphAlgorithm")) {
        int i = m_graphAlgoCombo->findText(s["graphAlgorithm"].toString());
        if (i >= 0) m_graphAlgoCombo->setCurrentIndex(i);
    }
    if (s.contains("maxNeighbors"))    m_maxNeighborsSpin->setValue(s["maxNeighbors"].toInt());
    if (s.contains("minMatchCount"))   m_minMatchCountSpin->setValue(s["minMatchCount"].toInt());
    if (s.contains("minOverlap"))      m_minOverlapSpin->setValue(s["minOverlap"].toDouble());
    if (s.contains("verifyMethod")) {
        int i = m_verifyMethodCombo->findText(s["verifyMethod"].toString());
        if (i >= 0) m_verifyMethodCombo->setCurrentIndex(i);
    }
    if (s.contains("verifyThreshold")) m_verifyThreshSpin->setValue(s["verifyThreshold"].toDouble());
    if (s.contains("pruneWeak"))       m_pruneWeakCheck->setChecked(s["pruneWeak"].toBool());
    if (s.contains("pruneThreshold"))  m_pruneThreshSpin->setValue(s["pruneThreshold"].toDouble());
    if (s.contains("threads"))         m_threadsSpin->setValue(s["threads"].toInt());
}

// ── emitSettingsNow ──────────────────────────────────────────────────────
void ObservationNetworkDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

