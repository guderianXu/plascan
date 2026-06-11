// =============================================================================
// 文件: ObservationNetworkDialog.cpp
// =============================================================================
#include "ObservationNetworkDialog.h"
#include "../widgets/ObservationNetworkView.h"
#include "graph/ObservationNetworkBuilder.h"
#include "ui_ObservationNetworkDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QTabWidget>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QMessageBox>

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
    Ui::ObservationNetworkDialog ui;
    ui.setupUi(this);

    m_tabs = ui.m_tabs;
    m_statsLabel = ui.m_statsLabel;
    m_presetCombo = ui.m_presetCombo;
    m_graphAlgoCombo = ui.m_graphAlgoCombo;
    m_maxNeighborsSpin = ui.m_maxNeighborsSpin;
    m_minMatchCountSpin = ui.m_minMatchCountSpin;
    m_minOverlapSpin = ui.m_minOverlapSpin;
    m_verifyMethodCombo = ui.m_verifyMethodCombo;
    m_verifyThreshSpin = ui.m_verifyThreshSpin;
    m_pruneWeakCheck = ui.m_pruneWeakCheck;
    m_pruneThreshSpin = ui.m_pruneThreshSpin;
    m_threadsSpin = ui.m_threadsSpin;
    m_netView = ui.m_netView;

    ui.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("执行构建"));

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

    connect(ui.previewBtn, &QPushButton::clicked, this, &ObservationNetworkDialog::onPreview);
    connect(ui.forceLayoutBtn, &QPushButton::clicked,
            m_netView, &ObservationNetworkView::startForceLayout);
    connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ObservationNetworkDialog::onRun);
    connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

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
