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

    _tabs = ui.m_tabs;
    _statsLabel = ui.m_statsLabel;
    _presetCombo = ui.m_presetCombo;
    _graphAlgoCombo = ui.m_graphAlgoCombo;
    _maxNeighborsSpin = ui.m_maxNeighborsSpin;
    _minMatchCountSpin = ui.m_minMatchCountSpin;
    _minOverlapSpin = ui.m_minOverlapSpin;
    _verifyMethodCombo = ui.m_verifyMethodCombo;
    _verifyThreshSpin = ui.m_verifyThreshSpin;
    _pruneWeakCheck = ui.m_pruneWeakCheck;
    _pruneThreshSpin = ui.m_pruneThreshSpin;
    _threadsSpin = ui.m_threadsSpin;
    _netView = ui.m_netView;

    ui.buttonBox->button(QDialogButtonBox::Ok)->setText(tr("执行构建"));

    // ── 信号连接 ──
    auto changed = [this]() { emitSettingsNow(); };
    connect(_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ObservationNetworkDialog::onPresetChanged);
    connect(_graphAlgoCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_minMatchCountSpin, QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);
    connect(_maxNeighborsSpin,  QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);
    connect(_minOverlapSpin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_verifyMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_verifyThreshSpin,  QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_pruneWeakCheck,    &QCheckBox::toggled, this, changed);
    connect(_pruneThreshSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_threadsSpin,       QOverload<int>::of(&QSpinBox::valueChanged),      this, changed);

    connect(ui.previewBtn, &QPushButton::clicked, this, &ObservationNetworkDialog::onPreview);
    connect(ui.forceLayoutBtn, &QPushButton::clicked,
            _netView, &ObservationNetworkView::startForceLayout);
    connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &ObservationNetworkDialog::onRun);
    connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    // 应用中预设（触发 onPresetChanged 设参数）
    onPresetChanged(_presetCombo->currentIndex());
}

// ── 注入数据 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::setMatchEdges(
    const QVector<xjw::MatchEdge>  &edges,
    const QStringList              &imageNames,
    const QVector<xjw::GpsCoord>   &gps)
{
    _matchEdges  = edges;
    _imageNames  = imageNames;
    _gpsCoords   = gps;

    const QString info = tr("已加载 %1 张图像，%2 个匹配对")
                         .arg(imageNames.size()).arg(edges.size());
    _statsLabel->setText(info);
}

// ── 预设切换 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::onPresetChanged(int idx)
{
    if (idx < 0 || idx >= 3) return;   // 自定义 → no-op

    const Preset &p = kPresets[idx];

    // 暂断 settingsChanged，避免多次触发
    _maxNeighborsSpin->blockSignals(true);
    _minMatchCountSpin->blockSignals(true);
    _minOverlapSpin->blockSignals(true);
    _pruneWeakCheck->blockSignals(true);
    _pruneThreshSpin->blockSignals(true);
    _verifyThreshSpin->blockSignals(true);

    _maxNeighborsSpin->setValue(p.k);
    _minMatchCountSpin->setValue(p.minMatch);
    _minOverlapSpin->setValue(p.minOverlap);
    _pruneWeakCheck->setChecked(p.pruneWeak);
    _pruneThreshSpin->setValue(p.pruneThresh);
    _verifyThreshSpin->setValue(p.verifyThresh);

    _maxNeighborsSpin->blockSignals(false);
    _minMatchCountSpin->blockSignals(false);
    _minOverlapSpin->blockSignals(false);
    _pruneWeakCheck->blockSignals(false);
    _pruneThreshSpin->blockSignals(false);
    _verifyThreshSpin->blockSignals(false);

    emitSettingsNow();
}

// ── 构建预览 ──────────────────────────────────────────────────────────────
void ObservationNetworkDialog::onPreview()
{
    if (_matchEdges.isEmpty())
    {
        QMessageBox::information(this, tr("无数据"),
            tr("尚未加载匹配数据，请先完成特征点匹配步骤再预览观测网络。"));
        return;
    }

    xjw::ObservationNetworkBuilder builder;
    // 将 Qt 容器转换为 std::vector
    std::vector<std::string> nodeNames;
    nodeNames.reserve(static_cast<size_t>(_imageNames.size()));
    for (const QString &name : _imageNames)
    {
        nodeNames.push_back(name.toStdString());
    }

    std::vector<xjw::MatchEdge> edges(_matchEdges.begin(), _matchEdges.end());
    std::vector<xjw::GpsCoord>  gps(_gpsCoords.begin(), _gpsCoords.end());

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

    _statsLabel->setText(statsText);

    _netView->setNetwork(net);
    _netView->startForceLayout();
    _tabs->setCurrentIndex(1);   // 切换到网络图标签
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
    int ai = _graphAlgoCombo->currentIndex();
    cfg.algorithm   = (ai >= 0 && ai < 5) ? algoMap[ai] : Algo::KNN;

    cfg.k           = _maxNeighborsSpin->value();
    cfg.minMatches  = _minMatchCountSpin->value();
    cfg.minOverlap  = _minOverlapSpin->value();
    cfg.pruneWeak   = _pruneWeakCheck->isChecked();
    cfg.pruneThresh = _pruneThreshSpin->value();
    return cfg;
}

// ── collectSettings ──────────────────────────────────────────────────────
QJsonObject ObservationNetworkDialog::collectSettings() const
{
    QJsonObject o;
    o["graphAlgorithm"]  = _graphAlgoCombo->currentText();
    o["graphAlgoIndex"]  = _graphAlgoCombo->currentIndex();
    o["maxNeighbors"]    = _maxNeighborsSpin->value();
    o["minMatchCount"]   = _minMatchCountSpin->value();
    o["minOverlap"]      = _minOverlapSpin->value();
    o["verifyMethod"]    = _verifyMethodCombo->currentText();
    o["verifyThreshold"] = _verifyThreshSpin->value();
    o["pruneWeak"]       = _pruneWeakCheck->isChecked();
    o["pruneThreshold"]  = _pruneThreshSpin->value();
    o["threads"]         = _threadsSpin->value();
    o["preset"]          = _presetCombo->currentText();
    return o;
}

// ── applySettings ────────────────────────────────────────────────────────
void ObservationNetworkDialog::applySettings(const QJsonObject &s)
{
    // 先切换自定义（避免 preset 覆盖即将设置的值）
    _presetCombo->setCurrentIndex(3);

    if (s.contains("graphAlgoIndex")) {
        _graphAlgoCombo->setCurrentIndex(s["graphAlgoIndex"].toInt());
    } else if (s.contains("graphAlgorithm")) {
        int i = _graphAlgoCombo->findText(s["graphAlgorithm"].toString());
        if (i >= 0) _graphAlgoCombo->setCurrentIndex(i);
    }
    if (s.contains("maxNeighbors"))    _maxNeighborsSpin->setValue(s["maxNeighbors"].toInt());
    if (s.contains("minMatchCount"))   _minMatchCountSpin->setValue(s["minMatchCount"].toInt());
    if (s.contains("minOverlap"))      _minOverlapSpin->setValue(s["minOverlap"].toDouble());
    if (s.contains("verifyMethod")) {
        int i = _verifyMethodCombo->findText(s["verifyMethod"].toString());
        if (i >= 0) _verifyMethodCombo->setCurrentIndex(i);
    }
    if (s.contains("verifyThreshold")) _verifyThreshSpin->setValue(s["verifyThreshold"].toDouble());
    if (s.contains("pruneWeak"))       _pruneWeakCheck->setChecked(s["pruneWeak"].toBool());
    if (s.contains("pruneThreshold"))  _pruneThreshSpin->setValue(s["pruneThreshold"].toDouble());
    if (s.contains("threads"))         _threadsSpin->setValue(s["threads"].toInt());
}

// ── emitSettingsNow ──────────────────────────────────────────────────────
void ObservationNetworkDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}
