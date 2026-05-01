#include "TriangulationDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>
#include <cmath>

TriangulationDialog::TriangulationDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("生成初始稀疏点云"));
    setMinimumWidth(500);

    auto *root = new QVBoxLayout(this);

    // ── 预设 ──
    {
        auto *box = new QGroupBox(tr("预设方案"));
        auto *fl = new QFormLayout(box);
        m_presetCombo = new QComboBox;
        m_presetCombo->addItems({tr("宽松 (更多点)"), tr("标准"), tr("严格 (高质量)"), tr("自定义")});
        m_presetCombo->setCurrentIndex(1);
        m_presetCombo->setToolTip(tr("选择预设后自动调整下方参数"));
        fl->addRow(tr("预设:"), m_presetCombo);
        root->addWidget(box);
    }

    // ── 三角化条件 ──
    {
        auto *box = new QGroupBox(tr("三角化条件"));
        auto *fl = new QFormLayout(box);

        m_minAngleSpin = new QDoubleSpinBox;
        m_minAngleSpin->setRange(0.1, 30.0);
        m_minAngleSpin->setDecimals(1);
        m_minAngleSpin->setSingleStep(0.5);
        m_minAngleSpin->setValue(2.0);
        m_minAngleSpin->setSuffix(QString::fromUtf8(" \u00b0"));
        m_minAngleSpin->setToolTip(tr("两条射线最小夹角。★ 推荐 2.0; 角度过小导致深度不确定性大"));
        fl->addRow(tr("最小交会角:"), m_minAngleSpin);

        m_reprojThreshSpin = new QDoubleSpinBox;
        m_reprojThreshSpin->setRange(0.1, 20.0);
        m_reprojThreshSpin->setDecimals(1);
        m_reprojThreshSpin->setSingleStep(0.5);
        m_reprojThreshSpin->setValue(2.0);
        m_reprojThreshSpin->setSuffix(tr(" px"));
        m_reprojThreshSpin->setToolTip(tr("三角化点反投影最大允许误差。★ 推荐 2.0 px"));
        fl->addRow(tr("重投影阈值:"), m_reprojThreshSpin);

        m_minObsSpin = new QSpinBox;
        m_minObsSpin->setRange(2, 50);
        m_minObsSpin->setValue(2);
        m_minObsSpin->setToolTip(tr("每个三维点至少需在多少张图像中被观测到。当前初始三角化推荐 2"));
        fl->addRow(tr("最少观测数:"), m_minObsSpin);

        m_ignoreTwoViewCheck = new QCheckBox(tr("忽略仅两视图观测"));
        m_ignoreTwoViewCheck->setChecked(false);
        m_ignoreTwoViewCheck->setToolTip(tr("开启后仅在两张图出现的特征点将不被三角化"));
        fl->addRow(m_ignoreTwoViewCheck);

        m_depthStabSpin = new QDoubleSpinBox;
        m_depthStabSpin->setRange(0.01, 100.0);
        m_depthStabSpin->setDecimals(2);
        m_depthStabSpin->setSingleStep(0.1);
        m_depthStabSpin->setValue(1.0);
        m_depthStabSpin->setToolTip(tr("深度估计标准差/深度值。★ 推荐 1.0; 越小越严格"));
        fl->addRow(tr("深度稳定性:"), m_depthStabSpin);

        root->addWidget(box);
    }

    // ── 过滤策略 ──
    {
        auto *box = new QGroupBox(tr("低质量点过滤"));
        auto *fl = new QFormLayout(box);

        m_filterModeCombo = new QComboBox;
        m_filterModeCombo->addItems({tr("不过滤"), tr("仅统计报告"), tr("标记但保留"), tr("直接移除")});
        m_filterModeCombo->setCurrentIndex(3);
        fl->addRow(tr("过滤模式:"), m_filterModeCombo);

        m_maxReprojErrSpin = new QDoubleSpinBox;
        m_maxReprojErrSpin->setRange(0.1, 50.0);
        m_maxReprojErrSpin->setDecimals(1);
        m_maxReprojErrSpin->setSingleStep(0.5);
        m_maxReprojErrSpin->setValue(4.0);
        m_maxReprojErrSpin->setSuffix(tr(" px"));
        m_maxReprojErrSpin->setToolTip(tr("★ 推荐 4.0 px"));
        fl->addRow(tr("过滤重投影阈值:"), m_maxReprojErrSpin);

        m_minAngleFiltSpin = new QDoubleSpinBox;
        m_minAngleFiltSpin->setRange(0.1, 30.0);
        m_minAngleFiltSpin->setDecimals(1);
        m_minAngleFiltSpin->setValue(1.5);
        m_minAngleFiltSpin->setSuffix(QString::fromUtf8(" \u00b0"));
        fl->addRow(tr("过滤最小角:"), m_minAngleFiltSpin);

        m_minTrackLenSpin = new QSpinBox;
        m_minTrackLenSpin->setRange(2, 50);
        m_minTrackLenSpin->setValue(3);
        fl->addRow(tr("最短 track:"), m_minTrackLenSpin);

        root->addWidget(box);
    }

    // ── 系统 ──
    {
        auto *box = new QGroupBox(tr("系统"));
        auto *fl = new QFormLayout(box);
        m_threadsSpin = new QSpinBox;
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        fl->addRow(tr("线程数:"), m_threadsSpin);

        m_overwriteResultCheck = new QCheckBox(tr("覆盖最近一次初始稀疏点云结果"), this);
        m_overwriteResultCheck->setChecked(false);
        m_overwriteResultCheck->setToolTip(tr("开启后会复用最近一次初始稀疏点云的输出目录，并在 DataTree 中覆盖该记录。"));
        fl->addRow(m_overwriteResultCheck);
        root->addWidget(box);
    }

    // ── 动态阈值建议 ──
    {
        auto *box = new QGroupBox(tr("动态阈值建议"));
        auto *fl = new QFormLayout(box);

        m_focalLenSpin = new QDoubleSpinBox;
        m_focalLenSpin->setRange(100.0, 100000.0);
        m_focalLenSpin->setDecimals(1);
        m_focalLenSpin->setValue(3000.0);
        m_focalLenSpin->setSuffix(tr(" px"));
        fl->addRow(tr("参考焦距:"), m_focalLenSpin);

        m_baselineSpin = new QDoubleSpinBox;
        m_baselineSpin->setRange(0.01, 10000.0);
        m_baselineSpin->setDecimals(2);
        m_baselineSpin->setValue(1.0);
        m_baselineSpin->setSuffix(tr(" m"));
        fl->addRow(tr("参考基线:"), m_baselineSpin);

        auto *sugRow = new QHBoxLayout;
        m_suggestBtn = new QPushButton(tr("计算建议"));
        sugRow->addWidget(m_suggestBtn);
        m_suggestLabel = new QLabel(tr("点击按钮获取建议"));
        sugRow->addWidget(m_suggestLabel, 1);
        fl->addRow(sugRow);

        root->addWidget(box);
    }

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("运行三角化"));
    root->addWidget(btnBox);

    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TriangulationDialog::onPresetChanged);
    connect(m_suggestBtn, &QPushButton::clicked,
            this, &TriangulationDialog::onSuggestThresholds);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_minAngleSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_reprojThreshSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_minObsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_ignoreTwoViewCheck, &QCheckBox::toggled, this, changed);
    connect(m_depthStabSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_filterModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_overwriteResultCheck, &QCheckBox::toggled, this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &TriangulationDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onPresetChanged(1);
}

void TriangulationDialog::onPresetChanged(int index)
{
    if (index == 3) return;
    struct P { double a; double r; int o; bool i; double d; int fm; double mr; double ma; int mt; };
    static const P kP[] = {
        {1.0, 4.0, 2, false, 2.0, 3, 8.0, 0.5, 2},
        {2.0, 2.0, 2, false, 1.0, 3, 4.0, 1.5, 2},
        {3.0, 1.0, 4, true,  0.5, 3, 2.0, 2.5, 4},
    };
    const auto &p = kP[index];
    m_minAngleSpin->setValue(p.a);
    m_reprojThreshSpin->setValue(p.r);
    m_minObsSpin->setValue(p.o);
    m_ignoreTwoViewCheck->setChecked(p.i);
    m_depthStabSpin->setValue(p.d);
    m_filterModeCombo->setCurrentIndex(p.fm);
    m_maxReprojErrSpin->setValue(p.mr);
    m_minAngleFiltSpin->setValue(p.ma);
    m_minTrackLenSpin->setValue(p.mt);
}

void TriangulationDialog::onSuggestThresholds()
{
    double f = m_focalLenSpin->value();
    double b = m_baselineSpin->value();
    double avgDepth = f * b / 50.0;
    double sugAngle = std::atan2(b, avgDepth) * 180.0 / M_PI;
    sugAngle = std::max(0.5, std::min(sugAngle, 10.0));
    double sugReproj = std::max(0.5, std::min(4.0, 2.0 * 3000.0 / f));

    m_suggestLabel->setText(
        tr("建议: 交会角>=%1, 重投影<=%2 px (f=%3, b=%4)")
            .arg(sugAngle, 0, 'f', 1).arg(sugReproj, 0, 'f', 1)
            .arg(f, 0, 'f', 0).arg(b, 0, 'f', 2));
    m_presetCombo->setCurrentIndex(3);
    m_minAngleSpin->setValue(sugAngle);
    m_reprojThreshSpin->setValue(sugReproj);
}

QJsonObject TriangulationDialog::collectSettings() const
{
    QJsonObject o;
    o["preset"]          = m_presetCombo->currentText();
    o["minAngle"]        = m_minAngleSpin->value();
    o["reprojThreshold"] = m_reprojThreshSpin->value();
    o["minObservations"] = m_minObsSpin->value();
    o["ignoreTwoView"]   = m_ignoreTwoViewCheck->isChecked();
    o["depthStability"]  = m_depthStabSpin->value();
    o["filterMode"]      = m_filterModeCombo->currentText();
    o["maxReprojError"]  = m_maxReprojErrSpin->value();
    o["minAngleFilter"]  = m_minAngleFiltSpin->value();
    o["minTrackLen"]     = m_minTrackLenSpin->value();
    o["threads"]         = m_threadsSpin->value();
    o["overwriteExistingResult"] = m_overwriteResultCheck->isChecked();
    return o;
}

void TriangulationDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("minAngle"))        m_minAngleSpin->setValue(s["minAngle"].toDouble());
    if (s.contains("reprojThreshold")) m_reprojThreshSpin->setValue(s["reprojThreshold"].toDouble());
    if (s.contains("minObservations")) m_minObsSpin->setValue(s["minObservations"].toInt());
    if (s.contains("ignoreTwoView"))   m_ignoreTwoViewCheck->setChecked(s["ignoreTwoView"].toBool());
    if (s.contains("depthStability"))  m_depthStabSpin->setValue(s["depthStability"].toDouble());
    if (s.contains("filterMode"))
    {
        int i = m_filterModeCombo->findText(s["filterMode"].toString());
        if (i >= 0) m_filterModeCombo->setCurrentIndex(i);
    }
    if (s.contains("maxReprojError"))  m_maxReprojErrSpin->setValue(s["maxReprojError"].toDouble());
    if (s.contains("minAngleFilter"))  m_minAngleFiltSpin->setValue(s["minAngleFilter"].toDouble());
    if (s.contains("minTrackLen"))     m_minTrackLenSpin->setValue(s["minTrackLen"].toInt());
    if (s.contains("threads"))         m_threadsSpin->setValue(s["threads"].toInt());
    if (s.contains("overwriteExistingResult")) m_overwriteResultCheck->setChecked(s["overwriteExistingResult"].toBool());
}

void TriangulationDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void TriangulationDialog::onRun() { emit runRequested(collectSettings()); accept(); }
