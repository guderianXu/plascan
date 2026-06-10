#include "TriangulationDialog.h"
#include "ui_TriangulationDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <cmath>

TriangulationDialog::TriangulationDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TriangulationDialog form;
    form.setupUi(this);

    m_presetCombo          = form.m_presetCombo;
    m_minAngleSpin         = form.m_minAngleSpin;
    m_reprojThreshSpin     = form.m_reprojThreshSpin;
    m_minObsSpin           = form.m_minObsSpin;
    m_ignoreTwoViewCheck   = form.m_ignoreTwoViewCheck;
    m_depthStabSpin        = form.m_depthStabSpin;
    m_filterModeCombo      = form.m_filterModeCombo;
    m_maxReprojErrSpin     = form.m_maxReprojErrSpin;
    m_minAngleFiltSpin     = form.m_minAngleFiltSpin;
    m_minTrackLenSpin      = form.m_minTrackLenSpin;
    m_threadsSpin          = form.m_threadsSpin;
    m_focalLenSpin         = form.m_focalLenSpin;
    m_baselineSpin         = form.m_baselineSpin;
    m_overwriteResultCheck = form.m_overwriteResultCheck;
    m_suggestBtn           = form.m_suggestBtn;
    m_suggestLabel         = form.m_suggestLabel;

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

    connect(form.m_runBtn, &QPushButton::clicked, this, &TriangulationDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

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
