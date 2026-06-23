#include "TriangulationDialog.h"
#include "ui_TriangulationDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QFormLayout>
#include <algorithm>
#include <cmath>

TriangulationDialog::TriangulationDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TriangulationDialog form;
    form.setupUi(this);

    _presetCombo = form.m_presetCombo;
    _minAngleSpin = form.m_minAngleSpin;
    _reprojThreshSpin = form.m_reprojThreshSpin;
    _minObsSpin = form.m_minObsSpin;
    _ignoreTwoViewCheck = form.m_ignoreTwoViewCheck;
    _depthStabSpin = form.m_depthStabSpin;
    _filterModeCombo = form.m_filterModeCombo;
    _maxReprojErrSpin = form.m_maxReprojErrSpin;
    _minAngleFiltSpin = form.m_minAngleFiltSpin;
    _minTrackLenSpin = form.m_minTrackLenSpin;
    _threadsSpin = form.m_threadsSpin;
    _focalLenSpin = form.m_focalLenSpin;
    _baselineSpin = form.m_baselineSpin;
    _overwriteResultCheck = form.m_overwriteResultCheck;
    _suggestBtn = form.m_suggestBtn;
    _suggestLabel = form.m_suggestLabel;

    auto hideUnsupportedField = [](QWidget *widget)
    {
        if (!widget || !widget->parentWidget())
        {
            return;
        }
        if (auto *layout = qobject_cast<QFormLayout *>(widget->parentWidget()->layout()))
        {
            if (auto *label = layout->labelForField(widget))
            {
                label->hide();
            }
        }
        widget->hide();
    };
    hideUnsupportedField(_depthStabSpin);
    hideUnsupportedField(_filterModeCombo);
    hideUnsupportedField(_maxReprojErrSpin);
    hideUnsupportedField(_minAngleFiltSpin);

    connect(_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &TriangulationDialog::onPresetChanged);
    connect(_suggestBtn, &QPushButton::clicked,
            this, &TriangulationDialog::onSuggestThresholds);

    auto changed = [this]() { emitSettingsNow(); };
    connect(_minAngleSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_reprojThreshSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_minObsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_ignoreTwoViewCheck, &QCheckBox::toggled, this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_overwriteResultCheck, &QCheckBox::toggled, this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &TriangulationDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);

    onPresetChanged(1);
}

void TriangulationDialog::onPresetChanged(int index)
{
    if (index == 3)
    {
        return;
    }
    struct P { double a; double r; int o; bool i; double d; int fm; double mr; double ma; int mt; };
    static const P kP[] = {
        {1.0, 4.0, 2, false, 2.0, 3, 8.0, 0.5, 2},
        {2.0, 2.0, 2, false, 1.0, 3, 4.0, 1.5, 2},
        {3.0, 1.0, 4, true,  0.5, 3, 2.0, 2.5, 4},
    };
    const auto &p = kP[index];
    _minAngleSpin->setValue(p.a);
    _reprojThreshSpin->setValue(p.r);
    _minObsSpin->setValue(p.o);
    _ignoreTwoViewCheck->setChecked(p.i);
    _depthStabSpin->setValue(p.d);
    _filterModeCombo->setCurrentIndex(p.fm);
    _maxReprojErrSpin->setValue(p.mr);
    _minAngleFiltSpin->setValue(p.ma);
    _minTrackLenSpin->setValue(p.mt);
}

void TriangulationDialog::onSuggestThresholds()
{
    double f = _focalLenSpin->value();
    double b = _baselineSpin->value();
    double avgDepth = f * b / 50.0;
    double sugAngle = std::atan2(b, avgDepth) * 180.0 / M_PI;
    sugAngle = std::max(0.5, std::min(sugAngle, 10.0));
    double sugReproj = std::max(0.5, std::min(4.0, 2.0 * 3000.0 / f));

    _suggestLabel->setText(
        tr("建议: 交会角>=%1, 重投影<=%2 px (f=%3, b=%4)")
            .arg(sugAngle, 0, 'f', 1).arg(sugReproj, 0, 'f', 1)
            .arg(f, 0, 'f', 0).arg(b, 0, 'f', 2));
    _presetCombo->setCurrentIndex(3);
    _minAngleSpin->setValue(sugAngle);
    _reprojThreshSpin->setValue(sugReproj);
}

QJsonObject TriangulationDialog::collectSettings() const
{
    QJsonObject o;
    o["preset"] = _presetCombo->currentText();
    o["minAngle"] = _minAngleSpin->value();
    o["reprojThreshold"] = _reprojThreshSpin->value();
    o["minObservations"] = _minObsSpin->value();
    o["ignoreTwoView"] = _ignoreTwoViewCheck->isChecked();
    o["minTrackLen"] = _minTrackLenSpin->value();
    o["threads"] = _threadsSpin->value();
    o["overwriteExistingResult"] = _overwriteResultCheck->isChecked();
    return o;
}

void TriangulationDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("minAngle"))
    {
        _minAngleSpin->setValue(s["minAngle"].toDouble());
    }
    if (s.contains("reprojThreshold"))
    {
        _reprojThreshSpin->setValue(s["reprojThreshold"].toDouble());
    }
    if (s.contains("minObservations"))
    {
        _minObsSpin->setValue(s["minObservations"].toInt());
    }
    if (s.contains("ignoreTwoView"))
    {
        _ignoreTwoViewCheck->setChecked(s["ignoreTwoView"].toBool());
    }
    if (s.contains("minTrackLen"))
    {
        _minTrackLenSpin->setValue(s["minTrackLen"].toInt());
    }
    if (s.contains("threads"))
    {
        _threadsSpin->setValue(s["threads"].toInt());
    }
    if (s.contains("overwriteExistingResult"))
    {
        _overwriteResultCheck->setChecked(s["overwriteExistingResult"].toBool());
    }
}

void TriangulationDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void TriangulationDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
