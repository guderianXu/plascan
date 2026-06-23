#include "DepthFusionDialog.h"
#include "ui_DepthFusionDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QPushButton>

DepthFusionDialog::DepthFusionDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::DepthFusionDialog form;
    form.setupUi(this);

    _fusionMethodCombo = form.m_fusionMethodCombo;
    _depthConsistSpin = form.m_depthConsistSpin;
    _minConsistViewSpin = form.m_minConsistViewSpin;
    _normalConsistSpin = form.m_normalConsistSpin;
    _voxelSizeSpin = form.m_voxelSizeSpin;
    _minConfidenceSpin = form.m_minConfidenceSpin;
    _maxReprojSpin = form.m_maxReprojSpin;
    _colorCheck = form.m_colorCheck;
    _normalCheck = form.m_normalCheck;
    _threadsSpin = form.m_threadsSpin;
    _cudaCheck = form.m_cudaCheck;
    _infoLabel = form.m_infoLabel;
    if (_cudaCheck)
    {
        _cudaCheck->setChecked(false);
        _cudaCheck->setEnabled(false);
        _cudaCheck->hide();
    }

    auto changed = [this]()
    {
        emitSettingsNow();
    };
    connect(_fusionMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_depthConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_minConsistViewSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_normalConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_voxelSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_minConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_maxReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_colorCheck, &QCheckBox::toggled, this, changed);
    connect(_normalCheck, &QCheckBox::toggled, this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &DepthFusionDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject DepthFusionDialog::collectSettings() const
{
    QJsonObject o;
    o["fusionMethod"] = _fusionMethodCombo->currentText();
    o["depthConsistency"] = _depthConsistSpin->value();
    o["minConsistentViews"] = _minConsistViewSpin->value();
    o["normalConsistency"] = _normalConsistSpin->value();
    o["voxelSize"] = _voxelSizeSpin->value();
    o["minConfidence"] = _minConfidenceSpin->value();
    o["maxReprojError"] = _maxReprojSpin->value();
    o["keepColor"] = _colorCheck->isChecked();
    o["keepNormals"] = _normalCheck->isChecked();
    o["threads"] = _threadsSpin->value();
    return o;
}

void DepthFusionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("fusionMethod"))
    {
        int i = _fusionMethodCombo->findText(s["fusionMethod"].toString());
        if (i >= 0)
        {
            _fusionMethodCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("depthConsistency"))
    {
        _depthConsistSpin->setValue(s["depthConsistency"].toDouble());
    }
    if (s.contains("minConsistentViews"))
    {
        _minConsistViewSpin->setValue(s["minConsistentViews"].toInt());
    }
    if (s.contains("normalConsistency"))
    {
        _normalConsistSpin->setValue(s["normalConsistency"].toDouble());
    }
    if (s.contains("voxelSize"))
    {
        _voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    }
    if (s.contains("minConfidence"))
    {
        _minConfidenceSpin->setValue(s["minConfidence"].toDouble());
    }
    if (s.contains("maxReprojError"))
    {
        _maxReprojSpin->setValue(s["maxReprojError"].toDouble());
    }
    if (s.contains("keepColor"))
    {
        _colorCheck->setChecked(s["keepColor"].toBool());
    }
    if (s.contains("keepNormals"))
    {
        _normalCheck->setChecked(s["keepNormals"].toBool());
    }
    if (s.contains("threads"))
    {
        _threadsSpin->setValue(s["threads"].toInt());
    }
}

void DepthFusionDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void DepthFusionDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
