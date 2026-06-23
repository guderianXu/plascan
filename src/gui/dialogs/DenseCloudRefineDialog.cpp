#include "DenseCloudRefineDialog.h"
#include "ui_DenseCloudRefineDialog.h"

#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QPushButton>

DenseCloudRefineDialog::DenseCloudRefineDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::DenseCloudRefineDialog form;
    form.setupUi(this);

    _sorGroup = form.m_sorGroup;
    _voxelGroup = form.m_voxelGroup;
    _normalGroup = form.m_normalGroup;
    _colorGroup = form.m_colorGroup;
    _sorKSpin = form.m_sorKSpin;
    _sorStdSpin = form.m_sorStdSpin;
    _voxelSizeSpin = form.m_voxelSizeSpin;
    _normalKSpin = form.m_normalKSpin;
    _smoothIterSpin = form.m_smoothIterSpin;
    _colorMethodCombo = form.m_colorMethodCombo;
    _threadsSpin = form.m_threadsSpin;

    auto changed = [this]()
    {
        emitSettingsNow();
    };
    connect(_sorGroup, &QGroupBox::toggled, this, changed);
    connect(_sorKSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_sorStdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_voxelGroup, &QGroupBox::toggled, this, changed);
    connect(_voxelSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_normalGroup, &QGroupBox::toggled, this, changed);
    connect(_normalKSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_smoothIterSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_colorGroup, &QGroupBox::toggled, this, changed);
    connect(_colorMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &DenseCloudRefineDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject DenseCloudRefineDialog::collectSettings() const
{
    QJsonObject o;
    o["sorEnabled"] = _sorGroup->isChecked();
    o["sorK"] = _sorKSpin->value();
    o["sorStdDev"] = _sorStdSpin->value();
    o["voxelEnabled"] = _voxelGroup->isChecked();
    o["voxelSize"] = _voxelSizeSpin->value();
    o["normalsEnabled"] = _normalGroup->isChecked();
    o["normalK"] = _normalKSpin->value();
    o["smoothIter"] = _smoothIterSpin->value();
    o["colorEnabled"] = _colorGroup->isChecked();
    o["colorMethod"] = _colorMethodCombo->currentText();
    o["threads"] = _threadsSpin->value();
    return o;
}

void DenseCloudRefineDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("sorEnabled"))
    {
        _sorGroup->setChecked(s["sorEnabled"].toBool());
    }
    if (s.contains("sorK"))
    {
        _sorKSpin->setValue(s["sorK"].toInt());
    }
    if (s.contains("sorStdDev"))
    {
        _sorStdSpin->setValue(s["sorStdDev"].toDouble());
    }
    if (s.contains("voxelEnabled"))
    {
        _voxelGroup->setChecked(s["voxelEnabled"].toBool());
    }
    if (s.contains("voxelSize"))
    {
        _voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    }
    if (s.contains("normalsEnabled"))
    {
        _normalGroup->setChecked(s["normalsEnabled"].toBool());
    }
    if (s.contains("normalK"))
    {
        _normalKSpin->setValue(s["normalK"].toInt());
    }
    if (s.contains("smoothIter"))
    {
        _smoothIterSpin->setValue(s["smoothIter"].toInt());
    }
    if (s.contains("colorEnabled"))
    {
        _colorGroup->setChecked(s["colorEnabled"].toBool());
    }
    if (s.contains("colorMethod"))
    {
        int i = _colorMethodCombo->findText(s["colorMethod"].toString());
        if (i >= 0)
        {
            _colorMethodCombo->setCurrentIndex(i);
        }
    }
    // 向后兼容旧键名
    if (!s.contains("normalsEnabled") && s.contains("estimateNormals"))
    {
        _normalGroup->setChecked(s["estimateNormals"].toBool());
    }
    if (!s.contains("colorEnabled") && s.contains("colorCorrection"))
    {
        _colorGroup->setChecked(s["colorCorrection"].toBool());
    }
    if (s.contains("threads"))
    {
        _threadsSpin->setValue(s["threads"].toInt());
    }
}

void DenseCloudRefineDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void DenseCloudRefineDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
