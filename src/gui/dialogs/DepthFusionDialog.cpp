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

    m_fusionMethodCombo  = form.m_fusionMethodCombo;
    m_depthConsistSpin   = form.m_depthConsistSpin;
    m_minConsistViewSpin = form.m_minConsistViewSpin;
    m_normalConsistSpin  = form.m_normalConsistSpin;
    m_voxelSizeSpin      = form.m_voxelSizeSpin;
    m_minConfidenceSpin  = form.m_minConfidenceSpin;
    m_maxReprojSpin      = form.m_maxReprojSpin;
    m_colorCheck         = form.m_colorCheck;
    m_normalCheck        = form.m_normalCheck;
    m_threadsSpin        = form.m_threadsSpin;
    m_cudaCheck          = form.m_cudaCheck;
    m_infoLabel          = form.m_infoLabel;
    if (m_cudaCheck)
    {
        m_cudaCheck->setChecked(false);
        m_cudaCheck->setEnabled(false);
        m_cudaCheck->hide();
    }

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_fusionMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_depthConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_minConsistViewSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_normalConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_voxelSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_minConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_maxReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_colorCheck, &QCheckBox::toggled, this, changed);
    connect(m_normalCheck, &QCheckBox::toggled, this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &DepthFusionDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject DepthFusionDialog::collectSettings() const
{
    QJsonObject o;
    o["fusionMethod"]       = m_fusionMethodCombo->currentText();
    o["depthConsistency"]   = m_depthConsistSpin->value();
    o["minConsistentViews"] = m_minConsistViewSpin->value();
    o["normalConsistency"]  = m_normalConsistSpin->value();
    o["voxelSize"]          = m_voxelSizeSpin->value();
    o["minConfidence"]      = m_minConfidenceSpin->value();
    o["maxReprojError"]     = m_maxReprojSpin->value();
    o["keepColor"]          = m_colorCheck->isChecked();
    o["keepNormals"]        = m_normalCheck->isChecked();
    o["threads"]            = m_threadsSpin->value();
    return o;
}

void DepthFusionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("fusionMethod"))
    {
        int i = m_fusionMethodCombo->findText(s["fusionMethod"].toString());
        if (i >= 0) m_fusionMethodCombo->setCurrentIndex(i);
    }
    if (s.contains("depthConsistency"))   m_depthConsistSpin->setValue(s["depthConsistency"].toDouble());
    if (s.contains("minConsistentViews")) m_minConsistViewSpin->setValue(s["minConsistentViews"].toInt());
    if (s.contains("normalConsistency"))  m_normalConsistSpin->setValue(s["normalConsistency"].toDouble());
    if (s.contains("voxelSize"))          m_voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    if (s.contains("minConfidence"))      m_minConfidenceSpin->setValue(s["minConfidence"].toDouble());
    if (s.contains("maxReprojError"))     m_maxReprojSpin->setValue(s["maxReprojError"].toDouble());
    if (s.contains("keepColor"))          m_colorCheck->setChecked(s["keepColor"].toBool());
    if (s.contains("keepNormals"))        m_normalCheck->setChecked(s["keepNormals"].toBool());
    if (s.contains("threads"))            m_threadsSpin->setValue(s["threads"].toInt());
}

void DepthFusionDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DepthFusionDialog::onRun() { emit runRequested(collectSettings()); accept(); }
