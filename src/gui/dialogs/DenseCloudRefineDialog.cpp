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

    m_sorGroup         = form.m_sorGroup;
    m_voxelGroup       = form.m_voxelGroup;
    m_normalGroup      = form.m_normalGroup;
    m_colorGroup       = form.m_colorGroup;
    m_sorKSpin         = form.m_sorKSpin;
    m_sorStdSpin       = form.m_sorStdSpin;
    m_voxelSizeSpin    = form.m_voxelSizeSpin;
    m_normalKSpin      = form.m_normalKSpin;
    m_smoothIterSpin   = form.m_smoothIterSpin;
    m_colorMethodCombo = form.m_colorMethodCombo;
    m_threadsSpin      = form.m_threadsSpin;

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_sorGroup,         &QGroupBox::toggled,                                     this, changed);
    connect(m_sorKSpin,         QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_sorStdSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged),     this, changed);
    connect(m_voxelGroup,       &QGroupBox::toggled,                                     this, changed);
    connect(m_voxelSizeSpin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged),     this, changed);
    connect(m_normalGroup,      &QGroupBox::toggled,                                     this, changed);
    connect(m_normalKSpin,      QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_smoothIterSpin,   QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_colorGroup,       &QGroupBox::toggled,                                     this, changed);
    connect(m_colorMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),      this, changed);
    connect(m_threadsSpin,      QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &DenseCloudRefineDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject DenseCloudRefineDialog::collectSettings() const
{
    QJsonObject o;
    o["sorEnabled"]      = m_sorGroup->isChecked();
    o["sorK"]            = m_sorKSpin->value();
    o["sorStdDev"]       = m_sorStdSpin->value();
    o["voxelEnabled"]    = m_voxelGroup->isChecked();
    o["voxelSize"]       = m_voxelSizeSpin->value();
    o["normalsEnabled"]  = m_normalGroup->isChecked();
    o["normalK"]         = m_normalKSpin->value();
    o["smoothIter"]      = m_smoothIterSpin->value();
    o["colorEnabled"]    = m_colorGroup->isChecked();
    o["colorMethod"]     = m_colorMethodCombo->currentText();
    o["threads"]         = m_threadsSpin->value();
    return o;
}

void DenseCloudRefineDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("sorEnabled"))    m_sorGroup->setChecked(s["sorEnabled"].toBool());
    if (s.contains("sorK"))          m_sorKSpin->setValue(s["sorK"].toInt());
    if (s.contains("sorStdDev"))     m_sorStdSpin->setValue(s["sorStdDev"].toDouble());
    if (s.contains("voxelEnabled"))  m_voxelGroup->setChecked(s["voxelEnabled"].toBool());
    if (s.contains("voxelSize"))     m_voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    if (s.contains("normalsEnabled"))m_normalGroup->setChecked(s["normalsEnabled"].toBool());
    if (s.contains("normalK"))       m_normalKSpin->setValue(s["normalK"].toInt());
    if (s.contains("smoothIter"))    m_smoothIterSpin->setValue(s["smoothIter"].toInt());
    if (s.contains("colorEnabled"))  m_colorGroup->setChecked(s["colorEnabled"].toBool());
    if (s.contains("colorMethod"))
    {
        int i = m_colorMethodCombo->findText(s["colorMethod"].toString());
        if (i >= 0) m_colorMethodCombo->setCurrentIndex(i);
    }
    // 向后兼容旧键名
    if (!s.contains("normalsEnabled") && s.contains("estimateNormals"))
        m_normalGroup->setChecked(s["estimateNormals"].toBool());
    if (!s.contains("colorEnabled") && s.contains("colorCorrection"))
        m_colorGroup->setChecked(s["colorCorrection"].toBool());
    if (s.contains("threads"))       m_threadsSpin->setValue(s["threads"].toInt());
}

void DenseCloudRefineDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DenseCloudRefineDialog::onRun() { emit runRequested(collectSettings()); accept(); }
