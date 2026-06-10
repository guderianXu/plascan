#include "TextureMappingDialog.h"
#include "ui_TextureMappingDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>

TextureMappingDialog::TextureMappingDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TextureMappingDialog form;
    form.setupUi(this);

    m_texSizeCombo      = form.m_texSizeCombo;
    m_blendCombo        = form.m_blendCombo;
    m_uvMethodCombo     = form.m_uvMethodCombo;
    m_colorCorrCheck    = form.m_colorCorrCheck;
    m_ghostFilterCheck  = form.m_ghostFilterCheck;
    m_seamsMarginSpin   = form.m_seamsMarginSpin;
    m_paddingSpin       = form.m_paddingSpin;
    m_keepUnmappedCheck = form.m_keepUnmappedCheck;
    m_threadsSpin       = form.m_threadsSpin;

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_texSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_blendCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_uvMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_colorCorrCheck, &QCheckBox::toggled, this, changed);
    connect(m_ghostFilterCheck, &QCheckBox::toggled, this, changed);
    connect(m_seamsMarginSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_paddingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_keepUnmappedCheck, &QCheckBox::toggled, this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &TextureMappingDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject TextureMappingDialog::collectSettings() const
{
    QJsonObject o;
    o["textureSize"]    = m_texSizeCombo->currentText().toInt();
    o["blendMethod"]    = m_blendCombo->currentText();
    o["uvMethod"]       = m_uvMethodCombo->currentText();
    o["colorCorrection"]= m_colorCorrCheck->isChecked();
    o["ghostFilter"]    = m_ghostFilterCheck->isChecked();
    o["seamsMargin"]    = m_seamsMarginSpin->value();
    o["padding"]        = m_paddingSpin->value();
    o["keepUnmapped"]   = m_keepUnmappedCheck->isChecked();
    o["threads"]        = m_threadsSpin->value();
    return o;
}

void TextureMappingDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("textureSize"))
    {
        int i = m_texSizeCombo->findText(QString::number(s["textureSize"].toInt()));
        if (i >= 0) m_texSizeCombo->setCurrentIndex(i);
    }
    if (s.contains("blendMethod"))
    {
        int i = m_blendCombo->findText(s["blendMethod"].toString());
        if (i >= 0) m_blendCombo->setCurrentIndex(i);
    }
    if (s.contains("uvMethod"))
    {
        int i = m_uvMethodCombo->findText(s["uvMethod"].toString());
        if (i >= 0) m_uvMethodCombo->setCurrentIndex(i);
    }
    if (s.contains("colorCorrection")) m_colorCorrCheck->setChecked(s["colorCorrection"].toBool());
    if (s.contains("ghostFilter"))     m_ghostFilterCheck->setChecked(s["ghostFilter"].toBool());
    if (s.contains("seamsMargin"))     m_seamsMarginSpin->setValue(s["seamsMargin"].toDouble());
    if (s.contains("padding"))         m_paddingSpin->setValue(s["padding"].toInt());
    if (s.contains("keepUnmapped"))    m_keepUnmappedCheck->setChecked(s["keepUnmapped"].toBool());
    if (s.contains("threads"))         m_threadsSpin->setValue(s["threads"].toInt());
}

void TextureMappingDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void TextureMappingDialog::onRun() { emit runRequested(collectSettings()); accept(); }
