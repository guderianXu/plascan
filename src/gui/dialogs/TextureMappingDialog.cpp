#include "TextureMappingDialog.h"
#include "ui_TextureMappingDialog.h"

#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QPushButton>
#include <QFormLayout>

TextureMappingDialog::TextureMappingDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TextureMappingDialog form;
    form.setupUi(this);

    _texSizeCombo = form.m_texSizeCombo;
    _blendCombo = form.m_blendCombo;
    _uvMethodCombo = form.m_uvMethodCombo;
    _colorCorrCheck = form.m_colorCorrCheck;
    _ghostFilterCheck = form.m_ghostFilterCheck;
    _seamsMarginSpin = form.m_seamsMarginSpin;
    _paddingSpin = form.m_paddingSpin;
    _keepUnmappedCheck = form.m_keepUnmappedCheck;
    _threadsSpin = form.m_threadsSpin;

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
    hideUnsupportedField(_colorCorrCheck);
    hideUnsupportedField(_ghostFilterCheck);
    hideUnsupportedField(_seamsMarginSpin);
    hideUnsupportedField(_threadsSpin);

    auto changed = [this]() { emitSettingsNow(); };
    connect(_texSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_blendCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_uvMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_paddingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_keepUnmappedCheck, &QCheckBox::toggled, this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &TextureMappingDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject TextureMappingDialog::collectSettings() const
{
    QJsonObject o;
    o["textureSize"] = _texSizeCombo->currentText().toInt();
    o["blendMethod"] = _blendCombo->currentText();
    o["uvMethod"] = _uvMethodCombo->currentText();
    o["padding"] = _paddingSpin->value();
    o["keepUnmapped"] = _keepUnmappedCheck->isChecked();
    return o;
}

void TextureMappingDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("textureSize"))
    {
        const int i = _texSizeCombo->findText(QString::number(s["textureSize"].toInt()));
        if (i >= 0)
        {
            _texSizeCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("blendMethod"))
    {
        const int i = _blendCombo->findText(s["blendMethod"].toString());
        if (i >= 0)
        {
            _blendCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("uvMethod"))
    {
        const int i = _uvMethodCombo->findText(s["uvMethod"].toString());
        if (i >= 0)
        {
            _uvMethodCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("padding"))
    {
        _paddingSpin->setValue(s["padding"].toInt());
    }
    if (s.contains("keepUnmapped"))
    {
        _keepUnmappedCheck->setChecked(s["keepUnmapped"].toBool());
    }
}

void TextureMappingDialog::emitSettingsNow()
{
    emit settingsChanged(collectSettings());
}

void TextureMappingDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
