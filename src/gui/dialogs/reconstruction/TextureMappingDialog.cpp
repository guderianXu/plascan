#include "reconstruction/TextureMappingDialog.h"
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

    _textureTypeCombo = form.m_textureTypeCombo;
    _sourceCombo = form.m_sourceCombo;
    _texSizeCombo = form.m_texSizeCombo;
    _blendCombo = form.m_blendCombo;
    _uvMethodCombo = form.m_uvMethodCombo;
    _imageDownscaleCombo = form.m_imageDownscaleCombo;
    _saveEachStepCheck = form.m_saveEachStepCheck;
    _holeFillCheck = form.m_holeFillCheck;
    _colorCorrCheck = form.m_colorCorrCheck;
    _ghostFilterCheck = form.m_ghostFilterCheck;
    _outOfFocusFilterCheck = form.m_outOfFocusFilterCheck;
    _useAssignedImagesCheck = form.m_useAssignedImagesCheck;
    _transferTextureCheck = form.m_transferTextureCheck;
    _seamsMarginSpin = form.m_seamsMarginSpin;
    _paddingSpin = form.m_paddingSpin;
    _keepUnmappedCheck = form.m_keepUnmappedCheck;

    auto changed = [this]() { emitSettingsNow(); };
    connect(_textureTypeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_texSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_blendCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_uvMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_imageDownscaleCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_saveEachStepCheck, &QCheckBox::toggled, this, changed);
    connect(_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(_colorCorrCheck, &QCheckBox::toggled, this, changed);
    connect(_ghostFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_outOfFocusFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_useAssignedImagesCheck, &QCheckBox::toggled, this, changed);
    connect(_transferTextureCheck, &QCheckBox::toggled, this, changed);
    connect(_seamsMarginSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_paddingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_keepUnmappedCheck, &QCheckBox::toggled, this, changed);

    connect(form.m_runBtn, &QPushButton::clicked, this, &TextureMappingDialog::onRun);
    connect(form.m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
}

QJsonObject TextureMappingDialog::collectSettings() const
{
    QJsonObject o;
    o["textureType"] = QStringLiteral("texture_mapping");
    o["sourceData"] = QStringLiteral("images");
    o["textureSize"] = _texSizeCombo->currentText().toInt();
    o["blendMethod"] = _blendCombo->currentText();
    o["blendMode"] = _blendCombo->currentIndex() == 1
        ? QStringLiteral("weighted_average")
        : (_blendCombo->currentIndex() == 2
               ? QStringLiteral("best_view")
               : QStringLiteral("natural"));
    o["uvMethod"] = _uvMethodCombo->currentText();
    o["mappingMode"] = QStringLiteral("auto_projective");
    o["imageDownscale"] = 1 << _imageDownscaleCombo->currentIndex();
    o["saveEachStep"] = _saveEachStepCheck->isChecked();
    o["holeFill"] = _holeFillCheck->isChecked();
    o["holeFillMode"] = _holeFillCheck->isChecked()
        ? QStringLiteral("texture_space_small_holes")
        : QStringLiteral("disabled");
    o["colorCorrection"] = _colorCorrCheck->isChecked();
    o["ghostFilter"] = _ghostFilterCheck->isChecked();
    o["outOfFocusFilter"] = _outOfFocusFilterCheck->isChecked();
    o["useAssignedImages"] = _useAssignedImagesCheck->isChecked();
    o["transferTexture"] = _transferTextureCheck->isChecked();
    o["sharpeningStrength"] = _seamsMarginSpin->value();
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
    if (s.contains("imageDownscale"))
    {
        const int downscale = s["imageDownscale"].toInt(1);
        _imageDownscaleCombo->setCurrentIndex(downscale >= 4 ? 2 : downscale >= 2 ? 1 : 0);
    }
    if (s.contains("blendMethod"))
    {
        const int i = _blendCombo->findText(s["blendMethod"].toString());
        if (i >= 0)
        {
            _blendCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("blendMode"))
    {
        const QString blend_mode = s["blendMode"].toString();
        _blendCombo->setCurrentIndex(
            blend_mode == QStringLiteral("weighted_average")
            ? 1
            : (blend_mode == QStringLiteral("best_view") ? 2 : 0));
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
    if (s.contains("saveEachStep")) _saveEachStepCheck->setChecked(s["saveEachStep"].toBool());
    if (s.contains("holeFill")) _holeFillCheck->setChecked(s["holeFill"].toBool(true));
    if (s.contains("holeFillMode"))
    {
        _holeFillCheck->setChecked(
            s["holeFillMode"].toString() != QStringLiteral("disabled"));
    }
    if (s.contains("colorCorrection")) _colorCorrCheck->setChecked(s["colorCorrection"].toBool(true));
    if (s.contains("ghostFilter")) _ghostFilterCheck->setChecked(s["ghostFilter"].toBool(true));
    if (s.contains("outOfFocusFilter")) _outOfFocusFilterCheck->setChecked(s["outOfFocusFilter"].toBool());
    if (s.contains("useAssignedImages")) _useAssignedImagesCheck->setChecked(s["useAssignedImages"].toBool());
    if (s.contains("transferTexture")) _transferTextureCheck->setChecked(s["transferTexture"].toBool());
    if (s.contains("sharpeningStrength")) _seamsMarginSpin->setValue(s["sharpeningStrength"].toDouble(1.0));
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
