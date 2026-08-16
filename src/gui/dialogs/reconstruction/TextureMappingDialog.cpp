#include "reconstruction/TextureMappingDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"
#include "ui_TextureMappingDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QSignalBlocker>
#include <QSpinBox>

TextureMappingDialog::TextureMappingDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TextureMappingDialog form;
    form.setupUi(this);
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    _texSizeCombo = form.m_texSizeCombo;
    _blendCombo = form.m_blendCombo;
    _imageDownscaleCombo = form.m_imageDownscaleCombo;
    _holeFillCheck = form.m_holeFillCheck;
    _colorCorrCheck = form.m_colorCorrCheck;
    _ghostFilterCheck = form.m_ghostFilterCheck;
    _outOfFocusFilterCheck = form.m_outOfFocusFilterCheck;
    _seamsMarginSpin = form.m_seamsMarginSpin;
    _paddingSpin = form.m_paddingSpin;
    _keepUnmappedCheck = form.m_keepUnmappedCheck;

    // Quality-first default. Persisted settings applied after construction may
    // still select x2/x4 explicitly for memory-constrained projects.
    _imageDownscaleCombo->setCurrentIndex(0);
    _colorCorrCheck->setChecked(false);
    _seamsMarginSpin->setValue(0.35);

    for (QComboBox *combo_box : {_texSizeCombo, _blendCombo, _imageDownscaleCombo})
    {
        xjw::gui::dialogs::configureWorkflowComboBox(combo_box);
    }
    for (QCheckBox *check_box : {
             _holeFillCheck,
             _colorCorrCheck,
             _ghostFilterCheck,
             _outOfFocusFilterCheck,
             _keepUnmappedCheck})
    {
        xjw::gui::dialogs::configureWorkflowCheckBox(check_box);
    }
    xjw::gui::dialogs::configureWorkflowInputWidget(_seamsMarginSpin);
    xjw::gui::dialogs::configureWorkflowInputWidget(_paddingSpin);
    xjw::gui::dialogs::configureWorkflowButtonBox(form.m_buttonBox);

    auto changed = [this]() { emitSettingsNow(); };
    connect(_texSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_blendCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_imageDownscaleCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(_colorCorrCheck, &QCheckBox::toggled, this, changed);
    connect(_ghostFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_outOfFocusFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_seamsMarginSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(_paddingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(_keepUnmappedCheck, &QCheckBox::toggled, this, changed);

    connect(form.m_buttonBox, &QDialogButtonBox::accepted,
            this, &TextureMappingDialog::onRun);
    connect(form.m_buttonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);
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
    o["uvMethod"] = QStringLiteral("自动投影（相机 chart）");
    o["mappingMode"] = QStringLiteral("auto_projective");
    o["imageDownscale"] = 1 << _imageDownscaleCombo->currentIndex();
    o["saveEachStep"] = false;
    o["holeFill"] = _holeFillCheck->isChecked();
    o["holeFillMode"] = _holeFillCheck->isChecked()
        ? QStringLiteral("neighbor_view_recovery")
        : QStringLiteral("disabled");
    o["colorCorrection"] = _colorCorrCheck->isChecked();
    o["ghostFilter"] = _ghostFilterCheck->isChecked();
    o["outOfFocusFilter"] = _outOfFocusFilterCheck->isChecked();
    o["useAssignedImages"] = false;
    o["transferTexture"] = false;
    o["sharpeningStrength"] = _seamsMarginSpin->value();
    o["padding"] = _paddingSpin->value();
    o["keepUnmapped"] = _keepUnmappedCheck->isChecked();
    return o;
}

void TextureMappingDialog::applySettings(const QJsonObject &s)
{
    const QSignalBlocker texture_size_blocker(_texSizeCombo);
    const QSignalBlocker blend_blocker(_blendCombo);
    const QSignalBlocker downscale_blocker(_imageDownscaleCombo);
    const QSignalBlocker hole_fill_blocker(_holeFillCheck);
    const QSignalBlocker color_correction_blocker(_colorCorrCheck);
    const QSignalBlocker ghost_filter_blocker(_ghostFilterCheck);
    const QSignalBlocker focus_filter_blocker(_outOfFocusFilterCheck);
    const QSignalBlocker sharpening_blocker(_seamsMarginSpin);
    const QSignalBlocker padding_blocker(_paddingSpin);
    const QSignalBlocker keep_unmapped_blocker(_keepUnmappedCheck);

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
    if (s.contains("padding"))
    {
        _paddingSpin->setValue(s["padding"].toInt());
    }
    if (s.contains("keepUnmapped"))
    {
        _keepUnmappedCheck->setChecked(s["keepUnmapped"].toBool());
    }
    if (s.contains("holeFill")) _holeFillCheck->setChecked(s["holeFill"].toBool(true));
    if (s.contains("holeFillMode"))
    {
        _holeFillCheck->setChecked(
            s["holeFillMode"].toString() != QStringLiteral("disabled"));
    }
    if (s.contains("colorCorrection")) _colorCorrCheck->setChecked(s["colorCorrection"].toBool(false));
    if (s.contains("ghostFilter")) _ghostFilterCheck->setChecked(s["ghostFilter"].toBool(true));
    if (s.contains("outOfFocusFilter")) _outOfFocusFilterCheck->setChecked(s["outOfFocusFilter"].toBool());
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
