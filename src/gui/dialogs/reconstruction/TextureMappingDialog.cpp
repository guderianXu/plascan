#include "reconstruction/TextureMappingDialog.h"
#include "shared/WorkflowParameterDialogStyle.h"
#include "ui_TextureMappingDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QSignalBlocker>

namespace
{

constexpr int kTextureMappingSettingsRevision = 3;

} // namespace

TextureMappingDialog::TextureMappingDialog(QWidget *parent)
    : QDialog(parent)
{
    Ui::TextureMappingDialog form;
    form.setupUi(this);
    xjw::gui::dialogs::configureWorkflowParameterDialog(this);

    _texSizeCombo = form.m_texSizeCombo;
    _imageDownscaleCombo = form.m_imageDownscaleCombo;
    _antiAliasingCombo = form.m_antiAliasingCombo;
    _holeFillCheck = form.m_holeFillCheck;
    _colorCorrCheck = form.m_colorCorrCheck;
    _ghostFilterCheck = form.m_ghostFilterCheck;
    _outOfFocusFilterCheck = form.m_outOfFocusFilterCheck;
    _seamsMarginSpin = form.m_seamsMarginSpin;

    _imageDownscaleCombo->setCurrentIndex(1);
    _colorCorrCheck->setChecked(false);
    _colorCorrCheck->setToolTip(tr(
        "仅用共同可见的同一三维点估计曝光；每个可靠连通分量独立校正，孤立视图保持原值，"
        "每张影像增益限制为 0.90–1.10。"));
    _seamsMarginSpin->setValue(1.0);

    for (QComboBox *combo_box : {
             _texSizeCombo,
             _imageDownscaleCombo,
             _antiAliasingCombo})
    {
        xjw::gui::dialogs::configureWorkflowComboBox(combo_box);
    }
    for (QCheckBox *check_box : {
             _holeFillCheck,
             _colorCorrCheck,
             _ghostFilterCheck,
             _outOfFocusFilterCheck})
    {
        xjw::gui::dialogs::configureWorkflowCheckBox(check_box);
    }
    xjw::gui::dialogs::configureWorkflowInputWidget(_seamsMarginSpin);
    xjw::gui::dialogs::configureWorkflowButtonBox(form.m_buttonBox, tr("生成"));

    auto changed = [this]() { emitSettingsNow(); };
    connect(_texSizeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_imageDownscaleCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_antiAliasingCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(_colorCorrCheck, &QCheckBox::toggled, this, changed);
    connect(_ghostFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_outOfFocusFilterCheck, &QCheckBox::toggled, this, changed);
    connect(_seamsMarginSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(form.m_buttonBox, &QDialogButtonBox::accepted,
            this, &TextureMappingDialog::onRun);
    connect(form.m_buttonBox, &QDialogButtonBox::rejected,
            this, &QDialog::reject);
}

QJsonObject TextureMappingDialog::collectSettings() const
{
    QJsonObject o;
    o["textureMappingSettingsRevision"] = kTextureMappingSettingsRevision;
    o["textureType"] = QStringLiteral("texture_mapping");
    o["sourceData"] = QStringLiteral("images");
    o["textureSize"] = _texSizeCombo->currentText().toInt();
    o["pipeline"] = QStringLiteral("recovered_natural_v1");
    o["blendMethod"] = QStringLiteral("Natural 多频段融合");
    o["blendMode"] = QStringLiteral("natural");
    o["uvMethod"] = QStringLiteral("Natural 映射（相机 chart）");
    o["mappingMode"] = QStringLiteral("natural_mapping");
    o["imageDownscale"] = 1 << _imageDownscaleCombo->currentIndex();
    o["antiAliasing"] = 1 << _antiAliasingCombo->currentIndex();
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
    o["padding"] = 2;
    o["keepUnmapped"] = true;
    return o;
}

void TextureMappingDialog::applySettings(const QJsonObject &s)
{
    const QSignalBlocker texture_size_blocker(_texSizeCombo);
    const QSignalBlocker downscale_blocker(_imageDownscaleCombo);
    const QSignalBlocker anti_aliasing_blocker(_antiAliasingCombo);
    const QSignalBlocker hole_fill_blocker(_holeFillCheck);
    const QSignalBlocker color_correction_blocker(_colorCorrCheck);
    const QSignalBlocker ghost_filter_blocker(_ghostFilterCheck);
    const QSignalBlocker focus_filter_blocker(_outOfFocusFilterCheck);
    const QSignalBlocker sharpening_blocker(_seamsMarginSpin);

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
    if (s.contains("antiAliasing"))
    {
        const int anti_aliasing = s["antiAliasing"].toInt(1);
        _antiAliasingCombo->setCurrentIndex(
            anti_aliasing >= 4 ? 2 : anti_aliasing >= 2 ? 1 : 0);
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
    if (s.contains("sharpeningStrength"))
    {
        double sharpening_strength =
            s["sharpeningStrength"].toDouble(1.0);
        const int settings_revision =
            s["textureMappingSettingsRevision"].toInt(0);
        if (settings_revision == 0 &&
            qFuzzyCompare(sharpening_strength, 0.35))
        {
            // The unversioned dialog persisted 0.35 as an implicit default.
            sharpening_strength = 1.0;
        }
        _seamsMarginSpin->setValue(sharpening_strength);
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
