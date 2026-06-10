// =============================================================================
// 文件: ModelGenerationDialog.cpp
// 模块: GUI / 对话框
// =============================================================================
#include "ModelGenerationDialog.h"
#include "ui_ModelGenerationDialog.h"

#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QJsonObject>

// ── 预设描述文本 ──────────────────────────────────────────────────────────────
static const char *kPresetDesc[3] = {
    "低精度：快速重建，适合预览。",
    "中精度（默认）：质量与速度平衡。",
    "高精度：网格更细致，耗时更长。"
};

// ── 预设默认值 ────────────────────────────────────────────────────────────────
struct Preset {
    int disparity;
    int gridRes;
    int meshRes;
    int smoothIter;
    double smoothLambda;
    double padding;
};
static const Preset kPresets[3] = {
    {  64, 256, 128, 1, 0.38, 0.03 },   // 低
    { 128, 512, 256, 2, 0.46, 0.05 },   // 中
    { 256, 1024, 512, 3, 0.52, 0.08 },  // 高
};

ModelGenerationDialog::ModelGenerationDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    applyPreset(1); // 默认中精度
}

void ModelGenerationDialog::setupUi()
{
    Ui::ModelGenerationDialog form;
    form.setupUi(this);

    m_presetCombo          = form.m_presetCombo;
    m_outputFormatCombo    = form.m_outputFormatCombo;
    m_disparitySpin        = form.m_disparitySpin;
    m_gridResSpin          = form.m_gridResSpin;
    m_meshResSpin          = form.m_meshResSpin;
    m_meshSmoothIterSpin   = form.m_meshSmoothIterSpin;
    m_meshSmoothLambdaSpin = form.m_meshSmoothLambdaSpin;
    m_meshPaddingSpin      = form.m_meshPaddingSpin;
    m_useCudaCheck         = form.m_useCudaCheck;
    m_outputColorCheck     = form.m_outputColorCheck;
    m_outputNormalCheck    = form.m_outputNormalCheck;
    m_presetDescLabel      = form.m_presetDescLabel;
    m_runtimeHintLabel     = form.m_runtimeHintLabel;
    m_okBtn                = form.m_okBtn;
    m_cancelBtn            = form.m_cancelBtn;

    // ── 信号 ───────────────────────────────────────────────────────────────
    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ModelGenerationDialog::onPresetChanged);
    connect(m_okBtn,     &QPushButton::clicked, this, &ModelGenerationDialog::onOk);
    connect(m_cancelBtn, &QPushButton::clicked, this, &ModelGenerationDialog::onCancel);
}

void ModelGenerationDialog::applyPreset(int preset)
{
    // 限制范围
    const int p = qBound(0, preset, 2);

    // 静默更新（避免循环触发 onPresetChanged）
    m_disparitySpin->blockSignals(true);
    m_gridResSpin->blockSignals(true);
    m_meshResSpin->blockSignals(true);
    m_meshSmoothIterSpin->blockSignals(true);
    m_meshSmoothLambdaSpin->blockSignals(true);
    m_meshPaddingSpin->blockSignals(true);

    m_disparitySpin->setValue(kPresets[p].disparity);
    m_gridResSpin->setValue(kPresets[p].gridRes);
    m_meshResSpin->setValue(kPresets[p].meshRes);
    m_meshSmoothIterSpin->setValue(kPresets[p].smoothIter);
    m_meshSmoothLambdaSpin->setValue(kPresets[p].smoothLambda);
    m_meshPaddingSpin->setValue(kPresets[p].padding);

    m_disparitySpin->blockSignals(false);
    m_gridResSpin->blockSignals(false);
    m_meshResSpin->blockSignals(false);
    m_meshSmoothIterSpin->blockSignals(false);
    m_meshSmoothLambdaSpin->blockSignals(false);
    m_meshPaddingSpin->blockSignals(false);

    m_presetDescLabel->setText(QString::fromUtf8(kPresetDesc[p]));
    if (p == 0) m_runtimeHintLabel->setText(QStringLiteral("约 10s - 40s（取决于点数与显卡）"));
    else if (p == 1) m_runtimeHintLabel->setText(QStringLiteral("约 30s - 2min（推荐）"));
    else m_runtimeHintLabel->setText(QStringLiteral("约 1min - 5min（高质量）"));
}

void ModelGenerationDialog::onPresetChanged(int index)
{
    applyPreset(index);
}

QJsonObject ModelGenerationDialog::collectSettings() const
{
    QJsonObject s;
    s[QStringLiteral("preset")]          = m_presetCombo->currentIndex();
    s[QStringLiteral("export_format")]   = m_outputFormatCombo->currentText();
    s[QStringLiteral("num_disparities")] = m_disparitySpin->value();
    s[QStringLiteral("grid_resolution")] = m_gridResSpin->value();
    s[QStringLiteral("mesh_resolution")] = m_meshResSpin->value();
    s[QStringLiteral("mesh_smooth_iterations")] = m_meshSmoothIterSpin->value();
    s[QStringLiteral("mesh_smooth_lambda")] = m_meshSmoothLambdaSpin->value();
    s[QStringLiteral("mesh_padding")] = m_meshPaddingSpin->value();
    s[QStringLiteral("use_cuda")]        = m_useCudaCheck->isChecked();
    s[QStringLiteral("output_color")]    = m_outputColorCheck->isChecked();
    s[QStringLiteral("output_normal")]   = m_outputNormalCheck->isChecked();
    return s;
}

void ModelGenerationDialog::applySettings(const QJsonObject &settings)
{
    if (settings.isEmpty()) return;

    const int preset = settings.value(QStringLiteral("preset")).toInt(1);
    m_presetCombo->setCurrentIndex(qBound(0, preset, 2));
    if (settings.contains(QStringLiteral("export_format")))
    {
        const int index = m_outputFormatCombo->findText(settings.value(QStringLiteral("export_format")).toString());
        if (index >= 0)
        {
            m_outputFormatCombo->setCurrentIndex(index);
        }
    }
    // preset combo 会触发 onPresetChanged → applyPreset；之后可以再覆盖具体值

    if (settings.contains(QStringLiteral("num_disparities")))
        m_disparitySpin->setValue(settings.value(QStringLiteral("num_disparities")).toInt(128));
    if (settings.contains(QStringLiteral("grid_resolution")))
        m_gridResSpin->setValue(settings.value(QStringLiteral("grid_resolution")).toInt(512));
    if (settings.contains(QStringLiteral("mesh_resolution")))
        m_meshResSpin->setValue(settings.value(QStringLiteral("mesh_resolution")).toInt(128));
    if (settings.contains(QStringLiteral("mesh_smooth_iterations")))
        m_meshSmoothIterSpin->setValue(settings.value(QStringLiteral("mesh_smooth_iterations")).toInt(2));
    if (settings.contains(QStringLiteral("mesh_smooth_lambda")))
        m_meshSmoothLambdaSpin->setValue(settings.value(QStringLiteral("mesh_smooth_lambda")).toDouble(0.5));
    if (settings.contains(QStringLiteral("mesh_padding")))
        m_meshPaddingSpin->setValue(settings.value(QStringLiteral("mesh_padding")).toDouble(0.05));
    if (settings.contains(QStringLiteral("use_cuda")))
        m_useCudaCheck->setChecked(settings.value(QStringLiteral("use_cuda")).toBool(true));
    if (settings.contains(QStringLiteral("output_color")))
        m_outputColorCheck->setChecked(settings.value(QStringLiteral("output_color")).toBool(true));
    if (settings.contains(QStringLiteral("output_normal")))
        m_outputNormalCheck->setChecked(settings.value(QStringLiteral("output_normal")).toBool(false));
}

void ModelGenerationDialog::onOk()
{
    emit accepted(collectSettings());
    QDialog::accept();
}

void ModelGenerationDialog::onCancel()
{
    QDialog::reject();
}
