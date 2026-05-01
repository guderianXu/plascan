// =============================================================================
// 文件: ModelGenerationDialog.cpp
// 模块: GUI / 对话框
// =============================================================================
#include "ModelGenerationDialog.h"

#include <QComboBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QDialogButtonBox>
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
    setWindowTitle(QStringLiteral("生成模型 - 参数设置"));
    setMinimumWidth(520);
    setupUi();
    applyPreset(1); // 默认中精度
}

void ModelGenerationDialog::setupUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(12);

    // ── 精度预设组 ─────────────────────────────────────────────────────────
    auto *presetGroup = new QGroupBox(QStringLiteral("重建精度"), this);
    auto *presetForm  = new QFormLayout(presetGroup);

    m_presetCombo = new QComboBox(this);
    m_presetCombo->addItem(QStringLiteral("低 (Low)"));
    m_presetCombo->addItem(QStringLiteral("中 (Medium)  — 默认"));
    m_presetCombo->addItem(QStringLiteral("高 (High)"));
    m_presetCombo->setCurrentIndex(1);
    presetForm->addRow(QStringLiteral("精度预设："), m_presetCombo);

    m_outputFormatCombo = new QComboBox(this);
    m_outputFormatCombo->addItem(QStringLiteral("PLY"));
    m_outputFormatCombo->addItem(QStringLiteral("OBJ"));
    m_outputFormatCombo->setCurrentIndex(0);
    m_outputFormatCombo->setToolTip(QStringLiteral("最终模型输出格式。PLY 为默认；OBJ 适合带纹理输出。"));
    presetForm->addRow(QStringLiteral("最终格式："), m_outputFormatCombo);

    m_presetDescLabel = new QLabel(this);
    m_presetDescLabel->setWordWrap(true);
    m_presetDescLabel->setStyleSheet(QStringLiteral("color: gray; font-size: 11px;"));
    presetForm->addRow(m_presetDescLabel);

    mainLayout->addWidget(presetGroup);

    // ── 高级参数组 ─────────────────────────────────────────────────────────
    auto *advGroup = new QGroupBox(QStringLiteral("重建参数（可与预设独立调整）"), this);
    auto *advForm  = new QFormLayout(advGroup);

    m_disparitySpin = new QSpinBox(this);
    m_disparitySpin->setRange(16, 512);
    m_disparitySpin->setSingleStep(16);
    m_disparitySpin->setToolTip(QStringLiteral("SGBM 视差搜索范围，必须为 16 的倍数。越大越慢但覆盖更深的景深差异。"));
    advForm->addRow(QStringLiteral("视差范围："), m_disparitySpin);

    m_gridResSpin = new QSpinBox(this);
    m_gridResSpin->setRange(64, 4096);
    m_gridResSpin->setSingleStep(64);
    m_gridResSpin->setToolTip(QStringLiteral("地形网格横向像素分辨率。越高细节越丰富，内存消耗越大。"));
    advForm->addRow(QStringLiteral("网格分辨率："), m_gridResSpin);

    m_meshResSpin = new QSpinBox(this);
    m_meshResSpin->setRange(64, 1024);
    m_meshResSpin->setSingleStep(32);
    m_meshResSpin->setToolTip(QStringLiteral("隐式场分辨率（最长轴体素数）。越高细节越多，耗时越长。"));
    advForm->addRow(QStringLiteral("网格体素分辨率："), m_meshResSpin);

    m_meshSmoothIterSpin = new QSpinBox(this);
    m_meshSmoothIterSpin->setRange(0, 10);
    m_meshSmoothIterSpin->setToolTip(QStringLiteral("Taubin 平滑迭代次数。0 表示不平滑。"));
    advForm->addRow(QStringLiteral("平滑迭代："), m_meshSmoothIterSpin);

    m_meshSmoothLambdaSpin = new QDoubleSpinBox(this);
    m_meshSmoothLambdaSpin->setRange(0.0, 1.0);
    m_meshSmoothLambdaSpin->setSingleStep(0.05);
    m_meshSmoothLambdaSpin->setDecimals(2);
    m_meshSmoothLambdaSpin->setToolTip(QStringLiteral("平滑步长，越大越平滑但会损失细节。"));
    advForm->addRow(QStringLiteral("平滑强度："), m_meshSmoothLambdaSpin);

    m_meshPaddingSpin = new QDoubleSpinBox(this);
    m_meshPaddingSpin->setRange(0.0, 0.30);
    m_meshPaddingSpin->setSingleStep(0.01);
    m_meshPaddingSpin->setDecimals(2);
    m_meshPaddingSpin->setToolTip(QStringLiteral("重建包围盒外扩比例。适度外扩可避免边界截断。"));
    advForm->addRow(QStringLiteral("包围盒外扩："), m_meshPaddingSpin);

    m_runtimeHintLabel = new QLabel(this);
    m_runtimeHintLabel->setWordWrap(true);
    m_runtimeHintLabel->setStyleSheet(QStringLiteral("color: #666; font-size: 11px;"));
    advForm->addRow(QStringLiteral("预计耗时："), m_runtimeHintLabel);

    mainLayout->addWidget(advGroup);

    // ── 选项组 ─────────────────────────────────────────────────────────────
    auto *optGroup = new QGroupBox(QStringLiteral("生成选项"), this);
    auto *optLayout = new QVBoxLayout(optGroup);

    m_useCudaCheck = new QCheckBox(QStringLiteral("使用 CUDA 加速（需要 NVIDIA GPU）"), this);
    m_useCudaCheck->setChecked(true); // 默认开启
    m_useCudaCheck->setToolTip(QStringLiteral("启用后，立体匹配阶段将使用 GPU 加速，速度显著提升。"));
    optLayout->addWidget(m_useCudaCheck);

    m_outputColorCheck = new QCheckBox(QStringLiteral("输出颜色（XYZRGB）"), this);
    m_outputColorCheck->setChecked(true);
    m_outputColorCheck->setToolTip(QStringLiteral("从原始影像采样 RGB 颜色写入点云/网格。"));
    optLayout->addWidget(m_outputColorCheck);

    m_outputNormalCheck = new QCheckBox(QStringLiteral("输出法向量"), this);
    m_outputNormalCheck->setChecked(false);
    m_outputNormalCheck->setToolTip(QStringLiteral("为网格顶点估算并写入法向量（计算量较大）。"));
    optLayout->addWidget(m_outputNormalCheck);

    mainLayout->addWidget(optGroup);

    // ── 按钮 ───────────────────────────────────────────────────────────────
    auto *btnBox = new QDialogButtonBox(this);
    m_okBtn     = btnBox->addButton(QStringLiteral("确定"), QDialogButtonBox::AcceptRole);
    m_cancelBtn = btnBox->addButton(QStringLiteral("取消"), QDialogButtonBox::RejectRole);
    mainLayout->addWidget(btnBox);

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
