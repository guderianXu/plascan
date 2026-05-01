#include "TextureMappingDialog.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QPushButton>

TextureMappingDialog::TextureMappingDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("纹理映射 (Texture Mapping)"));
    setMinimumWidth(460);

    auto *root = new QVBoxLayout(this);

    // ── 纹理参数 ──
    {
        auto *box = new QGroupBox(tr("纹理参数"));
        auto *fl = new QFormLayout(box);

        m_texSizeCombo = new QComboBox;
        m_texSizeCombo->addItems({"1024", "2048", "4096", "8192", "16384"});
        m_texSizeCombo->setCurrentIndex(2);
        m_texSizeCombo->setToolTip(
            tr("纹理图分辨率。★ 推荐 4096; 大模型可选 8192 但内存消耗倍增"));
        fl->addRow(tr("纹理分辨率:"), m_texSizeCombo);

        m_blendCombo = new QComboBox;
        m_blendCombo->addItems({
            tr("最佳视角优先"),
            tr("加权平均"),
            tr("多频段混合 (Multiband)")
        });
        m_blendCombo->setCurrentIndex(2);
        m_blendCombo->setToolTip(
            tr("多频段混合: 接缝最不明显; 最佳视角: 最锐利但有缝; 加权平均: 折中"));
        fl->addRow(tr("混合方式:"), m_blendCombo);

        m_uvMethodCombo = new QComboBox;
        m_uvMethodCombo->addItems({
            tr("自动 (基于面法线)"),
            tr("LSCM (最小二乘共形)"),
            tr("投射展开")
        });
        m_uvMethodCombo->setCurrentIndex(0);
        m_uvMethodCombo->setToolTip(tr("UV 参数化方法。通常自动即可"));
        fl->addRow(tr("UV 展开:"), m_uvMethodCombo);

        root->addWidget(box);
    }

    // ── 质量增强 ──
    {
        auto *box = new QGroupBox(tr("质量增强"));
        auto *fl = new QFormLayout(box);

        m_colorCorrCheck = new QCheckBox(tr("色彩一致性校正"));
        m_colorCorrCheck->setChecked(true);
        m_colorCorrCheck->setToolTip(tr("对来自不同图像的纹理块做全局色彩均衡"));
        fl->addRow(m_colorCorrCheck);

        m_ghostFilterCheck = new QCheckBox(tr("去除鬼影 (移动物体)"));
        m_ghostFilterCheck->setChecked(true);
        m_ghostFilterCheck->setToolTip(tr("检测并过滤不同时刻出现差异的区域"));
        fl->addRow(m_ghostFilterCheck);

        m_seamsMarginSpin = new QDoubleSpinBox;
        m_seamsMarginSpin->setRange(0.0, 20.0);
        m_seamsMarginSpin->setDecimals(1);
        m_seamsMarginSpin->setValue(3.0);
        m_seamsMarginSpin->setSuffix(tr(" px"));
        m_seamsMarginSpin->setToolTip(tr("接缝处过渡带宽度。★ 推荐 3 px"));
        fl->addRow(tr("接缝边距:"), m_seamsMarginSpin);

        m_paddingSpin = new QSpinBox;
        m_paddingSpin->setRange(0, 32);
        m_paddingSpin->setValue(4);
        m_paddingSpin->setToolTip(tr("纹理图内每个 chart 外扩像素数，避免采样越界"));
        fl->addRow(tr("填充边距:"), m_paddingSpin);

        m_keepUnmappedCheck = new QCheckBox(tr("保留无纹理区域 (默认色填充)"));
        m_keepUnmappedCheck->setChecked(true);
        fl->addRow(m_keepUnmappedCheck);

        root->addWidget(box);
    }

    // ── 系统 ──
    {
        auto *box = new QGroupBox(tr("系统"));
        auto *fl = new QFormLayout(box);
        m_threadsSpin = new QSpinBox;
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        fl->addRow(tr("线程数:"), m_threadsSpin);
        root->addWidget(box);
    }

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("开始纹理映射"));
    root->addWidget(btnBox);

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

    connect(btnBox, &QDialogButtonBox::accepted, this, &TextureMappingDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
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
