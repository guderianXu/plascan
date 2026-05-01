#include "ModelExportDialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QFileDialog>

ModelExportDialog::ModelExportDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("模型导出 (Export Model)"));
    setMinimumWidth(480);

    auto *root = new QVBoxLayout(this);

    // ── 格式与坐标系 ──
    {
        auto *box = new QGroupBox(tr("格式与坐标系"));
        auto *fl = new QFormLayout(box);

        m_formatCombo = new QComboBox;
        m_formatCombo->addItems({"OBJ (.obj)", "PLY (.ply)", "glTF (.gltf)", "FBX (.fbx)", "STL (.stl)"});
        m_formatCombo->setCurrentIndex(0);
        m_formatCombo->setToolTip(
            tr("OBJ: 兼容性最好; PLY: 含顶点属性; glTF: Web/实时渲染; FBX: 动画软件; STL: 3D 打印"));
        fl->addRow(tr("导出格式:"), m_formatCombo);

        m_coordSysCombo = new QComboBox;
        m_coordSysCombo->addItems({
            tr("原始 (不变换)"),
            tr("ENU (东-北-上)"),
            tr("NED (北-东-下)"),
            tr("OpenGL (右手 Y-up)"),
            tr("Unity (左手 Y-up)")
        });
        m_coordSysCombo->setCurrentIndex(0);
        m_coordSysCombo->setToolTip(tr("导出时的坐标系变换"));
        fl->addRow(tr("坐标系:"), m_coordSysCombo);

        m_upAxisCombo = new QComboBox;
        m_upAxisCombo->addItems({"Y", "Z"});
        m_upAxisCombo->setCurrentIndex(1);
        m_upAxisCombo->setToolTip(tr("Up 轴方向。摄影测量通常 Z-up, 图形引擎通常 Y-up"));
        fl->addRow(tr("Up 轴:"), m_upAxisCombo);

        root->addWidget(box);
    }

    // ── 包含数据 ──
    {
        auto *box = new QGroupBox(tr("包含数据"));
        auto *fl = new QFormLayout(box);

        m_includeTexCheck = new QCheckBox(tr("纹理"));
        m_includeTexCheck->setChecked(true);
        fl->addRow(m_includeTexCheck);

        m_includeNormalCheck = new QCheckBox(tr("法向量"));
        m_includeNormalCheck->setChecked(true);
        fl->addRow(m_includeNormalCheck);

        m_includeColorCheck = new QCheckBox(tr("顶点颜色"));
        m_includeColorCheck->setChecked(true);
        fl->addRow(m_includeColorCheck);

        root->addWidget(box);
    }

    // ── 简化 ──
    {
        auto *box = new QGroupBox(tr("导出时简化"));
        auto *fl = new QFormLayout(box);

        m_simplifyCheck = new QCheckBox(tr("启用简化"));
        m_simplifyCheck->setChecked(false);
        m_simplifyCheck->setToolTip(tr("导出前对网格做 QEM 简化"));
        fl->addRow(m_simplifyCheck);

        m_simplifyRatioSpin = new QDoubleSpinBox;
        m_simplifyRatioSpin->setRange(0.01, 1.0);
        m_simplifyRatioSpin->setDecimals(2);
        m_simplifyRatioSpin->setValue(0.50);
        m_simplifyRatioSpin->setToolTip(tr("目标保留面数比例"));
        fl->addRow(tr("保留比例:"), m_simplifyRatioSpin);

        root->addWidget(box);
    }

    // ── 输出路径 ──
    {
        auto *box = new QGroupBox(tr("输出路径"));
        auto *hl = new QHBoxLayout(box);
        m_outputPathEdit = new QLineEdit;
        m_outputPathEdit->setPlaceholderText(tr("选择输出目录或文件..."));
        hl->addWidget(m_outputPathEdit, 1);
        m_browseBtn = new QPushButton(tr("浏览..."));
        hl->addWidget(m_browseBtn);
        root->addWidget(box);
    }

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("导出"));
    root->addWidget(btnBox);

    connect(m_browseBtn, &QPushButton::clicked, this, &ModelExportDialog::onBrowseOutput);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_formatCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_coordSysCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_upAxisCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_includeTexCheck, &QCheckBox::toggled, this, changed);
    connect(m_includeNormalCheck, &QCheckBox::toggled, this, changed);
    connect(m_includeColorCheck, &QCheckBox::toggled, this, changed);
    connect(m_simplifyCheck, &QCheckBox::toggled, this, changed);
    connect(m_simplifyRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &ModelExportDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void ModelExportDialog::onBrowseOutput()
{
    QString path = QFileDialog::getSaveFileName(
        this, tr("选择导出路径"), QString(),
        tr("所有文件 (*)"));
    if (!path.isEmpty())
    {
        m_outputPathEdit->setText(path);
    }
}

QJsonObject ModelExportDialog::collectSettings() const
{
    QJsonObject o;
    o["format"]       = m_formatCombo->currentText();
    o["coordSystem"]  = m_coordSysCombo->currentText();
    o["upAxis"]       = m_upAxisCombo->currentText();
    o["includeTexture"] = m_includeTexCheck->isChecked();
    o["includeNormals"] = m_includeNormalCheck->isChecked();
    o["includeColor"]   = m_includeColorCheck->isChecked();
    o["simplify"]       = m_simplifyCheck->isChecked();
    o["simplifyRatio"]  = m_simplifyRatioSpin->value();
    o["outputPath"]     = m_outputPathEdit->text();
    return o;
}

void ModelExportDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("format"))
    {
        int i = m_formatCombo->findText(s["format"].toString());
        if (i >= 0) m_formatCombo->setCurrentIndex(i);
    }
    if (s.contains("coordSystem"))
    {
        int i = m_coordSysCombo->findText(s["coordSystem"].toString());
        if (i >= 0) m_coordSysCombo->setCurrentIndex(i);
    }
    if (s.contains("upAxis"))
    {
        int i = m_upAxisCombo->findText(s["upAxis"].toString());
        if (i >= 0) m_upAxisCombo->setCurrentIndex(i);
    }
    if (s.contains("includeTexture")) m_includeTexCheck->setChecked(s["includeTexture"].toBool());
    if (s.contains("includeNormals")) m_includeNormalCheck->setChecked(s["includeNormals"].toBool());
    if (s.contains("includeColor"))   m_includeColorCheck->setChecked(s["includeColor"].toBool());
    if (s.contains("simplify"))       m_simplifyCheck->setChecked(s["simplify"].toBool());
    if (s.contains("simplifyRatio"))  m_simplifyRatioSpin->setValue(s["simplifyRatio"].toDouble());
    if (s.contains("outputPath"))     m_outputPathEdit->setText(s["outputPath"].toString());
}

void ModelExportDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void ModelExportDialog::onRun() { emit runRequested(collectSettings()); accept(); }
