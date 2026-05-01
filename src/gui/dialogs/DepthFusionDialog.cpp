#include "DepthFusionDialog.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QFont>
#include <QDialogButtonBox>
#include <QPushButton>

DepthFusionDialog::DepthFusionDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("深度图融合生成密集点云"));
    setMinimumWidth(500);

    auto *root = new QVBoxLayout(this);
    root->setSpacing(8);

    // ── 说明 ──────────────────────────────────────────────────────────────────
    {
        auto *info = new QLabel(
            tr("将各视图的深度图按一致性检查合并为统一的密集三维点云。"
               "仅保留在多个视图中深度一致的点，以滤除噪声和误匹配。"),
            this);
        info->setWordWrap(true);
        QFont fi = info->font();
        fi.setPointSizeF(fi.pointSizeF() * 0.88);
        info->setFont(fi);
        auto fpal = info->palette();
        fpal.setColor(QPalette::WindowText, fpal.color(QPalette::Mid));
        info->setPalette(fpal);
        root->addWidget(info);
    }

    // ── 融合策略 ──
    {
        auto *box = new QGroupBox(tr("融合策略"));
        auto *fl = new QFormLayout(box);

        m_fusionMethodCombo = new QComboBox;
        m_fusionMethodCombo->addItems({
            tr("基于可见性 (Visibility)"),
            tr("TSDF 体积融合"),
            tr("简单深度平均")
        });
        m_fusionMethodCombo->setCurrentIndex(0);
        m_fusionMethodCombo->setToolTip(
            tr("Visibility: 利用多视图一致性过滤后合并，最常用"));
        fl->addRow(tr("融合方法:"), m_fusionMethodCombo);

        m_depthConsistSpin = new QDoubleSpinBox;
        m_depthConsistSpin->setRange(0.01, 50.0);
        m_depthConsistSpin->setDecimals(2);
        m_depthConsistSpin->setValue(1.0);
        m_depthConsistSpin->setToolTip(
            tr("相邻视图深度差异阈值 (像素)。★ 推荐 1.0; 严格时设小值"));
        fl->addRow(tr("深度一致性阈值:"), m_depthConsistSpin);

        m_minConsistViewSpin = new QSpinBox;
        m_minConsistViewSpin->setRange(2, 20);
        m_minConsistViewSpin->setValue(3);
        m_minConsistViewSpin->setToolTip(
            tr("至少需要多少个一致视图才保留该点。★ 推荐 3"));
        fl->addRow(tr("最少一致视图:"), m_minConsistViewSpin);

        m_normalConsistSpin = new QDoubleSpinBox;
        m_normalConsistSpin->setRange(0.0, 90.0);
        m_normalConsistSpin->setDecimals(1);
        m_normalConsistSpin->setValue(30.0);
        m_normalConsistSpin->setSuffix(QString::fromUtf8(" \u00b0"));
        m_normalConsistSpin->setToolTip(
            tr("法向量最大偏差角。★ 推荐 30; 越小筛选越严"));
        fl->addRow(tr("法向量一致性:"), m_normalConsistSpin);

        root->addWidget(box);
    }

    // ── 点云质量 ──
    {
        auto *box = new QGroupBox(tr("点云质量"));
        auto *fl = new QFormLayout(box);

        m_voxelSizeSpin = new QDoubleSpinBox;
        m_voxelSizeSpin->setRange(0.0, 100.0);
        m_voxelSizeSpin->setDecimals(3);
        m_voxelSizeSpin->setValue(0.0);
        m_voxelSizeSpin->setToolTip(tr("体素大小 (0=不做下采样)。设定后用于体素滤波去重"));
        fl->addRow(tr("体素大小:"), m_voxelSizeSpin);

        m_minConfidenceSpin = new QDoubleSpinBox;
        m_minConfidenceSpin->setRange(0.0, 1.0);
        m_minConfidenceSpin->setDecimals(2);
        m_minConfidenceSpin->setValue(0.5);
        m_minConfidenceSpin->setToolTip(tr("低于此置信度的点被丢弃。★ 推荐 0.5"));
        fl->addRow(tr("最小置信度:"), m_minConfidenceSpin);

        m_maxReprojSpin = new QDoubleSpinBox;
        m_maxReprojSpin->setRange(0.1, 50.0);
        m_maxReprojSpin->setDecimals(1);
        m_maxReprojSpin->setValue(2.0);
        m_maxReprojSpin->setSuffix(tr(" px"));
        m_maxReprojSpin->setToolTip(tr("融合后重投影误差超过此值的点被丢弃。★ 推荐 2.0"));
        fl->addRow(tr("最大重投影误差:"), m_maxReprojSpin);

        m_colorCheck = new QCheckBox(tr("保留颜色"));
        m_colorCheck->setChecked(true);
        fl->addRow(m_colorCheck);

        m_normalCheck = new QCheckBox(tr("保留法向量"));
        m_normalCheck->setChecked(true);
        fl->addRow(m_normalCheck);

        root->addWidget(box);
    }

    // ── 系统 ──
    {
        auto *box = new QGroupBox(tr("系统"));
        auto *fl = new QFormLayout(box);

        m_cudaCheck = new QCheckBox(tr("启用 CUDA 加速"));
        m_cudaCheck->setChecked(true);
        m_cudaCheck->setToolTip(tr("使用 GPU 加速深度图融合，速度显著提升"));
        fl->addRow(m_cudaCheck);

        m_threadsSpin = new QSpinBox;
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("CPU 线程数（CUDA 关闭时生效）。建议设为物理核心数"));
        fl->addRow(tr("线程数:"), m_threadsSpin);

        root->addWidget(box);
    }

    m_infoLabel = new QLabel(tr("融合点云将保存到项目目录，可在后续步骤中用于网格重建。"));
    m_infoLabel->setWordWrap(true);
    root->addWidget(m_infoLabel);

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("开始融合"));
    root->addWidget(btnBox);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_fusionMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_depthConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_minConsistViewSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_normalConsistSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_voxelSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_minConfidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_maxReprojSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_colorCheck, &QCheckBox::toggled, this, changed);
    connect(m_normalCheck, &QCheckBox::toggled, this, changed);
    connect(m_cudaCheck, &QCheckBox::toggled, this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &DepthFusionDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

QJsonObject DepthFusionDialog::collectSettings() const
{
    QJsonObject o;
    o["fusionMethod"]       = m_fusionMethodCombo->currentText();
    o["depthConsistency"]   = m_depthConsistSpin->value();
    o["minConsistentViews"] = m_minConsistViewSpin->value();
    o["normalConsistency"]  = m_normalConsistSpin->value();
    o["voxelSize"]          = m_voxelSizeSpin->value();
    o["minConfidence"]      = m_minConfidenceSpin->value();
    o["maxReprojError"]     = m_maxReprojSpin->value();
    o["keepColor"]          = m_colorCheck->isChecked();
    o["keepNormals"]        = m_normalCheck->isChecked();
    o["cuda"]               = m_cudaCheck->isChecked();
    o["threads"]            = m_threadsSpin->value();
    return o;
}

void DepthFusionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("fusionMethod"))
    {
        int i = m_fusionMethodCombo->findText(s["fusionMethod"].toString());
        if (i >= 0) m_fusionMethodCombo->setCurrentIndex(i);
    }
    if (s.contains("depthConsistency"))   m_depthConsistSpin->setValue(s["depthConsistency"].toDouble());
    if (s.contains("minConsistentViews")) m_minConsistViewSpin->setValue(s["minConsistentViews"].toInt());
    if (s.contains("normalConsistency"))  m_normalConsistSpin->setValue(s["normalConsistency"].toDouble());
    if (s.contains("voxelSize"))          m_voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    if (s.contains("minConfidence"))      m_minConfidenceSpin->setValue(s["minConfidence"].toDouble());
    if (s.contains("maxReprojError"))     m_maxReprojSpin->setValue(s["maxReprojError"].toDouble());
    if (s.contains("keepColor"))          m_colorCheck->setChecked(s["keepColor"].toBool());
    if (s.contains("keepNormals"))        m_normalCheck->setChecked(s["keepNormals"].toBool());
    if (s.contains("cuda"))               m_cudaCheck->setChecked(s["cuda"].toBool());
    if (s.contains("threads"))            m_threadsSpin->setValue(s["threads"].toInt());
}

void DepthFusionDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DepthFusionDialog::onRun() { emit runRequested(collectSettings()); accept(); }
