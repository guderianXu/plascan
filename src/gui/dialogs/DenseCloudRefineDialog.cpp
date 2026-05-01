#include "DenseCloudRefineDialog.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QLabel>
#include <QFont>
#include <QDialogButtonBox>
#include <QPushButton>

DenseCloudRefineDialog::DenseCloudRefineDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("密集点云后处理"));
    setMinimumWidth(480);

    auto *root = new QVBoxLayout(this);
    root->setSpacing(8);

    // ── 说明 ──────────────────────────────────────────────────────────────────
    {
        auto *info = new QLabel(
            tr("对密集点云依次执行离群点滤除、体素精简、法向量估计和颜色校正。"
               "勾选对应步骤的标题栏以启用，未勾选的步骤将被跳过。"),
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

    // ── 1. 统计离群点移除 (SOR) ───────────────────────────────────────────────
    {
        m_sorGroup = new QGroupBox(tr("1. 统计离群点移除 (SOR)（勾选后启用）"));
        m_sorGroup->setCheckable(true);
        m_sorGroup->setChecked(true);
        m_sorGroup->setToolTip(
            tr("移除偏离邻域均值超过 N 倍标准差的点。可有效清除孤立噪声点。"));
        auto *fl = new QFormLayout(m_sorGroup);

        m_sorKSpin = new QSpinBox;
        m_sorKSpin->setRange(2, 200);
        m_sorKSpin->setValue(30);
        m_sorKSpin->setToolTip(
            tr("统计时使用的 KNN 邻居数。★ 推荐 30；密集点云可适当增大至 50"));
        fl->addRow(tr("邻居数 (K):"), m_sorKSpin);

        m_sorStdSpin = new QDoubleSpinBox;
        m_sorStdSpin->setRange(0.1, 10.0);
        m_sorStdSpin->setDecimals(1);
        m_sorStdSpin->setSingleStep(0.5);
        m_sorStdSpin->setValue(2.0);
        m_sorStdSpin->setToolTip(
            tr("距离超过均值 + N×σ 的点被判为离群点。★ 推荐 2.0；"
               "减小可更激进地去噪，增大则更保守"));
        fl->addRow(tr("标准差倍数:"), m_sorStdSpin);

        root->addWidget(m_sorGroup);
    }

    // ── 2. 体素下采样 ─────────────────────────────────────────────────────────
    {
        m_voxelGroup = new QGroupBox(tr("2. 体素下采样（勾选后启用）"));
        m_voxelGroup->setCheckable(true);
        m_voxelGroup->setChecked(false);
        m_voxelGroup->setToolTip(
            tr("将空间划分为固定大小体素，每个体素内只保留质心点，降低点云密度。"));
        auto *fl = new QFormLayout(m_voxelGroup);

        m_voxelSizeSpin = new QDoubleSpinBox;
        m_voxelSizeSpin->setRange(0.001, 100.0);
        m_voxelSizeSpin->setDecimals(4);
        m_voxelSizeSpin->setSingleStep(0.001);
        m_voxelSizeSpin->setValue(0.005);
        m_voxelSizeSpin->setToolTip(
            tr("体素边长（米）。值越大采样越稀疏；典型值 0.005 m（5 mm）"));
        fl->addRow(tr("体素大小 (m):"), m_voxelSizeSpin);

        root->addWidget(m_voxelGroup);
    }

    // ── 3. 法向量估计 ─────────────────────────────────────────────────────────
    {
        m_normalGroup = new QGroupBox(tr("3. 法向量估计（勾选后启用）"));
        m_normalGroup->setCheckable(true);
        m_normalGroup->setChecked(true);
        m_normalGroup->setToolTip(
            tr("利用局部邻域 PCA 重新估计每个点的法向量，供网格重建使用。"));
        auto *fl = new QFormLayout(m_normalGroup);

        m_normalKSpin = new QSpinBox;
        m_normalKSpin->setRange(5, 200);
        m_normalKSpin->setValue(30);
        m_normalKSpin->setToolTip(
            tr("法向量估计时的邻域点数。★ 推荐 30；细节丰富的场景可适当减小"));
        fl->addRow(tr("邻域 K:"), m_normalKSpin);

        m_smoothIterSpin = new QSpinBox;
        m_smoothIterSpin->setRange(0, 20);
        m_smoothIterSpin->setValue(2);
        m_smoothIterSpin->setToolTip(
            tr("法向量平滑迭代次数，可减少法向量噪声。0 = 不平滑；★ 推荐 2"));
        fl->addRow(tr("平滑迭代:"), m_smoothIterSpin);

        root->addWidget(m_normalGroup);
    }

    // ── 4. 颜色校正 ───────────────────────────────────────────────────────────
    {
        m_colorGroup = new QGroupBox(tr("4. 颜色校正（勾选后启用）"));
        m_colorGroup->setCheckable(true);
        m_colorGroup->setChecked(false);
        m_colorGroup->setToolTip(
            tr("对多视图颜色进行全局一致性校正，减少因光照差异导致的色彩拼接痕迹。"));
        auto *fl = new QFormLayout(m_colorGroup);

        m_colorMethodCombo = new QComboBox;
        m_colorMethodCombo->addItems({
            tr("全局直方图均衡"),
            tr("增益补偿"),
            tr("多频段混合")});
        m_colorMethodCombo->setCurrentIndex(0);
        m_colorMethodCombo->setToolTip(
            tr("直方图均衡：整体亮度均衡，速度最快；"
               "增益补偿：逐图像调整增益，适合光照差异大的场景；"
               "多频段混合：最自然的过渡，但最慢"));
        fl->addRow(tr("校正方法:"), m_colorMethodCombo);

        root->addWidget(m_colorGroup);
    }

    // ── 系统 ──────────────────────────────────────────────────────────────────
    {
        auto *box = new QGroupBox(tr("系统"));
        auto *fl = new QFormLayout(box);

        m_threadsSpin = new QSpinBox;
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("CPU 并行线程数。建议设为物理核心数"));
        fl->addRow(tr("线程数:"), m_threadsSpin);

        root->addWidget(box);
    }

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("运行后处理"));
    root->addWidget(btnBox);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_sorGroup,         &QGroupBox::toggled,                                     this, changed);
    connect(m_sorKSpin,         QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_sorStdSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged),     this, changed);
    connect(m_voxelGroup,       &QGroupBox::toggled,                                     this, changed);
    connect(m_voxelSizeSpin,    QOverload<double>::of(&QDoubleSpinBox::valueChanged),     this, changed);
    connect(m_normalGroup,      &QGroupBox::toggled,                                     this, changed);
    connect(m_normalKSpin,      QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_smoothIterSpin,   QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);
    connect(m_colorGroup,       &QGroupBox::toggled,                                     this, changed);
    connect(m_colorMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),      this, changed);
    connect(m_threadsSpin,      QOverload<int>::of(&QSpinBox::valueChanged),              this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &DenseCloudRefineDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

QJsonObject DenseCloudRefineDialog::collectSettings() const
{
    QJsonObject o;
    o["sorEnabled"]      = m_sorGroup->isChecked();
    o["sorK"]            = m_sorKSpin->value();
    o["sorStdDev"]       = m_sorStdSpin->value();
    o["voxelEnabled"]    = m_voxelGroup->isChecked();
    o["voxelSize"]       = m_voxelSizeSpin->value();
    o["normalsEnabled"]  = m_normalGroup->isChecked();
    o["normalK"]         = m_normalKSpin->value();
    o["smoothIter"]      = m_smoothIterSpin->value();
    o["colorEnabled"]    = m_colorGroup->isChecked();
    o["colorMethod"]     = m_colorMethodCombo->currentText();
    o["threads"]         = m_threadsSpin->value();
    return o;
}

void DenseCloudRefineDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("sorEnabled"))    m_sorGroup->setChecked(s["sorEnabled"].toBool());
    if (s.contains("sorK"))          m_sorKSpin->setValue(s["sorK"].toInt());
    if (s.contains("sorStdDev"))     m_sorStdSpin->setValue(s["sorStdDev"].toDouble());
    if (s.contains("voxelEnabled"))  m_voxelGroup->setChecked(s["voxelEnabled"].toBool());
    if (s.contains("voxelSize"))     m_voxelSizeSpin->setValue(s["voxelSize"].toDouble());
    if (s.contains("normalsEnabled"))m_normalGroup->setChecked(s["normalsEnabled"].toBool());
    if (s.contains("normalK"))       m_normalKSpin->setValue(s["normalK"].toInt());
    if (s.contains("smoothIter"))    m_smoothIterSpin->setValue(s["smoothIter"].toInt());
    if (s.contains("colorEnabled"))  m_colorGroup->setChecked(s["colorEnabled"].toBool());
    if (s.contains("colorMethod"))
    {
        int i = m_colorMethodCombo->findText(s["colorMethod"].toString());
        if (i >= 0) m_colorMethodCombo->setCurrentIndex(i);
    }
    // 向后兼容旧键名
    if (!s.contains("normalsEnabled") && s.contains("estimateNormals"))
        m_normalGroup->setChecked(s["estimateNormals"].toBool());
    if (!s.contains("colorEnabled") && s.contains("colorCorrection"))
        m_colorGroup->setChecked(s["colorCorrection"].toBool());
    if (s.contains("threads"))       m_threadsSpin->setValue(s["threads"].toInt());
}

void DenseCloudRefineDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DenseCloudRefineDialog::onRun() { emit runRequested(collectSettings()); accept(); }
