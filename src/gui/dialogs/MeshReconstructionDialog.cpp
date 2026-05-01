#include "MeshReconstructionDialog.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>

MeshReconstructionDialog::MeshReconstructionDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("网格重建 (Mesh Reconstruction)"));
    setMinimumWidth(480);

    auto *root = new QVBoxLayout(this);

    // ── 输入点云 ──
    {
        auto *box = new QGroupBox(tr("输入点云"));
        auto *fl = new QFormLayout(box);

        auto *row = new QWidget(box);
        auto *hl = new QHBoxLayout(row);
        hl->setContentsMargins(0, 0, 0, 0);

        m_denseCloudCombo = new QComboBox(row);
        m_denseCloudCombo->setEditable(true);
        m_denseCloudCombo->setInsertPolicy(QComboBox::NoInsert);
        m_denseCloudCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        m_denseCloudCombo->setToolTip(tr("选择用于网格重建的密集点云文件（XYZ/PLY）。"));

        m_browseDenseBtn = new QPushButton(tr("浏览..."), row);

        hl->addWidget(m_denseCloudCombo, 1);
        hl->addWidget(m_browseDenseBtn);
        fl->addRow(tr("密集点云:"), row);

        root->addWidget(box);
    }

    // ── 重建方法 ──
    {
        auto *box = new QGroupBox(tr("重建方法"));
        auto *fl = new QFormLayout(box);

        m_methodCombo = new QComboBox;
        m_methodCombo->addItems({
            tr("Poisson Surface"),
            tr("Delaunay Triangulation"),
            tr("Ball Pivoting")
        });
        m_methodCombo->setCurrentIndex(0);
        m_methodCombo->setToolTip(
            tr("Poisson: 全局封闭曲面，适合 watertight 模型; Delaunay: 快速但需后处理; Ball Pivoting: 保留细节但慢"));
        fl->addRow(tr("方法:"), m_methodCombo);

        m_outputFormatCombo = new QComboBox;
        m_outputFormatCombo->addItems({tr("PLY"), tr("OBJ")});
        m_outputFormatCombo->setCurrentIndex(0);
        m_outputFormatCombo->setToolTip(tr("最终模型输出格式。PLY 为默认；OBJ 适合带纹理输出。"));
        fl->addRow(tr("最终格式:"), m_outputFormatCombo);

        m_qualityProfileCombo = new QComboBox;
        m_qualityProfileCombo->addItem(tr("细节优先 (清晰度高)"), QStringLiteral("detail"));
        m_qualityProfileCombo->addItem(tr("平衡 (推荐)"), QStringLiteral("balanced"));
        m_qualityProfileCombo->addItem(tr("轻量 (更快/更少面)"), QStringLiteral("lite"));
        m_qualityProfileCombo->setCurrentIndex(1);
        m_qualityProfileCombo->setToolTip(tr("控制网格重建参数组合：细节优先更清晰，轻量模式更省面数和时间。"));
        fl->addRow(tr("质量档位:"), m_qualityProfileCombo);

        m_octreeDepthSpin = new QSpinBox;
        m_octreeDepthSpin->setRange(4, 14);
        m_octreeDepthSpin->setValue(10);
        m_octreeDepthSpin->setToolTip(
            tr("Poisson 八叉树深度。★ 推荐 10; 增大提升细节但指数级增加内存"));
        fl->addRow(tr("八叉树深度:"), m_octreeDepthSpin);

        m_meshResSpin = new QDoubleSpinBox;
        m_meshResSpin->setRange(0.0, 1024.0);
        m_meshResSpin->setDecimals(0);
        m_meshResSpin->setSingleStep(32.0);
        m_meshResSpin->setValue(0.0);
        m_meshResSpin->setToolTip(tr("目标网格分辨率 (0=自动，建议 128~512)"));
        fl->addRow(tr("网格分辨率:"), m_meshResSpin);

        root->addWidget(box);
    }

    // ── 后处理 ──
    {
        auto *box = new QGroupBox(tr("后处理"));
        auto *fl = new QFormLayout(box);

        m_smoothIterSpin = new QSpinBox;
        m_smoothIterSpin->setRange(0, 50);
        m_smoothIterSpin->setValue(3);
        m_smoothIterSpin->setToolTip(tr("Taubin 平滑迭代次数。★ 推荐 2~4"));
        fl->addRow(tr("平滑迭代:"), m_smoothIterSpin);

        m_holeFillCheck = new QCheckBox(tr("启用补洞"));
        m_holeFillCheck->setChecked(true);
        fl->addRow(m_holeFillCheck);

        m_maxHoleSizeSpin = new QDoubleSpinBox;
        m_maxHoleSizeSpin->setRange(0.0, 100000.0);
        m_maxHoleSizeSpin->setDecimals(1);
        m_maxHoleSizeSpin->setValue(100.0);
        m_maxHoleSizeSpin->setToolTip(tr("最大补洞面积 (面数)。超过此值的洞不补"));
        fl->addRow(tr("最大洞面积:"), m_maxHoleSizeSpin);

        m_cleanCheck = new QCheckBox(tr("清除小连通体"));
        m_cleanCheck->setChecked(true);
        m_cleanCheck->setToolTip(tr("移除面数少于阈值的孤立碎片"));
        fl->addRow(m_cleanCheck);

        m_minFacesSpin = new QSpinBox;
        m_minFacesSpin->setRange(1, 100000);
        m_minFacesSpin->setValue(100);
        m_minFacesSpin->setToolTip(tr("小于此面数的连通体将被移除"));
        fl->addRow(tr("最小面数:"), m_minFacesSpin);

        m_voxelDensityCombo = new QComboBox;
        m_voxelDensityCombo->addItem(tr("粗 (更少三角)"), QStringLiteral("coarse"));
        m_voxelDensityCombo->addItem(tr("中"), QStringLiteral("medium"));
        m_voxelDensityCombo->addItem(tr("细 (更多细节)"), QStringLiteral("fine"));
        m_voxelDensityCombo->setCurrentIndex(1);
        m_voxelDensityCombo->setToolTip(tr("仅对封闭体 3D 重建分支生效。粗=更少小三角，细=保留更多细节。"));
        fl->addRow(tr("体素面密度:"), m_voxelDensityCombo);

        root->addWidget(box);
    }

    // ── 简化 ──
    {
        auto *box = new QGroupBox(tr("网格简化"));
        auto *fl = new QFormLayout(box);

        m_decimateCheck = new QCheckBox(tr("启用简化"));
        m_decimateCheck->setChecked(false);
        m_decimateCheck->setToolTip(tr("用二次误差度量 (QEM) 减少三角面数"));
        fl->addRow(m_decimateCheck);

        m_decimateRatioSpin = new QDoubleSpinBox;
        m_decimateRatioSpin->setRange(0.01, 1.0);
        m_decimateRatioSpin->setDecimals(2);
        m_decimateRatioSpin->setValue(0.50);
        m_decimateRatioSpin->setToolTip(tr("目标保留比例。0.5 表示保留50%的面"));
        fl->addRow(tr("保留比例:"), m_decimateRatioSpin);

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

    m_infoLabel = new QLabel(tr("输出: 项目目录下的 mesh 文件"));
    root->addWidget(m_infoLabel);

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("开始重建"));
    root->addWidget(btnBox);

    auto changed = [this]() { emitSettingsNow(); };
    connect(m_denseCloudCombo, &QComboBox::editTextChanged, this, changed);
    connect(m_methodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_qualityProfileCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_octreeDepthSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_meshResSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_smoothIterSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);
    connect(m_holeFillCheck, &QCheckBox::toggled, this, changed);
    connect(m_cleanCheck, &QCheckBox::toggled, this, changed);
    connect(m_voxelDensityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, changed);
    connect(m_decimateCheck, &QCheckBox::toggled, this, changed);
    connect(m_decimateRatioSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_threadsSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, changed);

    connect(m_browseDenseBtn, &QPushButton::clicked, this, [this]() {
        const QString initialPath = m_denseCloudCombo->currentText().trimmed();
        const QString initialDir = initialPath.isEmpty()
            ? QDir::homePath()
            : QFileInfo(initialPath).absolutePath();

        const QString chosenPath = QFileDialog::getOpenFileName(
            this,
            tr("选择密集点云"),
            initialDir,
            tr("点云文件 (*.xyz *.ply *.txt);;所有文件 (*)"));

        if (chosenPath.isEmpty())
        {
            return;
        }

        const QString normalized = QDir::cleanPath(chosenPath);
        const int index = m_denseCloudCombo->findData(normalized);
        if (index < 0)
        {
            m_denseCloudCombo->addItem(normalized, normalized);
        }
        m_denseCloudCombo->setCurrentText(normalized);
        emitSettingsNow();
    });

    connect(btnBox, &QDialogButtonBox::accepted, this, &MeshReconstructionDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

QJsonObject MeshReconstructionDialog::collectSettings() const
{
    QJsonObject o;
    o["method"]        = m_methodCombo->currentText();
    o["denseCloudPath"] = m_denseCloudCombo->currentText().trimmed();
    o["export_format"] = m_outputFormatCombo->currentText();
    o["qualityProfile"] = m_qualityProfileCombo->currentData().toString();
    o["octreeDepth"]   = m_octreeDepthSpin->value();
    o["meshResolution"]= m_meshResSpin->value();
    o["smoothIter"]    = m_smoothIterSpin->value();
    o["holeFill"]      = m_holeFillCheck->isChecked();
    o["maxHoleSize"]   = m_maxHoleSizeSpin->value();
    o["cleanSmall"]    = m_cleanCheck->isChecked();
    o["minFaces"]      = m_minFacesSpin->value();
    o["voxelDensity"]  = m_voxelDensityCombo->currentData().toString();
    o["decimate"]      = m_decimateCheck->isChecked();
    o["decimateRatio"] = m_decimateRatioSpin->value();
    o["threads"]       = m_threadsSpin->value();
    return o;
}

void MeshReconstructionDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("method"))
    {
        int i = m_methodCombo->findText(s["method"].toString());
        if (i >= 0) m_methodCombo->setCurrentIndex(i);
    }
    if (s.contains("denseCloudPath"))
    {
        m_denseCloudCombo->setCurrentText(QDir::cleanPath(s["denseCloudPath"].toString()));
    }
    if (s.contains("export_format"))
    {
        const int i = m_outputFormatCombo->findText(s["export_format"].toString());
        if (i >= 0) m_outputFormatCombo->setCurrentIndex(i);
    }
    if (s.contains("qualityProfile"))
    {
        const QString qualityProfile = s["qualityProfile"].toString();
        const int i = m_qualityProfileCombo->findData(qualityProfile);
        if (i >= 0)
        {
            m_qualityProfileCombo->setCurrentIndex(i);
        }
    }
    if (s.contains("octreeDepth"))    m_octreeDepthSpin->setValue(s["octreeDepth"].toInt());
    if (s.contains("meshResolution")) m_meshResSpin->setValue(s["meshResolution"].toDouble());
    if (s.contains("smoothIter"))     m_smoothIterSpin->setValue(s["smoothIter"].toInt());
    if (s.contains("holeFill"))       m_holeFillCheck->setChecked(s["holeFill"].toBool());
    if (s.contains("maxHoleSize"))    m_maxHoleSizeSpin->setValue(s["maxHoleSize"].toDouble());
    if (s.contains("cleanSmall"))     m_cleanCheck->setChecked(s["cleanSmall"].toBool());
    if (s.contains("minFaces"))       m_minFacesSpin->setValue(s["minFaces"].toInt());
    if (s.contains("voxelDensity"))
    {
        const QString density = s["voxelDensity"].toString();
        const int index = m_voxelDensityCombo->findData(density);
        if (index >= 0)
        {
            m_voxelDensityCombo->setCurrentIndex(index);
        }
    }
    if (s.contains("decimate"))       m_decimateCheck->setChecked(s["decimate"].toBool());
    if (s.contains("decimateRatio"))  m_decimateRatioSpin->setValue(s["decimateRatio"].toDouble());
    if (s.contains("threads"))        m_threadsSpin->setValue(s["threads"].toInt());
}

void MeshReconstructionDialog::setDenseCloudCandidates(const QStringList &paths)
{
    const QString current = QDir::cleanPath(m_denseCloudCombo->currentText().trimmed());

    m_denseCloudCombo->clear();
    for (const QString &path : paths)
    {
        const QString normalized = QDir::cleanPath(path);
        if (!normalized.isEmpty() && m_denseCloudCombo->findData(normalized) < 0)
        {
            m_denseCloudCombo->addItem(normalized, normalized);
        }
    }

    if (!current.isEmpty())
    {
        const int index = m_denseCloudCombo->findData(current);
        if (index >= 0)
        {
            m_denseCloudCombo->setCurrentIndex(index);
        }
        else
        {
            m_denseCloudCombo->setCurrentText(current);
        }
    }
    else if (m_denseCloudCombo->count() > 0)
    {
        m_denseCloudCombo->setCurrentIndex(0);
    }
}

void MeshReconstructionDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void MeshReconstructionDialog::onRun() { emit runRequested(collectSettings()); accept(); }
