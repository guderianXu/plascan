#include "DepthMapEstimateDialog.h"

#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>
#include <QFileInfo>
#include <QFont>
#include <QDialogButtonBox>
#include <QPushButton>

DepthMapEstimateDialog::DepthMapEstimateDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("深度图估计 (Depth Map Estimation)"));
    setMinimumWidth(520);

    auto *root = new QVBoxLayout(this);
    root->setSpacing(8);

    // ── 说明 ──────────────────────────────────────────────────────────────────
    {
        auto *info = new QLabel(
            tr("对每张输入影像，利用 PatchMatch 立体匹配算法与相邻视图计算逐像素深度图。"
               "预设档位会自动填充推荐参数；选择\"自定义\"后可自由调节所有参数。"),
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

    // ── 预设 ──
    {
        auto *box = new QGroupBox(tr("预设方案"));
        auto *fl = new QFormLayout(box);

        m_atResultCombo = new QComboBox;
        m_atResultCombo->setToolTip(tr("选择用于深度估计的 AT 结果（对应稀疏点云来源）。"));
        m_atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
        fl->addRow(tr("AT 结果:"), m_atResultCombo);

        m_presetCombo = new QComboBox;
        m_presetCombo->addItems({tr("快速"), tr("标准"), tr("精细"), tr("自定义")});
        m_presetCombo->setCurrentIndex(1);
        fl->addRow(tr("预设:"), m_presetCombo);
        root->addWidget(box);
    }

    // ── PatchMatch 参数 ──
    {
        auto *box = new QGroupBox(tr("PatchMatch 参数"));
        auto *fl = new QFormLayout(box);

        m_resScaleSpin = new QDoubleSpinBox;
        m_resScaleSpin->setRange(0.1, 1.0);
        m_resScaleSpin->setDecimals(2);
        m_resScaleSpin->setSingleStep(0.1);
        m_resScaleSpin->setValue(0.5);
        m_resScaleSpin->setToolTip(tr("图像分辨率缩放。★ 标准 0.5; 精细 1.0 (原始分辨率)"));
        fl->addRow(tr("分辨率缩放:"), m_resScaleSpin);

        m_iterationsSpin = new QSpinBox;
        m_iterationsSpin->setRange(1, 30);
        m_iterationsSpin->setValue(6);
        m_iterationsSpin->setToolTip(tr("PatchMatch 迭代次数。★ 推荐 6; 更多次收敛更好但更慢"));
        fl->addRow(tr("迭代次数:"), m_iterationsSpin);

        m_costFuncCombo = new QComboBox;
        m_costFuncCombo->addItems({"NCC", "Census", "SAD", "ZNCC"});
        m_costFuncCombo->setCurrentIndex(0);
        m_costFuncCombo->setToolTip(tr("NCC: 归一化互相关，平衡精度和速度; Census: 抗光照变化"));
        fl->addRow(tr("代价函数:"), m_costFuncCombo);

        m_propagCombo = new QComboBox;
        m_propagCombo->addItems({tr("棋盘格 (Checkerboard)"), tr("顺序 (Sequential)"), tr("红黑 (Red-Black)")});
        m_propagCombo->setCurrentIndex(0);
        m_propagCombo->setToolTip(tr("传播模式。棋盘格最适合 GPU 并行"));
        fl->addRow(tr("传播方式:"), m_propagCombo);

        m_patchSizeSpin = new QSpinBox;
        m_patchSizeSpin->setRange(3, 31);
        m_patchSizeSpin->setSingleStep(2);
        m_patchSizeSpin->setValue(11);
        m_patchSizeSpin->setToolTip(tr("匹配窗口大小 (奇数)。★ 推荐 11; 增大可减少噪声但丢失细节"));
        fl->addRow(tr("窗口大小:"), m_patchSizeSpin);

        m_minViewsSpin = new QSpinBox;
        m_minViewsSpin->setRange(2, 20);
        m_minViewsSpin->setValue(3);
        m_minViewsSpin->setToolTip(tr("参与深度估计的最少视图。★ 推荐 3"));
        fl->addRow(tr("最少视图:"), m_minViewsSpin);

        root->addWidget(box);
    }

    // ── 深度范围 ──
    {
        auto *box = new QGroupBox(tr("深度范围与置信度"));
        auto *fl = new QFormLayout(box);

        m_depthMinSpin = new QDoubleSpinBox;
        m_depthMinSpin->setRange(0.0, 100000.0);
        m_depthMinSpin->setDecimals(2);
        m_depthMinSpin->setValue(0.0);
        m_depthMinSpin->setToolTip(tr("最小深度 (0=自动)"));
        fl->addRow(tr("最小深度:"), m_depthMinSpin);

        m_depthMaxSpin = new QDoubleSpinBox;
        m_depthMaxSpin->setRange(0.0, 100000.0);
        m_depthMaxSpin->setDecimals(2);
        m_depthMaxSpin->setValue(0.0);
        m_depthMaxSpin->setToolTip(tr("最大深度 (0=自动)"));
        fl->addRow(tr("最大深度:"), m_depthMaxSpin);

        m_confidenceSpin = new QDoubleSpinBox;
        m_confidenceSpin->setRange(0.0, 1.0);
        m_confidenceSpin->setDecimals(2);
        m_confidenceSpin->setSingleStep(0.05);
        m_confidenceSpin->setValue(0.5);
        m_confidenceSpin->setToolTip(tr("低于此置信度的深度值被标记为无效。★ 推荐 0.5"));
        fl->addRow(tr("置信度阈值:"), m_confidenceSpin);

        m_normalMapCheck = new QCheckBox(tr("同时输出法向量图"));
        m_normalMapCheck->setChecked(true);
        fl->addRow(m_normalMapCheck);

        root->addWidget(box);
    }

    // ── GPU / 分块 ──
    {
        auto *box = new QGroupBox(tr("硬件与分块"));
        auto *fl = new QFormLayout(box);

        m_cudaCheck = new QCheckBox(tr("启用 CUDA 加速"));
        m_cudaCheck->setChecked(true);
        fl->addRow(m_cudaCheck);

        m_tileWSpin = new QSpinBox;
        m_tileWSpin->setRange(128, 4096);
        m_tileWSpin->setSingleStep(128);
        m_tileWSpin->setValue(1024);
        m_tileWSpin->setToolTip(tr("Tile 宽度 (px)。大显存可设更大"));
        fl->addRow(tr("Tile 宽度:"), m_tileWSpin);

        m_tileHSpin = new QSpinBox;
        m_tileHSpin->setRange(128, 4096);
        m_tileHSpin->setSingleStep(128);
        m_tileHSpin->setValue(1024);
        m_tileHSpin->setToolTip(tr("Tile 高度 (px)。大显存可设更大，小显存需相应减小"));
        fl->addRow(tr("Tile 高度:"), m_tileHSpin);

        m_threadsSpin = new QSpinBox;
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("CPU 线程数（CUDA 关闭时生效）。建议设为物理核心数"));
        fl->addRow(tr("线程数:"), m_threadsSpin);

        root->addWidget(box);
    }

    m_estimateLabel = nullptr; // 占位，不再显示"预计: -"

    auto *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    btnBox->button(QDialogButtonBox::Ok)->setText(tr("开始估计"));
    root->addWidget(btnBox);

    connect(m_presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &DepthMapEstimateDialog::onPresetChanged);

    // 任意参数变更时：若不在应用预设过程中，则自动切换到"自定义"
    auto changed = [this]() {
        if (!m_applyingPreset) {
            m_presetCombo->blockSignals(true);
            m_presetCombo->setCurrentIndex(3);
            m_presetCombo->blockSignals(false);
        }
        emitSettingsNow();
    };
    connect(m_resScaleSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_atResultCombo,  QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_iterationsSpin, QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_costFuncCombo,  QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_propagCombo,    QOverload<int>::of(&QComboBox::currentIndexChanged),  this, changed);
    connect(m_patchSizeSpin,  QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_minViewsSpin,   QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_depthMinSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_depthMaxSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_confidenceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, changed);
    connect(m_normalMapCheck, &QCheckBox::toggled,                                  this, changed);
    connect(m_cudaCheck,      &QCheckBox::toggled,                                  this, changed);
    connect(m_tileWSpin,      QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_tileHSpin,      QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);
    connect(m_threadsSpin,    QOverload<int>::of(&QSpinBox::valueChanged),          this, changed);

    connect(btnBox, &QDialogButtonBox::accepted, this, &DepthMapEstimateDialog::onRun);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    onPresetChanged(1);
}

void DepthMapEstimateDialog::setAvailableAtResults(const QJsonArray &atResults)
{
    if (!m_atResultCombo)
    {
        return;
    }

    m_atResultCombo->blockSignals(true);
    m_atResultCombo->clear();

    if (atResults.isEmpty())
    {
        m_atResultCombo->addItem(tr("（请先运行空三）"), -2);
        m_atResultCombo->setEnabled(false);
    }
    else
    {
        m_atResultCombo->setEnabled(true);
        m_atResultCombo->addItem(tr("最新 AT 结果（推荐）"), -1);
        for (int i = 0; i < atResults.size(); ++i)
        {
            const QJsonObject item = atResults.at(i).toObject();
            const int idx = item.value(QStringLiteral("index")).toInt(i);
            const int imgCnt = item.value(QStringLiteral("image_count")).toInt(0);
            const int ptsCnt = item.value(QStringLiteral("sparse_point_count")).toInt(0);
            const QString createdAt = item.value(QStringLiteral("created_at")).toString().left(10);
            const QString outDir = QFileInfo(item.value(QStringLiteral("output_dir")).toString()).fileName();
            const QString label = QStringLiteral("[%1] %2  %3 张影像  %4 点  (%5)")
                                      .arg(idx)
                                      .arg(outDir)
                                      .arg(imgCnt)
                                      .arg(ptsCnt)
                                      .arg(createdAt);
            m_atResultCombo->addItem(label, idx);
        }

        const int desired = m_pendingAtIndex;
        int comboIndex = -1;
        if (desired >= -1)
        {
            comboIndex = m_atResultCombo->findData(desired);
        }
        if (comboIndex < 0)
        {
            comboIndex = m_atResultCombo->findData(-1);
        }
        if (comboIndex < 0)
        {
            comboIndex = 0;
        }
        m_atResultCombo->setCurrentIndex(comboIndex);
    }

    m_atResultCombo->blockSignals(false);
    emitSettingsNow();
}

void DepthMapEstimateDialog::onPresetChanged(int index)
{
    if (index == 3) return;
    struct P { double rs; int it; int cf; int pr; int ps; int mv; double co; };
    static const P kP[] = {
        {0.25, 3, 0, 0, 7,  2, 0.3},
        {0.50, 6, 0, 0, 11, 3, 0.5},
        {1.00, 10,0, 0, 15, 4, 0.7},
    };
    const auto &p = kP[index];
    m_applyingPreset = true;
    m_resScaleSpin->setValue(p.rs);
    m_iterationsSpin->setValue(p.it);
    m_costFuncCombo->setCurrentIndex(p.cf);
    m_propagCombo->setCurrentIndex(p.pr);
    m_patchSizeSpin->setValue(p.ps);
    m_minViewsSpin->setValue(p.mv);
    m_confidenceSpin->setValue(p.co);
    m_applyingPreset = false;
    emitSettingsNow();
}

QJsonObject DepthMapEstimateDialog::collectSettings() const
{
    QJsonObject o;
    const QVariant atData = m_atResultCombo ? m_atResultCombo->currentData() : QVariant(-1);
    const int atIndex = atData.isValid() ? atData.toInt() : -1;
    o["at_index"] = atIndex;
    o["at_selection_mode"] = (atIndex < 0) ? QStringLiteral("latest") : QStringLiteral("fixed");
    o["preset"]       = m_presetCombo->currentText();
    o["resScale"]     = m_resScaleSpin->value();
    o["iterations"]   = m_iterationsSpin->value();
    o["costFunction"] = m_costFuncCombo->currentText();
    o["propagation"]  = m_propagCombo->currentText();
    o["patchSize"]    = m_patchSizeSpin->value();
    o["minViews"]     = m_minViewsSpin->value();
    o["depthMin"]     = m_depthMinSpin->value();
    o["depthMax"]     = m_depthMaxSpin->value();
    o["confidence"]   = m_confidenceSpin->value();
    o["normalMap"]    = m_normalMapCheck->isChecked();
    o["cuda"]         = m_cudaCheck->isChecked();
    o["tileWidth"]    = m_tileWSpin->value();
    o["tileHeight"]   = m_tileHSpin->value();
    o["threads"]      = m_threadsSpin->value();
    return o;
}

void DepthMapEstimateDialog::applySettings(const QJsonObject &s)
{
    if (s.contains("at_selection_mode") && s.value("at_selection_mode").toString() == QStringLiteral("latest"))
    {
        m_pendingAtIndex = -1;
    }
    else if (s.contains("at_index"))
    {
        m_pendingAtIndex = s.value("at_index").toInt(-1);
    }

    if (m_atResultCombo && m_atResultCombo->count() > 0)
    {
        const int idx = m_atResultCombo->findData(m_pendingAtIndex);
        if (idx >= 0)
        {
            m_atResultCombo->setCurrentIndex(idx);
        }
    }

    if (s.contains("resScale"))     m_resScaleSpin->setValue(s["resScale"].toDouble());
    if (s.contains("iterations"))   m_iterationsSpin->setValue(s["iterations"].toInt());
    if (s.contains("costFunction"))
    {
        int i = m_costFuncCombo->findText(s["costFunction"].toString());
        if (i >= 0) m_costFuncCombo->setCurrentIndex(i);
    }
    if (s.contains("propagation"))
    {
        int i = m_propagCombo->findText(s["propagation"].toString());
        if (i >= 0) m_propagCombo->setCurrentIndex(i);
    }
    if (s.contains("patchSize"))    m_patchSizeSpin->setValue(s["patchSize"].toInt());
    if (s.contains("minViews"))     m_minViewsSpin->setValue(s["minViews"].toInt());
    if (s.contains("depthMin"))     m_depthMinSpin->setValue(s["depthMin"].toDouble());
    if (s.contains("depthMax"))     m_depthMaxSpin->setValue(s["depthMax"].toDouble());
    if (s.contains("confidence"))   m_confidenceSpin->setValue(s["confidence"].toDouble());
    if (s.contains("normalMap"))    m_normalMapCheck->setChecked(s["normalMap"].toBool());
    if (s.contains("cuda"))         m_cudaCheck->setChecked(s["cuda"].toBool());
    if (s.contains("tileWidth"))    m_tileWSpin->setValue(s["tileWidth"].toInt());
    if (s.contains("tileHeight"))   m_tileHSpin->setValue(s["tileHeight"].toInt());
    if (s.contains("threads"))      m_threadsSpin->setValue(s["threads"].toInt());
}

void DepthMapEstimateDialog::emitSettingsNow() { emit settingsChanged(collectSettings()); }
void DepthMapEstimateDialog::onRun() { emit runRequested(collectSettings()); accept(); }
