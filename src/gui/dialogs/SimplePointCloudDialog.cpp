// =============================================================================
// 文件: SimplePointCloudDialog.cpp
// 说明: 一键创建稠密点云对话框实现
// =============================================================================
#include "SimplePointCloudDialog.h"
#include "ProjectManager.h"

#include <QBoxLayout>
#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QPushButton>
#include <QFrame>
#include <QStyle>

SimplePointCloudDialog::SimplePointCloudDialog(ProjectManager *projectManager,
                                               QWidget *parent)
    : QDialog(parent)
    , m_projectManager(projectManager)
{
    setWindowTitle(tr("创建密集点云"));
    setFixedSize(420, 340);
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    setupUi();
    loadAtInfo();
}

void SimplePointCloudDialog::setDefaultOutputDir(const QString &dir)
{
    m_outputDir = dir;
}

// ─────────────────────────────────────────────────────────────────────────────
void SimplePointCloudDialog::setupUi()
{
    auto *root = new QVBoxLayout(this);
    root->setSpacing(10);
    root->setContentsMargins(16, 14, 16, 12);

    // ── 顶部图标 + 说明 ──────────────────────────────────────────────────
    auto *titleRow = new QHBoxLayout;
    auto *iconLabel = new QLabel(this);
    iconLabel->setPixmap(
        style()->standardPixmap(QStyle::SP_FileDialogDetailedView)
              .scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    iconLabel->setFixedSize(32, 32);
    titleRow->addWidget(iconLabel);

    auto *titleLabel = new QLabel(tr("<b>从稀疏点云创建密集点云</b>"), this);
    titleRow->addWidget(titleLabel, 1);
    root->addLayout(titleRow);

    // ── 分隔线 ────────────────────────────────────────────────────────────
    auto *sep = new QFrame(this);
    sep->setFrameShape(QFrame::HLine);
    sep->setFrameShadow(QFrame::Sunken);
    root->addWidget(sep);

    // ── AT 状态信息 ───────────────────────────────────────────────────────
    m_infoLabel = new QLabel(tr("正在读取空三结果..."), this);
    m_infoLabel->setWordWrap(true);
    m_infoLabel->setStyleSheet(QStringLiteral("color: #555; font-size: 12px;"));
    root->addWidget(m_infoLabel);

    auto *atRow = new QHBoxLayout;
    atRow->addWidget(new QLabel(tr("输入稀疏点云："), this));
    m_atResultCombo = new QComboBox(this);
    atRow->addWidget(m_atResultCombo, 1);
    root->addLayout(atRow);

    // ── 质量档位 ──────────────────────────────────────────────────────────
    auto *qualRow = new QHBoxLayout;
    qualRow->addWidget(new QLabel(tr("处理质量："), this));
    m_qualityCombo = new QComboBox(this);
    m_qualityCombo->addItem(tr("快速（1/2 分辨率，速度优先）"), QStringLiteral("fast"));
    m_qualityCombo->addItem(tr("标准（原始分辨率，推荐）"),     QStringLiteral("standard"));
    m_qualityCombo->addItem(tr("精细（高精度，较慢）"),         QStringLiteral("quality"));
    m_qualityCombo->setCurrentIndex(1);
    m_qualityCombo->setToolTip(tr("快速：半分辨率 SGBM，速度约 4 倍\n"
                                    "标准：原始分辨率，精度与速度平衡\n"
                                    "精细：增大视差范围 + WLS 滤波，最高质量"));
    qualRow->addWidget(m_qualityCombo, 1);
    root->addLayout(qualRow);

    // ── 输出选项 ──────────────────────────────────────────────────────────
    m_colorsCheck = new QCheckBox(tr("生成彩色点云（从影像采样 RGB）"), this);
    m_colorsCheck->setChecked(true);
    root->addWidget(m_colorsCheck);

    m_meshCheck = new QCheckBox(tr("同时生成三角网格模型"), this);
    m_meshCheck->setChecked(false);
    root->addWidget(m_meshCheck);

    root->addStretch();

    // ── 按钮行 ────────────────────────────────────────────────────────────
    auto *btnRow = new QHBoxLayout;
    btnRow->addStretch();
    m_cancelBtn = new QPushButton(tr("取消"), this);
    m_startBtn  = new QPushButton(tr("开始生成密集点云"), this);
    m_startBtn->setDefault(true);
    m_startBtn->setFixedWidth(120);
    btnRow->addWidget(m_cancelBtn);
    btnRow->addWidget(m_startBtn);
    root->addLayout(btnRow);

    connect(m_cancelBtn, &QPushButton::clicked, this, &QDialog::reject);
    connect(m_startBtn,  &QPushButton::clicked, this, &SimplePointCloudDialog::onStartClicked);
    connect(m_atResultCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int)
    {
        updateAtInfoLabel();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
void SimplePointCloudDialog::loadAtInfo()
{
    if (!m_projectManager) {
        m_infoLabel->setText(tr("项目未打开。请先打开项目并完成空三。"));
        m_startBtn->setEnabled(false);
        return;
    }

    m_atResults = m_projectManager->getAvailableAtResults();
    m_atResultCombo->clear();
    if (m_atResults.isEmpty()) {
        m_infoLabel->setText(tr("\u26a0 \u672a\u627e\u5230\u7a7a\u4e09\u7ed3\u679c\u3002\n\u8bf7\u5148\u6267\u884c [\u7a7a\u4e2d\u4e09\u89d2\u6d4b\u91cf] \u4ee5\u83b7\u53d6\u76f8\u673a\u4f4d\u59ff\u3002"));
        m_startBtn->setEnabled(false);
        return;
    }

    for (const QJsonValue &value : m_atResults)
    {
        const QJsonObject item = value.toObject();
        m_atResultCombo->addItem(item.value(QStringLiteral("display_name")).toString(), item.value(QStringLiteral("index")).toInt(-1));
    }
    m_atResultCombo->setCurrentIndex(m_atResultCombo->count() - 1);
    updateAtInfoLabel();
    m_startBtn->setEnabled(true);
}

void SimplePointCloudDialog::updateAtInfoLabel()
{
    if (m_atResults.isEmpty() || !m_atResultCombo || m_atResultCombo->currentIndex() < 0)
    {
        m_bestAtIndex = -1;
        m_infoLabel->setText(tr("未选择空三结果。"));
        m_startBtn->setEnabled(false);
        return;
    }

    const QJsonObject current = m_atResults.at(m_atResultCombo->currentIndex()).toObject();
    m_bestAtIndex = current.value(QStringLiteral("index")).toInt(m_atResultCombo->currentData().toInt());

    const QString img0 = QFileInfo(current.value(QStringLiteral("image0")).toString()).fileName();
    const QString img1 = QFileInfo(current.value(QStringLiteral("image1")).toString()).fileName();
    const QString date = current.value(QStringLiteral("created_at")).toString().left(10);
    const int sparseCount = current.value(QStringLiteral("sparse_point_count")).toInt(0);
    const int imageCount  = current.value(QStringLiteral("image_count")).toInt(0);

    QString info = tr("✓ 已找到空三结果（%1 组）\n").arg(m_atResults.size());
    if (imageCount > 0)
        info += tr("影像数量：%1\n").arg(imageCount);
    if (sparseCount > 0)
        info += tr("稀疏点数：%1\n").arg(sparseCount);
    if (!img0.isEmpty() && !img1.isEmpty())
        info += tr("影像对：%1 ↔ %2\n").arg(img0, img1);
    if (!date.isEmpty())
        info += tr("生成时间：%1").arg(date);

    m_infoLabel->setText(info);
    m_startBtn->setEnabled(true);
}

// ─────────────────────────────────────────────────────────────────────────────
void SimplePointCloudDialog::onStartClicked()
{
    emit runRequested(buildSettings());
    accept();
}

// ─────────────────────────────────────────────────────────────────────────────
QJsonObject SimplePointCloudDialog::buildSettings() const
{
    const QString preset = m_qualityCombo
                           ? m_qualityCombo->currentData().toString()
                           : QStringLiteral("standard");

    QJsonObject s;
    s[QStringLiteral("at_index")]       = m_bestAtIndex;
    s[QStringLiteral("preset")]         = preset;
    s[QStringLiteral("output_colors")]  = m_colorsCheck ? m_colorsCheck->isChecked() : true;
    s[QStringLiteral("output_normals")] = true;

    // 根据质量档位设置 SGBM 参数
    if (preset == QStringLiteral("fast")) {
        s[QStringLiteral("num_disparities")]     = 64;
        s[QStringLiteral("block_size")]          = 11;
        s[QStringLiteral("uniqueness_ratio")]    = 5;
        s[QStringLiteral("speckle_window_size")] = 50;
        s[QStringLiteral("use_full_dp")]         = false;
        s[QStringLiteral("use_wls_filter")]      = false;
        s[QStringLiteral("processing_scale")]    = 0.5;
    } else if (preset == QStringLiteral("quality")) {
        s[QStringLiteral("num_disparities")]     = 256;
        s[QStringLiteral("block_size")]          = 7;
        s[QStringLiteral("uniqueness_ratio")]    = 15;
        s[QStringLiteral("speckle_window_size")] = 150;
        s[QStringLiteral("use_full_dp")]         = true;
        s[QStringLiteral("use_wls_filter")]      = true;
        s[QStringLiteral("processing_scale")]    = 1.0;
    } else { // standard
        s[QStringLiteral("num_disparities")]     = 128;
        s[QStringLiteral("block_size")]          = 9;
        s[QStringLiteral("uniqueness_ratio")]    = 10;
        s[QStringLiteral("speckle_window_size")] = 100;
        s[QStringLiteral("use_full_dp")]         = true;
        s[QStringLiteral("use_wls_filter")]      = true;
        s[QStringLiteral("processing_scale")]    = 1.0;
    }

    s[QStringLiteral("min_depth")]           = 0.001;
    s[QStringLiteral("max_depth")]           = 1e6;
    s[QStringLiteral("build_mesh")]          = m_meshCheck ? m_meshCheck->isChecked() : false;

    if (!m_outputDir.isEmpty())
        s[QStringLiteral("output_dir")] = m_outputDir;

    return s;
}
