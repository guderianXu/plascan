#include "SparseCloudPostProcessDialog.h"
#include "ui_SparseCloudPostProcessDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

// ---------------------------------------------------------------------------
// 构造
// ---------------------------------------------------------------------------

SparseCloudPostProcessDialog::SparseCloudPostProcessDialog(QWidget *parent)
    : QDialog(parent)
{
    setupUi();
    setWindowTitle(tr("稀疏点云后处理"));
}

// ---------------------------------------------------------------------------
// UI 构建
// ---------------------------------------------------------------------------

void SparseCloudPostProcessDialog::setupUi()
{
    setMinimumWidth(500);

    {
        Ui::SparseCloudPostProcessDialog ui;
        ui.setupUi(this);

        m_sourceCombo = ui.m_sourceCombo;
        m_statsLabel = ui.m_statsLabel;
        m_refineGroup = ui.m_refineGroup;
        m_spatialGroup = ui.m_spatialGroup;

        auto *qualityLayout = ui.pointQualityLayout;
        auto *refLayout = ui.refineForm;
        auto *spatLayout = ui.spatialForm;

        {
            auto *row = new QHBoxLayout;
            m_reprojCheck = new QCheckBox(tr("重投影误差 ≤"), this);
            m_reprojCheck->setChecked(true);
            m_reprojSpin = new QDoubleSpinBox(this);
            m_reprojSpin->setRange(0.1, 50.0);
            m_reprojSpin->setDecimals(1);
            m_reprojSpin->setSingleStep(0.5);
            m_reprojSpin->setValue(4.0);
            const QString reprojTip = tr("移除 BA 计算的逐点重投影误差超过此阈值的三维点。\n"
                                         "重投影误差越大说明该点定位越不精确。\n"
                                         "BA 质量较好时建议 2–3 px，初始三角化结果建议 4–6 px。");
            m_reprojCheck->setToolTip(reprojTip);
            m_reprojSpin->setToolTip(reprojTip);
            m_reprojSpin->setSuffix(tr(" px"));
            row->addWidget(m_reprojCheck);
            row->addWidget(m_reprojSpin);
            row->addStretch();
            qualityLayout->addLayout(row);
            connect(m_reprojCheck, &QCheckBox::toggled, m_reprojSpin, &QWidget::setEnabled);
        }

        {
            auto *row = new QHBoxLayout;
            m_trackCheck = new QCheckBox(tr("轨迹长度 ≥"), this);
            m_trackCheck->setChecked(true);
            m_trackSpin = new QSpinBox(this);
            m_trackSpin->setRange(2, 50);
            m_trackSpin->setValue(2);
            m_trackSpin->setSuffix(tr(" 次观测"));
            row->addWidget(m_trackCheck);
            row->addWidget(m_trackSpin);
            row->addStretch();
            qualityLayout->addLayout(row);
            const QString trackTip = tr("移除被少于此数量影像观测到的三维点。\n"
                                        "若所有点轨迹长度相同，此过滤器不会生效（安全保护）。\n"
                                        "建议：2（不过滤）、3（平衡）或 4（严格）。");
            m_trackCheck->setToolTip(trackTip);
            m_trackSpin->setToolTip(trackTip);
            connect(m_trackCheck, &QCheckBox::toggled, m_trackSpin, &QWidget::setEnabled);
        }

        {
            auto *row = new QHBoxLayout;
            m_angleCheck = new QCheckBox(tr("三角化角度 ≥"), this);
            m_angleCheck->setChecked(true);
            m_angleSpin = new QDoubleSpinBox(this);
            m_angleSpin->setRange(0.1, 30.0);
            m_angleSpin->setDecimals(1);
            m_angleSpin->setSingleStep(0.5);
            m_angleSpin->setValue(2.0);
            m_angleSpin->setSuffix(tr(" °"));
            row->addWidget(m_angleCheck);
            row->addWidget(m_angleSpin);
            row->addStretch();
            qualityLayout->addLayout(row);
            const QString angleTip = tr("移除所有观测射线最大交角小于此值的三维点。\n"
                                        "角度越大说明三角化几何条件越好。建议 2°（宽Song）至 5°（严格）。");
            m_angleCheck->setToolTip(angleTip);
            m_angleSpin->setToolTip(angleTip);
            connect(m_angleCheck, &QCheckBox::toggled, m_angleSpin, &QWidget::setEnabled);
        }

        {
            auto *row = new QHBoxLayout;
            m_statCheck = new QCheckBox(tr("统计离群"), this);
            m_statCheck->setChecked(true);
            m_statKSpin = new QSpinBox(this);
            m_statKSpin->setRange(2, 200);
            m_statKSpin->setValue(20);
            m_statStdSpin = new QDoubleSpinBox(this);
            m_statStdSpin->setRange(0.5, 10.0);
            m_statStdSpin->setDecimals(1);
            m_statStdSpin->setSingleStep(0.5);
            m_statStdSpin->setValue(2.0);
            row->addWidget(m_statCheck);
            row->addWidget(new QLabel(tr("K:"), this));
            row->addWidget(m_statKSpin);
            row->addWidget(new QLabel(tr("σ:"), this));
            row->addWidget(m_statStdSpin);
            row->addStretch();
            qualityLayout->addLayout(row);
            const QString statTip = tr("基于 KNN 邻域统计检测空间离群点。\n"
                                       "K：近邻数量，越大越稳定但越慢（建议 16–30）。\n"
                                       "σ：若某点到邻域中心的距离超过 均值 + σ×标准差则剪除（建议 2.0–2.5）。");
            m_statCheck->setToolTip(statTip);
            m_statKSpin->setToolTip(statTip);
            m_statStdSpin->setToolTip(statTip);
            connect(m_statCheck, &QCheckBox::toggled, m_statKSpin, &QWidget::setEnabled);
            connect(m_statCheck, &QCheckBox::toggled, m_statStdSpin, &QWidget::setEnabled);
        }

        {
            auto *row = new QHBoxLayout;
            m_densityCheck = new QCheckBox(tr("半径密度（慢）"), this);
            m_densityCheck->setChecked(false);
            m_densityRadiusSpin = new QDoubleSpinBox(this);
            m_densityRadiusSpin->setRange(0.01, 1000.0);
            m_densityRadiusSpin->setDecimals(3);
            m_densityRadiusSpin->setSingleStep(0.1);
            m_densityRadiusSpin->setValue(0.5);
            m_densityRadiusSpin->setEnabled(false);
            m_densityMinNbSpin = new QSpinBox(this);
            m_densityMinNbSpin->setRange(1, 50);
            m_densityMinNbSpin->setValue(5);
            m_densityMinNbSpin->setEnabled(false);
            row->addWidget(m_densityCheck);
            row->addWidget(new QLabel(tr("半径:"), this));
            row->addWidget(m_densityRadiusSpin);
            row->addWidget(new QLabel(tr("最少邻居:"), this));
            row->addWidget(m_densityMinNbSpin);
            row->addStretch();
            qualityLayout->addLayout(row);
            const QString densTip = tr("移除指定半径球内邻居数不足的孤立点。\n"
                                       "适合清理真正孤立的噪声点，但计算较慢。\n"
                                       "半径应与点云空间尺度匹配，最少邻居建议 3–6。");
            m_densityCheck->setToolTip(densTip);
            m_densityRadiusSpin->setToolTip(densTip);
            m_densityMinNbSpin->setToolTip(densTip);
            connect(m_densityCheck, &QCheckBox::toggled, m_densityRadiusSpin, &QWidget::setEnabled);
            connect(m_densityCheck, &QCheckBox::toggled, m_densityMinNbSpin, &QWidget::setEnabled);
        }

        m_iterRoundsSpin = new QSpinBox(m_refineGroup);
        m_iterRoundsSpin->setRange(1, 20);
        m_iterRoundsSpin->setValue(3);
        m_iterRoundsSpin->setToolTip(tr("每轮迭代内执行一次完整过滤，阈值逐轮收紧。建议 3–5 轮。"));
        refLayout->addRow(tr("精修轮数:"), m_iterRoundsSpin);

        m_retriangCheck = new QCheckBox(tr("独立轮次（每轮从原始点集重新过滤）"), m_refineGroup);
        m_retriangCheck->setChecked(false);
        m_retriangCheck->setToolTip(tr("不勾选（推荐）：每轮在上一轮保留的点上继续过滤，统计\n"
                                       "滤波随着离群点减少而逐渐收敛，效果更好。\n"
                                       "勾选：每轮均从原始完整点集重新过滤，各轮独立。\n"
                                       "注意：此选项不会重新执行三角化或光束法平差。"));
        refLayout->addRow(m_retriangCheck);

        m_normalConsCheck = new QCheckBox(tr("启用法向一致性检查"), m_refineGroup);
        m_normalConsCheck->setChecked(false);
        m_normalConsCheck->setToolTip(tr("额外检查三维点周围法向量的一致性，移除方向异常的点。\n"
                                         "计算较慢，适合密度较高的点云。"));
        refLayout->addRow(m_normalConsCheck);

        m_threadsSpin = new QSpinBox(m_refineGroup);
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("并行计算的线程数。建议与 CPU 核心数相近。"));
        refLayout->addRow(tr("线程数:"), m_threadsSpin);

        m_voxelSizeSpin = new QDoubleSpinBox(m_spatialGroup);
        m_voxelSizeSpin->setRange(0.0, 1000.0);
        m_voxelSizeSpin->setDecimals(3);
        m_voxelSizeSpin->setSingleStep(0.1);
        m_voxelSizeSpin->setValue(0.0);
        m_voxelSizeSpin->setSpecialValueText(tr("自动"));
        m_voxelSizeSpin->setToolTip(tr("将空间划分为该边长的体素，移除只含极少点的孤立体素。设 0（自动）时系统自动估算合理的体素尺度。"));
        spatLayout->addRow(tr("体素边长:"), m_voxelSizeSpin);

        m_minVoxelPtsSpin = new QSpinBox(m_spatialGroup);
        m_minVoxelPtsSpin->setRange(1, 20);
        m_minVoxelPtsSpin->setValue(2);
        m_minVoxelPtsSpin->setToolTip(tr("体素内点数低于此值时，该体素内所有点被将为孤立噪声并剪除。建议 2–3。"));
        spatLayout->addRow(tr("体素最少点数:"), m_minVoxelPtsSpin);

        m_localReprojCheck = new QCheckBox(tr("局部重投影过滤"), m_spatialGroup);
        m_localReprojCheck->setChecked(true);
        m_localReprojCheck->setToolTip(tr("在每个局部邻域内计算重投影误差统计，剪除超出阈值的异常点。"));
        spatLayout->addRow(m_localReprojCheck);

        m_reprojStdMulSpin = new QDoubleSpinBox(m_spatialGroup);
        m_reprojStdMulSpin->setRange(0.5, 10.0);
        m_reprojStdMulSpin->setDecimals(1);
        m_reprojStdMulSpin->setSingleStep(0.5);
        m_reprojStdMulSpin->setValue(2.5);
        m_reprojStdMulSpin->setToolTip(tr("局部重投影过滤的标准差倍数阈值。值越小过滤越激进（建议 2.0–3.0）。"));
        spatLayout->addRow(tr("重投影 σ 倍数:"), m_reprojStdMulSpin);

        m_dedupRadiusSpin = new QDoubleSpinBox(m_spatialGroup);
        m_dedupRadiusSpin->setRange(-1.0, 100.0);
        m_dedupRadiusSpin->setDecimals(3);
        m_dedupRadiusSpin->setSingleStep(0.01);
        m_dedupRadiusSpin->setValue(-1.0);
        m_dedupRadiusSpin->setSpecialValueText(tr("禁用"));
        m_dedupRadiusSpin->setToolTip(tr("移除距离小于此半径的重复点（仅保留其中一个）。设为“禁用”时不执行去重操作。"));
        spatLayout->addRow(tr("去重半径:"), m_dedupRadiusSpin);

        connect(m_localReprojCheck, &QCheckBox::toggled, m_reprojStdMulSpin, &QWidget::setEnabled);

        m_runButton = ui.buttonBox->button(QDialogButtonBox::Ok);
        m_runButton->setText(tr("运行后处理"));

        auto changed = [this]() { onAnyChanged(); };

        connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &SparseCloudPostProcessDialog::updateStatsLabel);
        connect(m_sourceCombo,       QOverload<int>::of(&QComboBox::currentIndexChanged),    this, changed);
        connect(m_reprojCheck,       &QCheckBox::toggled,                                    this, changed);
        connect(m_reprojSpin,        QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_trackCheck,        &QCheckBox::toggled,                                    this, changed);
        connect(m_trackSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_angleCheck,        &QCheckBox::toggled,                                    this, changed);
        connect(m_angleSpin,         QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_statCheck,         &QCheckBox::toggled,                                    this, changed);
        connect(m_statKSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_statStdSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_densityCheck,      &QCheckBox::toggled,                                    this, changed);
        connect(m_densityRadiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_densityMinNbSpin,  QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_refineGroup,       &QGroupBox::toggled,                                    this, changed);
        connect(m_iterRoundsSpin,    QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_retriangCheck,     &QCheckBox::toggled,                                    this, changed);
        connect(m_normalConsCheck,   &QCheckBox::toggled,                                    this, changed);
        connect(m_threadsSpin,       QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_spatialGroup,      &QGroupBox::toggled,                                    this, changed);
        connect(m_voxelSizeSpin,     QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_minVoxelPtsSpin,   QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
        connect(m_localReprojCheck,  &QCheckBox::toggled,                                    this, changed);
        connect(m_reprojStdMulSpin,  QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
        connect(m_dedupRadiusSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);

        connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &SparseCloudPostProcessDialog::onRun);
        connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

        return;
    }

    auto *root = new QVBoxLayout(this);
    root->setSpacing(8);

    // ── 说明 ──────────────────────────────────────────────────────────────────
    {
        auto *info = new QLabel(
            tr("过滤阈值基于光束法平差（BA）为每个三维点计算的逐点重投影误差、三角化角度和轨迹长度。"
               "悬停在各参数上可查看详细说明。"),
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

    // ── 输入来源 ─────────────────────────────────────────────────────────────
    {
        auto *form = new QFormLayout;
        m_sourceCombo = new QComboBox(this);
        m_sourceCombo->setToolTip(tr("选择要处理的稀疏点云。建议选经过光束法平差的最新结果。"));
        form->addRow(tr("输入稀疏点云:"), m_sourceCombo);

        m_statsLabel = new QLabel(this);
        m_statsLabel->setWordWrap(true);
        {
            QFont fs = m_statsLabel->font();
            fs.setPointSizeF(fs.pointSizeF() * 0.85);
            m_statsLabel->setFont(fs);
            auto spal = m_statsLabel->palette();
            spal.setColor(QPalette::WindowText, spal.color(QPalette::Mid));
            m_statsLabel->setPalette(spal);
        }
        form->addRow(QString(), m_statsLabel);
        root->addLayout(form);
    }

    // ── 1. 点级质量过滤 ───────────────────────────────────────────────────────
    {
        auto *box = new QGroupBox(tr("1. 点级质量过滤（必选，始终执行）"), this);
        auto *layout = new QVBoxLayout(box);
        layout->setSpacing(4);

        // 重投影误差
        {
            auto *row = new QHBoxLayout;
            m_reprojCheck = new QCheckBox(tr("重投影误差 ≤"), box);
            m_reprojCheck->setChecked(true);
            m_reprojSpin = new QDoubleSpinBox(box);
            m_reprojSpin->setRange(0.1, 50.0);
            m_reprojSpin->setDecimals(1);
            m_reprojSpin->setSingleStep(0.5);
            m_reprojSpin->setValue(4.0);
            const QString reprojTip = tr("移除 BA 计算的逐点重投影误差超过此阈值的三维点。\n"
                                         "重投影误差越大说明该点定位越不精确。\n"
                                         "BA 质量较好时建议 2–3 px，初始三角化结果建议 4–6 px。");
            m_reprojCheck->setToolTip(reprojTip);
            m_reprojSpin->setToolTip(reprojTip);
            m_reprojSpin->setSuffix(tr(" px"));
            row->addWidget(m_reprojCheck);
            row->addWidget(m_reprojSpin);
            row->addStretch();
            layout->addLayout(row);
            connect(m_reprojCheck, &QCheckBox::toggled, m_reprojSpin, &QWidget::setEnabled);
        }

        // 轨迹长度
        {
            auto *row = new QHBoxLayout;
            m_trackCheck = new QCheckBox(tr("轨迹长度 ≥"), box);
            m_trackCheck->setChecked(true);
            m_trackSpin = new QSpinBox(box);
            m_trackSpin->setRange(2, 50);
            m_trackSpin->setValue(2);
            m_trackSpin->setSuffix(tr(" 次观测"));
            row->addWidget(m_trackCheck);
            row->addWidget(m_trackSpin);
            row->addStretch();
            layout->addLayout(row);
            const QString trackTip = tr("移除被少于此数量影像观测到的三维点。\n"
                                        "若所有点轨迹长度相同，此过滤器不会生效（安全保护）。\n"
                                        "建议：2（不过滤）、3（平衡）或 4（严格）。");
            m_trackCheck->setToolTip(trackTip);
            m_trackSpin->setToolTip(trackTip);
            connect(m_trackCheck, &QCheckBox::toggled, m_trackSpin, &QWidget::setEnabled);
        }

        // 三角化角度
        {
            auto *row = new QHBoxLayout;
            m_angleCheck = new QCheckBox(tr("三角化角度 ≥"), box);
            m_angleCheck->setChecked(true);
            m_angleSpin = new QDoubleSpinBox(box);
            m_angleSpin->setRange(0.1, 30.0);
            m_angleSpin->setDecimals(1);
            m_angleSpin->setSingleStep(0.5);
            m_angleSpin->setValue(2.0);
            m_angleSpin->setSuffix(tr(" °"));
            row->addWidget(m_angleCheck);
            row->addWidget(m_angleSpin);
            row->addStretch();
            layout->addLayout(row);
            const QString angleTip = tr("移除所有观测射线最大交角小于此值的三维点。\n"
                                        "角度越大说明三角化几何条件越好。建议 2°（宽Song）至 5°（严格）。");
            m_angleCheck->setToolTip(angleTip);
            m_angleSpin->setToolTip(angleTip);
            connect(m_angleCheck, &QCheckBox::toggled, m_angleSpin, &QWidget::setEnabled);
        }

        // 统计离群
        {
            auto *row = new QHBoxLayout;
            m_statCheck = new QCheckBox(tr("统计离群"), box);
            m_statCheck->setChecked(true);
            m_statKSpin = new QSpinBox(box);
            m_statKSpin->setRange(2, 200);
            m_statKSpin->setValue(20);
            m_statStdSpin = new QDoubleSpinBox(box);
            m_statStdSpin->setRange(0.5, 10.0);
            m_statStdSpin->setDecimals(1);
            m_statStdSpin->setSingleStep(0.5);
            m_statStdSpin->setValue(2.0);
            row->addWidget(m_statCheck);
            row->addWidget(new QLabel(tr("K:"), box));
            row->addWidget(m_statKSpin);
            row->addWidget(new QLabel(tr("σ:"), box));
            row->addWidget(m_statStdSpin);
            row->addStretch();
            layout->addLayout(row);
            const QString statTip = tr("基于 KNN 邻域统计检测空间离群点。\n"
                                       "K：近邻数量，越大越稳定但越慢（建议 16–30）。\n"
                                       "σ：若某点到邻域中心的距离超过 均值 + σ×标准差则剪除（建议 2.0–2.5）。");
            m_statCheck->setToolTip(statTip);
            m_statKSpin->setToolTip(statTip);
            m_statStdSpin->setToolTip(statTip);
            connect(m_statCheck, &QCheckBox::toggled, m_statKSpin, &QWidget::setEnabled);
            connect(m_statCheck, &QCheckBox::toggled, m_statStdSpin, &QWidget::setEnabled);
        }

        // 半径密度（可选，默认关闭）
        {
            auto *row = new QHBoxLayout;
            m_densityCheck = new QCheckBox(tr("半径密度（慢）"), box);
            m_densityCheck->setChecked(false);
            m_densityRadiusSpin = new QDoubleSpinBox(box);
            m_densityRadiusSpin->setRange(0.01, 1000.0);
            m_densityRadiusSpin->setDecimals(3);
            m_densityRadiusSpin->setSingleStep(0.1);
            m_densityRadiusSpin->setValue(0.5);
            m_densityRadiusSpin->setEnabled(false);
            m_densityMinNbSpin = new QSpinBox(box);
            m_densityMinNbSpin->setRange(1, 50);
            m_densityMinNbSpin->setValue(5);
            m_densityMinNbSpin->setEnabled(false);
            row->addWidget(m_densityCheck);
            row->addWidget(new QLabel(tr("半径:"), box));
            row->addWidget(m_densityRadiusSpin);
            row->addWidget(new QLabel(tr("最少邻居:"), box));
            row->addWidget(m_densityMinNbSpin);
            row->addStretch();
            layout->addLayout(row);
            const QString densTip = tr("移除指定半径球内邻居数不足的孤立点。\n"
                                       "适合清理真正孤立的噪声点，但计算较慢。\n"
                                       "半径应与点云空间尺度匹配，最少邻居建议 3–6。");
            m_densityCheck->setToolTip(densTip);
            m_densityRadiusSpin->setToolTip(densTip);
            m_densityMinNbSpin->setToolTip(densTip);
            connect(m_densityCheck, &QCheckBox::toggled, m_densityRadiusSpin, &QWidget::setEnabled);
            connect(m_densityCheck, &QCheckBox::toggled, m_densityMinNbSpin, &QWidget::setEnabled);
        }

        root->addWidget(box);
    }

    // ── 迭代精修（可勾选，默认关闭）────────────────────────────────────────
    {
        m_refineGroup = new QGroupBox(tr("2. 迭代精修（勾选后启用）"), this);
        m_refineGroup->setCheckable(true);
        m_refineGroup->setChecked(false);
        m_refineGroup->setToolTip(tr("多轮迭代：每轮先按第一步阈值过滤，再逐步收紧阈值。\n"
                                     "适合在较粗糙的初始点云上做渐进式精化。\n"
                                     "若同时启用了第三步“空间去噪”，精修完成后会继续执行空间去噪。"));
        auto *refLayout = new QFormLayout(m_refineGroup);

        m_iterRoundsSpin = new QSpinBox(m_refineGroup);
        m_iterRoundsSpin->setRange(1, 20);
        m_iterRoundsSpin->setValue(3);
        m_iterRoundsSpin->setToolTip(tr("每轮迭代内执行一次完整过滤，阈值逐轮收紧。建议 3–5 轮。"));
        refLayout->addRow(tr("精修轮数:"), m_iterRoundsSpin);

        m_retriangCheck = new QCheckBox(tr("独立轮次（每轮从原始点集重新过滤）"), m_refineGroup);
        m_retriangCheck->setChecked(false);
        m_retriangCheck->setToolTip(tr("不勾选（推荐）：每轮在上一轮保留的点上继续过滤，统计\n"
                                       "滤波随着离群点减少而逐渐收敛，效果更好。\n"
                                       "勾选：每轮均从原始完整点集重新过滤，各轮独立。\n"
                                       "注意：此选项不会重新执行三角化或光束法平差。"));
        refLayout->addRow(m_retriangCheck);

        m_normalConsCheck = new QCheckBox(tr("启用法向一致性检查"), m_refineGroup);
        m_normalConsCheck->setChecked(false);
        m_normalConsCheck->setToolTip(tr("额外检查三维点周围法向量的一致性，移除方向异常的点。\n"
                                         "计算较慢，适合密度较高的点云。"));
        refLayout->addRow(m_normalConsCheck);

        m_threadsSpin = new QSpinBox(m_refineGroup);
        m_threadsSpin->setRange(1, 128);
        m_threadsSpin->setValue(8);
        m_threadsSpin->setToolTip(tr("并行计算的线程数。建议与 CPU 核心数相近。"));
        refLayout->addRow(tr("线程数:"), m_threadsSpin);

        root->addWidget(m_refineGroup);
    }

    // ── 空间清理（可勾选，默认关闭）────────────────────────────────────────
    {
        m_spatialGroup = new QGroupBox(tr("3. 空间去噪（勾选后启用）"), this);
        m_spatialGroup->setCheckable(true);
        m_spatialGroup->setChecked(false);
        m_spatialGroup->setToolTip(tr("基于体素网格移除孤立点簇、局部重投影异常点和近邻重复点。\n"
                                      "基于几何结构做清理，不依赖 BA 指标。\n"
                                      "注意：单独启用本步骤时，第一步点级过滤不会执行。"));
        auto *spatLayout = new QFormLayout(m_spatialGroup);

        m_voxelSizeSpin = new QDoubleSpinBox(m_spatialGroup);
        m_voxelSizeSpin->setRange(0.0, 1000.0);
        m_voxelSizeSpin->setDecimals(3);
        m_voxelSizeSpin->setSingleStep(0.1);
        m_voxelSizeSpin->setValue(0.0);
        m_voxelSizeSpin->setSpecialValueText(tr("自动"));
        m_voxelSizeSpin->setToolTip(tr("将空间划分为该边长的体素，移除只含极少点的孤立体素。设 0（自动）时系统自动估算合理的体素尺度。"));
        spatLayout->addRow(tr("体素边长:"), m_voxelSizeSpin);

        m_minVoxelPtsSpin = new QSpinBox(m_spatialGroup);
        m_minVoxelPtsSpin->setRange(1, 20);
        m_minVoxelPtsSpin->setValue(2);
        m_minVoxelPtsSpin->setToolTip(tr("体素内点数低于此值时，该体素内所有点被将为孤立噪声并剪除。建议 2–3。"));
        spatLayout->addRow(tr("体素最少点数:"), m_minVoxelPtsSpin);

        m_localReprojCheck = new QCheckBox(tr("局部重投影过滤"), m_spatialGroup);
        m_localReprojCheck->setChecked(true);
        m_localReprojCheck->setToolTip(tr("在每个局部邻域内计算重投影误差统计，剪除超出阈值的异常点。"));
        spatLayout->addRow(m_localReprojCheck);

        m_reprojStdMulSpin = new QDoubleSpinBox(m_spatialGroup);
        m_reprojStdMulSpin->setRange(0.5, 10.0);
        m_reprojStdMulSpin->setDecimals(1);
        m_reprojStdMulSpin->setSingleStep(0.5);
        m_reprojStdMulSpin->setValue(2.5);
        m_reprojStdMulSpin->setToolTip(tr("局部重投影过滤的标准差倍数阈值。值越小过滤越激进（建议 2.0–3.0）。"));
        spatLayout->addRow(tr("重投影 σ 倍数:"), m_reprojStdMulSpin);

        m_dedupRadiusSpin = new QDoubleSpinBox(m_spatialGroup);
        m_dedupRadiusSpin->setRange(-1.0, 100.0);
        m_dedupRadiusSpin->setDecimals(3);
        m_dedupRadiusSpin->setSingleStep(0.01);
        m_dedupRadiusSpin->setValue(-1.0);
        m_dedupRadiusSpin->setSpecialValueText(tr("禁用"));
        m_dedupRadiusSpin->setToolTip(tr("移除距离小于此半径的重复点（仅保留其中一个）。设为“禁用”时不执行去重操作。"));
        spatLayout->addRow(tr("去重半径:"), m_dedupRadiusSpin);

        connect(m_localReprojCheck, &QCheckBox::toggled, m_reprojStdMulSpin, &QWidget::setEnabled);

        root->addWidget(m_spatialGroup);
    }

    // ── 按钮 ─────────────────────────────────────────────────────────────────
    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    m_runButton = buttonBox->button(QDialogButtonBox::Ok);
    m_runButton->setText(tr("运行后处理"));
    root->addWidget(buttonBox);

    // ── 信号连接 ─────────────────────────────────────────────────────────────
    auto changed = [this]() { onAnyChanged(); };

    connect(m_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SparseCloudPostProcessDialog::updateStatsLabel);
    connect(m_sourceCombo,       QOverload<int>::of(&QComboBox::currentIndexChanged),    this, changed);
    connect(m_reprojCheck,       &QCheckBox::toggled,                                    this, changed);
    connect(m_reprojSpin,        QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_trackCheck,        &QCheckBox::toggled,                                    this, changed);
    connect(m_trackSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_angleCheck,        &QCheckBox::toggled,                                    this, changed);
    connect(m_angleSpin,         QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_statCheck,         &QCheckBox::toggled,                                    this, changed);
    connect(m_statKSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_statStdSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_densityCheck,      &QCheckBox::toggled,                                    this, changed);
    connect(m_densityRadiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_densityMinNbSpin,  QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_refineGroup,       &QGroupBox::toggled,                                    this, changed);
    connect(m_iterRoundsSpin,    QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_retriangCheck,     &QCheckBox::toggled,                                    this, changed);
    connect(m_normalConsCheck,   &QCheckBox::toggled,                                    this, changed);
    connect(m_threadsSpin,       QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_spatialGroup,      &QGroupBox::toggled,                                    this, changed);
    connect(m_voxelSizeSpin,     QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_minVoxelPtsSpin,   QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(m_localReprojCheck,  &QCheckBox::toggled,                                    this, changed);
    connect(m_reprojStdMulSpin,  QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(m_dedupRadiusSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &SparseCloudPostProcessDialog::onRun);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

// ---------------------------------------------------------------------------
// 公有接口
// ---------------------------------------------------------------------------

void SparseCloudPostProcessDialog::setAvailableSparseClouds(const QJsonArray &results)
{
    if (!m_sourceCombo)
    {
        return;
    }

    m_availableResults = results;
    m_sourceCombo->clear();
    for (const QJsonValue &value : results)
    {
        const QJsonObject item = value.toObject();
        const QString label = item.value(QStringLiteral("display_name")).toString(
            QFileInfo(item.value(QStringLiteral("sparse_cloud_xyz")).toString()).fileName());
        m_sourceCombo->addItem(label, item.value(QStringLiteral("index")).toInt(-1));
    }

    if (m_runButton)
    {
        m_runButton->setEnabled(m_sourceCombo->count() > 0);
    }
    applyPendingSourceSelection();
    if (m_sourceCombo->count() > 0 && m_sourceCombo->currentIndex() < 0)
    {
        m_sourceCombo->setCurrentIndex(m_sourceCombo->count() - 1);
    }
    updateStatsLabel();
}

void SparseCloudPostProcessDialog::applySettings(const QJsonObject &settings)
{
    m_pendingSourceIdx = settings.value(QStringLiteral("sourceAtIndex")).toInt(-1);
    applyPendingSourceSelection();

    if (settings.contains(QStringLiteral("filterByReprojError")))
        m_reprojCheck->setChecked(settings.value(QStringLiteral("filterByReprojError")).toBool());
    if (settings.contains(QStringLiteral("maxReprojError")))
        m_reprojSpin->setValue(settings.value(QStringLiteral("maxReprojError")).toDouble());

    if (settings.contains(QStringLiteral("filterByTrackLen")))
        m_trackCheck->setChecked(settings.value(QStringLiteral("filterByTrackLen")).toBool());
    if (settings.contains(QStringLiteral("minTrackLen")))
        m_trackSpin->setValue(settings.value(QStringLiteral("minTrackLen")).toInt());

    if (settings.contains(QStringLiteral("filterByTriAngle")))
        m_angleCheck->setChecked(settings.value(QStringLiteral("filterByTriAngle")).toBool());
    if (settings.contains(QStringLiteral("minTriAngleDeg")))
        m_angleSpin->setValue(settings.value(QStringLiteral("minTriAngleDeg")).toDouble());

    if (settings.contains(QStringLiteral("filterByStatistical")))
        m_statCheck->setChecked(settings.value(QStringLiteral("filterByStatistical")).toBool());
    if (settings.contains(QStringLiteral("statK")))
        m_statKSpin->setValue(settings.value(QStringLiteral("statK")).toInt());
    if (settings.contains(QStringLiteral("statStdDevMul")))
        m_statStdSpin->setValue(settings.value(QStringLiteral("statStdDevMul")).toDouble());

    if (settings.contains(QStringLiteral("filterByDensity")))
        m_densityCheck->setChecked(settings.value(QStringLiteral("filterByDensity")).toBool());
    if (settings.contains(QStringLiteral("densityRadius")))
        m_densityRadiusSpin->setValue(settings.value(QStringLiteral("densityRadius")).toDouble());
    if (settings.contains(QStringLiteral("densityMinNeighbors")))
        m_densityMinNbSpin->setValue(settings.value(QStringLiteral("densityMinNeighbors")).toInt());

    if (settings.contains(QStringLiteral("enableRefine")))
        m_refineGroup->setChecked(settings.value(QStringLiteral("enableRefine")).toBool());
    if (settings.contains(QStringLiteral("iterRounds")))
        m_iterRoundsSpin->setValue(settings.value(QStringLiteral("iterRounds")).toInt());
    if (settings.contains(QStringLiteral("retriangulate")))
        m_retriangCheck->setChecked(settings.value(QStringLiteral("retriangulate")).toBool());
    if (settings.contains(QStringLiteral("normalConsistency")))
        m_normalConsCheck->setChecked(settings.value(QStringLiteral("normalConsistency")).toBool());
    if (settings.contains(QStringLiteral("threads")))
        m_threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt());

    if (settings.contains(QStringLiteral("enableSpatialCleanup")))
        m_spatialGroup->setChecked(settings.value(QStringLiteral("enableSpatialCleanup")).toBool());
    if (settings.contains(QStringLiteral("voxelSize")))
        m_voxelSizeSpin->setValue(settings.value(QStringLiteral("voxelSize")).toDouble());
    if (settings.contains(QStringLiteral("minVoxelPoints")))
        m_minVoxelPtsSpin->setValue(settings.value(QStringLiteral("minVoxelPoints")).toInt());
    if (settings.contains(QStringLiteral("localReprojFilter")))
        m_localReprojCheck->setChecked(settings.value(QStringLiteral("localReprojFilter")).toBool());
    if (settings.contains(QStringLiteral("localReprojStdMul")))
        m_reprojStdMulSpin->setValue(settings.value(QStringLiteral("localReprojStdMul")).toDouble());
    if (settings.contains(QStringLiteral("deduplicationRadius")))
        m_dedupRadiusSpin->setValue(settings.value(QStringLiteral("deduplicationRadius")).toDouble());
}

// ---------------------------------------------------------------------------
// 私有
// ---------------------------------------------------------------------------

void SparseCloudPostProcessDialog::updateStatsLabel()
{
    if (!m_statsLabel)
        return;
    const int comboIdx = m_sourceCombo ? m_sourceCombo->currentIndex() : -1;
    if (comboIdx < 0 || comboIdx >= m_availableResults.size())
    {
        m_statsLabel->clear();
        return;
    }
    const QJsonObject item = m_availableResults.at(comboIdx).toObject();
    const int pts = item.value(QStringLiteral("sparse_point_count")).toInt(0);
    const QString opName = item.value(QStringLiteral("operation_display_name")).toString();
    const QJsonObject summary = item.value(QStringLiteral("operation_summary")).toObject();

    QStringList parts;
    if (pts > 0)
        parts << tr("%1 个三维点").arg(pts);
    if (!opName.isEmpty())
        parts << opName;
    if (!summary.isEmpty())
    {
        const int inp = summary.value(QStringLiteral("input_points")).toInt(0);
        const int rem = summary.value(QStringLiteral("removed_total")).toInt(0);
        if (inp > 0 && rem > 0)
            parts << tr("已从 %1 点移除 %2 点（%3%）")
                         .arg(inp).arg(rem)
                         .arg(100.0 * rem / inp, 0, 'f', 1);
    }
    m_statsLabel->setText(parts.join(QStringLiteral("  |  ")));
}

void SparseCloudPostProcessDialog::applyPendingSourceSelection()
{
    if (!m_sourceCombo || m_pendingSourceIdx < 0)
    {
        return;
    }
    for (int i = 0; i < m_sourceCombo->count(); ++i)
    {
        if (m_sourceCombo->itemData(i).toInt() == m_pendingSourceIdx)
        {
            m_sourceCombo->setCurrentIndex(i);
            m_pendingSourceIdx = -1;
            return;
        }
    }
}

QJsonObject SparseCloudPostProcessDialog::collectSettings() const
{
    const bool enableRefine  = m_refineGroup->isChecked();
    const bool enableSpatial = m_spatialGroup->isChecked();

    // 根据启用的步骤推导 mode，供 controller 路由
    QString mode;
    if (enableRefine)
    {
        mode = QStringLiteral("refine");
    }
    else if (enableSpatial)
    {
        mode = QStringLiteral("spatial_cleanup");
    }
    else
    {
        mode = QStringLiteral("outlier_removal");
    }

    QJsonObject s;
    s[QStringLiteral("sourceAtIndex")] = (m_sourceCombo && m_sourceCombo->currentIndex() >= 0)
                                             ? m_sourceCombo->currentData().toInt()
                                             : -1;
    s[QStringLiteral("mode")] = mode;

    // 点级滤波（所有后端通用）
    s[QStringLiteral("filterByReprojError")]  = m_reprojCheck->isChecked();
    s[QStringLiteral("maxReprojError")]       = m_reprojSpin->value();
    s[QStringLiteral("filterByTrackLen")]     = m_trackCheck->isChecked();
    s[QStringLiteral("minTrackLen")]          = m_trackSpin->value();
    s[QStringLiteral("filterByTriAngle")]     = m_angleCheck->isChecked();
    s[QStringLiteral("minTriAngleDeg")]       = m_angleSpin->value();
    s[QStringLiteral("filterByStatistical")]  = m_statCheck->isChecked();
    s[QStringLiteral("statK")]                = m_statKSpin->value();
    s[QStringLiteral("statStdDevMul")]        = m_statStdSpin->value();
    s[QStringLiteral("filterByDensity")]      = m_densityCheck->isChecked();
    s[QStringLiteral("densityRadius")]        = m_densityRadiusSpin->value();
    s[QStringLiteral("densityMinNeighbors")]  = m_densityMinNbSpin->value();

    // refine 后端需要的别名键
    s[QStringLiteral("knnNeighbors")]     = m_statKSpin->value();
    s[QStringLiteral("stdDevMultiplier")] = m_statStdSpin->value();
    s[QStringLiteral("minAngle")]         = m_angleSpin->value();

    // 迭代精修参数
    s[QStringLiteral("enableRefine")]      = enableRefine;
    s[QStringLiteral("iterRounds")]        = m_iterRoundsSpin->value();
    s[QStringLiteral("retriangulate")]     = m_retriangCheck->isChecked();
    s[QStringLiteral("normalConsistency")] = m_normalConsCheck->isChecked();
    s[QStringLiteral("threads")]           = m_threadsSpin->value();

    // 空间清理参数
    s[QStringLiteral("enableSpatialCleanup")] = enableSpatial;
    s[QStringLiteral("voxelSize")]            = m_voxelSizeSpin->value();
    s[QStringLiteral("minVoxelPoints")]       = m_minVoxelPtsSpin->value();
    s[QStringLiteral("localReprojFilter")]    = m_localReprojCheck->isChecked();
    s[QStringLiteral("localReprojStdMul")]    = m_reprojStdMulSpin->value();
    s[QStringLiteral("deduplicationRadius")]  = m_dedupRadiusSpin->value();

    return s;
}

void SparseCloudPostProcessDialog::onAnyChanged()
{
    emit settingsChanged(collectSettings());
}

void SparseCloudPostProcessDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
