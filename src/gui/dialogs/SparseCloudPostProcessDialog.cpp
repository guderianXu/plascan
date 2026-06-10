#include "SparseCloudPostProcessDialog.h"
#include "ui_SparseCloudPostProcessDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileInfo>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>

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

    Ui::SparseCloudPostProcessDialog ui;
    ui.setupUi(this);

    m_sourceCombo = ui.m_sourceCombo;
    m_statsLabel = ui.m_statsLabel;
    m_refineGroup = ui.m_refineGroup;
    m_spatialGroup = ui.m_spatialGroup;
    m_reprojCheck = ui.m_reprojCheck;
    m_reprojSpin = ui.m_reprojSpin;
    m_trackCheck = ui.m_trackCheck;
    m_trackSpin = ui.m_trackSpin;
    m_angleCheck = ui.m_angleCheck;
    m_angleSpin = ui.m_angleSpin;
    m_statCheck = ui.m_statCheck;
    m_statKSpin = ui.m_statKSpin;
    m_statStdSpin = ui.m_statStdSpin;
    m_densityCheck = ui.m_densityCheck;
    m_densityRadiusSpin = ui.m_densityRadiusSpin;
    m_densityMinNbSpin = ui.m_densityMinNbSpin;
    m_iterRoundsSpin = ui.m_iterRoundsSpin;
    m_retriangCheck = ui.m_retriangCheck;
    m_normalConsCheck = ui.m_normalConsCheck;
    m_threadsSpin = ui.m_threadsSpin;
    m_voxelSizeSpin = ui.m_voxelSizeSpin;
    m_minVoxelPtsSpin = ui.m_minVoxelPtsSpin;
    m_localReprojCheck = ui.m_localReprojCheck;
    m_reprojStdMulSpin = ui.m_reprojStdMulSpin;
    m_dedupRadiusSpin = ui.m_dedupRadiusSpin;

    connect(m_reprojCheck, &QCheckBox::toggled, m_reprojSpin, &QWidget::setEnabled);
    connect(m_trackCheck, &QCheckBox::toggled, m_trackSpin, &QWidget::setEnabled);
    connect(m_angleCheck, &QCheckBox::toggled, m_angleSpin, &QWidget::setEnabled);
    connect(m_statCheck, &QCheckBox::toggled, m_statKSpin, &QWidget::setEnabled);
    connect(m_statCheck, &QCheckBox::toggled, m_statStdSpin, &QWidget::setEnabled);
    connect(m_densityCheck, &QCheckBox::toggled, m_densityRadiusSpin, &QWidget::setEnabled);
    connect(m_densityCheck, &QCheckBox::toggled, m_densityMinNbSpin, &QWidget::setEnabled);
    connect(m_localReprojCheck, &QCheckBox::toggled, m_reprojStdMulSpin, &QWidget::setEnabled);

    m_reprojSpin->setEnabled(m_reprojCheck->isChecked());
    m_trackSpin->setEnabled(m_trackCheck->isChecked());
    m_angleSpin->setEnabled(m_angleCheck->isChecked());
    m_statKSpin->setEnabled(m_statCheck->isChecked());
    m_statStdSpin->setEnabled(m_statCheck->isChecked());
    m_densityRadiusSpin->setEnabled(m_densityCheck->isChecked());
    m_densityMinNbSpin->setEnabled(m_densityCheck->isChecked());
    m_reprojStdMulSpin->setEnabled(m_localReprojCheck->isChecked());

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
