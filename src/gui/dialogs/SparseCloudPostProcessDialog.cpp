#include "SparseCloudPostProcessDialog.h"
#include "ui_SparseCloudPostProcessDialog.h"

#include "project/SparseResultQuality.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
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

    _sourceModeCombo = new QComboBox(this);
    _sourceModeCombo->setObjectName(QStringLiteral("m_sourceModeCombo"));
    _sourceModeCombo->addItem(tr("项目结果"), QStringLiteral("project_result"));
    _sourceModeCombo->addItem(tr("外部 PLY 点云"), QStringLiteral("external_ply"));

    _sourceCombo = ui.m_sourceCombo;
    auto *sourceLabel = findChild<QLabel *>(QStringLiteral("sourceLabel"));
    if (sourceLabel)
    {
        sourceLabel->setText(tr("项目结果:"));
    }

    _externalPathEdit = new QLineEdit(this);
    _externalPathEdit->setObjectName(QStringLiteral("m_externalPathEdit"));
    _externalPathEdit->setPlaceholderText(tr("选择 .ply 稀疏点云文件"));
    _externalPathEdit->setToolTip(tr("外部 PLY 不包含 BA 质量字段时，仅执行几何/空间类过滤。"));
    _browseExternalButton = new QPushButton(tr("浏览..."), this);
    _browseExternalButton->setObjectName(QStringLiteral("m_browseExternalButton"));
    auto *externalPathWidget = new QWidget(this);
    auto *externalPathLayout = new QHBoxLayout(externalPathWidget);
    externalPathLayout->setContentsMargins(0, 0, 0, 0);
    externalPathLayout->addWidget(_externalPathEdit, 1);
    externalPathLayout->addWidget(_browseExternalButton);

    if (auto *sourceForm = findChild<QFormLayout *>(QStringLiteral("sourceForm")))
    {
        sourceForm->insertRow(0, tr("输入来源:"), _sourceModeCombo);
        sourceForm->insertRow(2, tr("外部 PLY:"), externalPathWidget);
    }

    _statsLabel = ui.m_statsLabel;
    _refineGroup = ui.m_refineGroup;
    _spatialGroup = ui.m_spatialGroup;
    _reprojCheck = ui.m_reprojCheck;
    _reprojSpin = ui.m_reprojSpin;
    _trackCheck = ui.m_trackCheck;
    _trackSpin = ui.m_trackSpin;
    _angleCheck = ui.m_angleCheck;
    _angleSpin = ui.m_angleSpin;
    _statCheck = ui.m_statCheck;
    _statKSpin = ui.m_statKSpin;
    _statStdSpin = ui.m_statStdSpin;
    _densityCheck = ui.m_densityCheck;
    _densityRadiusSpin = ui.m_densityRadiusSpin;
    _densityMinNbSpin = ui.m_densityMinNbSpin;
    _iterRoundsSpin = ui.m_iterRoundsSpin;
    _retriangCheck = ui.m_retriangCheck;
    _normalConsCheck = ui.m_normalConsCheck;
    _threadsSpin = ui.m_threadsSpin;
    _voxelSizeSpin = ui.m_voxelSizeSpin;
    _minVoxelPtsSpin = ui.m_minVoxelPtsSpin;
    _localReprojCheck = ui.m_localReprojCheck;
    _reprojStdMulSpin = ui.m_reprojStdMulSpin;
    _dedupRadiusSpin = ui.m_dedupRadiusSpin;

    connect(_reprojCheck, &QCheckBox::toggled, _reprojSpin, &QWidget::setEnabled);
    connect(_trackCheck, &QCheckBox::toggled, _trackSpin, &QWidget::setEnabled);
    connect(_angleCheck, &QCheckBox::toggled, _angleSpin, &QWidget::setEnabled);
    connect(_statCheck, &QCheckBox::toggled, _statKSpin, &QWidget::setEnabled);
    connect(_statCheck, &QCheckBox::toggled, _statStdSpin, &QWidget::setEnabled);
    connect(_densityCheck, &QCheckBox::toggled, _densityRadiusSpin, &QWidget::setEnabled);
    connect(_densityCheck, &QCheckBox::toggled, _densityMinNbSpin, &QWidget::setEnabled);
    connect(_localReprojCheck, &QCheckBox::toggled, _reprojStdMulSpin, &QWidget::setEnabled);

    _reprojSpin->setEnabled(_reprojCheck->isChecked());
    _trackSpin->setEnabled(_trackCheck->isChecked());
    _angleSpin->setEnabled(_angleCheck->isChecked());
    _statKSpin->setEnabled(_statCheck->isChecked());
    _statStdSpin->setEnabled(_statCheck->isChecked());
    _densityRadiusSpin->setEnabled(_densityCheck->isChecked());
    _densityMinNbSpin->setEnabled(_densityCheck->isChecked());
    _reprojStdMulSpin->setEnabled(_localReprojCheck->isChecked());

    _runButton = ui.buttonBox->button(QDialogButtonBox::Ok);
    _runButton->setText(tr("运行后处理"));

    auto changed = [this]() { onAnyChanged(); };

    connect(_sourceCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SparseCloudPostProcessDialog::updateStatsLabel);
    connect(_sourceModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SparseCloudPostProcessDialog::updateSourceModeUi);
    connect(_externalPathEdit, &QLineEdit::textChanged,
            this, &SparseCloudPostProcessDialog::updateRunButtonState);
    connect(_browseExternalButton, &QPushButton::clicked,
            this, &SparseCloudPostProcessDialog::browseExternalPly);
    connect(_sourceCombo,       QOverload<int>::of(&QComboBox::currentIndexChanged),    this, changed);
    connect(_sourceModeCombo,   QOverload<int>::of(&QComboBox::currentIndexChanged),    this, changed);
    connect(_externalPathEdit,  &QLineEdit::textChanged,                                this, changed);
    connect(_reprojCheck,       &QCheckBox::toggled,                                    this, changed);
    connect(_reprojSpin,        QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_trackCheck,        &QCheckBox::toggled,                                    this, changed);
    connect(_trackSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_angleCheck,        &QCheckBox::toggled,                                    this, changed);
    connect(_angleSpin,         QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_statCheck,         &QCheckBox::toggled,                                    this, changed);
    connect(_statKSpin,         QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_statStdSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_densityCheck,      &QCheckBox::toggled,                                    this, changed);
    connect(_densityRadiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_densityMinNbSpin,  QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_refineGroup,       &QGroupBox::toggled,                                    this, changed);
    connect(_iterRoundsSpin,    QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_retriangCheck,     &QCheckBox::toggled,                                    this, changed);
    connect(_normalConsCheck,   &QCheckBox::toggled,                                    this, changed);
    connect(_threadsSpin,       QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_spatialGroup,      &QGroupBox::toggled,                                    this, changed);
    connect(_voxelSizeSpin,     QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_minVoxelPtsSpin,   QOverload<int>::of(&QSpinBox::valueChanged),            this, changed);
    connect(_localReprojCheck,  &QCheckBox::toggled,                                    this, changed);
    connect(_reprojStdMulSpin,  QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);
    connect(_dedupRadiusSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged),   this, changed);

    connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &SparseCloudPostProcessDialog::onRun);
    connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    updateSourceModeUi();
}

// ---------------------------------------------------------------------------
// 公有接口
// ---------------------------------------------------------------------------

void SparseCloudPostProcessDialog::setAvailableSparseClouds(const QJsonArray &results)
{
    if (!_sourceCombo)
    {
        return;
    }

    _programmaticUpdate = true;
    QJsonArray filteredResults;
    _sourceCombo->clear();
    for (const QJsonValue &value : results)
    {
        const QJsonObject record = value.toObject();
        if (!xjw::gui::project::isProductionSparseResult(record))
        {
            continue;
        }

        filteredResults.append(record);
        const QString label = record.value(QStringLiteral("display_name")).toString(
            QFileInfo(record.value(QStringLiteral("sparse_cloud_xyz")).toString()).fileName());
        _sourceCombo->addItem(label, record.value(QStringLiteral("index")).toInt(-1));
    }
    _availableResults = filteredResults;

    applyPendingSourceSelection();
    if (_sourceCombo->count() > 0 && _sourceCombo->currentIndex() < 0)
    {
        _sourceCombo->setCurrentIndex(_sourceCombo->count() - 1);
    }
    _programmaticUpdate = false;
    updateStatsLabel();
    updateRunButtonState();
}

void SparseCloudPostProcessDialog::applySettings(const QJsonObject &settings)
{
    _programmaticUpdate = true;
    const QString sourceKind = settings.value(QStringLiteral("sourceKind")).toString();
    if (!sourceKind.isEmpty() && _sourceModeCombo)
    {
        const int modeIndex = _sourceModeCombo->findData(sourceKind);
        if (modeIndex >= 0)
        {
            _sourceModeCombo->setCurrentIndex(modeIndex);
        }
    }
    if (_externalPathEdit && settings.contains(QStringLiteral("externalSparseCloudPath")))
    {
        _externalPathEdit->setText(settings.value(QStringLiteral("externalSparseCloudPath")).toString());
    }

    _pendingSourceIdx = settings.value(QStringLiteral("sourceAtIndex")).toInt(-1);
    applyPendingSourceSelection();

    if (settings.contains(QStringLiteral("filterByReprojError")))
        _reprojCheck->setChecked(settings.value(QStringLiteral("filterByReprojError")).toBool());
    if (settings.contains(QStringLiteral("maxReprojError")))
        _reprojSpin->setValue(settings.value(QStringLiteral("maxReprojError")).toDouble());

    if (settings.contains(QStringLiteral("filterByTrackLen")))
        _trackCheck->setChecked(settings.value(QStringLiteral("filterByTrackLen")).toBool());
    if (settings.contains(QStringLiteral("minTrackLen")))
        _trackSpin->setValue(settings.value(QStringLiteral("minTrackLen")).toInt());

    if (settings.contains(QStringLiteral("filterByTriAngle")))
        _angleCheck->setChecked(settings.value(QStringLiteral("filterByTriAngle")).toBool());
    if (settings.contains(QStringLiteral("minTriAngleDeg")))
        _angleSpin->setValue(settings.value(QStringLiteral("minTriAngleDeg")).toDouble());

    if (settings.contains(QStringLiteral("filterByStatistical")))
        _statCheck->setChecked(settings.value(QStringLiteral("filterByStatistical")).toBool());
    if (settings.contains(QStringLiteral("statK")))
        _statKSpin->setValue(settings.value(QStringLiteral("statK")).toInt());
    if (settings.contains(QStringLiteral("statStdDevMul")))
        _statStdSpin->setValue(settings.value(QStringLiteral("statStdDevMul")).toDouble());

    if (settings.contains(QStringLiteral("filterByDensity")))
        _densityCheck->setChecked(settings.value(QStringLiteral("filterByDensity")).toBool());
    if (settings.contains(QStringLiteral("densityRadius")))
        _densityRadiusSpin->setValue(settings.value(QStringLiteral("densityRadius")).toDouble());
    if (settings.contains(QStringLiteral("densityMinNeighbors")))
        _densityMinNbSpin->setValue(settings.value(QStringLiteral("densityMinNeighbors")).toInt());

    if (settings.contains(QStringLiteral("enableRefine")))
        _refineGroup->setChecked(settings.value(QStringLiteral("enableRefine")).toBool());
    if (settings.contains(QStringLiteral("iterRounds")))
        _iterRoundsSpin->setValue(settings.value(QStringLiteral("iterRounds")).toInt());
    if (settings.contains(QStringLiteral("retriangulate")))
        _retriangCheck->setChecked(settings.value(QStringLiteral("retriangulate")).toBool());
    if (settings.contains(QStringLiteral("normalConsistency")))
        _normalConsCheck->setChecked(settings.value(QStringLiteral("normalConsistency")).toBool());
    if (settings.contains(QStringLiteral("threads")))
        _threadsSpin->setValue(settings.value(QStringLiteral("threads")).toInt());

    if (settings.contains(QStringLiteral("enableSpatialCleanup")))
        _spatialGroup->setChecked(settings.value(QStringLiteral("enableSpatialCleanup")).toBool());
    if (settings.contains(QStringLiteral("voxelSize")))
        _voxelSizeSpin->setValue(settings.value(QStringLiteral("voxelSize")).toDouble());
    if (settings.contains(QStringLiteral("minVoxelPoints")))
        _minVoxelPtsSpin->setValue(settings.value(QStringLiteral("minVoxelPoints")).toInt());
    if (settings.contains(QStringLiteral("localReprojFilter")))
        _localReprojCheck->setChecked(settings.value(QStringLiteral("localReprojFilter")).toBool());
    if (settings.contains(QStringLiteral("localReprojStdMul")))
        _reprojStdMulSpin->setValue(settings.value(QStringLiteral("localReprojStdMul")).toDouble());
    if (settings.contains(QStringLiteral("deduplicationRadius")))
        _dedupRadiusSpin->setValue(settings.value(QStringLiteral("deduplicationRadius")).toDouble());

    _programmaticUpdate = false;
    updateSourceModeUi();
}

// ---------------------------------------------------------------------------
// 私有
// ---------------------------------------------------------------------------

void SparseCloudPostProcessDialog::updateStatsLabel()
{
    if (!_statsLabel)
        return;
    if (usingExternalPly())
    {
        const QString path = _externalPathEdit ? _externalPathEdit->text().trimmed() : QString();
        _statsLabel->setText(path.isEmpty()
            ? tr("外部 PLY 将按纯几何点云处理，不使用 BA 重投影误差、轨迹长度和三角化角度。")
            : tr("外部 PLY: %1").arg(path));
        return;
    }

    const int comboIdx = _sourceCombo ? _sourceCombo->currentIndex() : -1;
    if (comboIdx < 0 || comboIdx >= _availableResults.size())
    {
        _statsLabel->clear();
        return;
    }
    const QJsonObject item = _availableResults.at(comboIdx).toObject();
    int pts = item.value(QStringLiteral("sparse_point_count")).toInt(0);
    if (pts <= 0)
    {
        pts = item.value(QStringLiteral("point_count")).toInt(
            item.value(QStringLiteral("quality")).toObject().value(QStringLiteral("point_count")).toInt(0));
    }
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
    _statsLabel->setText(parts.join(QStringLiteral("  |  ")));
}

bool SparseCloudPostProcessDialog::usingExternalPly() const
{
    return _sourceModeCombo &&
           _sourceModeCombo->currentData().toString() == QLatin1String("external_ply");
}

void SparseCloudPostProcessDialog::updateRunButtonState()
{
    if (!_runButton)
    {
        return;
    }

    if (usingExternalPly())
    {
        const QString path = _externalPathEdit ? _externalPathEdit->text().trimmed() : QString();
        _runButton->setEnabled(!path.isEmpty() && QFileInfo::exists(path));
    }
    else
    {
        _runButton->setEnabled(_sourceCombo && _sourceCombo->count() > 0);
    }
}

void SparseCloudPostProcessDialog::updateSourceModeUi()
{
    const bool external = usingExternalPly();
    if (_sourceCombo)
    {
        _sourceCombo->setEnabled(!external);
    }
    if (_externalPathEdit)
    {
        _externalPathEdit->setEnabled(external);
    }
    if (_browseExternalButton)
    {
        _browseExternalButton->setEnabled(external);
    }

    const bool hasQualityMetrics = !external;
    if (_reprojCheck) _reprojCheck->setEnabled(hasQualityMetrics);
    if (_reprojSpin) _reprojSpin->setEnabled(hasQualityMetrics && _reprojCheck->isChecked());
    if (_trackCheck) _trackCheck->setEnabled(hasQualityMetrics);
    if (_trackSpin) _trackSpin->setEnabled(hasQualityMetrics && _trackCheck->isChecked());
    if (_angleCheck) _angleCheck->setEnabled(hasQualityMetrics);
    if (_angleSpin) _angleSpin->setEnabled(hasQualityMetrics && _angleCheck->isChecked());
    if (_localReprojCheck) _localReprojCheck->setEnabled(hasQualityMetrics);
    if (_reprojStdMulSpin) _reprojStdMulSpin->setEnabled(hasQualityMetrics && _localReprojCheck->isChecked());

    updateStatsLabel();
    updateRunButtonState();
}

void SparseCloudPostProcessDialog::browseExternalPly()
{
    const QString startDir = QFileInfo(_externalPathEdit ? _externalPathEdit->text() : QString()).absolutePath();
    const QString path = QFileDialog::getOpenFileName(this,
                                                      tr("选择外部 PLY 稀疏点云"),
                                                      startDir,
                                                      tr("PLY 点云 (*.ply);;所有文件 (*.*)"));
    if (!path.isEmpty() && _externalPathEdit)
    {
        _externalPathEdit->setText(QDir::cleanPath(path));
    }
}

void SparseCloudPostProcessDialog::applyPendingSourceSelection()
{
    if (!_sourceCombo || _pendingSourceIdx < 0)
    {
        return;
    }
    for (int i = 0; i < _sourceCombo->count(); ++i)
    {
        if (_sourceCombo->itemData(i).toInt() == _pendingSourceIdx)
        {
            _sourceCombo->setCurrentIndex(i);
            _pendingSourceIdx = -1;
            return;
        }
    }
}

QJsonObject SparseCloudPostProcessDialog::collectSettings() const
{
    const bool enableRefine  = _refineGroup->isChecked();
    const bool enableSpatial = _spatialGroup->isChecked();

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
    const bool external = usingExternalPly();
    const bool hasQualityMetrics = !external;
    s[QStringLiteral("sourceKind")] = external ? QStringLiteral("external_ply")
                                               : QStringLiteral("project_result");
    s[QStringLiteral("externalSparseCloudPath")] =
        (external && _externalPathEdit)
            ? QDir::cleanPath(_externalPathEdit->text().trimmed())
            : QString();
    s[QStringLiteral("sourceAtIndex")] = (_sourceCombo && _sourceCombo->currentIndex() >= 0)
                                             ? (external ? -1 : _sourceCombo->currentData().toInt())
                                             : -1;
    s[QStringLiteral("mode")] = mode;

    // 点级滤波（所有后端通用）
    s[QStringLiteral("filterByReprojError")]  = hasQualityMetrics && _reprojCheck->isChecked();
    s[QStringLiteral("maxReprojError")]       = _reprojSpin->value();
    s[QStringLiteral("filterByTrackLen")]     = hasQualityMetrics && _trackCheck->isChecked();
    s[QStringLiteral("minTrackLen")]          = _trackSpin->value();
    s[QStringLiteral("filterByTriAngle")]     = hasQualityMetrics && _angleCheck->isChecked();
    s[QStringLiteral("minTriAngleDeg")]       = _angleSpin->value();
    s[QStringLiteral("filterByStatistical")]  = _statCheck->isChecked();
    s[QStringLiteral("statK")]                = _statKSpin->value();
    s[QStringLiteral("statStdDevMul")]        = _statStdSpin->value();
    s[QStringLiteral("filterByDensity")]      = _densityCheck->isChecked();
    s[QStringLiteral("densityRadius")]        = _densityRadiusSpin->value();
    s[QStringLiteral("densityMinNeighbors")]  = _densityMinNbSpin->value();

    // refine 后端需要的别名键
    s[QStringLiteral("knnNeighbors")]     = _statKSpin->value();
    s[QStringLiteral("stdDevMultiplier")] = _statStdSpin->value();
    s[QStringLiteral("minAngle")]         = _angleSpin->value();

    // 迭代精修参数
    s[QStringLiteral("enableRefine")]      = enableRefine;
    s[QStringLiteral("iterRounds")]        = _iterRoundsSpin->value();
    s[QStringLiteral("retriangulate")]     = _retriangCheck->isChecked();
    s[QStringLiteral("normalConsistency")] = _normalConsCheck->isChecked();
    s[QStringLiteral("threads")]           = _threadsSpin->value();

    // 空间清理参数
    s[QStringLiteral("enableSpatialCleanup")] = enableSpatial;
    s[QStringLiteral("voxelSize")]            = _voxelSizeSpin->value();
    s[QStringLiteral("minVoxelPoints")]       = _minVoxelPtsSpin->value();
    s[QStringLiteral("localReprojFilter")]    = hasQualityMetrics && _localReprojCheck->isChecked();
    s[QStringLiteral("localReprojStdMul")]    = _reprojStdMulSpin->value();
    s[QStringLiteral("deduplicationRadius")]  = _dedupRadiusSpin->value();

    return s;
}

void SparseCloudPostProcessDialog::onAnyChanged()
{
    if (_programmaticUpdate)
    {
        return;
    }
    emit settingsChanged(collectSettings());
}

void SparseCloudPostProcessDialog::onRun()
{
    emit runRequested(collectSettings());
    accept();
}
